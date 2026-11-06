use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

static ACTIVE_SANDBOX_PGIDS: OnceLock<Mutex<Vec<(i32, String)>>> = OnceLock::new();

fn active_pgids() -> &'static Mutex<Vec<(i32, String)>> {
    ACTIVE_SANDBOX_PGIDS.get_or_init(|| Mutex::new(Vec::new()))
}

fn register_active_pgid(pgid: i32, workspace: &Path) {
    if pgid <= 1 {
        return;
    }
    if let Ok(mut guard) = active_pgids().lock() {
        let ws = workspace.display().to_string();
        if !guard.iter().any(|(p, _)| *p == pgid) {
            guard.push((pgid, ws));
        }
    }
}

fn unregister_active_pgid(pgid: i32) {
    if let Ok(mut guard) = active_pgids().lock() {
        guard.retain(|(p, _)| *p != pgid);
    }
}

/// Discovers all live (non-zombie) PIDs on the host whose process group ID (`pgrp`)
/// or parent PID (`ppid`) matches `target_pgid`.
pub fn list_live_pids_in_process_group(target_pgid: i32) -> Vec<i32> {
    if target_pgid <= 1 {
        return Vec::new();
    }
    let mut live_pids = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return live_pids;
    };
    for entry in entries.flatten() {
        let fname = entry.file_name();
        let Some(name_str) = fname.to_str() else {
            continue;
        };
        let Ok(pid) = name_str.parse::<i32>() else {
            continue;
        };
        let stat_path = entry.path().join("stat");
        let Ok(stat_str) = std::fs::read_to_string(stat_path) else {
            continue;
        };
        // Format: pid (comm) state ppid pgrp session ...
        let Some(rparen) = stat_str.rfind(')') else {
            continue;
        };
        let rest = stat_str[rparen + 1..].trim();
        let fields: Vec<&str> = rest.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let state = fields[0];
        if state == "Z" || state == "X" {
            // Zombie or dead process is already terminated and only awaiting parent wait()
            continue;
        }
        let ppid = fields[1].parse::<i32>().unwrap_or(-1);
        let pgrp = fields[2].parse::<i32>().unwrap_or(-1);
        if pid == target_pgid || pgrp == target_pgid || ppid == target_pgid {
            live_pids.push(pid);
        }
    }
    live_pids
}

/// Explicitly sends SIGKILL to the entire sandbox process group (`-pgid`) and all descendant
/// PIDs, and confirms exit before returning.
pub fn terminate_and_confirm_process_group(pgid: i32) {
    if pgid <= 1 {
        return;
    }
    let send_kills = |target: i32| {
        let descendants = list_live_pids_in_process_group(target);
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(target),
            nix::sys::signal::Signal::SIGKILL,
        );
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(-target),
            nix::sys::signal::Signal::SIGKILL,
        );
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(target),
            nix::sys::signal::Signal::SIGKILL,
        );
        for dpid in descendants {
            let _ = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(dpid),
                nix::sys::signal::Signal::SIGKILL,
            );
        }
    };

    send_kills(pgid);
    for _ in 0..100 {
        let remaining = list_live_pids_in_process_group(pgid);
        if remaining.is_empty() {
            break;
        }
        send_kills(pgid);
        std::thread::sleep(Duration::from_millis(10));
    }
    unregister_active_pgid(pgid);
}

/// Terminates and confirms exit for all active sandbox process groups matching a workspace prefix
/// (or all active sandbox process groups if `workspace_prefix` is None).
pub fn terminate_all_active_sandbox_process_groups(workspace_prefix: Option<&Path>) -> usize {
    let targets: Vec<i32> = if let Ok(guard) = active_pgids().lock() {
        guard
            .iter()
            .filter(|(_, ws)| match workspace_prefix {
                Some(pfx) => ws.starts_with(&pfx.display().to_string()),
                None => true,
            })
            .map(|(p, _)| *p)
            .collect()
    } else {
        Vec::new()
    };
    let count = targets.len();
    for pgid in targets {
        terminate_and_confirm_process_group(pgid);
    }
    count
}

struct SandboxProcessGroupGuard {
    pgid: Option<i32>,
    armed: bool,
}

impl SandboxProcessGroupGuard {
    fn new(pgid: Option<i32>) -> Self {
        Self { pgid, armed: true }
    }

    fn disarm(&mut self) {
        if let Some(p) = self.pgid {
            unregister_active_pgid(p);
        }
        self.armed = false;
    }
}

impl Drop for SandboxProcessGroupGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Some(pgid) = self.pgid {
                terminate_and_confirm_process_group(pgid);
            }
        }
    }
}

#[async_trait]
pub trait ExecutionAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn status_summary(&self) -> String;

    async fn execute_isolated(
        &self,
        command: &str,
        args: &[&str],
        stdin: Option<&str>,
        workspace_dir: &Path,
        timeout: Duration,
    ) -> Result<ExecutionResult>;
}

/// Linux Bubblewrap (bwrap) isolation adapter
pub struct BubblewrapAdapter {
    bwrap_path: Option<PathBuf>,
}

impl BubblewrapAdapter {
    pub fn probe() -> Self {
        let bwrap_path = which::which("bwrap").ok();
        Self { bwrap_path }
    }
}

#[async_trait]
impl ExecutionAdapter for BubblewrapAdapter {
    fn name(&self) -> &str {
        "bwrap"
    }

    fn is_available(&self) -> bool {
        self.bwrap_path.is_some()
    }

    fn status_summary(&self) -> String {
        if let Some(ref p) = self.bwrap_path {
            format!("Available at {}", p.display())
        } else {
            "Unavailable (bwrap binary not found)".to_string()
        }
    }

    async fn execute_isolated(
        &self,
        command: &str,
        args: &[&str],
        stdin: Option<&str>,
        workspace_dir: &Path,
        timeout: Duration,
    ) -> Result<ExecutionResult> {
        let bwrap = match self.bwrap_path {
            Some(ref p) => p,
            None => bail!("Bubblewrap isolation unavailable on this system"),
        };

        let start = std::time::Instant::now();
        let ws_str = workspace_dir.to_str().context("Invalid workspace path")?;
        let results_file_path = workspace_dir.join("results.json");
        let results_file_str = results_file_path
            .to_str()
            .unwrap_or("results.json")
            .to_string();

        let mut cmd = Command::new(bwrap);
        cmd.arg("--ro-bind")
            .arg("/")
            .arg("/")
            .arg("--dev")
            .arg("/dev")
            .arg("--proc")
            .arg("/proc")
            .arg("--tmpfs")
            .arg("/tmp")
            .arg("--tmpfs")
            .arg("/home/cgegout/.cache");

        // If `command` is an executable path under `/tmp` outside `workspace_dir` (e.g. in a test tempdir),
        // bind its parent directory read-only so `--tmpfs /tmp` does not mask the compiled binary.
        let cmd_path = Path::new(command);
        if cmd_path.is_absolute() && command.starts_with("/tmp/") && !command.starts_with(ws_str) {
            if let Some(parent) = cmd_path.parent() {
                if parent.exists() {
                    if let Some(parent_str) = parent.to_str() {
                        cmd.arg("--ro-bind").arg(parent_str).arg(parent_str);
                    }
                }
            }
        }

        cmd.arg("--bind")
            .arg(ws_str)
            .arg(ws_str)
            .arg("--chdir")
            .arg(ws_str)
            .arg("--setenv")
            .arg("TRIAD_RESULTS_FILE")
            .arg(&results_file_str)
            .arg("--setenv")
            .arg("RAYON_NUM_THREADS")
            .arg("2")
            .arg("--setenv")
            .arg("OMP_NUM_THREADS")
            .arg("2")
            .arg("--unshare-all");

        // Only share network for cargo builds if needed; generated research tools run with
        // network unshared (isolated) and restricted capabilities.
        if command == "cargo" {
            cmd.arg("--share-net");
        }

        cmd.arg("--die-with-parent")
            .arg("--")
            .arg(command)
            .args(args);

        // Place bwrap and its sandboxed children into a dedicated host process group (PGID = child PID)
        cmd.process_group(0);
        cmd.kill_on_drop(true);
        cmd.stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn isolated command {}", command))?;

        let child_pgid = child.id().map(|id| id as i32);
        if let Some(pgid) = child_pgid {
            register_active_pgid(pgid, workspace_dir);
        }
        let mut pg_guard = SandboxProcessGroupGuard::new(child_pgid);

        if let Some(input) = stdin {
            if let Some(mut cin) = child.stdin.take() {
                let _ = cin.write_all(input.as_bytes()).await;
            }
        }

        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let stdout_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(ref mut out) = stdout_pipe {
                let _ = out.read_to_end(&mut buf).await;
            }
            buf
        });
        let stderr_task = tokio::spawn(async move {
            let mut buf = Vec::new();
            if let Some(ref mut err) = stderr_pipe {
                let _ = err.read_to_end(&mut buf).await;
            }
            buf
        });

        let wait_res = tokio::time::timeout(timeout, child.wait()).await;
        let elapsed = start.elapsed().as_millis() as u64;

        match wait_res {
            Ok(Ok(status)) => {
                // Ensure any lingering background grandchildren in the sandbox PGID are cleaned up
                if let Some(pgid) = child_pgid {
                    terminate_and_confirm_process_group(pgid);
                }
                pg_guard.disarm();
                let stdout_bytes = stdout_task.await.unwrap_or_default();
                let stderr_bytes = stderr_task.await.unwrap_or_default();
                let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
                let stderr = String::from_utf8_lossy(&stderr_bytes).to_string();
                Ok(ExecutionResult {
                    exit_code: status.code().unwrap_or(-1),
                    stdout,
                    stderr,
                    duration_ms: elapsed,
                })
            }
            Ok(Err(e)) => {
                if let Some(pgid) = child_pgid {
                    terminate_and_confirm_process_group(pgid);
                }
                pg_guard.disarm();
                stdout_task.abort();
                stderr_task.abort();
                bail!("Isolated command execution error: {}", e)
            }
            Err(_) => {
                // Timeout: explicitly kill the entire sandbox process group, reap child, and confirm exit
                let _ = child.start_kill();
                if let Some(pgid) = child_pgid {
                    terminate_and_confirm_process_group(pgid);
                }
                let _ = child.wait().await;
                if let Some(pgid) = child_pgid {
                    terminate_and_confirm_process_group(pgid);
                }
                pg_guard.disarm();
                stdout_task.abort();
                stderr_task.abort();
                bail!(
                    "SANDBOX_TIMED_OUT: Isolated command timed out after {:?} (sandbox process group {:?} killed and exit confirmed)",
                    timeout,
                    child_pgid
                )
            }
        }
    }
}

/// Antigravity CLI adapter (`agy`)
pub struct AntigravityAdapter {
    binary_path: Option<PathBuf>,
}

impl AntigravityAdapter {
    pub fn probe(configured_binary: &str) -> Self {
        let binary_path = if !configured_binary.trim().is_empty() {
            Some(PathBuf::from(configured_binary))
        } else {
            which::which("agy").ok().or_else(|| which::which("antigravity").ok())
        };
        Self { binary_path }
    }
}

#[async_trait]
impl ExecutionAdapter for AntigravityAdapter {
    fn name(&self) -> &str {
        "antigravity"
    }

    fn is_available(&self) -> bool {
        self.binary_path.is_some()
    }

    fn status_summary(&self) -> String {
        if let Some(ref p) = self.binary_path {
            format!("Available at {} (CLI runtime detected)", p.display())
        } else {
            "Unavailable (antigravity/agy binary not found in PATH or config)".to_string()
        }
    }

    async fn execute_isolated(
        &self,
        command: &str,
        args: &[&str],
        stdin: Option<&str>,
        workspace_dir: &Path,
        timeout: Duration,
    ) -> Result<ExecutionResult> {
        // Antigravity execution adapter routes through bubblewrap for strong process isolation
        let bwrap = BubblewrapAdapter::probe();
        if bwrap.is_available() {
            bwrap.execute_isolated(command, args, stdin, workspace_dir, timeout).await
        } else {
            bail!("Execution isolation failed: bubblewrap boundary is required for safe execution")
        }
    }
}

/// Simple which resolver helper
pub mod which {
    use std::path::PathBuf;

    pub fn which(name: &str) -> Result<PathBuf, ()> {
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path_var) {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_bubblewrap_execution() {
        let adapter = BubblewrapAdapter::probe();
        if !adapter.is_available() {
            eprintln!("bwrap not installed, skipping test");
            return;
        }

        let dir = tempdir().unwrap();
        let res = adapter.execute_isolated("echo", &["triad_bwrap_ok"], None, dir.path(), Duration::from_secs(5)).await.unwrap();
        assert_eq!(res.exit_code, 0);
        assert!(res.stdout.contains("triad_bwrap_ok"));
    }
}
