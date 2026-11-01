use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
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

        let mut cmd = Command::new(bwrap);
        cmd.arg("--ro-bind").arg("/").arg("/")
            .arg("--dev").arg("/dev")
            .arg("--proc").arg("/proc")
            .arg("--tmpfs").arg("/tmp")
            .arg("--tmpfs").arg("/home/cgegout/.cache")
            .arg("--bind").arg(ws_str).arg(ws_str)
            .arg("--chdir").arg(ws_str)
            .arg("--unshare-all")
            .arg("--share-net")
            .arg("--die-with-parent")
            .arg("--")
            .arg(command)
            .args(args);

        cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()
            .with_context(|| format!("Failed to spawn isolated command {}", command))?;

        if let Some(input) = stdin {
            if let Some(mut cin) = child.stdin.take() {
                cin.write_all(input.as_bytes()).await?;
            }
        }

        let output_res = tokio::time::timeout(timeout, child.wait_with_output()).await;
        let elapsed = start.elapsed().as_millis() as u64;

        match output_res {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(ExecutionResult {
                    exit_code: output.status.code().unwrap_or(-1),
                    stdout,
                    stderr,
                    duration_ms: elapsed,
                })
            }
            Ok(Err(e)) => bail!("Isolated command execution error: {}", e),
            Err(_) => bail!("Isolated command timed out after {:?}", timeout),
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
