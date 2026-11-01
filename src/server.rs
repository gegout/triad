use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use crate::ipc::{IpcRequest, IpcResponse};

pub fn default_socket_path() -> PathBuf {
    let uid = nix::unistd::getuid();
    let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/run/user/{}", uid)));

    if runtime_dir.exists() {
        runtime_dir.join("triad.sock")
    } else {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp"));
        home.join(".triad").join("triad.sock")
    }
}

pub struct SingleInstanceLock {
    lock_file: PathBuf,
}

impl SingleInstanceLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        let parent = path.parent().context("Invalid lock path")?;
        fs::create_dir_all(parent)?;
        if path.exists() {
            // Check if process is really alive
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(pid) = content.trim().parse::<i32>() {
                    // Send signal 0 to test
                    if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok() {
                        bail!("Triad daemon is already running (PID: {})", pid);
                    }
                }
            }
            let _ = fs::remove_file(path);
        }

        fs::write(path, std::process::id().to_string())?;
        Ok(Self {
            lock_file: path.to_path_buf(),
        })
    }
}

impl Drop for SingleInstanceLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_file);
    }
}

pub async fn send_ipc_request(req: &IpcRequest, sock_path: &Path) -> Result<IpcResponse> {
    if !sock_path.exists() {
        bail!("Triad daemon is not running (socket not found at {})", sock_path.display());
    }

    let mut stream = UnixStream::connect(sock_path).await
        .with_context(|| format!("Failed to connect to Triad socket at {}", sock_path.display()))?;

    let req_bytes = serde_json::to_vec(req)?;
    let len_prefix = (req_bytes.len() as u32).to_le_bytes();
    stream.write_all(&len_prefix).await?;
    stream.write_all(&req_bytes).await?;

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let resp_len = u32::from_le_bytes(len_buf) as usize;

    let mut resp_buf = vec![0u8; resp_len];
    stream.read_exact(&mut resp_buf).await?;

    let resp: IpcResponse = serde_json::from_slice(&resp_buf)?;
    Ok(resp)
}

pub async fn write_ipc_response(stream: &mut UnixStream, resp: &IpcResponse) -> Result<()> {
    let bytes = serde_json::to_vec(resp)?;
    let len_prefix = (bytes.len() as u32).to_le_bytes();
    stream.write_all(&len_prefix).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}

pub async fn read_ipc_request(stream: &mut UnixStream) -> Result<IpcRequest> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let req_len = u32::from_le_bytes(len_buf) as usize;

    let mut req_buf = vec![0u8; req_len];
    stream.read_exact(&mut req_buf).await?;

    let req: IpcRequest = serde_json::from_slice(&req_buf)?;
    Ok(req)
}
