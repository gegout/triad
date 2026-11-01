use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::execution::ExecutionAdapter;
use crate::store::Store;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProposal {
    pub name: String,
    pub description: String,
    pub schema_json: String,
    pub rust_source: String,
    pub test_code: String,
}

pub struct ToolManager {
    store: Store,
    workspace_dir: PathBuf,
    execution_adapter: Arc<dyn ExecutionAdapter>,
}

impl ToolManager {
    pub fn new(store: Store, workspace_dir: PathBuf, execution_adapter: Arc<dyn ExecutionAdapter>) -> Self {
        Self {
            store,
            workspace_dir,
            execution_adapter,
        }
    }

    /// Complete lifecycle:
    /// Contract -> Generate Source -> Compile -> Run Tests -> Endorse -> Register -> Execute
    pub async fn build_and_register_tool(
        &self,
        proposal: &ToolProposal,
        endorser_1: &str,
        endorser_2: &str,
    ) -> Result<i64> {
        let valid_agents = ["Explorer", "Sceptic", "Builder"];
        if !valid_agents.contains(&endorser_1) || !valid_agents.contains(&endorser_2) {
            bail!("Constitutional invariant violation: endorsers must be among Explorer, Sceptic, Builder");
        }
        if endorser_1 == endorser_2 {
            bail!("Constitutional invariant violation: tool endorsement requires two distinct agent identities");
        }

        if !self.execution_adapter.is_available() {
            bail!("Cannot build or register tools: execution isolation adapter is unavailable");
        }

        let tools_ws = self.workspace_dir.join("tools").join(&proposal.name);
        fs::create_dir_all(&tools_ws)
            .with_context(|| format!("Failed to create tool workspace at {}", tools_ws.display()))?;

        // Write Cargo.toml for standalone tool
        let cargo_toml = format!(r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
"#, proposal.name);

        fs::write(tools_ws.join("Cargo.toml"), cargo_toml)?;
        let src_dir = tools_ws.join("src");
        fs::create_dir_all(&src_dir)?;

        // Combine rust source with tests
        let full_source = format!("{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{}\n}}\n",
            proposal.rust_source, proposal.test_code);
        fs::write(src_dir.join("main.rs"), &full_source)?;

        // Compute source hash
        let mut hasher = Sha256::new();
        hasher.update(full_source.as_bytes());
        let hash = hasher.finalize();
        let source_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

        // Run cargo test in isolated container
        let test_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["test", "--quiet"],
            None,
            &tools_ws,
            Duration::from_secs(120),
        ).await;

        let test_passed = match test_res {
            Ok(ref res) => {
                if res.exit_code != 0 {
                    eprintln!("Tool test exited with code {}:\nSTDOUT:\n{}\nSTDERR:\n{}", res.exit_code, res.stdout, res.stderr);
                }
                res.exit_code == 0
            }
            Err(ref e) => {
                eprintln!("Tool test failed execution: {}", e);
                false
            }
        };

        if !test_passed {
            bail!("Tool tests failed; registration aborted. Repair required.");
        }

        // Build release binary
        let build_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["build", "--release", "--quiet"],
            None,
            &tools_ws,
            Duration::from_secs(120),
        ).await?;

        if build_res.exit_code != 0 {
            bail!("Tool compilation failed: {}", build_res.stderr);
        }

        let bin_path = tools_ws.join("target").join("release").join(&proposal.name);
        let bin_str = bin_path.to_str().unwrap_or_default();

        let next_version = match self.store.get_tool_by_name(&proposal.name)? {
            Some(existing) => existing.version + 1,
            None => 1,
        };

        let tool_id = self.store.register_tool(
            &proposal.name,
            next_version,
            &proposal.description,
            &proposal.schema_json,
            &full_source,
            bin_str,
            &source_hash,
            Some(endorser_1),
            Some(endorser_2),
            test_passed,
        )?;

        Ok(tool_id)
    }

    pub async fn execute_tool(&self, name: &str, input_json: &str) -> Result<String> {
        let tool = self.store.get_tool_by_name(name)?
            .context(format!("Tool '{}' not found in registry", name))?;

        if tool.status != "active" {
            bail!("Tool '{}' is not active (status: {})", name, tool.status);
        }

        let bin_path = Path::new(&tool.binary_path);
        if !bin_path.exists() {
            bail!("Tool binary does not exist at {}", tool.binary_path);
        }

        let start = std::time::Instant::now();
        let res = self.execution_adapter.execute_isolated(
            &tool.binary_path,
            &[],
            Some(input_json),
            &self.workspace_dir,
            Duration::from_secs(30),
        ).await;

        let elapsed = start.elapsed().as_millis() as u64;

        match res {
            Ok(out) => {
                self.store.record_tool_execution(
                    name,
                    tool.version,
                    input_json,
                    Some(&out.stdout),
                    if out.stderr.is_empty() { None } else { Some(&out.stderr) },
                    out.exit_code,
                    elapsed,
                )?;
                if out.exit_code == 0 {
                    Ok(out.stdout)
                } else {
                    bail!("Tool exited with code {}: {}", out.exit_code, out.stderr)
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                self.store.record_tool_execution(
                    name,
                    tool.version,
                    input_json,
                    None,
                    Some(&err_msg),
                    -1,
                    elapsed,
                )?;
                bail!("Tool execution error: {}", err_msg);
            }
        }
    }
}
