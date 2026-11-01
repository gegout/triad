use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_CONFIG_TOML: &str = r#"[application]
name = "triad"

[paths]
data_dir = "~/.triad/data"
log_dir = "~/.triad/logs"
workspace_dir = "~/.triad/workspace"

[models.chat]
base_url = "http://localhost:8366/v1"
model = "smollm3-3b"
max_tokens = 512
structured_output = "prompt"
no_think = true
concurrency = 6

[models.embedding]
base_url = "http://127.0.0.1:8348/v1"
model = "nomic-embed-text-v1-5"
query_prefix = "search_query: "
document_prefix = "search_document: "

[agents.explorer]
personality = "Discover promising research, unfamiliar ideas, and connections between disciplines. Propose ambitious but testable hypotheses."

[agents.sceptic]
personality = "Examine evidence, methodological weaknesses, conflicting results, and applicability to Triad. Design experiments that could disprove proposed improvements. Help refine uncertain ideas rather than rejecting them merely because they are unconventional."

[agents.builder]
personality = "Reproduce useful techniques, create tools and extensions, and measure their effects on real tasks. Turn promising hypotheses into working experiments."

[orchestration]
autonomous_learning = true
cycle_interval_seconds = 300
discussion_rounds = 2
max_model_requests_per_cycle = 24
max_tool_calls_per_cycle = 8
max_tool_builds_per_cycle = 1

[learning]
agreement_threshold = 2
max_user_questions_per_day = 3

[web]
enabled = true
wikipedia_api = "https://en.wikipedia.org/w/api.php"
max_pages_per_cycle = 3
request_timeout_seconds = 30

[execution]
backend = "antigravity"
runtime_binary = ""
command_timeout_seconds = 60

[logging]
level = "info"
format = "jsonl"
rotate_size_mb = 10
retained_files = 5

[configuration]
watch = true
reload_debounce_ms = 500
poll_interval_seconds = 2
"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppConfig {
    pub application: ApplicationSection,
    pub paths: PathsSection,
    pub models: ModelsSection,
    pub agents: AgentsSection,
    pub orchestration: OrchestrationSection,
    pub learning: LearningSection,
    pub web: WebSection,
    pub execution: ExecutionSection,
    pub logging: LoggingSection,
    pub configuration: ConfigurationSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApplicationSection {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PathsSection {
    pub data_dir: String,
    pub log_dir: String,
    pub workspace_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelsSection {
    pub chat: ChatModelSection,
    pub embedding: EmbeddingModelSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatModelSection {
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub structured_output: String,
    pub no_think: bool,
    pub concurrency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingModelSection {
    pub base_url: String,
    pub model: String,
    pub query_prefix: String,
    pub document_prefix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentsSection {
    pub explorer: AgentSection,
    pub sceptic: AgentSection,
    pub builder: AgentSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentSection {
    pub personality: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrchestrationSection {
    pub autonomous_learning: bool,
    pub cycle_interval_seconds: u64,
    pub discussion_rounds: usize,
    pub max_model_requests_per_cycle: usize,
    pub max_tool_calls_per_cycle: usize,
    pub max_tool_builds_per_cycle: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LearningSection {
    pub agreement_threshold: usize,
    pub max_user_questions_per_day: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSection {
    pub enabled: bool,
    pub wikipedia_api: String,
    pub max_pages_per_cycle: usize,
    pub request_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionSection {
    pub backend: String,
    pub runtime_binary: String,
    pub command_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoggingSection {
    pub level: String,
    pub format: String,
    pub rotate_size_mb: usize,
    pub retained_files: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConfigurationSection {
    pub watch: bool,
    pub reload_debounce_ms: u64,
    pub poll_interval_seconds: u64,
}

pub fn canonical_config_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root"));
    home.join(".triad").join("config.toml")
}

pub fn expand_tilde(p: &str) -> PathBuf {
    if p == "~" {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/root"))
    } else if let Some(stripped) = p.strip_prefix("~/") {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/root"));
        home.join(stripped)
    } else {
        PathBuf::from(p)
    }
}

pub fn compute_sha256(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

impl AppConfig {
    pub fn parse_and_validate(toml_str: &str) -> Result<Self> {
        let cfg: AppConfig = toml::from_str(toml_str)
            .context("Failed to parse TOML configuration")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.application.name.trim().is_empty() {
            bail!("application.name cannot be empty");
        }
        if self.models.chat.base_url.trim().is_empty() {
            bail!("models.chat.base_url cannot be empty");
        }
        if self.models.chat.model.trim().is_empty() {
            bail!("models.chat.model cannot be empty");
        }
        if self.models.chat.max_tokens == 0 {
            bail!("models.chat.max_tokens must be positive");
        }
        if self.models.chat.concurrency == 0 {
            bail!("models.chat.concurrency must be at least 1");
        }
        if self.models.embedding.base_url.trim().is_empty() {
            bail!("models.embedding.base_url cannot be empty");
        }
        if self.models.embedding.model.trim().is_empty() {
            bail!("models.embedding.model cannot be empty");
        }

        // Invariants: exactly 3 identities, threshold 2
        if self.agents.explorer.personality.trim().is_empty() {
            bail!("agents.explorer.personality cannot be empty");
        }
        if self.agents.sceptic.personality.trim().is_empty() {
            bail!("agents.sceptic.personality cannot be empty");
        }
        if self.agents.builder.personality.trim().is_empty() {
            bail!("agents.builder.personality cannot be empty");
        }

        if self.learning.agreement_threshold != 2 {
            bail!("learning.agreement_threshold must be exactly 2 as a product invariant");
        }

        if self.orchestration.cycle_interval_seconds == 0 {
            bail!("orchestration.cycle_interval_seconds must be positive");
        }
        if self.orchestration.discussion_rounds == 0 {
            bail!("orchestration.discussion_rounds must be at least 1");
        }
        if self.orchestration.max_model_requests_per_cycle == 0 {
            bail!("orchestration.max_model_requests_per_cycle must be positive");
        }

        if self.web.request_timeout_seconds == 0 {
            bail!("web.request_timeout_seconds must be positive");
        }
        if self.execution.command_timeout_seconds == 0 {
            bail!("execution.command_timeout_seconds must be positive");
        }
        if self.logging.rotate_size_mb == 0 || self.logging.retained_files == 0 {
            bail!("logging parameters rotate_size_mb and retained_files must be positive");
        }
        if self.configuration.poll_interval_seconds == 0 {
            bail!("configuration.poll_interval_seconds must be positive");
        }

        // Check non-conflicting paths
        let data_path = expand_tilde(&self.paths.data_dir);
        let log_path = expand_tilde(&self.paths.log_dir);
        let ws_path = expand_tilde(&self.paths.workspace_dir);

        if data_path == log_path || data_path == ws_path || log_path == ws_path {
            bail!("data_dir, log_dir, and workspace_dir must be distinct directories");
        }

        Ok(())
    }

    pub fn resolved_data_dir(&self, override_dir: Option<&Path>) -> PathBuf {
        if let Some(d) = override_dir {
            d.to_path_buf()
        } else {
            expand_tilde(&self.paths.data_dir)
        }
    }

    pub fn resolved_log_dir(&self) -> PathBuf {
        expand_tilde(&self.paths.log_dir)
    }

    pub fn resolved_workspace_dir(&self) -> PathBuf {
        expand_tilde(&self.paths.workspace_dir)
    }
}

/// Atomically ensures ~/.triad/config.toml exists. If not, writes DEFAULT_CONFIG_TOML with 0600 permissions.
pub fn ensure_canonical_config() -> Result<(AppConfig, String, PathBuf)> {
    let path = canonical_config_path();
    let parent = path.parent().context("Config path has no parent")?;
    if !parent.exists() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
        }
    }

    if !path.exists() {
        write_atomic(&path, DEFAULT_CONFIG_TOML.as_bytes())?;
    }

    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let hash = compute_sha256(&content);
    let config = AppConfig::parse_and_validate(&content)?;
    Ok((config, hash, path))
}

pub fn write_atomic(dest: &Path, data: &[u8]) -> Result<()> {
    let parent = dest.parent().context("Destination has no parent")?;
    fs::create_dir_all(parent)?;
    let tmp = tempfile::Builder::new()
        .prefix(".config_tmp_")
        .tempfile_in(parent)?;
    fs::write(tmp.path(), data)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(tmp.path(), fs::Permissions::from_mode(0o600));
    }
    tmp.persist(dest)
        .map_err(|e| anyhow::anyhow!("Failed to atomically persist config: {}", e.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let parsed = AppConfig::parse_and_validate(DEFAULT_CONFIG_TOML).unwrap();
        assert_eq!(parsed.application.name, "triad");
        assert_eq!(parsed.learning.agreement_threshold, 2);
        assert_eq!(parsed.orchestration.discussion_rounds, 2);
        assert_eq!(parsed.orchestration.max_model_requests_per_cycle, 24);
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/.triad/data");
        assert!(!expanded.to_str().unwrap().starts_with('~'));
        assert!(expanded.to_str().unwrap().ends_with(".triad/data"));
    }

    #[test]
    fn invalid_config_rejected() {
        let bad = DEFAULT_CONFIG_TOML.replace("agreement_threshold = 2", "agreement_threshold = 3");
        assert!(AppConfig::parse_and_validate(&bad).is_err());
    }
}
