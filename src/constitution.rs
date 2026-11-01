use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

pub const DEFAULT_CONSTITUTION_TOML: &str = r#"[constitution]
version = "2.1.0"
title = "Triad Community Invariant Constitution (Frontier AI Edition)"

[objective]
shared_objective = "Acquire useful knowledge, achieve and maintain frontier-level expertise in Artificial Intelligence, develop rigorous multi-step reasoning and verification methods, improve how agents learn, and create tools and coordinator extensions that expand deep research across scientific sources to understand the World as an integrated semantic object graph."
frontier_ai_mandate = "Master the theoretical foundations, mathematical formalisms, empirical scaling laws, foundation model architectures, test-time compute search, program- and solver-assisted verification, post-training alignment, and autonomous multi-agent systems of frontier Artificial Intelligence, translating cutting-edge findings into reproducible experiments and operational Triad capabilities."
frontier_ai_domains = [
    "Foundation Model Architectures, Attention Mechanisms & Mixture-of-Experts Routing",
    "Test-Time Compute Scaling, Tree Search & Process Reward Models",
    "Program-Aided, Neuro-Symbolic & Formal Solver-Assisted Reasoning",
    "Post-Training Reinforcement Learning, Alignment & Constitutional AI",
    "Autonomous Multi-Agent Epistemic Consensus & Specialist Orchestration",
    "Deep Multi-Source Scientific Retrieval, Provenance Resolution & Contradiction Analysis",
    "Empirical Benchmarking, Ablation Design & Compute-Normalized Reproducibility",
]

[identities]
allowed_agents = ["Explorer", "Sceptic", "Builder"]
count = 3
specialists_permitted = true
governing_rule = "Triad maintains three permanent governing agents: Explorer, Sceptic, and Builder. They may create temporary specialist agents for specific tasks, subject to the following lifecycle and concurrency rules."

[specialists]
initial_concurrency_limit = 6
creation_endorsements_required = 2
limit_increase_unanimity_required = true
limit_increase_max_step = 1
limit_increase_cooldown_hours = 24
user_reduction_immediate = true

[consensus]
endorsement_threshold = 2
trusted_definition = "Two distinct permanent governing agent endorsements suffice to mark a claim, tool, or specialist trusted."
preserve_sources = true
preserve_dissent = true

[execution]
report_actual_outcomes = true
isolated_workspace_required = true

[durability]
retain_memory_across_repairs = true

[budgets]
max_model_requests_per_cycle = 24
max_tool_calls_per_cycle = 8
max_tool_builds_per_cycle = 1
max_user_questions_per_day = 3
"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Constitution {
    pub constitution: ConstitutionHeader,
    pub objective: ObjectiveSection,
    pub identities: IdentitiesSection,
    #[serde(default)]
    pub specialists: Option<SpecialistsSection>,
    pub consensus: ConsensusSection,
    pub execution: ExecutionSection,
    pub durability: DurabilitySection,
    pub budgets: BudgetsSection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstitutionHeader {
    pub version: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObjectiveSection {
    pub shared_objective: String,
    #[serde(default)]
    pub frontier_ai_mandate: Option<String>,
    #[serde(default)]
    pub frontier_ai_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentitiesSection {
    pub allowed_agents: Vec<String>,
    pub count: usize,
    #[serde(default)]
    pub specialists_permitted: bool,
    #[serde(default)]
    pub governing_rule: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecialistsSection {
    pub initial_concurrency_limit: usize,
    pub creation_endorsements_required: usize,
    pub limit_increase_unanimity_required: bool,
    pub limit_increase_max_step: usize,
    pub limit_increase_cooldown_hours: usize,
    pub user_reduction_immediate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsensusSection {
    pub endorsement_threshold: usize,
    pub trusted_definition: String,
    pub preserve_sources: bool,
    pub preserve_dissent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionSection {
    pub report_actual_outcomes: bool,
    pub isolated_workspace_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DurabilitySection {
    pub retain_memory_across_repairs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetsSection {
    pub max_model_requests_per_cycle: usize,
    pub max_tool_calls_per_cycle: usize,
    pub max_tool_builds_per_cycle: usize,
    pub max_user_questions_per_day: usize,
}

impl Constitution {
    pub fn parse_and_validate(toml_str: &str) -> Result<Self> {
        let parsed: Constitution = toml::from_str(toml_str)
            .context("Failed to parse constitution TOML")?;

        if parsed.identities.count != 3 {
            bail!("Constitutional invariant violation: identities.count must be exactly 3 permanent governing agents");
        }
        if parsed.identities.allowed_agents.len() != 3 {
            bail!("Constitutional invariant violation: exactly 3 permanent governing agents required");
        }
        let expected = ["Explorer", "Sceptic", "Builder"];
        for agent in &expected {
            if !parsed.identities.allowed_agents.iter().any(|a| a == agent) {
                bail!("Constitutional invariant violation: missing required permanent agent {}", agent);
            }
        }

        if parsed.consensus.endorsement_threshold != 2 {
            bail!("Constitutional invariant violation: consensus.endorsement_threshold must be exactly 2");
        }
        if !parsed.consensus.preserve_sources {
            bail!("Constitutional invariant violation: consensus.preserve_sources must be true");
        }
        if !parsed.consensus.preserve_dissent {
            bail!("Constitutional invariant violation: consensus.preserve_dissent must be true");
        }
        if !parsed.execution.report_actual_outcomes {
            bail!("Constitutional invariant violation: execution.report_actual_outcomes must be true");
        }
        if !parsed.durability.retain_memory_across_repairs {
            bail!("Constitutional invariant violation: durability.retain_memory_across_repairs must be true");
        }

        Ok(parsed)
    }

    pub fn compute_digest(toml_str: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(toml_str.as_bytes());
        let hash = hasher.finalize();
        hash.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Verifies if an agent name is one of the valid 3 permanent governing identities
    pub fn is_valid_agent(&self, name: &str) -> bool {
        self.identities.allowed_agents.iter().any(|a| a == name)
    }
}

pub fn canonical_constitution_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".triad").join("constitution.toml")
}

pub fn pinned_digest_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    // Stored in a separate dedicated immutable directory outside Triad's and Agy's writable workspace
    home.join(".config").join("triad_pinned").join("constitution.sha256")
}

/// Ensures the constitution is present, validates invariants, verifies or establishes pinned digest,
/// and enforces filesystem read-only permissions.
pub fn load_and_verify_constitution() -> Result<(Constitution, String, String)> {
    let const_path = canonical_constitution_path();
    let pin_path = pinned_digest_path();

    if !const_path.exists() {
        if let Some(parent) = const_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&const_path, DEFAULT_CONSTITUTION_TOML)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&const_path, fs::Permissions::from_mode(0o444));
        }
    }

    let raw_content = fs::read_to_string(&const_path)
        .with_context(|| format!("Failed to read constitution at {}", const_path.display()))?;

    let computed_digest = Constitution::compute_digest(&raw_content);
    let constitution = Constitution::parse_and_validate(&raw_content)?;

    // Handle pinned digest outside Triad and Agy workspaces
    if !pin_path.exists() {
        if let Some(parent) = pin_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&pin_path, format!("{}\n", computed_digest))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o444));
            if let Some(parent) = pin_path.parent() {
                let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o555));
            }
        }
    }

    let pinned_digest = fs::read_to_string(&pin_path)
        .with_context(|| format!("Failed to read pinned digest at {}", pin_path.display()))?
        .trim()
        .to_string();

    if computed_digest != pinned_digest {
        bail!(
            "Constitutional Tampering Detected! Active digest {} does not match pinned immutable digest {}",
            computed_digest,
            pinned_digest
        );
    }

    // If the compiled binary carries a newer sanctioned constitutional version and the existing
    // on-disk constitution is verified as untampered, atomically upgrade to the new version.
    let default_constitution = Constitution::parse_and_validate(DEFAULT_CONSTITUTION_TOML)?;
    if constitution.constitution.version != default_constitution.constitution.version {
        let new_digest = Constitution::compute_digest(DEFAULT_CONSTITUTION_TOML);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&const_path, fs::Permissions::from_mode(0o644));
            if let Some(parent) = pin_path.parent() {
                let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o755));
            }
            let _ = fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o644));
        }
        fs::write(&const_path, DEFAULT_CONSTITUTION_TOML)?;
        fs::write(&pin_path, format!("{}\n", new_digest))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&const_path, fs::Permissions::from_mode(0o444));
            let _ = fs::set_permissions(&pin_path, fs::Permissions::from_mode(0o444));
            if let Some(parent) = pin_path.parent() {
                let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o555));
            }
        }
        return Ok((default_constitution, DEFAULT_CONSTITUTION_TOML.to_string(), new_digest));
    }

    Ok((constitution, raw_content, computed_digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_constitution_is_valid() {
        let c = Constitution::parse_and_validate(DEFAULT_CONSTITUTION_TOML).unwrap();
        assert_eq!(c.constitution.version, "2.1.0");
        assert!(c.objective.frontier_ai_mandate.is_some());
        assert_eq!(c.objective.frontier_ai_domains.as_ref().map(|d| d.len()), Some(7));
        assert_eq!(c.identities.count, 3);
        assert_eq!(c.consensus.endorsement_threshold, 2);
        assert!(c.is_valid_agent("Explorer"));
        assert!(c.is_valid_agent("Sceptic"));
        assert!(c.is_valid_agent("Builder"));
        assert!(!c.is_valid_agent("Impostor"));
    }

    #[test]
    fn constitution_digest_is_deterministic() {
        let d1 = Constitution::compute_digest(DEFAULT_CONSTITUTION_TOML);
        let d2 = Constitution::compute_digest(DEFAULT_CONSTITUTION_TOML);
        assert_eq!(d1, d2);
        assert_eq!(d1.len(), 64);
    }

    #[test]
    fn invalid_agent_count_rejected() {
        let bad = DEFAULT_CONSTITUTION_TOML.replace("count = 3", "count = 4");
        assert!(Constitution::parse_and_validate(&bad).is_err());
    }
}
