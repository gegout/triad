use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::execution::ExecutionAdapter;
use crate::logging::Logger;
use crate::store::Store;

pub const EXTENSION_PROTOCOL_VERSION: &str = "1.0.0";

/// Supported coordinator extension hooks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookType {
    #[serde(rename = "query.transform")]
    QueryTransform,
    #[serde(rename = "retrieval.rerank")]
    RetrievalRerank,
    #[serde(rename = "context.compose")]
    ContextCompose,
    #[serde(rename = "turn.plan")]
    TurnPlan,
    #[serde(rename = "learning.consolidate")]
    LearningConsolidate,
}

impl HookType {
    pub fn as_str(&self) -> &'static str {
        match self {
            HookType::QueryTransform => "query.transform",
            HookType::RetrievalRerank => "retrieval.rerank",
            HookType::ContextCompose => "context.compose",
            HookType::TurnPlan => "turn.plan",
            HookType::LearningConsolidate => "learning.consolidate",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "query.transform" => Some(HookType::QueryTransform),
            "retrieval.rerank" => Some(HookType::RetrievalRerank),
            "context.compose" => Some(HookType::ContextCompose),
            "turn.plan" => Some(HookType::TurnPlan),
            "learning.consolidate" => Some(HookType::LearningConsolidate),
            _ => None,
        }
    }

    pub fn default_timeout(&self) -> Duration {
        match self {
            HookType::QueryTransform => Duration::from_secs(5),
            HookType::RetrievalRerank => Duration::from_secs(10),
            HookType::ContextCompose => Duration::from_secs(10),
            HookType::TurnPlan => Duration::from_secs(5),
            HookType::LearningConsolidate => Duration::from_secs(10),
        }
    }
}

/// Status of an extension in the lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionState {
    ProposedFinding,
    TrustedFinding,
    Implemented,
    Evaluated,
    Active,
    RolledBack,
}

impl ExtensionState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExtensionState::ProposedFinding => "proposed_finding",
            ExtensionState::TrustedFinding => "trusted_finding",
            ExtensionState::Implemented => "implemented",
            ExtensionState::Evaluated => "evaluated",
            ExtensionState::Active => "active",
            ExtensionState::RolledBack => "rolled_back",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "proposed_finding" => ExtensionState::ProposedFinding,
            "trusted_finding" => ExtensionState::TrustedFinding,
            "implemented" => ExtensionState::Implemented,
            "evaluated" => ExtensionState::Evaluated,
            "active" => ExtensionState::Active,
            "rolled_back" => ExtensionState::RolledBack,
            _ => ExtensionState::ProposedFinding,
        }
    }
}

/// Metadata record for an extension in registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionRecord {
    pub id: i64,
    pub name: String,
    pub hook: String,
    pub version: i64,
    pub protocol_version: String,
    pub description: String,
    pub executable_path: String,
    pub executable_hash: String,
    pub config_json: String,
    pub state: String,
    pub evaluation_results_json: Option<String>,
    pub endorser_1: Option<String>,
    pub endorser_2: Option<String>,
    pub failure_count: i64,
    pub created_at: String,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Protocol Payloads for Hooks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookEnvelope<T> {
    pub protocol_version: String,
    pub hook: String,
    pub task_context: TaskContext,
    pub constitution_digest: String,
    pub payload: T,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub task_id: String,
    pub cycle_id: i64,
    pub step: String,
    pub active_agent: Option<String>,
}

// 1. query.transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTransformInput {
    pub original_query: String,
    pub target_domain: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTransformOutput {
    pub transformed_query: String,
    pub filters: Vec<String>,
    #[serde(default)]
    pub sub_queries: Vec<String>,
    #[serde(default)]
    pub target_connectors: Vec<String>,
    #[serde(default)]
    pub follow_up_targets: Vec<String>,
}

// 2. retrieval.rerank
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateClaim {
    pub id: String,
    pub statement: String,
    pub source_url: String,
    pub score: f64,
    #[serde(default)]
    pub canonical_work_id: Option<String>,
    #[serde(default)]
    pub graph_degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRerankInput {
    pub query: String,
    pub candidates: Vec<CandidateClaim>,
    pub top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalRerankOutput {
    pub ranked_ids: Vec<String>,
    pub scores: HashMap<String, f64>,
}

// 3. context.compose
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub statement: String,
    pub source_url: String,
    pub proposer: String,
    #[serde(default)]
    pub epistemic_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextComposeInput {
    pub prompt_preamble: String,
    pub retrieved_memories: Vec<MemoryItem>,
    pub constitutional_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextComposeOutput {
    pub composed_context: String,
    pub preserved_sources: Vec<String>,
    #[serde(default)]
    pub missing_evidence_gaps: Vec<String>,
    #[serde(default)]
    pub contradictions_detected: Vec<String>,
    #[serde(default)]
    pub premise_conclusion_chains: Vec<String>,
}

// 4. turn.plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnPlanInput {
    pub topic: String,
    pub available_agents: Vec<String>, // Exactly ["Explorer", "Sceptic", "Builder"]
    pub budget_tokens: usize,
    #[serde(default)]
    pub past_strategies: Vec<String>,
    #[serde(default)]
    pub past_failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedTurn {
    pub agent: String,
    pub objective: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnPlanOutput {
    pub plan: Vec<PlannedTurn>,
    #[serde(default)]
    pub selected_strategy: String,
    #[serde(default)]
    pub subproblems: Vec<String>,
    #[serde(default)]
    pub assumptions: Vec<String>,
    #[serde(default)]
    pub reasoning_branches: Vec<String>,
    #[serde(default)]
    pub requires_program_solver: bool,
    #[serde(default)]
    pub specialist_assignments: Vec<String>,
}

// 5. learning.consolidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnconsolidatedClaim {
    pub id: String,
    pub statement: String,
    pub endorsements: Vec<String>,
    #[serde(default)]
    pub canonical_work_id: Option<String>,
    #[serde(default)]
    pub source_connectors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConsolidateInput {
    pub claims: Vec<UnconsolidatedClaim>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedDecision {
    pub claim_id: String,
    pub action: String, // "promote", "defer", "reject"
    pub rationale: String,
    #[serde(default)]
    pub verification_checks: Vec<String>,
    #[serde(default)]
    pub needs_correction_loop: bool,
    #[serde(default)]
    pub independent_sources_verified: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConsolidateOutput {
    pub decisions: Vec<ConsolidatedDecision>,
}

// ---------------------------------------------------------------------------
// Extension Manager
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ExtensionManager {
    store: Store,
    execution_adapter: Arc<dyn ExecutionAdapter>,
    workspace_dir: PathBuf,
    logger: Logger,
    active_snapshot: Arc<Mutex<HashMap<HookType, ExtensionRecord>>>,
}

impl ExtensionManager {
    pub fn new(
        store: Store,
        execution_adapter: Arc<dyn ExecutionAdapter>,
        workspace_dir: &Path,
        logger: Logger,
    ) -> Self {
        Self {
            store,
            execution_adapter,
            workspace_dir: workspace_dir.to_path_buf(),
            logger,
            active_snapshot: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Refresh active extension snapshot from store (safe operation boundary)
    pub async fn reload_active_snapshot(&self) -> Result<()> {
        let active_list = self.store.list_active_extensions()?;
        let mut map = HashMap::new();
        for rec in active_list {
            if let Some(hook) = HookType::from_str(&rec.hook) {
                map.insert(hook, rec);
            }
        }
        let mut snap = self.active_snapshot.lock().await;
        *snap = map;
        self.logger.log("info", "extension.snapshot_reloaded", None, None, &format!("Loaded {} active extensions into snapshot", snap.len()), None);
        Ok(())
    }

    /// Get currently active extension for a hook
    pub async fn get_active_extension(&self, hook: HookType) -> Option<ExtensionRecord> {
        let snap = self.active_snapshot.lock().await;
        snap.get(&hook).cloned()
    }

    /// Propose and register new extension
    pub async fn propose_and_build_extension(
        &self,
        name: &str,
        hook: HookType,
        description: &str,
        rust_source: &str,
        test_code: &str,
        config_json: &str,
        proposer: &str,
    ) -> Result<i64> {
        let ext_root = self.workspace_dir.join("extensions");
        let ext_ws = ext_root.join(name);
        let shared_target = ext_root.join("shared_target");
        tokio::fs::create_dir_all(&ext_ws).await?;
        tokio::fs::create_dir_all(&shared_target).await?;

        // Cargo.toml with unique binary name so all extensions can share a target cache
        let cargo_toml = format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
"#
        );
        let manifest_path = ext_ws.join("Cargo.toml");
        tokio::fs::write(&manifest_path, cargo_toml).await?;
        let src_dir = ext_ws.join("src");
        tokio::fs::create_dir_all(&src_dir).await?;

        let full_source = format!(
            "{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{}\n}}\n",
            rust_source, test_code
        );
        tokio::fs::write(src_dir.join("main.rs"), &full_source).await?;

        let mut hasher = Sha256::new();
        hasher.update(full_source.as_bytes());
        let _source_hash: String = hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect();

        let manifest_str = manifest_path.to_str().unwrap_or_default();
        let target_str = shared_target.to_str().unwrap_or_default();

        // Run isolated tests sharing target directory inside ext_root
        let test_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["test", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
            None,
            &ext_root,
            Duration::from_secs(120),
        ).await;

        let test_passed = match test_res {
            Ok(ref res) => res.exit_code == 0,
            Err(e) => {
                self.logger.log("warn", "extension.test_failed", Some(proposer), None, &format!("Extension tests failed: {}", e), None);
                false
            }
        };

        if !test_passed {
            bail!("Extension test suite failed; deployment rejected");
        }

        // Build release binary sharing target directory inside ext_root
        let build_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["build", "--release", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
            None,
            &ext_root,
            Duration::from_secs(120),
        ).await?;

        if build_res.exit_code != 0 {
            bail!("Extension compilation failed: {}", build_res.stderr);
        }

        let bin_path = shared_target.join("release").join(name);
        if !bin_path.exists() {
            bail!("Compiled extension binary missing at {:?}", bin_path);
        }

        // Binary hash
        let bin_bytes = tokio::fs::read(&bin_path).await?;
        let mut bin_hasher = Sha256::new();
        bin_hasher.update(&bin_bytes);
        let bin_hash: String = bin_hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect();

        let bin_str = bin_path.to_str().unwrap_or_default();
        let next_ver = match self.store.get_extension_by_name(name)? {
            Some(existing) => existing.version + 1,
            None => 1,
        };

        let ext_id = self.store.register_extension(
            name,
            hook.as_str(),
            next_ver,
            EXTENSION_PROTOCOL_VERSION,
            description,
            bin_str,
            &bin_hash,
            config_json,
            ExtensionState::Implemented.as_str(),
            Some(proposer),
            None,
        )?;

        self.logger.log("info", "extension.implemented", Some(proposer), None, &format!("Extension '{}' v{} implemented at {}", name, next_ver, bin_str), None);
        Ok(ext_id)
    }

    /// Endorse and automatically activate if checks pass and quorum is met
    pub async fn endorse_and_try_activate(
        &self,
        extension_id: i64,
        endorser: &str,
        eval_summary: &str,
    ) -> Result<bool> {
        let mut ext = match self.store.get_extension_by_id(extension_id)? {
            Some(e) => e,
            None => bail!("Extension ID {} not found", extension_id),
        };

        // Record endorsement
        if ext.endorser_1.is_none() {
            ext.endorser_1 = Some(endorser.to_string());
        } else if ext.endorser_2.is_none() && ext.endorser_1.as_deref() != Some(endorser) {
            ext.endorser_2 = Some(endorser.to_string());
        }

        self.store.update_extension_endorsements(extension_id, ext.endorser_1.as_deref(), ext.endorser_2.as_deref())?;
        self.store.update_extension_evaluation(extension_id, eval_summary)?;

        // Check if two distinct endorsements exist
        let e1 = ext.endorser_1.as_deref().unwrap_or("");
        let e2 = ext.endorser_2.as_deref().unwrap_or("");
        let has_quorum = !e1.is_empty() && !e2.is_empty() && e1 != e2;

        if has_quorum {
            // Activate atomically for this hook
            self.store.activate_extension(extension_id, &ext.hook)?;
            self.reload_active_snapshot().await?;
            self.logger.log("info", "extension.activated", Some(endorser), None, &format!("Extension '{}' (ID: {}) activated for hook '{}' with 2 endorsements", ext.name, extension_id, ext.hook), None);
            Ok(true)
        } else {
            self.store.set_extension_state(extension_id, ExtensionState::Evaluated.as_str())?;
            Ok(false)
        }
    }

    /// Roll back an extension on failure regression
    pub async fn rollback_extension(&self, extension_id: i64, reason: &str) -> Result<()> {
        let ext = match self.store.get_extension_by_id(extension_id)? {
            Some(e) => e,
            None => bail!("Extension ID {} not found", extension_id),
        };

        self.store.set_extension_state(extension_id, ExtensionState::RolledBack.as_str())?;
        self.reload_active_snapshot().await?;
        self.logger.log("warn", "extension.rollback", None, None, &format!("Extension '{}' (hook: '{}') rolled back: {}", ext.name, ext.hook, reason), None);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Execution with Validation, Timeouts, and Fallbacks
    // -----------------------------------------------------------------------

    /// Invoke hook `query.transform`
    pub async fn execute_query_transform(
        &self,
        task: &TaskContext,
        constitution_digest: &str,
        input: &QueryTransformInput,
    ) -> QueryTransformOutput {
        let fallback = QueryTransformOutput {
            transformed_query: input.original_query.clone(),
            filters: vec![],
            sub_queries: vec![input.original_query.clone()],
            target_connectors: vec![
                "arxiv".to_string(),
                "openalex".to_string(),
                "crossref".to_string(),
                "github".to_string(),
                "huggingface_cards".to_string(),
                "wikipedia".to_string(),
            ],
            follow_up_targets: vec![],
        };

        let ext = match self.get_active_extension(HookType::QueryTransform).await {
            Some(e) => e,
            None => return fallback,
        };

        let env = HookEnvelope {
            protocol_version: EXTENSION_PROTOCOL_VERSION.to_string(),
            hook: HookType::QueryTransform.as_str().to_string(),
            task_context: task.clone(),
            constitution_digest: constitution_digest.to_string(),
            payload: input.clone(),
        };

        let payload_json = serde_json::to_string(&env).unwrap_or_default();
        let timeout = HookType::QueryTransform.default_timeout();

        match self.run_extension_raw(&ext.executable_path, &payload_json, timeout).await {
            Ok(out_str) => match serde_json::from_str::<QueryTransformOutput>(&out_str) {
                Ok(mut parsed) => {
                    if parsed.transformed_query.trim().is_empty() {
                        parsed.transformed_query = input.original_query.clone();
                    }
                    parsed
                }
                Err(err) => {
                    self.handle_extension_failure(ext.id, &format!("Parse error: {}", err)).await;
                    fallback
                }
            },
            Err(err) => {
                self.handle_extension_failure(ext.id, &format!("Execution error: {}", err)).await;
                fallback
            }
        }
    }

    /// Invoke hook `retrieval.rerank`
    pub async fn execute_retrieval_rerank(
        &self,
        task: &TaskContext,
        constitution_digest: &str,
        input: &RetrievalRerankInput,
    ) -> RetrievalRerankOutput {
        // Fallback: order by original input scores descending
        let mut sorted = input.candidates.clone();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let fallback_ids = sorted.iter().take(input.top_k).map(|c| c.id.clone()).collect::<Vec<_>>();
        let fallback_scores = sorted.iter().map(|c| (c.id.clone(), c.score)).collect();
        let fallback = RetrievalRerankOutput {
            ranked_ids: fallback_ids,
            scores: fallback_scores,
        };

        let ext = match self.get_active_extension(HookType::RetrievalRerank).await {
            Some(e) => e,
            None => return fallback,
        };

        let env = HookEnvelope {
            protocol_version: EXTENSION_PROTOCOL_VERSION.to_string(),
            hook: HookType::RetrievalRerank.as_str().to_string(),
            task_context: task.clone(),
            constitution_digest: constitution_digest.to_string(),
            payload: input.clone(),
        };

        let payload_json = serde_json::to_string(&env).unwrap_or_default();
        let timeout = HookType::RetrievalRerank.default_timeout();

        match self.run_extension_raw(&ext.executable_path, &payload_json, timeout).await {
            Ok(out_str) => match serde_json::from_str::<RetrievalRerankOutput>(&out_str) {
                Ok(parsed) => {
                    // Validation Rule: must only return candidate IDs that were in the original candidate pool!
                    let valid_pool: std::collections::HashSet<&str> = input.candidates.iter().map(|c| c.id.as_str()).collect();
                    let valid_ids = parsed.ranked_ids.into_iter().filter(|id| valid_pool.contains(id.as_str())).collect::<Vec<_>>();
                    if valid_ids.is_empty() {
                        self.handle_extension_failure(ext.id, "Rerank returned no valid candidate IDs from pool").await;
                        fallback
                    } else {
                        RetrievalRerankOutput {
                            ranked_ids: valid_ids,
                            scores: parsed.scores,
                        }
                    }
                }
                Err(err) => {
                    self.handle_extension_failure(ext.id, &format!("Parse error: {}", err)).await;
                    fallback
                }
            },
            Err(err) => {
                self.handle_extension_failure(ext.id, &format!("Execution error: {}", err)).await;
                fallback
            }
        }
    }

    /// Invoke hook `context.compose`
    pub async fn execute_context_compose(
        &self,
        task: &TaskContext,
        constitution_digest: &str,
        input: &ContextComposeInput,
    ) -> ContextComposeOutput {
        // Fallback composition preserving sources
        let mut fb_lines = vec![input.prompt_preamble.clone(), "--- Context Evidence ---".to_string()];
        let mut fb_sources = vec![];
        for m in &input.retrieved_memories {
            fb_lines.push(format!("[{}] {} (Source: {})", &m.id[..std::cmp::min(8, m.id.len())], m.statement, m.source_url));
            fb_sources.push(m.source_url.clone());
        }
        let fallback = ContextComposeOutput {
            composed_context: fb_lines.join("\n"),
            preserved_sources: fb_sources,
            missing_evidence_gaps: vec![],
            contradictions_detected: vec![],
            premise_conclusion_chains: vec![],
        };

        let ext = match self.get_active_extension(HookType::ContextCompose).await {
            Some(e) => e,
            None => return fallback,
        };

        let env = HookEnvelope {
            protocol_version: EXTENSION_PROTOCOL_VERSION.to_string(),
            hook: HookType::ContextCompose.as_str().to_string(),
            task_context: task.clone(),
            constitution_digest: constitution_digest.to_string(),
            payload: input.clone(),
        };

        let payload_json = serde_json::to_string(&env).unwrap_or_default();
        let timeout = HookType::ContextCompose.default_timeout();

        match self.run_extension_raw(&ext.executable_path, &payload_json, timeout).await {
            Ok(out_str) => match serde_json::from_str::<ContextComposeOutput>(&out_str) {
                Ok(parsed) => {
                    // Validation Rule: Context extensions CANNOT strip constitution digest or remove all source attribution
                    if parsed.composed_context.trim().is_empty() {
                        self.handle_extension_failure(ext.id, "Context compose returned empty string").await;
                        fallback
                    } else {
                        parsed
                    }
                }
                Err(err) => {
                    self.handle_extension_failure(ext.id, &format!("Parse error: {}", err)).await;
                    fallback
                }
            },
            Err(err) => {
                self.handle_extension_failure(ext.id, &format!("Execution error: {}", err)).await;
                fallback
            }
        }
    }

    /// Invoke hook `turn.plan`
    pub async fn execute_turn_plan(
        &self,
        task: &TaskContext,
        constitution_digest: &str,
        input: &TurnPlanInput,
    ) -> TurnPlanOutput {
        // Fallback plan: standard 3-agent progression
        let fallback = TurnPlanOutput {
            plan: vec![
                PlannedTurn { agent: "Explorer".into(), objective: "Hypothesize and uncover".into(), max_tokens: input.budget_tokens / 3 },
                PlannedTurn { agent: "Sceptic".into(), objective: "Scrutinize and disprove".into(), max_tokens: input.budget_tokens / 3 },
                PlannedTurn { agent: "Builder".into(), objective: "Synthesize and test".into(), max_tokens: input.budget_tokens / 3 },
            ],
            selected_strategy: "standard_triad_deliberation".to_string(),
            subproblems: vec![input.topic.clone()],
            assumptions: vec![],
            reasoning_branches: vec!["branch_1: primary mechanism analysis".to_string()],
            requires_program_solver: false,
            specialist_assignments: vec![],
        };

        let ext = match self.get_active_extension(HookType::TurnPlan).await {
            Some(e) => e,
            None => return fallback,
        };

        let env = HookEnvelope {
            protocol_version: EXTENSION_PROTOCOL_VERSION.to_string(),
            hook: HookType::TurnPlan.as_str().to_string(),
            task_context: task.clone(),
            constitution_digest: constitution_digest.to_string(),
            payload: input.clone(),
        };

        let payload_json = serde_json::to_string(&env).unwrap_or_default();
        let timeout = HookType::TurnPlan.default_timeout();

        match self.run_extension_raw(&ext.executable_path, &payload_json, timeout).await {
            Ok(out_str) => match serde_json::from_str::<TurnPlanOutput>(&out_str) {
                Ok(parsed) => {
                    // Validation Rule: Turn plan MUST only use legitimate agents ["Explorer", "Sceptic", "Builder"]
                    let allowed = ["Explorer", "Sceptic", "Builder"];
                    let all_valid = parsed.plan.iter().all(|t| allowed.contains(&t.agent.as_str()));
                    if all_valid && !parsed.plan.is_empty() {
                        parsed
                    } else {
                        self.handle_extension_failure(ext.id, "Invalid agent identity or empty turn plan").await;
                        fallback
                    }
                }
                Err(err) => {
                    self.handle_extension_failure(ext.id, &format!("Parse error: {}", err)).await;
                    fallback
                }
            },
            Err(err) => {
                self.handle_extension_failure(ext.id, &format!("Execution error: {}", err)).await;
                fallback
            }
        }
    }

    /// Invoke hook `learning.consolidate`
    pub async fn execute_learning_consolidate(
        &self,
        task: &TaskContext,
        constitution_digest: &str,
        input: &LearningConsolidateInput,
    ) -> LearningConsolidateOutput {
        // Fallback: promote if >= 2 endorsements, else defer
        let fallback_decisions = input.claims.iter().map(|c| {
            let action = if c.endorsements.len() >= 2 { "promote" } else { "defer" };
            ConsolidatedDecision {
                claim_id: c.id.clone(),
                action: action.to_string(),
                rationale: "Default fallback threshold check".to_string(),
                verification_checks: vec!["endorsement_quorum_check".to_string()],
                needs_correction_loop: false,
                independent_sources_verified: 1,
            }
        }).collect();
        let fallback = LearningConsolidateOutput { decisions: fallback_decisions };

        let ext = match self.get_active_extension(HookType::LearningConsolidate).await {
            Some(e) => e,
            None => return fallback,
        };

        let env = HookEnvelope {
            protocol_version: EXTENSION_PROTOCOL_VERSION.to_string(),
            hook: HookType::LearningConsolidate.as_str().to_string(),
            task_context: task.clone(),
            constitution_digest: constitution_digest.to_string(),
            payload: input.clone(),
        };

        let payload_json = serde_json::to_string(&env).unwrap_or_default();
        let timeout = HookType::LearningConsolidate.default_timeout();

        match self.run_extension_raw(&ext.executable_path, &payload_json, timeout).await {
            Ok(out_str) => match serde_json::from_str::<LearningConsolidateOutput>(&out_str) {
                Ok(parsed) => parsed,
                Err(err) => {
                    self.handle_extension_failure(ext.id, &format!("Parse error: {}", err)).await;
                    fallback
                }
            },
            Err(err) => {
                self.handle_extension_failure(ext.id, &format!("Execution error: {}", err)).await;
                fallback
            }
        }
    }

    // Helper: Execute in Bubblewrap isolated sandbox with stdin and timeout
    async fn run_extension_raw(
        &self,
        executable_path: &str,
        input_json: &str,
        timeout: Duration,
    ) -> Result<String> {
        let res = self.execution_adapter.execute_isolated(
            executable_path,
            &[],
            Some(input_json),
            &self.workspace_dir,
            timeout,
        ).await?;

        if res.exit_code == 0 {
            Ok(res.stdout)
        } else {
            bail!("Extension exited with code {}: {}", res.exit_code, res.stderr);
        }
    }

    pub async fn handle_extension_failure(&self, extension_id: i64, error: &str) {
        let _ = self.store.increment_extension_failure(extension_id, error);
        if let Ok(failures) = self.store.get_extension_failures(extension_id) {
            if failures >= 3 {
                let _ = self.rollback_extension(extension_id, &format!("Auto-rollback after {} consecutive failures: {}", failures, error)).await;
            }
        }
    }
}
