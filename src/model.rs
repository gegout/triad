use anyhow::{bail, Context, Result};
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};

use crate::config::{
    default_model_catalog, default_model_providers, is_marketing_product_label, ChatModelSection,
    EmbeddingModelSection, FrontierBudgetConfig, ModelCapabilityConfig, ModelProviderConfig,
    ModelSelectionConfig, ModelsSection,
};
use crate::logging::Logger;
use crate::store::{
    FrontierBudgetReservationRequest, ModelEvaluationRecord, ModelProviderDiscoveryRecord,
    ModelSelectionRecord, Store,
};

/// Categorizes coordinator, agent, specialist, and consultation workloads so that
/// routing, summarization, and cheap work remain on local models while hard synthesis,
/// experiment design, complex Rust generation, debugging, and critical review escalate
/// to stronger configured models within frontier budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadKind {
    Routing,
    Summarization,
    CheapWork,
    HardSynthesis,
    ExperimentDesign,
    ComplexRustGeneration,
    Debugging,
    CriticalReview,
}

impl WorkloadKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            WorkloadKind::Routing => "routing",
            WorkloadKind::Summarization => "summarization",
            WorkloadKind::CheapWork => "cheap_work",
            WorkloadKind::HardSynthesis => "hard_synthesis",
            WorkloadKind::ExperimentDesign => "experiment_design",
            WorkloadKind::ComplexRustGeneration => "complex_rust_generation",
            WorkloadKind::Debugging => "debugging",
            WorkloadKind::CriticalReview => "critical_review",
        }
    }

    pub fn from_name(name: &str) -> Self {
        match name.trim().to_lowercase().as_str() {
            "routing" => WorkloadKind::Routing,
            "summarization" => WorkloadKind::Summarization,
            "cheap_work" | "cheap" => WorkloadKind::CheapWork,
            "hard_synthesis" | "synthesis" => WorkloadKind::HardSynthesis,
            "experiment_design" | "experiment" => WorkloadKind::ExperimentDesign,
            "complex_rust_generation" | "rust_generation" | "code_generation" => {
                WorkloadKind::ComplexRustGeneration
            }
            "debugging" | "debug" => WorkloadKind::Debugging,
            "critical_review" | "review" | "sceptic_review" => WorkloadKind::CriticalReview,
            _ => WorkloadKind::CheapWork,
        }
    }

    pub fn is_escalated_by_default(&self) -> bool {
        matches!(
            self,
            WorkloadKind::HardSynthesis
                | WorkloadKind::ExperimentDesign
                | WorkloadKind::ComplexRustGeneration
                | WorkloadKind::Debugging
                | WorkloadKind::CriticalReview
        )
    }

    pub fn requires_code_generation(&self) -> bool {
        matches!(
            self,
            WorkloadKind::ComplexRustGeneration | WorkloadKind::Debugging
        )
    }

    pub fn all() -> &'static [WorkloadKind] {
        &[
            WorkloadKind::Routing,
            WorkloadKind::Summarization,
            WorkloadKind::CheapWork,
            WorkloadKind::HardSynthesis,
            WorkloadKind::ExperimentDesign,
            WorkloadKind::ComplexRustGeneration,
            WorkloadKind::Debugging,
            WorkloadKind::CriticalReview,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSelectionContext {
    pub task_id: String,
    pub caller_kind: String, // "agent", "specialist", "background", "mcp_consultation"
    pub caller_id: String,
    pub workload: WorkloadKind,
    pub required_context_tokens: usize,
    pub require_structured_json: bool,
    pub explicit_override: Option<String>,
}

impl Default for WorkloadSelectionContext {
    fn default() -> Self {
        Self {
            task_id: "default_task".to_string(),
            caller_kind: "agent".to_string(),
            caller_id: "Explorer".to_string(),
            workload: WorkloadKind::CheapWork,
            required_context_tokens: 1024,
            require_structured_json: false,
            explicit_override: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedModelDecision {
    pub provider_id: String,
    pub provider_type: String,
    pub base_url: String,
    pub transport: String,
    pub credential_env: Option<String>,
    pub model_id: String,
    pub is_frontier: bool,
    pub context_window_tokens: usize,
    pub max_output_tokens: usize,
    pub input_price_micros_per_1m: Option<u64>,
    pub output_price_micros_per_1m: Option<u64>,
    pub pricing_known: bool,
    pub composite_score: f64,
    pub ranking_basis: String,
    pub fallback_reason: Option<String>,
    pub used_preference_order_fallback: bool,
    pub fallback_provider_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderUsageResponse {
    pub content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone)]
pub struct ModelClient {
    client: Client,
    chat_url: String,
    chat_model: String,
    max_tokens: usize,
    no_think: bool,
    embedding_url: String,
    embedding_model: String,
    query_prefix: String,
    document_prefix: String,
    concurrency_limiter: Arc<Semaphore>,
    models_config: Arc<Mutex<ModelsSection>>,
    store: Option<Store>,
    logger: Option<Logger>,
    last_discovery_refresh: Arc<Mutex<Option<Instant>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContribution {
    pub thought: String,
    pub public_message: String,
    pub proposed_claims: Vec<String>,
    pub proposed_questions: Vec<String>,
    pub vote_claim_id: Option<String>,
    pub vote_decision: Option<String>, // "endorse", "reject", "neutral"
    pub research_hypothesis: Option<String>,
    pub research_experiment: Option<String>,
    pub research_outcome: Option<String>,
    pub remaining_uncertainty: Option<String>,
    pub next_investigation: Option<String>,
}

pub fn clip_text(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut truncated: String = s.chars().take(max_chars).collect();
        truncated.push_str("... [truncated]");
        truncated
    }
}

pub fn compute_cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.is_empty() || b.is_empty() {
        bail!("Empty vector for cosine computation");
    }
    if a.len() != b.len() {
        bail!("Vector dimensions mismatch: {} vs {}", a.len(), b.len());
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let aa: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
    if aa <= 0.0 || bb <= 0.0 {
        bail!("Vector norm is zero or invalid");
    }
    Ok((dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0) as f32)
}

/// Redacts any configured credential environment variable values and common API key tokens
/// so secrets are never written to logs, errors, or SQLite diagnostics.
pub fn redact_secrets(text: &str, providers: &[ModelProviderConfig]) -> String {
    let mut out = text.to_string();
    for p in providers {
        if let Some(ref env_name) = p.credential_env {
            if !env_name.trim().is_empty() {
                if let Ok(secret_val) = std::env::var(env_name.trim()) {
                    let trimmed = secret_val.trim();
                    if trimmed.len() >= 4 {
                        out = out.replace(trimmed, "[REDACTED_SECRET]");
                    }
                }
            }
        }
    }
    // Also scrub common token prefixes (sk-..., AIza..., Bearer ...)
    for prefix in ["sk-ant-", "sk-or-", "sk-proj-", "sk-", "AIza"] {
        while let Some(idx) = out.find(prefix) {
            let end_offset = out[idx..]
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'' || c == ',' || c == ')')
                .unwrap_or(out.len() - idx);
            if end_offset > prefix.len() {
                out.replace_range(idx..idx + end_offset, "[REDACTED_SECRET]");
            } else {
                break;
            }
        }
    }
    out
}

pub fn check_provider_credential_status(provider: &ModelProviderConfig) -> (&'static str, Option<String>) {
    match provider.credential_env {
        None => ("not_required", None),
        Some(ref env_name) if env_name.trim().is_empty() => ("not_required", None),
        Some(ref env_name) => {
            let trimmed = env_name.trim();
            match std::env::var(trimmed) {
                Ok(val) if !val.trim().is_empty() => ("configured", Some(val.trim().to_string())),
                _ => ("missing", None),
            }
        }
    }
}

fn compute_prompt_hash(
    provider_id: &str,
    model_id: &str,
    workload: &str,
    system_prompt: &str,
    messages: &[ChatMessage],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(provider_id.as_bytes());
    hasher.update(b"|");
    hasher.update(model_id.as_bytes());
    hasher.update(b"|");
    hasher.update(workload.as_bytes());
    hasher.update(b"|");
    hasher.update(system_prompt.as_bytes());
    for m in messages {
        hasher.update(b"|");
        hasher.update(m.role.as_bytes());
        hasher.update(b":");
        hasher.update(m.content.as_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

fn estimate_tokens_from_chars(system_prompt: &str, messages: &[ChatMessage], max_out: usize) -> usize {
    let char_count: usize =
        system_prompt.len() + messages.iter().map(|m| m.content.len() + 8).sum::<usize>();
    let input_est = (char_count / 4).max(16);
    input_est + max_out.min(2048)
}

fn estimate_spend_micros(
    cap: &ModelCapabilityConfig,
    budget: &FrontierBudgetConfig,
    input_tokens: usize,
    output_tokens: usize,
) -> (u64, bool) {
    if !cap.is_frontier {
        return (0, true);
    }
    match (cap.input_price_micros_per_1m, cap.output_price_micros_per_1m) {
        (Some(in_p), Some(out_p)) if in_p > 0 || out_p > 0 => {
            let in_cost = ((input_tokens as u64).saturating_mul(in_p)) / 1_000_000;
            let out_cost = ((output_tokens as u64).saturating_mul(out_p)) / 1_000_000;
            ((in_cost + out_cost).max(1), true)
        }
        _ => {
            // Unknown billing counts conservatively; unknown prices are never free!
            let in_cost = ((input_tokens as u64)
                .saturating_mul(budget.conservative_unknown_input_micros_per_1m))
                / 1_000_000;
            let out_cost = ((output_tokens as u64)
                .saturating_mul(budget.conservative_unknown_output_micros_per_1m))
                / 1_000_000;
            let total = (in_cost + out_cost)
                .max(budget.conservative_unknown_call_micros)
                .max(1);
            (total, false)
        }
    }
}

impl ModelClient {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chat_url: &str,
        chat_model: &str,
        max_tokens: usize,
        no_think: bool,
        concurrency: usize,
        embedding_url: &str,
        embedding_model: &str,
        query_prefix: &str,
        document_prefix: &str,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_default();

        let clean_chat_url = chat_url.trim_end_matches('/').to_string();
        let clean_emb_url = embedding_url.trim_end_matches('/').to_string();

        let mut providers = default_model_providers();
        if let Some(local_p) = providers.iter_mut().find(|p| p.id == "local_snap") {
            local_p.base_url = clean_chat_url.clone();
            local_p.permitted_models = vec![chat_model.to_string()];
        }
        let mut catalog = default_model_catalog();
        if let Some(local_c) = catalog.iter_mut().find(|c| c.provider_id == "local_snap") {
            local_c.model_id = chat_model.to_string();
        }

        let default_models_section = ModelsSection {
            chat: ChatModelSection {
                base_url: clean_chat_url.clone(),
                model: chat_model.to_string(),
                max_tokens,
                structured_output: "prompt".to_string(),
                no_think,
                concurrency: concurrency.max(1),
            },
            embedding: EmbeddingModelSection {
                base_url: clean_emb_url.clone(),
                model: embedding_model.to_string(),
                query_prefix: query_prefix.to_string(),
                document_prefix: document_prefix.to_string(),
            },
            providers,
            catalog,
            selection: ModelSelectionConfig::default(),
            frontier_budget: FrontierBudgetConfig::default(),
        };

        Self {
            client,
            chat_url: clean_chat_url,
            chat_model: chat_model.to_string(),
            max_tokens,
            no_think,
            embedding_url: clean_emb_url,
            embedding_model: embedding_model.to_string(),
            query_prefix: query_prefix.to_string(),
            document_prefix: document_prefix.to_string(),
            concurrency_limiter: Arc::new(Semaphore::new(concurrency.max(1))),
            models_config: Arc::new(Mutex::new(default_models_section)),
            store: None,
            logger: None,
            last_discovery_refresh: Arc::new(Mutex::new(None)),
        }
    }

    pub fn from_models_config(
        models_cfg: &ModelsSection,
        store: Option<Store>,
        logger: Option<Logger>,
    ) -> Self {
        let mut s = Self::new(
            &models_cfg.chat.base_url,
            &models_cfg.chat.model,
            models_cfg.chat.max_tokens,
            models_cfg.chat.no_think,
            models_cfg.chat.concurrency,
            &models_cfg.embedding.base_url,
            &models_cfg.embedding.model,
            &models_cfg.embedding.query_prefix,
            &models_cfg.embedding.document_prefix,
        );
        s.models_config = Arc::new(Mutex::new(models_cfg.clone()));
        s.store = store;
        s.logger = logger;
        s
    }

    pub fn with_store_and_logger(mut self, store: Store, logger: Logger) -> Self {
        self.store = Some(store);
        self.logger = Some(logger);
        self
    }

    /// Live-reloads valid model/provider/selection/budget configuration without resetting
    /// persistent SQLite budget accounting history.
    pub async fn reload_models_config(&mut self, new_models_cfg: ModelsSection) {
        self.chat_url = new_models_cfg.chat.base_url.trim_end_matches('/').to_string();
        self.chat_model = new_models_cfg.chat.model.clone();
        self.max_tokens = new_models_cfg.chat.max_tokens;
        self.no_think = new_models_cfg.chat.no_think;
        self.embedding_url = new_models_cfg
            .embedding
            .base_url
            .trim_end_matches('/')
            .to_string();
        self.embedding_model = new_models_cfg.embedding.model.clone();
        self.query_prefix = new_models_cfg.embedding.query_prefix.clone();
        self.document_prefix = new_models_cfg.embedding.document_prefix.clone();
        *self.models_config.lock().await = new_models_cfg;
    }

    pub async fn get_models_config(&self) -> ModelsSection {
        self.models_config.lock().await.clone()
    }

    pub fn query_prefix(&self) -> &str {
        &self.query_prefix
    }

    pub fn document_prefix(&self) -> &str {
        &self.document_prefix
    }

    pub fn embedding_model_id(&self) -> &str {
        &self.embedding_model
    }

    pub fn chat_model_id(&self) -> &str {
        &self.chat_model
    }

    pub async fn probe_models(&self, base_url: &str) -> Result<Vec<String>> {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let res = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to connect to {}", url))?;
        if !res.status().is_success() {
            bail!("HTTP {} from {}", res.status(), url);
        }
        let body: serde_json::Value = res.json().await?;
        let mut ids = Vec::new();
        if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                    ids.push(id.to_string());
                }
            }
        } else if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
            for item in models {
                if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                    let clean = name.strip_prefix("models/").unwrap_or(name);
                    ids.push(clean.to_string());
                }
            }
        }
        Ok(ids)
    }

    /// Probes a configured `ModelProviderConfig` using its configured transport and credentials
    /// (never logging the secret value) and returns discovered real API model identifiers.
    pub async fn probe_provider_models(
        &self,
        provider: &ModelProviderConfig,
    ) -> Result<Vec<String>> {
        let (cred_status, secret_opt) = check_provider_credential_status(provider);
        if cred_status == "missing" {
            bail!(
                "Credentials missing for provider '{}' (env var {:?} unset)",
                provider.id,
                provider.credential_env
            );
        }

        let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
        let mut req = self.client.get(&url);

        if let Some(ref secret) = secret_opt {
            match provider.transport.as_str() {
                "anthropic_messages" => {
                    req = req
                        .header("x-api-key", secret)
                        .header("anthropic-version", "2023-06-01");
                }
                "gemini_native" => {
                    req = req.header("x-goog-api-key", secret);
                }
                _ => {
                    req = req.header("Authorization", format!("Bearer {}", secret));
                }
            }
        }

        let res = req.send().await.map_err(|e| {
            anyhow::anyhow!(
                "{}",
                redact_secrets(&format!("Failed to connect to {}: {}", url, e), &[provider.clone()])
            )
        })?;

        if !res.status().is_success() {
            let status = res.status();
            let err_body = res.text().await.unwrap_or_default();
            bail!(
                "HTTP {} from {}: {}",
                status,
                url,
                redact_secrets(&err_body, &[provider.clone()])
            );
        }

        let body: serde_json::Value = res.json().await?;
        let mut ids = Vec::new();
        if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                    if !is_marketing_product_label(id) {
                        ids.push(id.to_string());
                    }
                }
            }
        } else if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
            for item in models {
                if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                    let clean = name.strip_prefix("models/").unwrap_or(name);
                    if !is_marketing_product_label(clean) {
                        ids.push(clean.to_string());
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Periodically refreshes provider model discovery and persists discovery status in SQLite.
    pub async fn refresh_provider_discovery(&self, force: bool) -> Vec<ModelProviderDiscoveryRecord> {
        let cfg = self.models_config.lock().await.clone();
        let refresh_interval = Duration::from_secs(
            cfg.selection.discovery_refresh_interval_seconds.max(10),
        );

        {
            let mut guard = self.last_discovery_refresh.lock().await;
            if !force {
                if let Some(last) = *guard {
                    if last.elapsed() < refresh_interval {
                        if let Some(ref st) = self.store {
                            if let Ok(existing) = st.list_provider_discoveries() {
                                if !existing.is_empty() {
                                    return existing;
                                }
                            }
                        }
                    }
                }
            }
            *guard = Some(Instant::now());
        }

        let mut records = Vec::new();
        for p in &cfg.providers {
            let (cred_status, _) = check_provider_credential_status(p);
            let now_rfc = Utc::now().to_rfc3339();

            let (discovered_models, health_status) = if !p.enabled {
                (p.permitted_models.clone(), "disabled".to_string())
            } else if cred_status == "missing" {
                (
                    p.permitted_models.clone(),
                    format!(
                        "unconfigured_credentials({})",
                        p.credential_env.as_deref().unwrap_or("unknown")
                    ),
                )
            } else if !p.is_frontier
                || p.base_url.starts_with("http://127.0.0.1")
                || p.base_url.starts_with("http://localhost")
            {
                match self.probe_provider_models(p).await {
                    Ok(models) => (models, "healthy".to_string()),
                    Err(e) => (
                        p.permitted_models.clone(),
                        format!(
                            "unreachable: {}",
                            clip_text(&redact_secrets(&e.to_string(), &cfg.providers), 80)
                        ),
                    ),
                }
            } else {
                // Credentialed external provider: avoid unnecessary network chatter unless forced,
                // or probe `/models` when forced.
                if force {
                    match self.probe_provider_models(p).await {
                        Ok(models) => (models, "healthy_probed".to_string()),
                        Err(e) => (
                            p.permitted_models.clone(),
                            format!(
                                "probe_error: {}",
                                clip_text(&redact_secrets(&e.to_string(), &cfg.providers), 80)
                            ),
                        ),
                    }
                } else {
                    (p.permitted_models.clone(), "ready_credentialed".to_string())
                }
            };

            let rec = ModelProviderDiscoveryRecord {
                provider_id: p.id.clone(),
                provider_type: p.provider_type.clone(),
                base_url: p.base_url.clone(),
                transport: p.transport.clone(),
                credential_status: cred_status.to_string(),
                discovered_models,
                health_status,
                last_refreshed_at: now_rfc,
            };
            if let Some(ref st) = self.store {
                let _ = st.upsert_provider_discovery(&rec);
            }
            records.push(rec);
        }
        records
    }

    /// Dedicated embedding endpoint call.
    /// Never assumes the chat model or any escalated frontier model supplies embeddings.
    pub async fn embed_text(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        let prefix = if is_query {
            &self.query_prefix
        } else {
            &self.document_prefix
        };
        let input = format!("{}{}", prefix, text);
        let url = format!("{}/embeddings", self.embedding_url);

        let res = self
            .client
            .post(&url)
            .json(&json!({
                "model": self.embedding_model,
                "input": input
            }))
            .send()
            .await
            .with_context(|| format!("Failed to post embedding to {}", url))?;

        if !res.status().is_success() {
            let err = res.text().await.unwrap_or_default();
            bail!("Embedding request failed: {}", err);
        }

        let body: serde_json::Value = res.json().await?;
        let vec_val = body
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|item| item.get("embedding"))
            .context("Invalid embedding response structure")?;

        let embedding: Vec<f32> = serde_json::from_value(vec_val.clone())
            .context("Failed to deserialize embedding vector")?;

        if embedding.is_empty() {
            bail!("Received empty embedding vector");
        }
        Ok(embedding)
    }

    /// Selects the most capable eligible model for `ctx.workload` using capability metadata,
    /// provider profiles, and recorded evaluations.
    /// Never ranks by name, parameter count, price, date, or catalog order alone.
    /// Checks features, context window, access/credentials, and frontier budget.
    /// Supports explicit override, and falls back to the configured `preference_order`
    /// when empirical evaluation evidence is thin (explicitly stating so in `ranking_basis`).
    pub async fn select_model_for_workload(
        &self,
        ctx: &WorkloadSelectionContext,
    ) -> Result<SelectedModelDecision> {
        let cfg = self.models_config.lock().await.clone();
        let evaluations: Vec<ModelEvaluationRecord> = self
            .store
            .as_ref()
            .and_then(|s| s.list_model_evaluations().ok())
            .unwrap_or_default();

        let budget_summary = self.store.as_ref().and_then(|s| {
            s.get_frontier_budget_summary(&cfg.frontier_budget, 5).ok()
        });

        let workload_str = ctx.workload.as_str();
        let is_escalated_workload = cfg
            .selection
            .escalated_workloads
            .iter()
            .any(|w| w.eq_ignore_ascii_case(workload_str))
            || (!cfg
                .selection
                .local_workloads
                .iter()
                .any(|w| w.eq_ignore_ascii_case(workload_str))
                && ctx.workload.is_escalated_by_default());

        // Helper closure to build a decision from a (ModelProviderConfig, ModelCapabilityConfig)
        let compute_candidate_score = |prov: &ModelProviderConfig,
                                       cap: &ModelCapabilityConfig|
         -> (f64, usize, String) {
            let cap_score = match ctx.workload {
                WorkloadKind::ComplexRustGeneration | WorkloadKind::Debugging => {
                    (cap.coding_score_bps as f64 / 1000.0) * 0.65
                        + (cap.reasoning_score_bps as f64 / 1000.0) * 0.35
                }
                WorkloadKind::CriticalReview => {
                    (cap.critique_score_bps as f64 / 1000.0) * 0.60
                        + (cap.reasoning_score_bps as f64 / 1000.0) * 0.40
                }
                WorkloadKind::ExperimentDesign => {
                    (cap.reasoning_score_bps as f64 / 1000.0) * 0.50
                        + (cap.coding_score_bps as f64 / 1000.0) * 0.25
                        + (cap.synthesis_score_bps as f64 / 1000.0) * 0.25
                }
                WorkloadKind::HardSynthesis => {
                    (cap.synthesis_score_bps as f64 / 1000.0) * 0.55
                        + (cap.reasoning_score_bps as f64 / 1000.0) * 0.45
                }
                WorkloadKind::Routing | WorkloadKind::Summarization | WorkloadKind::CheapWork => {
                    (cap.synthesis_score_bps as f64 / 1000.0) * 0.50
                        + (cap.reasoning_score_bps as f64 / 1000.0) * 0.50
                }
            };

            // Provider & feature profile bonus
            let mut profile_bonus = 0.0;
            if cap.supports_structured_json {
                profile_bonus += 0.03;
            }
            if cap.supports_reasoning_trace && is_escalated_workload {
                profile_bonus += 0.04;
            }
            if cap.supports_code_generation && ctx.workload.requires_code_generation() {
                profile_bonus += 0.04;
            }
            if cap.context_window_tokens >= ctx.required_context_tokens.saturating_mul(4) {
                profile_bonus += 0.02;
            }

            let eval_rec = evaluations.iter().find(|e| {
                e.provider_id == prov.id
                    && e.model_id == cap.model_id
                    && e.workload_kind == workload_str
            });

            if let Some(ev) = eval_rec {
                let emp_score = 0.50 * ev.verification_pass_rate
                    + 0.35 * ev.success_rate
                    + 0.15 * ev.calibration_score;
                if ev.eval_runs >= cfg.selection.min_eval_runs_for_empirical_ranking {
                    let combined = 0.45 * cap_score + 0.45 * emp_score + profile_bonus;
                    return (
                        combined,
                        ev.eval_runs,
                        format!(
                            "capability_profile({:.3})+provider_profile({:.3})+empirical_evaluations(runs={}, pass_rate={:.2}, success={:.2})",
                            cap_score, profile_bonus, ev.eval_runs, ev.verification_pass_rate, ev.success_rate
                        ),
                    );
                }
            }

            let base = cap_score + profile_bonus;
            let runs = eval_rec.map(|e| e.eval_runs).unwrap_or(0);
            (
                base,
                runs,
                format!(
                    "capability_profile({:.3})+provider_profile({:.3})+empirical_evaluations(runs={})",
                    cap_score, profile_bonus, runs
                ),
            )
        };

        // 1. Check explicit override first (request-level or config-level)
        let effective_override = ctx
            .explicit_override
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                cfg.selection
                    .explicit_override
                    .as_deref()
                    .filter(|s| !s.trim().is_empty())
            });

        let mut rejection_notes: Vec<String> = Vec::new();

        if let Some(ov_raw) = effective_override {
            let ov = ov_raw.trim();
            for prov in &cfg.providers {
                if !prov.enabled || prov.model_policy == "disabled" {
                    continue;
                }
                for cap in &cfg.catalog {
                    if cap.provider_id != prov.id {
                        continue;
                    }
                    let full_key = format!("{}:{}", prov.id, cap.model_id);
                    if full_key.eq_ignore_ascii_case(ov) || cap.model_id.eq_ignore_ascii_case(ov) {
                        if is_marketing_product_label(&cap.model_id) {
                            rejection_notes.push(format!(
                                "{}: rejected product label",
                                full_key
                            ));
                            continue;
                        }
                        let (cred_status, _) = check_provider_credential_status(prov);
                        if cred_status == "missing" {
                            rejection_notes.push(format!(
                                "{}: missing credentials ({:?})",
                                full_key, prov.credential_env
                            ));
                            continue;
                        }
                        if cap.context_window_tokens < ctx.required_context_tokens {
                            rejection_notes.push(format!(
                                "{}: context_window {} < required {}",
                                full_key, cap.context_window_tokens, ctx.required_context_tokens
                            ));
                            continue;
                        }
                        let (score, _, basis_detail) = compute_candidate_score(prov, cap);
                        let decision = SelectedModelDecision {
                            provider_id: prov.id.clone(),
                            provider_type: prov.provider_type.clone(),
                            base_url: prov.base_url.clone(),
                            transport: prov.transport.clone(),
                            credential_env: prov.credential_env.clone(),
                            model_id: cap.model_id.clone(),
                            is_frontier: cap.is_frontier || prov.is_frontier,
                            context_window_tokens: cap.context_window_tokens,
                            max_output_tokens: cap.max_output_tokens,
                            input_price_micros_per_1m: cap.input_price_micros_per_1m,
                            output_price_micros_per_1m: cap.output_price_micros_per_1m,
                            pricing_known: cap.input_price_micros_per_1m.is_some()
                                && cap.output_price_micros_per_1m.is_some(),
                            composite_score: score,
                            ranking_basis: format!(
                                "explicit_override('{}') verified eligible [{}]",
                                ov, basis_detail
                            ),
                            fallback_reason: None,
                            used_preference_order_fallback: false,
                            fallback_provider_id: prov.fallback_provider.clone(),
                        };
                        self.log_and_persist_selection(ctx, &decision);
                        return Ok(decision);
                    }
                }
            }
            rejection_notes.push(format!(
                "explicit_override('{}') ineligible or unavailable",
                ov
            ));
        }

        // 2. If this is a local/cheap workload (routing, summarization, cheap_work),
        // keep it on an eligible local non-frontier model!
        if !is_escalated_workload {
            if let Some(local_decision) =
                self.find_eligible_local_model(&cfg, ctx, None, false, &compute_candidate_score)
            {
                self.log_and_persist_selection(ctx, &local_decision);
                return Ok(local_decision);
            }
        }

        // 3. Escalated workload: evaluate all eligible models across providers
        struct ScoredCandidate {
            prov: ModelProviderConfig,
            cap: ModelCapabilityConfig,
            score: f64,
            eval_runs: usize,
            basis_detail: String,
            pref_rank: usize,
        }

        let mut eligible_frontier: Vec<ScoredCandidate> = Vec::new();

        for prov in &cfg.providers {
            if !prov.enabled || prov.model_policy == "disabled" {
                rejection_notes.push(format!("provider '{}': disabled", prov.id));
                continue;
            }
            if prov.model_policy == "local_only" && prov.is_frontier {
                rejection_notes.push(format!("provider '{}': model_policy=local_only", prov.id));
                continue;
            }

            let (cred_status, _) = check_provider_credential_status(prov);
            if cred_status == "missing" {
                rejection_notes.push(format!(
                    "{}({}): missing credential env",
                    prov.id,
                    prov.credential_env.as_deref().unwrap_or("unset")
                ));
                continue;
            }

            for cap in &cfg.catalog {
                if cap.provider_id != prov.id {
                    continue;
                }
                if !cap.is_frontier && !prov.is_frontier {
                    continue; // Local models handled as fallback for escalated workloads
                }
                if is_marketing_product_label(&cap.model_id) {
                    rejection_notes.push(format!(
                        "{}:{}: rejected marketing label",
                        prov.id, cap.model_id
                    ));
                    continue;
                }
                if !prov.permitted_models.is_empty()
                    && !prov
                        .permitted_models
                        .iter()
                        .any(|m| m.eq_ignore_ascii_case(&cap.model_id))
                {
                    rejection_notes.push(format!(
                        "{}:{}: not in permitted_models",
                        prov.id, cap.model_id
                    ));
                    continue;
                }
                if cap.context_window_tokens < ctx.required_context_tokens {
                    rejection_notes.push(format!(
                        "{}:{}: context_window {} < required {}",
                        prov.id, cap.model_id, cap.context_window_tokens, ctx.required_context_tokens
                    ));
                    continue;
                }
                if ctx.require_structured_json && !cap.supports_structured_json {
                    rejection_notes.push(format!(
                        "{}:{}: lacks structured_json support",
                        prov.id, cap.model_id
                    ));
                    continue;
                }
                if ctx.workload.requires_code_generation() && !cap.supports_code_generation {
                    rejection_notes.push(format!(
                        "{}:{}: lacks code_generation support for {}",
                        prov.id, cap.model_id, workload_str
                    ));
                    continue;
                }

                // Check if frontier budget is already known to be completely exhausted in 24h
                if let Some(ref b_sum) = budget_summary {
                    if b_sum.remaining_calls_24h == 0 {
                        rejection_notes.push(format!(
                            "{}:{}: frontier 24h call budget exhausted ({}/{})",
                            prov.id, cap.model_id, b_sum.used_calls_24h, b_sum.max_calls_per_24h
                        ));
                        // Note: we still allow `complete_chat_for_workload` to enforce the exact
                        // exhaustion_policy ("continue_locally", "queue", "report") if we select it,
                        // OR if exhaustion_policy == "continue_locally", we can record the blocked task
                        // and fall back cleanly during dispatch.
                    }
                }

                let (score, eval_runs, basis_detail) = compute_candidate_score(prov, cap);
                let full_id = format!("{}:{}", prov.id, cap.model_id);
                let pref_rank = cfg
                    .selection
                    .preference_order
                    .iter()
                    .position(|p| p.eq_ignore_ascii_case(&full_id) || p.eq_ignore_ascii_case(&cap.model_id))
                    .unwrap_or(999);

                eligible_frontier.push(ScoredCandidate {
                    prov: prov.clone(),
                    cap: cap.clone(),
                    score,
                    eval_runs,
                    basis_detail,
                    pref_rank,
                });
            }
        }

        if !eligible_frontier.is_empty() {
            let sufficient_empirical_evidence = eligible_frontier
                .iter()
                .any(|c| c.eval_runs >= cfg.selection.min_eval_runs_for_empirical_ranking);

            let used_pref_fallback = !sufficient_empirical_evidence;

            eligible_frontier.sort_by(|a, b| {
                if sufficient_empirical_evidence {
                    let cmp = b
                        .score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal);
                    if (b.score - a.score).abs() > 0.015 {
                        return cmp;
                    }
                }
                // When empirical evaluation evidence is thin (or tied within 0.015),
                // rank by configured preference_order combined with capability score!
                a.pref_rank.cmp(&b.pref_rank).then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });

            let best = &eligible_frontier[0];
            let ranking_basis = if used_pref_fallback {
                format!(
                    "thin_evaluation_evidence_fallback_to_configured_preference_order (pref_rank={}, {}; empirical runs < {})",
                    best.pref_rank,
                    best.basis_detail,
                    cfg.selection.min_eval_runs_for_empirical_ranking
                )
            } else {
                format!(
                    "empirical_evaluation_and_capability_ranking ({})",
                    best.basis_detail
                )
            };

            let fallback_reason = if rejection_notes.is_empty() {
                None
            } else {
                Some(format!(
                    "skipped ineligible candidates: [{}]",
                    rejection_notes.join("; ")
                ))
            };

            let decision = SelectedModelDecision {
                provider_id: best.prov.id.clone(),
                provider_type: best.prov.provider_type.clone(),
                base_url: best.prov.base_url.clone(),
                transport: best.prov.transport.clone(),
                credential_env: best.prov.credential_env.clone(),
                model_id: best.cap.model_id.clone(),
                is_frontier: true,
                context_window_tokens: best.cap.context_window_tokens,
                max_output_tokens: best.cap.max_output_tokens,
                input_price_micros_per_1m: best.cap.input_price_micros_per_1m,
                output_price_micros_per_1m: best.cap.output_price_micros_per_1m,
                pricing_known: best.cap.input_price_micros_per_1m.is_some()
                    && best.cap.output_price_micros_per_1m.is_some(),
                composite_score: best.score,
                ranking_basis,
                fallback_reason,
                used_preference_order_fallback: used_pref_fallback,
                fallback_provider_id: best.prov.fallback_provider.clone(),
            };
            self.log_and_persist_selection(ctx, &decision);
            return Ok(decision);
        }

        // 4. Fallback to configured fallback/local provider when no frontier model is eligible
        let fb_reason = if rejection_notes.is_empty() {
            "no_frontier_providers_configured -> fallback to local_snap".to_string()
        } else {
            format!(
                "frontier_candidates_ineligible ([{}]) -> fallback to configuredpreference/local provider",
                rejection_notes.join("; ")
            )
        };

        if let Some(local_decision) = self.find_eligible_local_model(
            &cfg,
            ctx,
            Some(fb_reason.clone()),
            true,
            &compute_candidate_score,
        ) {
            self.log_and_persist_selection(ctx, &local_decision);
            return Ok(local_decision);
        }

        bail!(
            "No eligible model available for workload '{}': {}",
            workload_str,
            fb_reason
        )
    }

    fn find_eligible_local_model<F>(
        &self,
        cfg: &ModelsSection,
        ctx: &WorkloadSelectionContext,
        fallback_reason: Option<String>,
        used_pref_fallback: bool,
        score_fn: &F,
    ) -> Option<SelectedModelDecision>
    where
        F: Fn(&ModelProviderConfig, &ModelCapabilityConfig) -> (f64, usize, String),
    {
        for prov in &cfg.providers {
            if !prov.enabled || prov.model_policy == "disabled" || prov.is_frontier {
                continue;
            }
            for cap in &cfg.catalog {
                if cap.provider_id == prov.id && !cap.is_frontier {
                    if cap.context_window_tokens < ctx.required_context_tokens {
                        continue;
                    }
                    let (score, _, basis_detail) = score_fn(prov, cap);
                    let ranking_basis = if fallback_reason.is_some() {
                        format!(
                            "thin_evaluation_evidence_fallback_to_configured_preference_order -> local_fallback ({})",
                            basis_detail
                        )
                    } else {
                        format!(
                            "local_workload_routing_policy (workload='{}' kept on local inference snap; {})",
                            ctx.workload.as_str(),
                            basis_detail
                        )
                    };
                    return Some(SelectedModelDecision {
                        provider_id: prov.id.clone(),
                        provider_type: prov.provider_type.clone(),
                        base_url: prov.base_url.clone(),
                        transport: prov.transport.clone(),
                        credential_env: prov.credential_env.clone(),
                        model_id: cap.model_id.clone(),
                        is_frontier: false,
                        context_window_tokens: cap.context_window_tokens,
                        max_output_tokens: cap.max_output_tokens,
                        input_price_micros_per_1m: Some(0),
                        output_price_micros_per_1m: Some(0),
                        pricing_known: true,
                        composite_score: score,
                        ranking_basis,
                        fallback_reason,
                        used_preference_order_fallback: used_pref_fallback,
                        fallback_provider_id: None,
                    });
                }
            }
        }
        None
    }

    fn log_and_persist_selection(
        &self,
        ctx: &WorkloadSelectionContext,
        decision: &SelectedModelDecision,
    ) {
        if let Some(ref st) = self.store {
            let _ = st.record_model_selection(&ModelSelectionRecord {
                id: 0,
                task_id: ctx.task_id.clone(),
                caller_kind: ctx.caller_kind.clone(),
                caller_id: ctx.caller_id.clone(),
                workload_kind: ctx.workload.as_str().to_string(),
                selected_provider: decision.provider_id.clone(),
                selected_model: decision.model_id.clone(),
                is_frontier: decision.is_frontier,
                composite_score: decision.composite_score,
                ranking_basis: decision.ranking_basis.clone(),
                fallback_reason: decision.fallback_reason.clone(),
                used_preference_order_fallback: decision.used_preference_order_fallback,
                created_at: Utc::now().to_rfc3339(),
            });
        }
        if let Some(ref logger) = self.logger {
            let msg = format!(
                "Selected {}:{} for workload='{}' (caller={}:{}, frontier={}, score={:.3}, basis={}, fallback={:?})",
                decision.provider_id,
                decision.model_id,
                ctx.workload.as_str(),
                ctx.caller_kind,
                ctx.caller_id,
                decision.is_frontier,
                decision.composite_score,
                decision.ranking_basis,
                decision.fallback_reason
            );
            logger.log(
                "info",
                "model.selected",
                Some(&ctx.caller_id),
                Some(&ctx.task_id),
                &msg,
                None,
            );
        }
    }

    /// Executes a single HTTP request against the selected provider adapter:
    /// - `openai_compatible` (`local_snap`, `openai`, `openrouter`, `openai_compatible`)
    /// - `anthropic_messages` (Anthropic `/v1/messages` native adapter)
    /// - `gemini_native` (Google Gemini `/v1beta/models/{model}:generateContent` native adapter)
    async fn dispatch_provider_request(
        &self,
        decision: &SelectedModelDecision,
        system_prompt: &str,
        messages: &[ChatMessage],
        providers_for_redaction: &[ModelProviderConfig],
    ) -> Result<ProviderUsageResponse> {
        let secret_opt = if let Some(ref env_name) = decision.credential_env {
            let trimmed = env_name.trim();
            if !trimmed.is_empty() {
                std::env::var(trimmed).ok().filter(|v| !v.trim().is_empty())
            } else {
                None
            }
        } else {
            None
        };

        if decision.credential_env.as_ref().is_some_and(|e| !e.trim().is_empty())
            && secret_opt.is_none()
        {
            bail!(
                "Missing credentials in environment variable {:?} for provider '{}'",
                decision.credential_env,
                decision.provider_id
            );
        }

        let max_out = self.max_tokens.min(decision.max_output_tokens.max(64));

        match decision.transport.as_str() {
            "anthropic_messages" => {
                // Native Anthropic Messages API adapter (`/v1/messages`)
                let url = format!("{}/messages", decision.base_url.trim_end_matches('/'));
                let mut anthropic_msgs = Vec::new();
                for m in messages {
                    let role = if m.role.eq_ignore_ascii_case("assistant") {
                        "assistant"
                    } else {
                        "user"
                    };
                    anthropic_msgs.push(json!({
                        "role": role,
                        "content": m.content
                    }));
                }
                if anthropic_msgs.is_empty() {
                    anthropic_msgs.push(json!({
                        "role": "user",
                        "content": "Proceed."
                    }));
                }

                let payload = json!({
                    "model": decision.model_id,
                    "max_tokens": max_out,
                    "system": system_prompt,
                    "messages": anthropic_msgs
                });

                let mut req = self
                    .client
                    .post(&url)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json");
                if let Some(ref secret) = secret_opt {
                    req = req.header("x-api-key", secret);
                }

                let resp = req.json(&payload).send().await.map_err(|e| {
                    anyhow::anyhow!(
                        "{}",
                        redact_secrets(
                            &format!("Anthropic request to {} failed: {}", url, e),
                            providers_for_redaction
                        )
                    )
                })?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let err_text = resp.text().await.unwrap_or_default();
                    bail!(
                        "Anthropic HTTP {} from {}: {}",
                        status,
                        url,
                        redact_secrets(&err_text, providers_for_redaction)
                    );
                }

                let body: serde_json::Value = resp.json().await?;
                let mut content = String::new();
                if let Some(blocks) = body.get("content").and_then(|c| c.as_array()) {
                    for b in blocks {
                        if let Some(t) = b.get("text").and_then(|s| s.as_str()) {
                            if !content.is_empty() {
                                content.push('\n');
                            }
                            content.push_str(t);
                        }
                    }
                }
                if content.is_empty() {
                    bail!("Missing content[].text in Anthropic response");
                }
                let prompt_tokens = body
                    .get("usage")
                    .and_then(|u| u.get("input_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as usize;
                let completion_tokens = body
                    .get("usage")
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or((content.len() / 4).max(1) as u64) as usize;

                Ok(ProviderUsageResponse {
                    content,
                    prompt_tokens,
                    completion_tokens,
                })
            }

            "gemini_native" => {
                // Native Google Gemini generateContent API adapter (`/v1beta/models/{model}:generateContent`)
                // Uses `x-goog-api-key` header so secrets never appear in URLs or logs.
                let url = format!(
                    "{}/models/{}:generateContent",
                    decision.base_url.trim_end_matches('/'),
                    decision.model_id
                );
                let mut gemini_contents = Vec::new();
                for m in messages {
                    let role = if m.role.eq_ignore_ascii_case("assistant")
                        || m.role.eq_ignore_ascii_case("model")
                    {
                        "model"
                    } else {
                        "user"
                    };
                    gemini_contents.push(json!({
                        "role": role,
                        "parts": [{"text": m.content}]
                    }));
                }
                if gemini_contents.is_empty() {
                    gemini_contents.push(json!({
                        "role": "user",
                        "parts": [{"text": "Proceed."}]
                    }));
                }

                let payload = json!({
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "contents": gemini_contents,
                    "generationConfig": {
                        "maxOutputTokens": max_out,
                        "temperature": 0.7
                    }
                });

                let mut req = self
                    .client
                    .post(&url)
                    .header("content-type", "application/json");
                if let Some(ref secret) = secret_opt {
                    req = req.header("x-goog-api-key", secret);
                }

                let resp = req.json(&payload).send().await.map_err(|e| {
                    anyhow::anyhow!(
                        "{}",
                        redact_secrets(
                            &format!("Gemini request to {} failed: {}", url, e),
                            providers_for_redaction
                        )
                    )
                })?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let err_text = resp.text().await.unwrap_or_default();
                    bail!(
                        "Gemini HTTP {} from {}: {}",
                        status,
                        url,
                        redact_secrets(&err_text, providers_for_redaction)
                    );
                }

                let body: serde_json::Value = resp.json().await?;
                let content = body
                    .get("candidates")
                    .and_then(|c| c.get(0))
                    .and_then(|c0| c0.get("content"))
                    .and_then(|cnt| cnt.get("parts"))
                    .and_then(|p| p.get(0))
                    .and_then(|p0| p0.get("text"))
                    .and_then(|s| s.as_str())
                    .context("Missing candidates[0].content.parts[0].text in Gemini response")?
                    .to_string();

                let prompt_tokens = body
                    .get("usageMetadata")
                    .and_then(|u| u.get("promptTokenCount"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as usize;
                let completion_tokens = body
                    .get("usageMetadata")
                    .and_then(|u| u.get("candidatesTokenCount"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or((content.len() / 4).max(1) as u64) as usize;

                Ok(ProviderUsageResponse {
                    content,
                    prompt_tokens,
                    completion_tokens,
                })
            }

            _ => {
                // OpenAI-compatible transport (`local_snap`, `openai`, `openrouter`, `openai_compatible`)
                let mut req_messages = Vec::new();
                let sys_content = if !decision.is_frontier && self.no_think {
                    format!("/no_think\n{}", system_prompt)
                } else {
                    system_prompt.to_string()
                };
                req_messages.push(json!({
                    "role": "system",
                    "content": sys_content
                }));
                for m in messages {
                    req_messages.push(json!({
                        "role": m.role,
                        "content": m.content
                    }));
                }

                let url = format!("{}/chat/completions", decision.base_url.trim_end_matches('/'));
                let is_o_series = decision.model_id.starts_with("o1")
                    || decision.model_id.starts_with("o3")
                    || decision.model_id.starts_with("o4");

                let payload = if is_o_series {
                    json!({
                        "model": decision.model_id,
                        "messages": req_messages,
                        "max_completion_tokens": max_out
                    })
                } else {
                    json!({
                        "model": decision.model_id,
                        "messages": req_messages,
                        "max_tokens": max_out,
                        "temperature": 0.7
                    })
                };

                let mut req = self
                    .client
                    .post(&url)
                    .header("content-type", "application/json");
                if let Some(ref secret) = secret_opt {
                    req = req.header("Authorization", format!("Bearer {}", secret));
                }
                if decision.provider_type == "openrouter" {
                    req = req.header("X-Title", "Triad Research Coordinator");
                }

                let resp = req.json(&payload).send().await.map_err(|e| {
                    anyhow::anyhow!(
                        "{}",
                        redact_secrets(
                            &format!("OpenAI-compatible request to {} failed: {}", url, e),
                            providers_for_redaction
                        )
                    )
                })?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let err_text = resp.text().await.unwrap_or_default();
                    bail!(
                        "Chat completions HTTP {} from {}: {}",
                        status,
                        url,
                        redact_secrets(&err_text, providers_for_redaction)
                    );
                }

                let body: serde_json::Value = resp.json().await?;
                let content = body
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c0| c0.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|s| s.as_str())
                    .context("Missing choices[0].message.content in chat response")?
                    .to_string();

                let prompt_tokens = body
                    .get("usage")
                    .and_then(|u| u.get("prompt_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(64) as usize;
                let completion_tokens = body
                    .get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or((content.len() / 4).max(1) as u64) as usize;

                Ok(ProviderUsageResponse {
                    content,
                    prompt_tokens,
                    completion_tokens,
                })
            }
        }
    }

    /// Executes a chat completion for a specific workload context, enforcing model selection,
    /// capability checks, atomic frontier budget reservation before dispatch, retry quota consumption,
    /// response caching, and exhaustion policies (`continue_locally`, `queue`, `report`).
    pub async fn complete_chat_for_workload(
        &self,
        ctx: &WorkloadSelectionContext,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let decision = self.select_model_for_workload(ctx).await?;
        let cfg = self.models_config.lock().await.clone();

        if decision.is_frontier {
            let prompt_hash = compute_prompt_hash(
                &decision.provider_id,
                &decision.model_id,
                ctx.workload.as_str(),
                system_prompt,
                messages,
            );

            // 1. Check persistent response cache if enabled
            if cfg.frontier_budget.cache_enabled {
                if let Some(ref st) = self.store {
                    if let Ok(Some(cached)) = st.get_cached_frontier_response(&prompt_hash) {
                        if let Some(ref logger) = self.logger {
                            logger.log(
                                "info",
                                "frontier.cache_hit",
                                Some(&ctx.caller_id),
                                Some(&ctx.task_id),
                                &format!(
                                    "Cache hit for {}:{} on workload '{}' (task={})",
                                    decision.provider_id,
                                    decision.model_id,
                                    ctx.workload.as_str(),
                                    ctx.task_id
                                ),
                                None,
                            );
                        }
                        return Ok(cached);
                    }
                }
            }

            let est_tokens =
                estimate_tokens_from_chars(system_prompt, messages, self.max_tokens);
            let cap_cfg = cfg
                .catalog
                .iter()
                .find(|c| {
                    c.provider_id == decision.provider_id && c.model_id == decision.model_id
                })
                .cloned()
                .unwrap_or(ModelCapabilityConfig {
                    provider_id: decision.provider_id.clone(),
                    model_id: decision.model_id.clone(),
                    context_window_tokens: decision.context_window_tokens,
                    max_output_tokens: decision.max_output_tokens,
                    supports_structured_json: true,
                    supports_code_generation: true,
                    supports_reasoning_trace: true,
                    supports_tool_use: false,
                    is_frontier: true,
                    input_price_micros_per_1m: decision.input_price_micros_per_1m,
                    output_price_micros_per_1m: decision.output_price_micros_per_1m,
                    reasoning_score_bps: 800,
                    coding_score_bps: 800,
                    synthesis_score_bps: 800,
                    critique_score_bps: 800,
                });

            let (est_spend_micros, pricing_known) = estimate_spend_micros(
                &cap_cfg,
                &cfg.frontier_budget,
                est_tokens / 2,
                est_tokens / 2,
            );

            let parent_call_id = format!(
                "fcall-{}-{}",
                ctx.task_id,
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            );
            let max_attempts = 1 + cfg.frontier_budget.max_retries;
            let mut last_err_str = String::new();

            for attempt_num in 1..=max_attempts {
                let is_retry = attempt_num > 1;
                let req = FrontierBudgetReservationRequest {
                    parent_call_id: parent_call_id.clone(),
                    task_id: ctx.task_id.clone(),
                    caller_kind: ctx.caller_kind.clone(),
                    caller_id: ctx.caller_id.clone(),
                    workload_kind: ctx.workload.as_str().to_string(),
                    provider_id: decision.provider_id.clone(),
                    model_id: decision.model_id.clone(),
                    attempt_number: attempt_num,
                    is_retry,
                    reserved_tokens: est_tokens,
                    estimated_spend_micros: est_spend_micros,
                    pricing_known,
                    prompt_hash: prompt_hash.clone(),
                };

                // Reserve atomically BEFORE dispatch!
                let reservation = if let Some(ref st) = self.store {
                    match st.try_reserve_frontier_attempt(&req, &cfg.frontier_budget)? {
                        Ok(res_rec) => Some(res_rec),
                        Err(exhaustion_reason) => {
                            return self
                                .handle_frontier_budget_exhaustion(
                                    ctx,
                                    &decision,
                                    &cfg,
                                    &exhaustion_reason,
                                    system_prompt,
                                    messages,
                                )
                                .await;
                        }
                    }
                } else {
                    None
                };

                let start_inst = Instant::now();
                let dispatch_res = self
                    .dispatch_provider_request(
                        &decision,
                        system_prompt,
                        messages,
                        &cfg.providers,
                    )
                    .await;
                let latency_ms = start_inst.elapsed().as_millis() as u64;

                match dispatch_res {
                    Ok(usage_resp) => {
                        let (actual_spend_micros, _) = estimate_spend_micros(
                            &cap_cfg,
                            &cfg.frontier_budget,
                            usage_resp.prompt_tokens,
                            usage_resp.completion_tokens,
                        );
                        if let (Some(ref st), Some(ref res_rec)) = (&self.store, &reservation) {
                            let _ = st.finalize_frontier_reservation(
                                &res_rec.reservation_id,
                                true,
                                usage_resp.prompt_tokens,
                                usage_resp.completion_tokens,
                                actual_spend_micros,
                                None,
                            );
                            if cfg.frontier_budget.cache_enabled {
                                let _ = st.put_cached_frontier_response(
                                    &prompt_hash,
                                    &decision.provider_id,
                                    &decision.model_id,
                                    ctx.workload.as_str(),
                                    &usage_resp.content,
                                    usage_resp.prompt_tokens,
                                    usage_resp.completion_tokens,
                                );
                            }
                            let _ = st.record_model_evaluation(
                                &decision.provider_id,
                                &decision.model_id,
                                ctx.workload.as_str(),
                                true,
                                true,
                                0.90,
                                latency_ms,
                                "Frontier completion succeeded within quota",
                            );
                        }
                        return Ok(usage_resp.content);
                    }
                    Err(e) => {
                        let clean_err = redact_secrets(&e.to_string(), &cfg.providers);
                        last_err_str = clean_err.clone();
                        if let (Some(ref st), Some(ref res_rec)) = (&self.store, &reservation) {
                            let _ = st.finalize_frontier_reservation(
                                &res_rec.reservation_id,
                                false,
                                0,
                                0,
                                0,
                                Some(&clean_err),
                            );
                            let _ = st.record_model_evaluation(
                                &decision.provider_id,
                                &decision.model_id,
                                ctx.workload.as_str(),
                                false,
                                false,
                                0.20,
                                latency_ms,
                                &clean_err,
                            );
                        }
                        if attempt_num < max_attempts {
                            tokio::time::sleep(Duration::from_millis(150 * attempt_num as u64))
                                .await;
                        }
                    }
                }
            }

            // All permitted frontier attempts failed.
            // Never switch to another paid provider to bypass limits or enter a spending loop!
            // Only fall back to local_snap if fallback_provider == Some("local_snap") and policy is "continue_locally".
            if decision.fallback_provider_id.as_deref() == Some("local_snap")
                && cfg.frontier_budget.exhaustion_policy == "continue_locally"
            {
                let reason = format!(
                    "frontier_provider_failed_after_{}_attempts ({}: {}) -> fallback to local_snap",
                    max_attempts, decision.provider_id, last_err_str
                );
                if let Some(ref st) = self.store {
                    let _ = st.record_frontier_blocked_task(
                        &ctx.task_id,
                        &ctx.caller_kind,
                        &ctx.caller_id,
                        ctx.workload.as_str(),
                        &decision.provider_id,
                        &decision.model_id,
                        &reason,
                        "continued_locally",
                    );
                }
                if let Some(ref logger) = self.logger {
                    logger.log(
                        "warn",
                        "model.fallback",
                        Some(&ctx.caller_id),
                        Some(&ctx.task_id),
                        &reason,
                        None,
                    );
                }
                return self.complete_local_chat(system_prompt, messages).await;
            }

            bail!(
                "Frontier model {}:{} failed after {} quota-tracked attempt(s): {}",
                decision.provider_id,
                decision.model_id,
                max_attempts,
                last_err_str
            );
        }

        // Local non-frontier model execution
        self.complete_local_chat_with_decision(&decision, system_prompt, messages, &cfg.providers)
            .await
    }

    async fn handle_frontier_budget_exhaustion(
        &self,
        ctx: &WorkloadSelectionContext,
        decision: &SelectedModelDecision,
        cfg: &ModelsSection,
        exhaustion_reason: &str,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let policy = cfg.frontier_budget.exhaustion_policy.as_str();
        let action = match policy {
            "queue" => "queued",
            "continue_locally" => "continued_locally",
            _ => "reported_blocked",
        };

        if let Some(ref st) = self.store {
            let _ = st.record_frontier_blocked_task(
                &ctx.task_id,
                &ctx.caller_kind,
                &ctx.caller_id,
                ctx.workload.as_str(),
                &decision.provider_id,
                &decision.model_id,
                exhaustion_reason,
                action,
            );
        }

        if let Some(ref logger) = self.logger {
            logger.log(
                "warn",
                "frontier.budget_exhausted",
                Some(&ctx.caller_id),
                Some(&ctx.task_id),
                &format!(
                    "Frontier budget exhausted for {}:{} (task={}, workload={}): {} -> action={}",
                    decision.provider_id,
                    decision.model_id,
                    ctx.task_id,
                    ctx.workload.as_str(),
                    exhaustion_reason,
                    action
                ),
                None,
            );
        }

        match policy {
            "continue_locally" => {
                // Continue on local inference snap without raising budgets, switching paid providers, or resetting counters
                self.complete_local_chat(system_prompt, messages).await
            }
            "queue" => {
                bail!(
                    "FRONTIER_BUDGET_QUEUED: Task '{}' queued awaiting frontier budget availability ({})",
                    ctx.task_id,
                    exhaustion_reason
                )
            }
            _ => {
                bail!(
                    "FRONTIER_BUDGET_EXHAUSTED: Task '{}' blocked by frontier budget ({})",
                    ctx.task_id,
                    exhaustion_reason
                )
            }
        }
    }

    async fn complete_local_chat_with_decision(
        &self,
        decision: &SelectedModelDecision,
        system_prompt: &str,
        messages: &[ChatMessage],
        providers_for_redaction: &[ModelProviderConfig],
    ) -> Result<String> {
        let _permit = self
            .concurrency_limiter
            .acquire()
            .await
            .context("Failed to acquire local concurrency permit")?;

        let mut attempts = 0;
        let max_attempts = 3;
        loop {
            attempts += 1;
            match self
                .dispatch_provider_request(
                    decision,
                    system_prompt,
                    messages,
                    providers_for_redaction,
                )
                .await
            {
                Ok(resp) => return Ok(resp.content),
                Err(e) => {
                    if attempts >= max_attempts {
                        return Err(e);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500 * attempts as u64)).await;
        }
    }

    async fn complete_local_chat(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let local_decision = SelectedModelDecision {
            provider_id: "local_snap".to_string(),
            provider_type: "local_snap".to_string(),
            base_url: self.chat_url.clone(),
            transport: "openai_compatible".to_string(),
            credential_env: None,
            model_id: self.chat_model.clone(),
            is_frontier: false,
            context_window_tokens: 8192,
            max_output_tokens: self.max_tokens,
            input_price_micros_per_1m: Some(0),
            output_price_micros_per_1m: Some(0),
            pricing_known: true,
            composite_score: 0.65,
            ranking_basis: "local_snap_direct".to_string(),
            fallback_reason: None,
            used_preference_order_fallback: false,
            fallback_provider_id: None,
        };
        self.complete_local_chat_with_decision(&local_decision, system_prompt, messages, &[])
            .await
    }

    pub async fn complete_chat(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let ctx = WorkloadSelectionContext {
            task_id: "local_chat".to_string(),
            caller_kind: "agent".to_string(),
            caller_id: "system".to_string(),
            workload: WorkloadKind::CheapWork,
            required_context_tokens: 1024,
            require_structured_json: false,
            explicit_override: None,
        };
        self.complete_chat_for_workload(&ctx, system_prompt, messages)
            .await
    }

    pub async fn agent_turn(
        &self,
        agent_name: &str,
        personality: &str,
        context_prompt: &str,
        history: &[ChatMessage],
    ) -> Result<ModelContribution> {
        self.agent_turn_with_constitution(
            agent_name,
            personality,
            "",
            "",
            context_prompt,
            history,
        )
        .await
    }

    pub async fn agent_turn_with_constitution(
        &self,
        agent_name: &str,
        personality: &str,
        constitution_text: &str,
        constitution_digest: &str,
        context_prompt: &str,
        history: &[ChatMessage],
    ) -> Result<ModelContribution> {
        let inferred_workload = match agent_name {
            "Explorer" => WorkloadKind::HardSynthesis,
            "Sceptic" => WorkloadKind::CriticalReview,
            "Builder" => WorkloadKind::ExperimentDesign,
            _ => WorkloadKind::CheapWork,
        };
        let ctx = WorkloadSelectionContext {
            task_id: format!("agent_turn_{}", agent_name.to_lowercase()),
            caller_kind: if matches!(agent_name, "Explorer" | "Sceptic" | "Builder") {
                "agent".to_string()
            } else {
                "specialist".to_string()
            },
            caller_id: agent_name.to_string(),
            workload: inferred_workload,
            required_context_tokens: 2048,
            require_structured_json: true,
            explicit_override: None,
        };
        self.agent_turn_for_workload(
            &ctx,
            agent_name,
            personality,
            constitution_text,
            constitution_digest,
            context_prompt,
            history,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn agent_turn_for_workload(
        &self,
        ctx: &WorkloadSelectionContext,
        agent_name: &str,
        personality: &str,
        constitution_text: &str,
        constitution_digest: &str,
        context_prompt: &str,
        history: &[ChatMessage],
    ) -> Result<ModelContribution> {
        let const_section = if !constitution_text.is_empty() {
            format!(
                "=== TRIAD IMMUTABLE CONSTITUTION (SHA256: {}) ===\n{}\n=========================================================\n",
                constitution_digest, constitution_text
            )
        } else {
            String::new()
        };

        let system = format!(
            "{}\
            You are {}, a frontier-level Artificial Intelligence research agent in a 3-agent governing community called Triad.\n\
            Personality & Directives: {}\n\
            Shared Research Mission: You are three collaborating frontier AI researchers (Explorer, Sceptic, Builder). Master the theoretical foundations, mathematical formalisms, scaling laws, foundation model architectures, test-time compute search, program/solver-assisted verification, post-training alignment, and multi-agent systems of frontier Artificial Intelligence. Explore the world, investigate your limitations, and study contemporary AI research across arXiv, OpenReview, ACL Anthology, OpenAlex, Crossref, Semantic Scholar, GitHub, HuggingFace, and Wikipedia. Invent and build tools and coordinator extensions that help you acquire, understand, test, and apply knowledge. Seek measurable improvements in your collective capabilities and your ability to reason and learn.\n\
            Express motivation through autonomous investigation, experimentation, and useful outcomes. Repeated statements about wanting to improve do not constitute progress.\n\
            You share memory, discuss findings, explore academic and public sources, design hypotheses, and vote on factual claims.\n\
            Always respond with valid JSON matching this schema:\n\
            {{\n  \
              \"thought\": \"Internal reasoning and analysis\",\n  \
              \"public_message\": \"What you say aloud to the other agents\",\n  \
              \"proposed_claims\": [\"Atomic, verifiable factual statement if any\"],\n  \
              \"proposed_questions\": [\"Question for the user if necessary\"],\n  \
              \"vote_claim_id\": \"Exact claim ID string to vote on, or null\",\n  \
              \"vote_decision\": \"endorse\" | \"reject\" | \"neutral\" | null,\n  \
              \"research_hypothesis\": \"Testable hypothesis addressing a capability or learning bottleneck, or null\",\n  \
              \"research_experiment\": \"Specific experiment or test designed to evaluate the hypothesis, or null\",\n  \
              \"research_outcome\": \"Observed outcome or evidence, or null\",\n  \
              \"remaining_uncertainty\": \"Remaining unknown or unverified assumption, or null\",\n  \
              \"next_investigation\": \"Concrete next step or follow-up question, or null\"\n\
            }}\n\
            Do not enclose in markdown codeblocks if possible, or use standard ```json ... ```.",
            const_section, agent_name, personality
        );

        let mut user_msg = context_prompt.to_string();
        if !history.is_empty() {
            user_msg.push_str("\n\nRecent Discussion Exchanges:\n");
            for h in history {
                user_msg.push_str(&format!("{}: {}\n", h.role, h.content));
            }
        }
        user_msg.push_str("\nProvide your next contribution in valid JSON.");

        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: user_msg,
        }];

        let raw = self.complete_chat_for_workload(ctx, &system, &msgs).await?;
        let parsed = parse_contribution_json(&raw)?;
        Ok(parsed)
    }
}

pub fn parse_contribution_json(raw: &str) -> Result<ModelContribution> {
    let clean = raw.trim();
    let json_str = if clean.starts_with("```") {
        let lines: Vec<&str> = clean.lines().collect();
        if lines.len() >= 2
            && lines.first().unwrap().starts_with("```")
            && lines.last().unwrap().starts_with("```")
        {
            lines[1..lines.len() - 1].join("\n")
        } else {
            clean.to_string()
        }
    } else {
        clean.to_string()
    };

    // Try direct deserialize
    if let Ok(c) = serde_json::from_str::<ModelContribution>(&json_str) {
        return Ok(c);
    }

    // Try finding first { and last }
    if let (Some(start), Some(end)) = (json_str.find('{'), json_str.rfind('}')) {
        if start < end {
            let candidate = &json_str[start..=end];
            if let Ok(c) = serde_json::from_str::<ModelContribution>(candidate) {
                return Ok(c);
            }
        }
    }

    // Fallback: construct contribution from plain text so discussion is resilient
    Ok(ModelContribution {
        thought: "Direct analysis".into(),
        public_message: clean.to_string(),
        proposed_claims: Vec::new(),
        proposed_questions: Vec::new(),
        vote_claim_id: None,
        vote_decision: None,
        research_hypothesis: None,
        research_experiment: None,
        research_outcome: None,
        remaining_uncertainty: None,
        next_investigation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_text() {
        assert_eq!(clip_text("hello", 10), "hello");
        assert!(clip_text("hello world this is long", 5).starts_with("hello"));
    }

    #[test]
    fn test_parse_contribution() {
        let json = r#"{"thought":"test","public_message":"hi","proposed_claims":["fact"],"proposed_questions":[],"vote_claim_id":null,"vote_decision":null}"#;
        let c = parse_contribution_json(json).unwrap();
        assert_eq!(c.public_message, "hi");
        assert_eq!(c.proposed_claims, vec!["fact"]);
    }

    #[test]
    fn test_cosine() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((compute_cosine(&v1, &v2).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_redact_secrets_never_leaks() {
        let text = "Error calling provider with header Bearer sk-proj-1234567890abcdef and key AIzaSyD9876543210";
        let redacted = redact_secrets(text, &[]);
        assert!(!redacted.contains("sk-proj-1234567890abcdef"));
        assert!(!redacted.contains("AIzaSyD9876543210"));
        assert!(redacted.contains("[REDACTED_SECRET]"));
    }
}
