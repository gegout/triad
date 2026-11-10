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

[models.selection]
# Explicit override ("provider_id:model_id") when set; leave empty for capability+evaluation selection
explicit_override = ""
# Preference order fallback when empirical evaluation evidence is thin
preference_order = [
    "anthropic:claude-sonnet-4-20250514",
    "openai:gpt-4.1",
    "gemini:gemini-2.5-pro",
    "openrouter:deepseek/deepseek-r1",
    "local_snap:smollm3-3b"
]
min_eval_runs_for_empirical_ranking = 2
discovery_refresh_interval_seconds = 3600
local_workloads = ["routing", "summarization", "cheap_work"]
escalated_workloads = [
    "hard_synthesis",
    "experiment_design",
    "complex_rust_generation",
    "debugging",
    "critical_review"
]

[models.frontier_budget]
# Persistent, atomic accounting shared across all agents, specialists, background cycles, and MCP consultations.
# Defaults: 3 attempts per rolling 24h, 1 per task, concurrency 1, one retry. Retries consume quota.
max_calls_per_24h = 3
max_calls_per_task = 1
max_concurrency = 1
max_retries = 1
max_tokens_per_24h = 64000
max_tokens_per_task = 16000
max_spend_micros_per_24h = 2000000
# Unknown billing counts conservatively; unknown prices are never free.
conservative_unknown_call_micros = 250000
conservative_unknown_input_micros_per_1m = 15000000
conservative_unknown_output_micros_per_1m = 75000000
# On exhaustion: "continue_locally", "queue", or "report"
exhaustion_policy = "continue_locally"
cache_enabled = true
max_consecutive_failures_before_halt = 2
smart_pacing_enabled = true
pacing_utilization_target_pct = 90
pacing_burst_allowance = 2
workload_balancing_enabled = true
model_balancing_enabled = true

[[models.providers]]
id = "local_snap"
provider_type = "local_snap"
enabled = true
base_url = "http://localhost:8366/v1"
transport = "openai_compatible"
model_policy = "all"
permitted_models = ["smollm3-3b"]
is_frontier = false

[[models.providers]]
id = "openai"
provider_type = "openai"
enabled = true
base_url = "https://api.openai.com/v1"
transport = "openai_compatible"
credential_env = "OPENAI_API_KEY"
model_policy = "escalate_only"
permitted_models = ["gpt-4.1", "o4-mini", "gpt-4o"]
fallback_provider = "local_snap"
is_frontier = true

[[models.providers]]
id = "anthropic"
provider_type = "anthropic"
enabled = true
base_url = "https://api.anthropic.com/v1"
transport = "anthropic_messages"
credential_env = "ANTHROPIC_API_KEY"
model_policy = "escalate_only"
permitted_models = ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"]
fallback_provider = "local_snap"
is_frontier = true

[[models.providers]]
id = "gemini"
provider_type = "gemini"
enabled = true
base_url = "https://generativelanguage.googleapis.com/v1beta"
transport = "gemini_native"
credential_env = "GEMINI_API_KEY"
model_policy = "escalate_only"
permitted_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
fallback_provider = "local_snap"
is_frontier = true

[[models.providers]]
id = "openrouter"
provider_type = "openrouter"
enabled = true
base_url = "https://openrouter.ai/api/v1"
transport = "openai_compatible"
credential_env = "OPENROUTER_API_KEY"
model_policy = "escalate_only"
permitted_models = ["deepseek/deepseek-r1", "qwen/qwen-2.5-coder-32b-instruct"]
fallback_provider = "local_snap"
is_frontier = true

[[models.providers]]
id = "openai_compatible"
provider_type = "openai_compatible"
enabled = false
base_url = "http://127.0.0.1:8000/v1"
transport = "openai_compatible"
credential_env = "OPENAI_COMPAT_API_KEY"
model_policy = "escalate_only"
permitted_models = []
fallback_provider = "local_snap"
is_frontier = true


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
enforce_specialist_state_machine = true
specialist_timeout_seconds = 120

[learning]
agreement_threshold = 2
max_user_questions_per_day = 3

[web]
enabled = true
wikipedia_api = "https://en.wikipedia.org/w/api.php"
max_pages_per_cycle = 3
request_timeout_seconds = 30
source_refresh_interval_seconds = 86400
max_source_retries = 3
source_retry_backoff_seconds = 300
reservation_lease_seconds = 1800
max_download_bytes = 5242880
cache_ttl_seconds = 3600
max_expansion_depth = 2
max_expansion_volume_per_cycle = 6
max_section_context_tokens = 1200

[execution]
backend = "antigravity"
runtime_binary = ""
command_timeout_seconds = 60
max_tool_memory_mb = 512
restrict_tool_network = true

[logging]
level = "info"
format = "jsonl"
rotate_size_mb = 10
retained_files = 5

[configuration]
watch = true
reload_debounce_ms = 500
poll_interval_seconds = 2

[domain]
name = "frontier_ai_research_and_verification"
display_name = "Frontier AI Research, Reasoning & Verification"
evaluation_set_path = "evaluation/external_benchmark.json"
evaluation_interval_cycles = 5
sources = ["arxiv", "openalex", "semantic_scholar", "crossref", "openreview", "acl_anthology", "dblp", "hal", "biorxiv", "medrxiv", "ssrn", "pubmed", "github", "huggingface", "zenodo", "papers_with_code", "institutional", "web_fetch", "wikipedia"]
ontology = ["concept", "research_work", "dataset", "implementation", "source", "tool", "supports", "contradicts", "discovered_from", "uses_tool", "corroborates"]
executable_checks = ["real_implementation", "simulation", "solver_call", "numerical_comparison", "data_reanalysis", "statistical_replication", "constraint_solving"]
scope = [
    "Foundation model architectures, attention mechanisms, and sparse Mixture-of-Experts (MoE) routing",
    "Test-time compute scaling, tree search, Process Reward Models (PRMs), and token-budget parity normalization",
    "Program-aided, neuro-symbolic, and formal solver-assisted verification",
    "Hierarchical task network planning (HDDL 2.1) and multi-agent epistemic consensus",
    "Active learning, statistical leverage sampling (ALEVS), and domain generalization",
    "Multi-source scientific retrieval, provenance deduplication, and compute-normalized benchmarking"
]
exclusions = [
    "Clinical medical diagnosis or patient treatment prescriptions",
    "Regulated legal, tax, or live financial trading execution",
    "Full-parameter on-device weight fine-tuning or gradient backpropagation (hardware constrained to local 3B inference + Rust tool/extension synthesis)",
    "Unverified claims lacking source provenance or executable solver checks"
]
terminology = [
    "retrieved_finding: Claim directly backed by an indexed canonical research work or trusted domain source",
    "expert_synthesis: Multi-agent (Explorer, Sceptic, Builder) inference combining retrieved evidence with caller constraints",
    "execution_verified_result: Result confirmed by compiling and executing an isolated Rust solver or verification tool with exit code 0",
    "token_budget_parity: Normalizing accuracy comparisons across reasoning methods by matching total test-time inference tokens",
    "epistemic_consensus: Constitutional rule requiring 2-of-3 distinct permanent governing agent endorsements"
]
research_objectives = [
    "Verify frontier AI claims under strict compute and token-budget parity",
    "Translate theoretical mechanisms into zero-overhead coordinator hook extensions and sandboxed Rust verification solvers",
    "Expand multi-source corroboration across arXiv, OpenReview, OpenAlex, Semantic Scholar, ACL Anthology, and GitHub",
    "Maintain calibrated uncertainty estimates grounded in empirical ablation and negative-result tracking"
]
evaluation_criteria = [
    "Answer correctness against held-out external benchmarks",
    "Problem-solving success verified by sandboxed executable checks",
    "Calibration of stated uncertainty (1 - Expected Calibration Error / Brier score)",
    "Citation accuracy and source provenance integrity"
]

[consultation]
default_depth = "standard"
max_depth = "deep"
quick_timeout_seconds = 30
standard_timeout_seconds = 90
deep_timeout_seconds = 300
standard_async_default = true
max_specialists_per_consultation = 3
http_bind_addr = "127.0.0.1:8390"
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
    #[serde(default)]
    pub domain: DomainSection,
    #[serde(default)]
    pub consultation: ConsultationSection,
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
    #[serde(default = "default_model_providers")]
    pub providers: Vec<ModelProviderConfig>,
    #[serde(default = "default_model_catalog")]
    pub catalog: Vec<ModelCapabilityConfig>,
    #[serde(default)]
    pub selection: ModelSelectionConfig,
    #[serde(default)]
    pub frontier_budget: FrontierBudgetConfig,
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

fn default_provider_transport() -> String {
    "openai_compatible".to_string()
}

fn default_model_policy() -> String {
    "escalate_only".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelProviderConfig {
    pub id: String,
    /// Provider kind: "local_snap", "openai_compatible", "openai", "openrouter", "gemini", "anthropic"
    pub provider_type: String,
    #[serde(default = "default_true_bool")]
    pub enabled: bool,
    pub base_url: String,
    /// Wire transport: "openai_compatible", "anthropic_messages", "gemini_native"
    #[serde(default = "default_provider_transport")]
    pub transport: String,
    /// Environment variable name holding the API key (never the raw secret).
    #[serde(default)]
    pub credential_env: Option<String>,
    /// Policy: "all", "local_only", "escalate_only", "disabled"
    #[serde(default = "default_model_policy")]
    pub model_policy: String,
    /// Permitted real API model identifiers (empty means any model in catalog for this provider).
    #[serde(default)]
    pub permitted_models: Vec<String>,
    /// Fallback provider ID when this provider is unavailable or out of budget.
    #[serde(default)]
    pub fallback_provider: Option<String>,
    #[serde(default)]
    pub is_frontier: bool,
}

fn default_cap_score_bps() -> u32 {
    750
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCapabilityConfig {
    pub provider_id: String,
    pub model_id: String,
    pub context_window_tokens: usize,
    pub max_output_tokens: usize,
    #[serde(default = "default_true_bool")]
    pub supports_structured_json: bool,
    #[serde(default = "default_true_bool")]
    pub supports_code_generation: bool,
    #[serde(default)]
    pub supports_reasoning_trace: bool,
    #[serde(default)]
    pub supports_tool_use: bool,
    #[serde(default)]
    pub is_frontier: bool,
    /// Input price in micro-USD ($1e-6) per 1M tokens. None means unknown billing/pricing.
    #[serde(default)]
    pub input_price_micros_per_1m: Option<u64>,
    /// Output price in micro-USD ($1e-6) per 1M tokens. None means unknown billing/pricing.
    #[serde(default)]
    pub output_price_micros_per_1m: Option<u64>,
    #[serde(default = "default_cap_score_bps")]
    pub reasoning_score_bps: u32,
    #[serde(default = "default_cap_score_bps")]
    pub coding_score_bps: u32,
    #[serde(default = "default_cap_score_bps")]
    pub synthesis_score_bps: u32,
    #[serde(default = "default_cap_score_bps")]
    pub critique_score_bps: u32,
}

pub fn default_model_providers() -> Vec<ModelProviderConfig> {
    vec![
        ModelProviderConfig {
            id: "local_snap".to_string(),
            provider_type: "local_snap".to_string(),
            enabled: true,
            base_url: "http://localhost:8366/v1".to_string(),
            transport: "openai_compatible".to_string(),
            credential_env: None,
            model_policy: "all".to_string(),
            permitted_models: vec!["smollm3-3b".to_string()],
            fallback_provider: None,
            is_frontier: false,
        },
        ModelProviderConfig {
            id: "openai".to_string(),
            provider_type: "openai".to_string(),
            enabled: true,
            base_url: "https://api.openai.com/v1".to_string(),
            transport: "openai_compatible".to_string(),
            credential_env: Some("OPENAI_API_KEY".to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec![
                "gpt-4.1".to_string(),
                "o4-mini".to_string(),
                "gpt-4o".to_string(),
            ],
            fallback_provider: Some("local_snap".to_string()),
            is_frontier: true,
        },
        ModelProviderConfig {
            id: "anthropic".to_string(),
            provider_type: "anthropic".to_string(),
            enabled: true,
            base_url: "https://api.anthropic.com/v1".to_string(),
            transport: "anthropic_messages".to_string(),
            credential_env: Some("ANTHROPIC_API_KEY".to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec![
                "claude-sonnet-4-20250514".to_string(),
                "claude-3-7-sonnet-20250219".to_string(),
            ],
            fallback_provider: Some("local_snap".to_string()),
            is_frontier: true,
        },
        ModelProviderConfig {
            id: "gemini".to_string(),
            provider_type: "gemini".to_string(),
            enabled: true,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            transport: "gemini_native".to_string(),
            credential_env: Some("GEMINI_API_KEY".to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec![
                "gemini-2.5-pro".to_string(),
                "gemini-2.5-flash".to_string(),
            ],
            fallback_provider: Some("local_snap".to_string()),
            is_frontier: true,
        },
        ModelProviderConfig {
            id: "openrouter".to_string(),
            provider_type: "openrouter".to_string(),
            enabled: true,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            transport: "openai_compatible".to_string(),
            credential_env: Some("OPENROUTER_API_KEY".to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec![
                "deepseek/deepseek-r1".to_string(),
                "qwen/qwen-2.5-coder-32b-instruct".to_string(),
            ],
            fallback_provider: Some("local_snap".to_string()),
            is_frontier: true,
        },
        ModelProviderConfig {
            id: "openai_compatible".to_string(),
            provider_type: "openai_compatible".to_string(),
            enabled: false,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            transport: "openai_compatible".to_string(),
            credential_env: Some("OPENAI_COMPAT_API_KEY".to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec![],
            fallback_provider: Some("local_snap".to_string()),
            is_frontier: true,
        },
    ]
}

pub fn default_model_catalog() -> Vec<ModelCapabilityConfig> {
    vec![
        ModelCapabilityConfig {
            provider_id: "local_snap".to_string(),
            model_id: "smollm3-3b".to_string(),
            context_window_tokens: 8192,
            max_output_tokens: 2048,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: false,
            is_frontier: false,
            input_price_micros_per_1m: Some(0),
            output_price_micros_per_1m: Some(0),
            reasoning_score_bps: 620,
            coding_score_bps: 600,
            synthesis_score_bps: 640,
            critique_score_bps: 630,
        },
        ModelCapabilityConfig {
            provider_id: "anthropic".to_string(),
            model_id: "claude-sonnet-4-20250514".to_string(),
            context_window_tokens: 200_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(3_000_000),
            output_price_micros_per_1m: Some(15_000_000),
            reasoning_score_bps: 950,
            coding_score_bps: 965,
            synthesis_score_bps: 955,
            critique_score_bps: 960,
        },
        ModelCapabilityConfig {
            provider_id: "anthropic".to_string(),
            model_id: "claude-3-7-sonnet-20250219".to_string(),
            context_window_tokens: 200_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(3_000_000),
            output_price_micros_per_1m: Some(15_000_000),
            reasoning_score_bps: 930,
            coding_score_bps: 940,
            synthesis_score_bps: 935,
            critique_score_bps: 935,
        },
        ModelCapabilityConfig {
            provider_id: "openai".to_string(),
            model_id: "gpt-4.1".to_string(),
            context_window_tokens: 128_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(2_000_000),
            output_price_micros_per_1m: Some(8_000_000),
            reasoning_score_bps: 940,
            coding_score_bps: 950,
            synthesis_score_bps: 945,
            critique_score_bps: 945,
        },
        ModelCapabilityConfig {
            provider_id: "openai".to_string(),
            model_id: "o4-mini".to_string(),
            context_window_tokens: 128_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(1_100_000),
            output_price_micros_per_1m: Some(4_400_000),
            reasoning_score_bps: 945,
            coding_score_bps: 935,
            synthesis_score_bps: 920,
            critique_score_bps: 940,
        },
        ModelCapabilityConfig {
            provider_id: "openai".to_string(),
            model_id: "gpt-4o".to_string(),
            context_window_tokens: 128_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: false,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(2_500_000),
            output_price_micros_per_1m: Some(10_000_000),
            reasoning_score_bps: 900,
            coding_score_bps: 905,
            synthesis_score_bps: 910,
            critique_score_bps: 905,
        },
        ModelCapabilityConfig {
            provider_id: "gemini".to_string(),
            model_id: "gemini-2.5-pro".to_string(),
            context_window_tokens: 1_000_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(1_250_000),
            output_price_micros_per_1m: Some(10_000_000),
            reasoning_score_bps: 945,
            coding_score_bps: 940,
            synthesis_score_bps: 950,
            critique_score_bps: 940,
        },
        ModelCapabilityConfig {
            provider_id: "gemini".to_string(),
            model_id: "gemini-2.5-flash".to_string(),
            context_window_tokens: 1_000_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: Some(300_000),
            output_price_micros_per_1m: Some(2_500_000),
            reasoning_score_bps: 880,
            coding_score_bps: 875,
            synthesis_score_bps: 885,
            critique_score_bps: 880,
        },
        ModelCapabilityConfig {
            provider_id: "openrouter".to_string(),
            model_id: "deepseek/deepseek-r1".to_string(),
            context_window_tokens: 128_000,
            max_output_tokens: 16_384,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: false,
            is_frontier: true,
            input_price_micros_per_1m: None, // Unknown dynamic OpenRouter pricing -> conservative accounting
            output_price_micros_per_1m: None,
            reasoning_score_bps: 935,
            coding_score_bps: 925,
            synthesis_score_bps: 930,
            critique_score_bps: 935,
        },
        ModelCapabilityConfig {
            provider_id: "openrouter".to_string(),
            model_id: "qwen/qwen-2.5-coder-32b-instruct".to_string(),
            context_window_tokens: 32_768,
            max_output_tokens: 8_192,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: false,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: None,
            output_price_micros_per_1m: None,
            reasoning_score_bps: 860,
            coding_score_bps: 920,
            synthesis_score_bps: 850,
            critique_score_bps: 865,
        },
    ]
}

pub fn default_model_preference_order() -> Vec<String> {
    vec![
        "anthropic:claude-sonnet-4-20250514".to_string(),
        "openai:gpt-4.1".to_string(),
        "gemini:gemini-2.5-pro".to_string(),
        "openrouter:deepseek/deepseek-r1".to_string(),
        "local_snap:smollm3-3b".to_string(),
    ]
}

fn default_min_eval_runs() -> usize {
    2
}

fn default_discovery_refresh_seconds() -> u64 {
    3600
}

pub fn default_local_workloads() -> Vec<String> {
    vec![
        "routing".to_string(),
        "summarization".to_string(),
        "cheap_work".to_string(),
    ]
}

pub fn default_escalated_workloads() -> Vec<String> {
    vec![
        "hard_synthesis".to_string(),
        "experiment_design".to_string(),
        "complex_rust_generation".to_string(),
        "debugging".to_string(),
        "critical_review".to_string(),
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelSelectionConfig {
    #[serde(default)]
    pub explicit_override: Option<String>,
    #[serde(default = "default_model_preference_order")]
    pub preference_order: Vec<String>,
    #[serde(default = "default_min_eval_runs")]
    pub min_eval_runs_for_empirical_ranking: usize,
    #[serde(default = "default_discovery_refresh_seconds")]
    pub discovery_refresh_interval_seconds: u64,
    #[serde(default = "default_local_workloads")]
    pub local_workloads: Vec<String>,
    #[serde(default = "default_escalated_workloads")]
    pub escalated_workloads: Vec<String>,
}

impl Default for ModelSelectionConfig {
    fn default() -> Self {
        Self {
            explicit_override: None,
            preference_order: default_model_preference_order(),
            min_eval_runs_for_empirical_ranking: default_min_eval_runs(),
            discovery_refresh_interval_seconds: default_discovery_refresh_seconds(),
            local_workloads: default_local_workloads(),
            escalated_workloads: default_escalated_workloads(),
        }
    }
}

fn default_frontier_max_calls_per_24h() -> usize {
    3
}

fn default_frontier_max_calls_per_task() -> usize {
    1
}

fn default_frontier_max_concurrency() -> usize {
    1
}

fn default_frontier_max_retries() -> usize {
    1
}

fn default_frontier_max_tokens_per_24h() -> usize {
    64_000
}

fn default_frontier_max_tokens_per_task() -> usize {
    16_000
}

fn default_frontier_max_spend_micros_per_24h() -> u64 {
    2_000_000 // $2.00 per rolling 24h
}

fn default_conservative_unknown_call_micros() -> u64 {
    250_000 // $0.25 minimum conservative floor per unknown-priced call
}

fn default_conservative_unknown_input_micros_per_1m() -> u64 {
    15_000_000 // $15.00 / 1M input tokens
}

fn default_conservative_unknown_output_micros_per_1m() -> u64 {
    75_000_000 // $75.00 / 1M output tokens
}

fn default_exhaustion_policy() -> String {
    "continue_locally".to_string()
}

fn default_max_consecutive_failures() -> usize {
    2
}

fn default_pacing_utilization_target_pct() -> usize {
    90
}

fn default_pacing_burst_allowance() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrontierBudgetConfig {
    #[serde(default = "default_frontier_max_calls_per_24h")]
    pub max_calls_per_24h: usize,
    #[serde(default = "default_frontier_max_calls_per_task")]
    pub max_calls_per_task: usize,
    #[serde(default = "default_frontier_max_concurrency")]
    pub max_concurrency: usize,
    #[serde(default = "default_frontier_max_retries")]
    pub max_retries: usize,
    #[serde(default = "default_frontier_max_tokens_per_24h")]
    pub max_tokens_per_24h: usize,
    #[serde(default = "default_frontier_max_tokens_per_task")]
    pub max_tokens_per_task: usize,
    #[serde(default = "default_frontier_max_spend_micros_per_24h")]
    pub max_spend_micros_per_24h: u64,
    #[serde(default = "default_conservative_unknown_call_micros")]
    pub conservative_unknown_call_micros: u64,
    #[serde(default = "default_conservative_unknown_input_micros_per_1m")]
    pub conservative_unknown_input_micros_per_1m: u64,
    #[serde(default = "default_conservative_unknown_output_micros_per_1m")]
    pub conservative_unknown_output_micros_per_1m: u64,
    #[serde(default = "default_exhaustion_policy")]
    pub exhaustion_policy: String,
    #[serde(default = "default_true_bool")]
    pub cache_enabled: bool,
    #[serde(default = "default_max_consecutive_failures")]
    pub max_consecutive_failures_before_halt: usize,
    #[serde(default = "default_true_bool")]
    pub smart_pacing_enabled: bool,
    #[serde(default = "default_pacing_utilization_target_pct")]
    pub pacing_utilization_target_pct: usize,
    #[serde(default = "default_pacing_burst_allowance")]
    pub pacing_burst_allowance: usize,
    #[serde(default = "default_true_bool")]
    pub workload_balancing_enabled: bool,
    #[serde(default = "default_true_bool")]
    pub model_balancing_enabled: bool,
}

impl Default for FrontierBudgetConfig {
    fn default() -> Self {
        Self {
            max_calls_per_24h: default_frontier_max_calls_per_24h(),
            max_calls_per_task: default_frontier_max_calls_per_task(),
            max_concurrency: default_frontier_max_concurrency(),
            max_retries: default_frontier_max_retries(),
            max_tokens_per_24h: default_frontier_max_tokens_per_24h(),
            max_tokens_per_task: default_frontier_max_tokens_per_task(),
            max_spend_micros_per_24h: default_frontier_max_spend_micros_per_24h(),
            conservative_unknown_call_micros: default_conservative_unknown_call_micros(),
            conservative_unknown_input_micros_per_1m: default_conservative_unknown_input_micros_per_1m(),
            conservative_unknown_output_micros_per_1m: default_conservative_unknown_output_micros_per_1m(),
            exhaustion_policy: default_exhaustion_policy(),
            cache_enabled: true,
            max_consecutive_failures_before_halt: default_max_consecutive_failures(),
            smart_pacing_enabled: true,
            pacing_utilization_target_pct: default_pacing_utilization_target_pct(),
            pacing_burst_allowance: default_pacing_burst_allowance(),
            workload_balancing_enabled: true,
            model_balancing_enabled: true,
        }
    }
}

/// Returns true if `model_id` is a marketing/product label rather than a real API model identifier.
pub fn is_marketing_product_label(model_id: &str) -> bool {
    let trimmed = model_id.trim();
    if trimmed.is_empty() || trimmed.contains(' ') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    matches!(
        lower.as_str(),
        "chatgpt"
            | "chatgpt-plus"
            | "claude"
            | "claude-ai"
            | "gemini"
            | "gemini-advanced"
            | "bard"
            | "copilot"
            | "openai"
            | "anthropic"
    )
}

/// Returns true if an `env_var_name` looks like a raw secret value rather than an environment variable identifier.
pub fn looks_like_raw_secret(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    let lower = t.to_lowercase();
    lower.starts_with("sk-")
        || lower.starts_with("aiza")
        || lower.starts_with("bearer ")
        || t.chars().any(|c| !c.is_ascii_alphanumeric() && c != '_')
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

fn default_true_bool() -> bool {
    true
}

fn default_specialist_timeout_seconds() -> u64 {
    120
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OrchestrationSection {
    pub autonomous_learning: bool,
    pub cycle_interval_seconds: u64,
    pub discussion_rounds: usize,
    pub max_model_requests_per_cycle: usize,
    pub max_tool_calls_per_cycle: usize,
    pub max_tool_builds_per_cycle: usize,
    #[serde(default = "default_true_bool")]
    pub enforce_specialist_state_machine: bool,
    #[serde(default = "default_specialist_timeout_seconds")]
    pub specialist_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LearningSection {
    pub agreement_threshold: usize,
    pub max_user_questions_per_day: usize,
}

fn default_source_refresh_interval_seconds() -> u64 {
    86400
}

fn default_max_source_retries() -> usize {
    3
}

fn default_source_retry_backoff_seconds() -> u64 {
    300
}

fn default_reservation_lease_seconds() -> u64 {
    1800
}

fn default_max_download_bytes() -> usize {
    10_485_760 // 10 MB safe ingestion download cap for arXiv LaTeX tarballs and OA PDFs
}

fn default_cache_ttl_seconds() -> u64 {
    3600
}

fn default_max_expansion_depth() -> usize {
    2
}

fn default_max_expansion_volume_per_cycle() -> usize {
    6
}

fn default_max_section_context_tokens() -> usize {
    1200
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WebSection {
    pub enabled: bool,
    pub wikipedia_api: String,
    pub max_pages_per_cycle: usize,
    pub request_timeout_seconds: u64,
    #[serde(default = "default_source_refresh_interval_seconds")]
    pub source_refresh_interval_seconds: u64,
    #[serde(default = "default_max_source_retries")]
    pub max_source_retries: usize,
    #[serde(default = "default_source_retry_backoff_seconds")]
    pub source_retry_backoff_seconds: u64,
    #[serde(default = "default_reservation_lease_seconds")]
    pub reservation_lease_seconds: u64,
    #[serde(default = "default_max_download_bytes")]
    pub max_download_bytes: usize,
    #[serde(default = "default_cache_ttl_seconds")]
    pub cache_ttl_seconds: u64,
    #[serde(default = "default_max_expansion_depth")]
    pub max_expansion_depth: usize,
    #[serde(default = "default_max_expansion_volume_per_cycle")]
    pub max_expansion_volume_per_cycle: usize,
    #[serde(default = "default_max_section_context_tokens")]
    pub max_section_context_tokens: usize,
}

fn default_max_tool_memory_mb() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionSection {
    pub backend: String,
    pub runtime_binary: String,
    pub command_timeout_seconds: u64,
    #[serde(default = "default_max_tool_memory_mb")]
    pub max_tool_memory_mb: usize,
    #[serde(default = "default_true_bool")]
    pub restrict_tool_network: bool,
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

fn default_topic_priority() -> i64 {
    80
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainTopicSeed {
    pub topic: String,
    pub discipline: String,
    #[serde(default = "default_topic_priority")]
    pub priority: i64,
    pub hypothesis: Option<String>,
}

pub fn default_domain_sources() -> Vec<String> {
    vec![
        "arxiv".to_string(),
        "openalex".to_string(),
        "semantic_scholar".to_string(),
        "crossref".to_string(),
        "openreview".to_string(),
        "acl_anthology".to_string(),
        "dblp".to_string(),
        "hal".to_string(),
        "biorxiv".to_string(),
        "medrxiv".to_string(),
        "ssrn".to_string(),
        "pubmed".to_string(),
        "github".to_string(),
        "huggingface".to_string(),
        "zenodo".to_string(),
        "papers_with_code".to_string(),
        "institutional".to_string(),
        "web_fetch".to_string(),
        "wikipedia".to_string(),
    ]
}

pub fn default_domain_ontology() -> Vec<String> {
    vec![
        "concept".to_string(),
        "research_work".to_string(),
        "dataset".to_string(),
        "implementation".to_string(),
        "source".to_string(),
        "tool".to_string(),
        "supports".to_string(),
        "contradicts".to_string(),
        "discovered_from".to_string(),
        "uses_tool".to_string(),
        "corroborates".to_string(),
    ]
}

pub fn default_domain_topics() -> Vec<DomainTopicSeed> {
    vec![
        DomainTopicSeed {
            topic: "Test-Time Compute Scaling and Process Reward Models".to_string(),
            discipline: "Frontier AI: Reasoning & Search".to_string(),
            priority: 98,
            hypothesis: Some("Step-level process verification and tree search outperform single-pass decoding at matched compute budgets".to_string()),
        },
        DomainTopicSeed {
            topic: "Mixture-of-Experts Routing and Sparse Foundation Architectures".to_string(),
            discipline: "Frontier AI: Foundation Models".to_string(),
            priority: 95,
            hypothesis: Some("Load-balanced sparse expert routing scales model capacity sub-linearly with active inference FLOPs".to_string()),
        },
        DomainTopicSeed {
            topic: "Program-Aided Formal Verification and Neuro-Symbolic Synthesis".to_string(),
            discipline: "Frontier AI: Formal Reasoning".to_string(),
            priority: 94,
            hypothesis: Some("Translating natural-language reasoning steps into executable solver constraints eliminates arithmetic and logical drift".to_string()),
        },
        DomainTopicSeed {
            topic: "Post-Training Reinforcement Learning and Constitutional Alignment".to_string(),
            discipline: "Frontier AI: Alignment & RL".to_string(),
            priority: 92,
            hypothesis: Some("Verifiable reward signals and constitutional invariants improve multi-step reasoning faithfulness without sycophancy".to_string()),
        },
        DomainTopicSeed {
            topic: "Autonomous Multi-Agent Epistemic Consensus and Specialist Orchestration".to_string(),
            discipline: "Frontier AI: Multi-Agent Systems".to_string(),
            priority: 91,
            hypothesis: Some("Independent multi-agent verification with bounded specialist delegation reduces correlated error cascades".to_string()),
        },
        DomainTopicSeed {
            topic: "Adaptive Retrieval and Cross-Repository Provenance Deduplication".to_string(),
            discipline: "Information Retrieval".to_string(),
            priority: 90,
            hypothesis: Some("Re-ranking retrieved memories while penalizing duplicate canonical preprints prevents false corroboration".to_string()),
        },
        DomainTopicSeed {
            topic: "Formal Graph Reasoning and Density".to_string(),
            discipline: "Mathematics & Graph Theory".to_string(),
            priority: 85,
            hypothesis: Some("Higher clustering coefficient in semantic graph improves multi-hop deduction".to_string()),
        },
        DomainTopicSeed {
            topic: "Sandboxed Execution Isolation and Safety".to_string(),
            discipline: "Computer Systems".to_string(),
            priority: 80,
            hypothesis: Some("Container isolation guarantees deterministic tool evaluation without host leakage".to_string()),
        },
        DomainTopicSeed {
            topic: "Cognitive Architecture and Working Memory Bounds".to_string(),
            discipline: "Cognitive Science".to_string(),
            priority: 75,
            hypothesis: Some("Explicitly separating working hypotheses from verified claims reduces hallucination".to_string()),
        },
        DomainTopicSeed {
            topic: "Active Learning and Uncertainty Sampling".to_string(),
            discipline: "Machine Learning".to_string(),
            priority: 70,
            hypothesis: Some("Sampling sources at high semantic divergence maximizes novel information acquisition".to_string()),
        },
        DomainTopicSeed {
            topic: "Hierarchical Planning and Subgoal Decomposition".to_string(),
            discipline: "AI Planning".to_string(),
            priority: 65,
            hypothesis: Some("Decomposing open-ended questions into atomic subtasks boosts completion rates".to_string()),
        },
        DomainTopicSeed {
            topic: "Neuro-Symbolic Tool Synthesis".to_string(),
            discipline: "AI Engineering".to_string(),
            priority: 60,
            hypothesis: Some("Generating strongly typed Rust tools prevents runtime execution failures".to_string()),
        },
    ]
}

fn default_connector_enabled() -> bool {
    true
}

fn default_connector_priority() -> i64 {
    80
}

fn default_connector_rate_limit_ms() -> u64 {
    500
}

fn default_connector_max_results() -> usize {
    3
}

fn default_connector_prefer_full_text() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainConnectorConfig {
    pub name: String,
    #[serde(default)]
    pub category: String, // "preprints", "metadata", "venues", "code_and_artifacts", "institutional", "orientation_only", "web_fetch"
    #[serde(default = "default_connector_enabled")]
    pub enabled: bool,
    #[serde(default = "default_connector_priority")]
    pub priority: i64,
    #[serde(default = "default_connector_rate_limit_ms")]
    pub rate_limit_ms: u64,
    #[serde(default = "default_connector_max_results")]
    pub max_results_per_query: usize,
    #[serde(default = "default_connector_prefer_full_text")]
    pub prefer_full_text: bool,
    #[serde(default)]
    pub orientation_only: bool,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub rss_feeds: Vec<String>,
}

pub fn default_domain_connectors() -> Vec<DomainConnectorConfig> {
    vec![
        // 1. Preprints
        DomainConnectorConfig {
            name: "arxiv".to_string(),
            category: "preprints".to_string(),
            enabled: true,
            priority: 100,
            rate_limit_ms: 3000, // arXiv ToU: max 1 request every 3 seconds
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://export.arxiv.org/api/query".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "biorxiv".to_string(),
            category: "preprints".to_string(),
            enabled: true,
            priority: 72,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/prefixes/10.1101/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "medrxiv".to_string(),
            category: "preprints".to_string(),
            enabled: true,
            priority: 70,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/prefixes/10.1101/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "ssrn".to_string(),
            category: "preprints".to_string(),
            enabled: true,
            priority: 68,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/prefixes/10.2139/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "hal".to_string(),
            category: "preprints".to_string(),
            enabled: true,
            priority: 75,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.archives-ouvertes.fr/search/".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        // 2. Metadata (citation graphs, author disambiguation, version linking)
        DomainConnectorConfig {
            name: "openalex".to_string(),
            category: "metadata".to_string(),
            enabled: true,
            priority: 95,
            rate_limit_ms: 250,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.openalex.org/works".to_string()),
            api_key_env: Some("OPENALEX_API_KEY".to_string()),
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "crossref".to_string(),
            category: "metadata".to_string(),
            enabled: true,
            priority: 90,
            rate_limit_ms: 250,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "semantic_scholar".to_string(),
            category: "metadata".to_string(),
            enabled: true,
            priority: 92,
            rate_limit_ms: 1000,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.semanticscholar.org/graph/v1/paper/search".to_string()),
            api_key_env: Some("S2_API_KEY".to_string()),
            urls: vec![],
            rss_feeds: vec![],
        },
        // 3. Venues
        DomainConnectorConfig {
            name: "openreview".to_string(),
            category: "venues".to_string(),
            enabled: true,
            priority: 88,
            rate_limit_ms: 500,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api2.openreview.net/notes/search".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "acl_anthology".to_string(),
            category: "venues".to_string(),
            enabled: true,
            priority: 86,
            rate_limit_ms: 500,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/prefixes/10.18653/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "dblp".to_string(),
            category: "venues".to_string(),
            enabled: true,
            priority: 82,
            rate_limit_ms: 1000,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://dblp.org/search/publ/api".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "pubmed".to_string(),
            category: "venues".to_string(),
            enabled: true,
            priority: 74,
            rate_limit_ms: 400,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://eutils.ncbi.nlm.nih.gov/entrez/eutils".to_string()),
            api_key_env: Some("NCBI_API_KEY".to_string()),
            urls: vec![],
            rss_feeds: vec![],
        },
        // 4. Code and artifacts
        DomainConnectorConfig {
            name: "github".to_string(),
            category: "code_and_artifacts".to_string(),
            enabled: true,
            priority: 85,
            rate_limit_ms: 1000,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.github.com".to_string()),
            api_key_env: Some("GITHUB_TOKEN".to_string()),
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "huggingface".to_string(),
            category: "code_and_artifacts".to_string(),
            enabled: true,
            priority: 84,
            rate_limit_ms: 500,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://huggingface.co/api".to_string()),
            api_key_env: Some("HF_TOKEN".to_string()),
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "zenodo".to_string(),
            category: "code_and_artifacts".to_string(),
            enabled: true,
            priority: 78,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://zenodo.org/api/records".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        DomainConnectorConfig {
            name: "papers_with_code".to_string(),
            category: "code_and_artifacts".to_string(),
            enabled: true,
            priority: 76,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://huggingface.co/api/daily_papers".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        // 5. Institutional (lab publications, technical reports, standards bodies, patents, regulatory documents)
        DomainConnectorConfig {
            name: "institutional".to_string(),
            category: "institutional".to_string(),
            enabled: true,
            priority: 70,
            rate_limit_ms: 500,
            max_results_per_query: 2,
            prefer_full_text: true,
            orientation_only: false,
            base_url: Some("https://api.crossref.org/works".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        // 6. Orientation only (Wikipedia and domain wikis for discovery and reference harvesting, never as primary evidence)
        DomainConnectorConfig {
            name: "wikipedia".to_string(),
            category: "orientation_only".to_string(),
            enabled: true,
            priority: 40,
            rate_limit_ms: 300,
            max_results_per_query: 2,
            prefer_full_text: false,
            orientation_only: true,
            base_url: Some("https://en.wikipedia.org/w/api.php".to_string()),
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec![],
        },
        // 7. Generic web fetch for configured URL lists and RSS feeds
        DomainConnectorConfig {
            name: "web_fetch".to_string(),
            category: "web_fetch".to_string(),
            enabled: true,
            priority: 65,
            rate_limit_ms: 500,
            max_results_per_query: 3,
            prefer_full_text: true,
            orientation_only: false,
            base_url: None,
            api_key_env: None,
            urls: vec![],
            rss_feeds: vec!["https://rss.arxiv.org/rss/cs.AI".to_string()],
        },
    ]
}

pub fn default_domain_executable_checks() -> Vec<String> {
    vec![
        "real_implementation".to_string(),
        "simulation".to_string(),
        "solver_call".to_string(),
        "numerical_comparison".to_string(),
        "data_reanalysis".to_string(),
        "statistical_replication".to_string(),
        "constraint_solving".to_string(),
    ]
}

pub fn default_domain_evaluation_set_path() -> String {
    "evaluation/external_benchmark.json".to_string()
}

pub fn default_evaluation_interval_cycles() -> u64 {
    5
}

pub fn default_eval_case_kind() -> String {
    "question".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainEvaluationCase {
    pub id: String,
    pub source_origin: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default = "default_eval_case_kind")]
    pub case_kind: String, // "question", "problem", "review_case"
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub in_triad_strong_coverage: bool,
    pub question: String,
    #[serde(default)]
    pub objective: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    pub expected_keywords: Vec<String>,
    #[serde(default)]
    pub expected_citation_markers: Vec<String>,
    #[serde(default)]
    pub requires_solver: bool,
    #[serde(default)]
    pub expects_contradiction_detection: bool,
    #[serde(default)]
    pub data_kind: Option<String>, // "real_data" or "synthetic_data"
    pub ground_truth_summary: String,
}

/// Checks if an evaluation origin, author, or identifier indicates that Triad or its internal agents
/// (`Explorer`, `Sceptic`, `Builder`, or temporary specialists) authored the evaluation item.
/// Triad never authors its own evaluation set.
pub fn is_triad_authored_evaluation_marker(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    if lower.is_empty() {
        return false;
    }
    let exact_agents = ["explorer", "sceptic", "builder", "triad", "coordinator", "specialist"];
    if exact_agents.contains(&lower.as_str()) {
        return true;
    }
    let forbidden_markers = [
        "triad_authored",
        "triad-authored",
        "authored by triad",
        "authored_by_triad",
        "self_authored",
        "self-authored",
        "self_generated",
        "self-generated",
        "agent_authored",
        "agent-authored",
        "agent_generated",
        "agent-generated",
        "triad_internal",
        "triad-internal",
        "triad_agent",
        "triad-agent",
        "autonomous_synthesis",
        "generated by explorer",
        "generated by sceptic",
        "generated by builder",
        "authored by explorer",
        "authored by sceptic",
        "authored by builder",
        "authored by specialist",
        "triad://",
        "triad:self",
    ];
    forbidden_markers.iter().any(|m| lower.contains(m))
}

pub fn validate_evaluation_case_external(case: &DomainEvaluationCase) -> Result<()> {
    if case.id.trim().is_empty() {
        bail!("Evaluation case id cannot be empty");
    }
    if case.source_origin.trim().is_empty() {
        bail!(
            "Evaluation case '{}' must declare an external source_origin",
            case.id
        );
    }
    if is_triad_authored_evaluation_marker(&case.id)
        || is_triad_authored_evaluation_marker(&case.source_origin)
        || case
            .author
            .as_deref()
            .map(is_triad_authored_evaluation_marker)
            .unwrap_or(false)
    {
        bail!(
            "Evaluation case '{}' rejected: Triad never authors its own evaluation set (detected Triad/agent authorship in origin='{}', author={:?})",
            case.id,
            case.source_origin,
            case.author
        );
    }
    let ck = case.case_kind.trim();
    if !ck.is_empty() && !matches!(ck, "question" | "problem" | "review_case") {
        bail!(
            "Evaluation case '{}' has invalid case_kind '{}'; must be 'question', 'problem', or 'review_case'",
            case.id,
            ck
        );
    }
    if let Some(ref dk) = case.data_kind {
        let dkt = dk.trim();
        if !dkt.is_empty() && !matches!(dkt, "real_data" | "synthetic_data") {
            bail!(
                "Evaluation case '{}' has invalid data_kind '{}'; must be 'real_data' or 'synthetic_data'",
                case.id,
                dkt
            );
        }
    }
    if case.question.trim().is_empty() {
        bail!("Evaluation case '{}' question cannot be empty", case.id);
    }
    if case.ground_truth_summary.trim().is_empty() && case.expected_keywords.is_empty() {
        bail!(
            "Evaluation case '{}' must supply known answers or rubrics (ground_truth_summary and expected_keywords)",
            case.id
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DomainSection {
    pub name: String,
    pub display_name: String,
    #[serde(default = "default_domain_evaluation_set_path")]
    pub evaluation_set_path: String,
    #[serde(default = "default_evaluation_interval_cycles")]
    pub evaluation_interval_cycles: u64,
    #[serde(default)]
    pub evaluation_cases: Vec<DomainEvaluationCase>,
    #[serde(default = "default_domain_sources")]
    pub sources: Vec<String>,
    #[serde(default = "default_domain_connectors")]
    pub connectors: Vec<DomainConnectorConfig>,
    #[serde(default = "default_domain_ontology")]
    pub ontology: Vec<String>,
    #[serde(default = "default_domain_executable_checks")]
    pub executable_checks: Vec<String>,
    #[serde(default = "default_domain_topics")]
    pub topics: Vec<DomainTopicSeed>,
    pub scope: Vec<String>,
    pub exclusions: Vec<String>,
    pub terminology: Vec<String>,
    pub research_objectives: Vec<String>,
    pub evaluation_criteria: Vec<String>,
}

impl DomainSection {
    /// Returns the prioritized list of enabled connector configs for this domain.
    /// A connector is active if `enabled == true` in `self.connectors` AND (if `self.sources` is non-empty)
    /// its name (or canonical alias) appears in `self.sources`.
    pub fn effective_enabled_connectors(&self) -> Vec<DomainConnectorConfig> {
        let mut list = Vec::new();
        for c in &self.connectors {
            if !c.enabled {
                continue;
            }
            let in_sources = self.sources.is_empty()
                || self.sources.iter().any(|s| {
                    let sl = s.trim().to_lowercase();
                    let cl = c.name.trim().to_lowercase();
                    sl == cl
                        || (sl == "huggingface_cards" && cl == "huggingface")
                        || (sl == "huggingface" && cl == "huggingface_cards")
                });
            if in_sources {
                list.push(c.clone());
            }
        }
        // Also include any connector explicitly listed in `self.sources` that wasn't in `self.connectors`
        for s in &self.sources {
            let sl = s.trim().to_lowercase();
            let canonical_name = if sl == "huggingface_cards" {
                "huggingface"
            } else {
                sl.as_str()
            };
            let already_configured = self
                .connectors
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(canonical_name));
            if !already_configured {
                list.push(DomainConnectorConfig {
                    name: canonical_name.to_string(),
                    category: "custom".to_string(),
                    enabled: true,
                    priority: 50,
                    rate_limit_ms: 500,
                    max_results_per_query: 2,
                    prefer_full_text: true,
                    orientation_only: canonical_name == "wikipedia",
                    base_url: None,
                    api_key_env: None,
                    urls: vec![],
                    rss_feeds: vec![],
                });
            }
        }
        list.sort_by(|a, b| b.priority.cmp(&a.priority));
        list
    }

    pub fn is_connector_enabled(&self, name: &str) -> bool {
        let nl = name.trim().to_lowercase();
        let canonical = if nl == "huggingface_cards" || nl == "model_cards" {
            "huggingface"
        } else {
            nl.as_str()
        };
        self.effective_enabled_connectors()
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(canonical))
    }
}

impl Default for DomainSection {
    fn default() -> Self {
        Self {
            name: "frontier_ai_research_and_verification".to_string(),
            display_name: "Frontier AI Research, Reasoning & Verification".to_string(),
            evaluation_set_path: default_domain_evaluation_set_path(),
            evaluation_interval_cycles: default_evaluation_interval_cycles(),
            evaluation_cases: Vec::new(),
            sources: default_domain_sources(),
            connectors: default_domain_connectors(),
            ontology: default_domain_ontology(),
            executable_checks: default_domain_executable_checks(),
            topics: default_domain_topics(),
            scope: vec![
                "Foundation model architectures, attention mechanisms, and sparse Mixture-of-Experts (MoE) routing".to_string(),
                "Test-time compute scaling, tree search, Process Reward Models (PRMs), and token-budget parity normalization".to_string(),
                "Program-aided, neuro-symbolic, and formal solver-assisted verification".to_string(),
                "Hierarchical task network planning (HDDL 2.1) and multi-agent epistemic consensus".to_string(),
                "Active learning, statistical leverage sampling (ALEVS), and domain generalization".to_string(),
                "Multi-source scientific retrieval, provenance deduplication, and compute-normalized benchmarking".to_string(),
            ],
            exclusions: vec![
                "Clinical medical diagnosis or patient treatment prescriptions".to_string(),
                "Regulated legal, tax, or live financial trading execution".to_string(),
                "Full-parameter on-device weight fine-tuning or gradient backpropagation (hardware constrained to local 3B inference + Rust tool/extension synthesis)".to_string(),
                "Unverified claims lacking source provenance or executable solver checks".to_string(),
            ],
            terminology: vec![
                "retrieved_finding: Claim directly backed by an indexed canonical research work or trusted domain source".to_string(),
                "expert_synthesis: Multi-agent (Explorer, Sceptic, Builder) inference combining retrieved evidence with caller constraints".to_string(),
                "execution_verified_result: Result confirmed by compiling and executing an isolated Rust solver or verification tool with exit code 0".to_string(),
                "token_budget_parity: Normalizing accuracy comparisons across reasoning methods by matching total test-time inference tokens".to_string(),
                "epistemic_consensus: Constitutional rule requiring 2-of-3 distinct permanent governing agent endorsements".to_string(),
            ],
            research_objectives: vec![
                "Verify frontier AI claims under strict compute and token-budget parity".to_string(),
                "Translate theoretical mechanisms into zero-overhead coordinator hook extensions and sandboxed Rust verification solvers".to_string(),
                "Expand multi-source corroboration across arXiv, OpenReview, OpenAlex, Semantic Scholar, ACL Anthology, and GitHub".to_string(),
                "Maintain calibrated uncertainty estimates grounded in empirical ablation and negative-result tracking".to_string(),
            ],
            evaluation_criteria: vec![
                "Answer correctness against held-out external benchmarks".to_string(),
                "Problem-solving success verified by sandboxed executable checks".to_string(),
                "Calibration of stated uncertainty (1 - Expected Calibration Error / Brier score)".to_string(),
                "Citation accuracy and source provenance integrity".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsultationSection {
    pub default_depth: String,
    pub max_depth: String,
    pub quick_timeout_seconds: u64,
    pub standard_timeout_seconds: u64,
    pub deep_timeout_seconds: u64,
    pub standard_async_default: bool,
    pub max_specialists_per_consultation: usize,
    pub http_bind_addr: String,
}

impl Default for ConsultationSection {
    fn default() -> Self {
        Self {
            default_depth: "standard".to_string(),
            max_depth: "deep".to_string(),
            quick_timeout_seconds: 30,
            standard_timeout_seconds: 90,
            deep_timeout_seconds: 300,
            standard_async_default: true,
            max_specialists_per_consultation: 3,
            http_bind_addr: "127.0.0.1:8390".to_string(),
        }
    }
}

pub fn depth_rank(depth: &str) -> Option<u8> {
    match depth {
        "quick" => Some(1),
        "standard" => Some(2),
        "deep" => Some(3),
        _ => None,
    }
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
        if is_marketing_product_label(&self.models.chat.model) {
            bail!(
                "models.chat.model '{}' must be a real API model identifier, not a marketing product label",
                self.models.chat.model
            );
        }
        if is_marketing_product_label(&self.models.embedding.model) {
            bail!(
                "models.embedding.model '{}' must be a real API model identifier, not a marketing product label",
                self.models.embedding.model
            );
        }

        for p in &self.models.providers {
            if p.id.trim().is_empty() {
                bail!("models.providers[].id cannot be empty");
            }
            if p.base_url.trim().is_empty() {
                bail!("models.providers['{}'].base_url cannot be empty", p.id);
            }
            if !matches!(
                p.provider_type.as_str(),
                "local_snap" | "openai_compatible" | "openai" | "openrouter" | "gemini" | "anthropic"
            ) {
                bail!(
                    "models.providers['{}'].provider_type '{}' is invalid",
                    p.id,
                    p.provider_type
                );
            }
            if !matches!(
                p.transport.as_str(),
                "openai_compatible" | "anthropic_messages" | "gemini_native"
            ) {
                bail!(
                    "models.providers['{}'].transport '{}' must be 'openai_compatible', 'anthropic_messages', or 'gemini_native'",
                    p.id,
                    p.transport
                );
            }
            if !matches!(
                p.model_policy.as_str(),
                "all" | "local_only" | "escalate_only" | "disabled"
            ) {
                bail!(
                    "models.providers['{}'].model_policy '{}' must be 'all', 'local_only', 'escalate_only', or 'disabled'",
                    p.id,
                    p.model_policy
                );
            }
            if let Some(ref env_name) = p.credential_env {
                if looks_like_raw_secret(env_name) {
                    bail!(
                        "models.providers['{}'].credential_env must be an environment variable name, never a raw secret",
                        p.id
                    );
                }
            }
            for m in &p.permitted_models {
                if is_marketing_product_label(m) {
                    bail!(
                        "models.providers['{}'].permitted_models entry '{}' is a marketing label; use real API model identifiers",
                        p.id,
                        m
                    );
                }
            }
        }

        for c in &self.models.catalog {
            if is_marketing_product_label(&c.model_id) {
                bail!(
                    "models.catalog model_id '{}' must be a real API model identifier, not a marketing product label",
                    c.model_id
                );
            }
            if c.context_window_tokens == 0 || c.max_output_tokens == 0 {
                bail!(
                    "models.catalog['{}:{}'] context_window_tokens and max_output_tokens must be positive",
                    c.provider_id,
                    c.model_id
                );
            }
        }

        if let Some(ref ov) = self.models.selection.explicit_override {
            let trimmed = ov.trim();
            if !trimmed.is_empty() {
                let model_part = trimmed.split(':').next_back().unwrap_or(trimmed);
                if is_marketing_product_label(model_part) {
                    bail!(
                        "models.selection.explicit_override '{}' must use a real API model identifier",
                        trimmed
                    );
                }
            }
        }

        let fb = &self.models.frontier_budget;
        if fb.max_calls_per_24h == 0
            || fb.max_calls_per_task == 0
            || fb.max_concurrency == 0
            || fb.max_tokens_per_24h == 0
            || fb.max_tokens_per_task == 0
            || fb.max_spend_micros_per_24h == 0
        {
            bail!("models.frontier_budget limits (calls, concurrency, tokens, spend) must be positive");
        }
        if fb.conservative_unknown_call_micros == 0
            || fb.conservative_unknown_input_micros_per_1m == 0
            || fb.conservative_unknown_output_micros_per_1m == 0
        {
            bail!("models.frontier_budget conservative unknown pricing floors must be > 0 (unknown prices are never free)");
        }
        if !matches!(
            fb.exhaustion_policy.as_str(),
            "continue_locally" | "queue" | "report"
        ) {
            bail!(
                "models.frontier_budget.exhaustion_policy '{}' must be 'continue_locally', 'queue', or 'report'",
                fb.exhaustion_policy
            );
        }
        if fb.pacing_utilization_target_pct == 0 || fb.pacing_utilization_target_pct > 100 {
            bail!("models.frontier_budget.pacing_utilization_target_pct must be between 1 and 100");
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

        if self.domain.name.trim().is_empty() {
            bail!("domain.name cannot be empty");
        }
        if self.domain.scope.is_empty() {
            bail!("domain.scope must contain at least one scope entry");
        }
        if self.domain.evaluation_set_path.trim().is_empty() && self.domain.evaluation_cases.is_empty() {
            bail!("domain must supply an evaluation set via domain.evaluation_set_path or domain.evaluation_cases");
        }
        if is_triad_authored_evaluation_marker(&self.domain.evaluation_set_path) {
            bail!(
                "domain.evaluation_set_path '{}' rejected: Triad never authors its own evaluation set",
                self.domain.evaluation_set_path
            );
        }
        if self.domain.evaluation_interval_cycles == 0 {
            bail!("domain.evaluation_interval_cycles must be positive");
        }
        for ec in &self.domain.evaluation_cases {
            validate_evaluation_case_external(ec)?;
        }

        let def_rank = depth_rank(&self.consultation.default_depth)
            .context("consultation.default_depth must be 'quick', 'standard', or 'deep'")?;
        let max_rank = depth_rank(&self.consultation.max_depth)
            .context("consultation.max_depth must be 'quick', 'standard', or 'deep'")?;
        if def_rank > max_rank {
            bail!("consultation.default_depth ({}) cannot exceed consultation.max_depth ({})",
                self.consultation.default_depth, self.consultation.max_depth);
        }
        if self.consultation.quick_timeout_seconds == 0
            || self.consultation.standard_timeout_seconds == 0
            || self.consultation.deep_timeout_seconds == 0
        {
            bail!("consultation timeouts must be positive");
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
