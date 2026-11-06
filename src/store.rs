use anyhow::{bail, Context, Result};
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSource {
    pub id: i64,
    pub url: String,
    pub title: String,
    pub content: String,
    pub revision: Option<String>,
    pub retrieved_at: String,
}

fn default_source_support_status() -> String {
    "single_source".to_string()
}

fn default_corroboration_count() -> usize {
    1
}

fn default_experimental_verification_status() -> String {
    "unverified".to_string()
}

fn default_finding_kind() -> String {
    "source_supported".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub id: String, // stable SHA256 of statement
    pub statement: String,
    pub source_url: String,
    pub proposer: String,
    pub status: String, // Trust attribute: "unresolved", "trusted", "rejected", "superseded"
    pub version: i64,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default = "default_source_support_status")]
    pub source_support_status: String, // "unsupported", "single_source", "corroborated", "contested"
    #[serde(default = "default_corroboration_count")]
    pub source_corroboration_count: usize,
    #[serde(default = "default_experimental_verification_status")]
    pub experimental_verification_status: String, // "unverified", "verified", "failed"
    #[serde(default)]
    pub verification_evidence: Option<String>,
    #[serde(default = "default_finding_kind")]
    pub finding_kind: String, // "source_supported" vs "locally_reproduced"
    #[serde(default)]
    pub source_fidelity_summary: Option<String>, // e.g. "full-text: 1, low-fidelity: 0, abstract-only: 1"
    #[serde(default)]
    pub execution_id: Option<i64>, // host-verified tool_executions.id for locally_reproduced empirical claims
    #[serde(default)]
    pub assumptions_text: Option<String>,
    #[serde(default)]
    pub negative_results_text: Option<String>,
    #[serde(default)]
    pub reproducibility_artifacts_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrackedSourceRecord {
    pub canonical_id: String,
    pub canonical_url: String,
    pub title: String,
    pub normalized_title: String,
    pub provider_id: Option<String>,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub version: String,
    pub versions_seen: Vec<String>,
    pub content_hash: String,
    pub source_kind: String, // "paper", "dataset", "implementation"
    pub domain: String,
    pub observed_urls: Vec<String>,
    pub observed_connectors: Vec<String>,
    pub route_observation_count: usize,
    pub discovery_status: String, // "discovered"
    pub discovered_at: String,
    pub ingestion_status: String, // "pending", "ingested", "failed"
    pub ingested_at: Option<String>,
    pub research_status: String, // "pending", "reserved", "completed", "failed"
    pub reserved_by: Option<String>,
    pub reservation_token: Option<String>,
    pub reserved_at: Option<String>,
    pub completed_at: Option<String>,
    pub completed_version: Option<String>,
    pub completed_content_hash: Option<String>,
    pub failure_count: usize,
    pub last_failure_reason: Option<String>,
    pub last_failed_at: Option<String>,
    pub retry_eligible: bool,
    pub next_retry_at: Option<String>,
    pub revisit_reason: Option<String>,
    pub has_contradiction: bool,
    pub contradiction_details: Option<String>,
    pub investigated_questions: Vec<String>,
    #[serde(default = "default_ingestion_completeness")]
    pub ingestion_completeness: String, // "full_text", "abstract_only", "metadata_only", "orientation_only"
    #[serde(default)]
    pub full_text_format: Option<String>, // "html", "latex", "pdf", "markdown", "abstract"
    #[serde(default)]
    pub topic: Option<String>,
    #[serde(default)]
    pub is_orientation_only: bool,
    #[serde(default)]
    pub section_count: usize,
    #[serde(default)]
    pub equations_preserved: usize,
    #[serde(default)]
    pub tables_found: usize,
    #[serde(default)]
    pub low_fidelity: bool,
    #[serde(default)]
    pub low_fidelity_reason: Option<String>,
    pub updated_at: String,
}

fn default_ingestion_completeness() -> String {
    "abstract_only".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredSourceSection {
    pub id: i64,
    pub canonical_source_id: String,
    pub version: String,
    pub connector: String,
    pub section_path: String,
    pub section_kind: String, // "hierarchy", "method", "algorithm", "equation", "theorem", "assumption", "experiment", "limitation", "reference"
    pub source_location: String,
    pub content: String,
    pub token_estimate: usize,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectorHealthRecord {
    pub connector: String,
    pub category: String,
    pub status: String, // "healthy", "degraded", "unreachable", "rate_limited"
    pub last_successful_fetch_at: Option<String>,
    pub last_attempted_at: Option<String>,
    pub last_error: Option<String>,
    pub total_requests: usize,
    pub total_successes: usize,
    pub total_failures: usize,
    pub discovered_count: usize,
    pub ingested_count: usize,
    pub full_text_count: usize,
    pub abstract_only_count: usize,
    pub avg_latency_ms: u64,
    pub rate_limit_ms: u64,
    pub is_orientation_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TopicCoverageSummary {
    pub topic: String,
    pub discovered_count: usize,
    pub ingested_count: usize,
    pub full_text_count: usize,
    pub abstract_only_count: usize,
    pub orientation_only_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorroborationSummary {
    pub target_id: String,
    pub total_independent_sources: usize,
    pub independent_papers: usize,
    pub independent_datasets: usize,
    pub independent_implementations: usize,
    pub total_route_observations: usize,
    pub canonical_source_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceTrackingDiagnostics {
    pub total_tracked_sources: usize,
    pub discovered_count: usize,
    pub ingested_count: usize,
    pub full_text_ingested_count: usize,
    pub abstract_only_ingested_count: usize,
    pub orientation_only_count: usize,
    pub total_stored_sections: usize,
    pub total_equations_preserved: usize,
    pub total_tables_found: usize,
    pub low_fidelity_count: usize,
    pub pending_research_count: usize,
    pub reserved_research_count: usize,
    pub completed_research_count: usize,
    pub failed_research_count: usize,
    pub retry_eligible_count: usize,
    pub contradiction_flagged_count: usize,
    pub independent_papers_count: usize,
    pub independent_datasets_count: usize,
    pub independent_implementations_count: usize,
    pub total_route_observations: usize,
    pub duplicate_route_observations_deduped: usize,
    pub connector_health: Vec<ConnectorHealthRecord>,
    pub topic_coverage: Vec<TopicCoverageSummary>,
    pub recent_sources: Vec<TrackedSourceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRecord {
    pub id: i64,
    pub claim_id: String,
    pub claim_version: i64,
    pub agent: String,
    pub decision: String, // "endorse", "reject"
    pub cycle_id: i64,
    pub rationale: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbedding {
    pub memory_type: String, // "claim", "source", "user_info", "procedure"
    pub target_id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserQuestionRecord {
    pub id: i64,
    pub agent: String,
    pub question: String,
    pub answer: Option<String>,
    pub answered_at: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: String,
    pub kind: String, // "ask", "challenge", "tell"
    pub description: String,
    pub status: String, // "queued", "running", "completed", "failed", "cancelled"
    pub result: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRecord {
    pub id: i64,
    pub name: String,
    pub version: i64,
    pub description: String,
    pub schema_json: String,
    pub source_code: String,
    pub binary_path: String,
    pub source_hash: String,
    pub status: String, // "draft", "active", "retired"
    pub endorser_1: Option<String>,
    pub endorser_2: Option<String>,
    pub test_passed: bool,
    pub created_at: String,
    #[serde(default)]
    pub experiment_spec_id: Option<String>,
    #[serde(default)]
    pub sceptic_verification_notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentMetricSpec {
    pub name: String,
    pub result_key: String,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub direction: String,
    pub success_criterion: String,
}

fn default_spec_domain() -> String {
    "frontier_ai_research_and_verification".to_string()
}

fn default_spec_check_kind() -> String {
    "simulation".to_string()
}

fn default_spec_fidelity_status() -> String {
    "self_contained_computational".to_string()
}

fn default_sceptic_status() -> String {
    "pending".to_string()
}

fn default_budget_runtime_secs() -> u64 {
    30
}

fn default_budget_memory_mb() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpecification {
    #[serde(default)]
    pub id: String,
    pub name: String,
    #[serde(default = "default_spec_domain")]
    pub domain: String,
    #[serde(default = "default_spec_check_kind")]
    pub executable_check_kind: String,
    #[serde(default)]
    pub data_kind: String, // "real_data" or "synthetic_data"
    pub research_question: String,
    pub falsifiable_hypothesis: String,
    pub mechanism_under_test: String,
    pub inputs_description: String,
    pub assumptions: Vec<String>,
    pub baseline_description: String,
    pub controls: Vec<String>,
    #[serde(default)]
    pub held_out_cases: Vec<String>,
    pub metrics: Vec<ExperimentMetricSpec>,
    #[serde(default = "default_budget_runtime_secs")]
    pub budget_max_runtime_secs: u64,
    #[serde(default = "default_budget_memory_mb")]
    pub budget_max_memory_mb: usize,
    pub seeds: Vec<i64>,
    pub reproducibility_requirements: String,
    pub expected_outputs: Vec<String>,
    pub known_limitations: Vec<String>,
    #[serde(default)]
    pub cited_source_ids: Vec<String>,
    #[serde(default)]
    pub cited_passage_ids: Vec<i64>,
    #[serde(default = "default_spec_fidelity_status")]
    pub source_fidelity_status: String,
    #[serde(default)]
    pub under_specified: bool,
    #[serde(default)]
    pub missing_source_details: Vec<String>,
    #[serde(default)]
    pub reingestion_requested: bool,
    #[serde(default)]
    pub builder_agent: String,
    #[serde(default)]
    pub sceptic_agent: String,
    #[serde(default = "default_sceptic_status")]
    pub sceptic_review_status: String,
    #[serde(default)]
    pub sceptic_review_notes: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierBudgetReservationRequest {
    pub parent_call_id: String,
    pub task_id: String,
    pub caller_kind: String, // "agent", "specialist", "background", "mcp_consultation"
    pub caller_id: String,
    pub workload_kind: String,
    pub provider_id: String,
    pub model_id: String,
    pub attempt_number: usize, // 1 = initial, 2 = retry #1, ...
    pub is_retry: bool,
    pub reserved_tokens: usize,
    pub estimated_spend_micros: u64,
    pub pricing_known: bool,
    pub prompt_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierBudgetLedgerRecord {
    pub reservation_id: String,
    pub parent_call_id: String,
    pub task_id: String,
    pub caller_kind: String,
    pub caller_id: String,
    pub workload_kind: String,
    pub provider_id: String,
    pub model_id: String,
    pub attempt_number: usize,
    pub is_retry: bool,
    pub status: String, // "reserved", "completed", "failed", "aborted_after_dispatch"
    pub reserved_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub estimated_spend_micros: u64,
    pub actual_spend_micros: u64,
    pub pricing_known: bool,
    pub prompt_hash: String,
    pub created_at_epoch_ms: i64,
    pub created_at: String,
    pub completed_at: Option<String>,
    pub error_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierBlockedTaskRecord {
    pub id: i64,
    pub task_id: String,
    pub caller_kind: String,
    pub caller_id: String,
    pub workload_kind: String,
    pub requested_provider: String,
    pub requested_model: String,
    pub exhaustion_reason: String,
    pub exhaustion_action: String, // "queued", "continued_locally", "reported_blocked"
    pub status: String,            // "queued", "resolved", "blocked", "continued_locally"
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierBudgetSummary {
    pub max_calls_per_24h: usize,
    pub used_calls_24h: usize,
    pub remaining_calls_24h: usize,
    pub max_calls_per_task: usize,
    pub max_concurrency: usize,
    pub active_concurrency: usize,
    pub max_retries: usize,
    pub max_tokens_per_24h: usize,
    pub used_tokens_24h: usize,
    pub remaining_tokens_24h: usize,
    pub max_tokens_per_task: usize,
    pub max_spend_micros_per_24h: u64,
    pub used_spend_micros_24h: u64,
    pub remaining_spend_micros_24h: u64,
    pub exhaustion_policy: String,
    pub cache_hits_total: usize,
    pub total_historical_ledger_entries: usize,
    pub recent_ledger: Vec<FrontierBudgetLedgerRecord>,
    pub blocked_tasks: Vec<FrontierBlockedTaskRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEvaluationRecord {
    pub provider_id: String,
    pub model_id: String,
    pub workload_kind: String,
    pub eval_runs: usize,
    pub success_rate: f64,
    pub verification_pass_rate: f64,
    pub calibration_score: f64,
    pub avg_latency_ms: u64,
    pub notes: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProviderDiscoveryRecord {
    pub provider_id: String,
    pub provider_type: String,
    pub base_url: String,
    pub transport: String,
    pub credential_status: String, // "not_required", "configured", "missing"
    pub discovered_models: Vec<String>,
    pub health_status: String,
    pub last_refreshed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionRecord {
    pub id: i64,
    pub task_id: String,
    pub caller_kind: String,
    pub caller_id: String,
    pub workload_kind: String,
    pub selected_provider: String,
    pub selected_model: String,
    pub is_frontier: bool,
    pub composite_score: f64,
    pub ranking_basis: String,
    pub fallback_reason: Option<String>,
    pub used_preference_order_fallback: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    pub id: String,
    pub label: String,
    pub node_type: String, // "concept", "source", "entity", "tool"
    pub properties_json: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEdge {
    pub id: i64,
    pub source_id: String,
    pub target_id: String,
    pub relation: String, // "explains", "discovered_from", "uses_tool", "relates_to", "supports"
    pub weight: f64,
    pub created_at: String,
}

fn default_empty_json_array() -> String {
    "[]".to_string()
}

fn default_empty_json_obj() -> String {
    "{}".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub id: i64,
    pub tool_name: String,
    pub tool_version: i64,
    #[serde(default)]
    pub tool_hash: String,
    pub input_json: String,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default = "default_empty_json_array")]
    pub dependencies_json: String,
    pub output_json: Option<String>,
    pub error: Option<String>,
    pub exit_code: i32,
    pub duration_ms: i64,
    #[serde(default = "default_empty_json_array")]
    pub artifacts_json: String,
    #[serde(default = "default_empty_json_obj")]
    pub measured_results_json: String,
    #[serde(default)]
    pub specialist_id: Option<String>,
    #[serde(default)]
    pub immutable_evidence_hash: String,
    pub executed_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistPendingCommit {
    pub id: String,
    pub specialist_id: String,
    pub specialist_name: String,
    pub phase: String, // "phase2_validate_and_commit", "committed", "rejected"
    pub finding_kind: String, // "locally_reproduced" vs "source_supported"
    pub conclusion: String,
    pub assumptions: String,
    pub negative_results: String,
    pub is_negative_result: bool,
    pub execution_id: Option<i64>,
    pub claimed_measurements_json: String,
    pub reproducibility_artifacts_json: String,
    pub source_passage_ids: Vec<i64>,
    pub canonical_source_ids: Vec<String>,
    pub independent_source_count: usize,
    pub source_fidelity_summary: String,
    pub validation_status: String, // "pending", "committed", "rejected"
    pub rejection_reason: Option<String>,
    pub committed_claim_id: Option<String>,
    pub commit_idempotency_key: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistPhase2Proposal {
    pub specialist_id: String,
    pub finding_kind: String, // "locally_reproduced" or "source_supported"
    pub conclusion: String,
    pub assumptions: String,
    pub negative_results: String,
    pub is_negative_result: bool,
    pub execution_id: Option<i64>,
    pub claimed_measurements: serde_json::Value,
    pub source_passage_ids: Vec<i64>,
    pub canonical_source_ids: Vec<String>,
    pub cycle_id: i64,
    pub domain: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistCommitOutcome {
    pub commit_id: String,
    pub validation_status: String, // "committed" or "rejected"
    pub rejection_reason: Option<String>,
    pub committed_claim_id: Option<String>,
    pub trusted: bool,
    pub finding_kind: String,
    pub independent_source_count: usize,
    pub source_fidelity_summary: String,
    pub deduplicated_existing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchBacklogItem {
    pub id: i64,
    pub topic: String,
    pub discipline: String,
    pub priority: i64,
    pub hypothesis: Option<String>,
    pub status: String, // "pending", "investigating", "completed", "failed"
    pub created_at: String,
    pub updated_at: String,
}

fn default_epistemic_experiment_note() -> String {
    "Numerical agreement supports an experiment; it is not a proof.".to_string()
}

fn default_outcome_status() -> String {
    "completed".to_string()
}

fn default_experiment_data_kind() -> String {
    "synthetic_data".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchExperimentRecord {
    pub id: i64,
    pub cycle_id: i64,
    #[serde(default = "default_spec_domain")]
    pub domain: String,
    #[serde(default = "default_experiment_data_kind")]
    pub data_kind: String, // "real_data" or "synthetic_data"
    pub hypothesis: String,
    pub conditions: String,
    pub attempted_method: String,
    pub observed_outcome: String,
    pub lessons_learned: String,
    pub is_failure: bool,
    pub justification_to_retry: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub spec_id: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub execution_id: Option<i64>,
    #[serde(default = "default_empty_json_obj")]
    pub baseline_comparison_json: String,
    #[serde(default = "default_empty_json_obj")]
    pub held_out_evaluation_json: String,
    #[serde(default = "default_empty_json_obj")]
    pub measured_results_json: String,
    #[serde(default = "default_spec_fidelity_status")]
    pub source_fidelity_status: String,
    #[serde(default)]
    pub under_specified: bool,
    #[serde(default = "default_empty_json_array")]
    pub missing_source_details_json: String,
    #[serde(default = "default_outcome_status")]
    pub outcome_status: String,
    #[serde(default = "default_epistemic_experiment_note")]
    pub epistemic_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMapItem {
    pub id: i64,
    pub capability_name: String,
    pub status: String, // "demonstrated", "struggles", "missing"
    pub description: String,
    pub unlocks_research: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchSynthesisRecord {
    pub id: i64,
    pub question: String,
    pub sub_queries_json: String,
    pub canonical_works_json: String,
    pub competing_findings: String,
    pub limitations_summary: String,
    pub verification_checks_json: String,
    pub experiment_translation: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsultationRecord {
    pub id: String,
    pub domain: String,
    pub requested_tool: String,
    pub effective_tool: String,
    pub escalated_from: Option<String>,
    pub escalation_reason: Option<String>,
    pub depth: String,
    pub state: String, // "queued", "running", "awaiting_input", "completed", "failed", "cancelled"
    pub progress: String,
    pub input_json: String,
    pub pending_question: Option<String>,
    pub clarification_history_json: String,
    pub result_json: Option<String>,
    pub artifacts_json: String,
    pub error: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsultationFeedbackRecord {
    pub id: i64,
    pub consultation_id: String,
    pub domain: String,
    pub consumer: String,
    pub applied_advice: String,
    pub observed_outcome: String,
    pub succeeded: bool,
    pub discrepancies: String,
    pub metrics_json: String,
    pub status: String, // always "attributable_evaluation_evidence", never promoted to trusted claim on its own
    pub created_at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TopicScoreSplit {
    pub item_count: usize,
    pub triad_correctness: f64,
    pub baseline_correctness: f64,
    pub triad_citation_accuracy: f64,
    pub baseline_citation_accuracy: f64,
    pub triad_source_diversity: f64,
    pub baseline_source_diversity: f64,
    pub triad_contradiction_detection: f64,
    pub baseline_contradiction_detection: f64,
    pub triad_uncertainty_calibration: f64, // 1.0 - Brier error (higher is better)
    pub baseline_uncertainty_calibration: f64,
    pub triad_calibration_brier: f64,       // Brier error (lower is better)
    pub baseline_calibration_brier: f64,
    pub triad_problem_solving: f64,
    pub baseline_problem_solving: f64,
}

fn default_baseline_provider_id() -> String {
    "local_snap".to_string()
}

fn default_baseline_model_id() -> String {
    "smollm3-3b".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRunRecord {
    pub id: i64,
    pub domain: String,
    pub benchmark_name: String,
    pub item_count: usize,
    pub triad_correctness: f64,
    pub baseline_correctness: f64,
    pub triad_problem_solving: f64,
    pub baseline_problem_solving: f64,
    pub triad_calibration_brier: f64,
    pub baseline_calibration_brier: f64,
    pub triad_citation_accuracy: f64,
    pub baseline_citation_accuracy: f64,
    #[serde(default)]
    pub triad_source_diversity: f64,
    #[serde(default)]
    pub baseline_source_diversity: f64,
    #[serde(default)]
    pub triad_contradiction_detection: f64,
    #[serde(default)]
    pub baseline_contradiction_detection: f64,
    #[serde(default)]
    pub triad_uncertainty_calibration: f64,
    #[serde(default)]
    pub baseline_uncertainty_calibration: f64,
    #[serde(default = "default_baseline_provider_id")]
    pub baseline_provider_id: String,
    #[serde(default = "default_baseline_model_id")]
    pub baseline_model_id: String,
    #[serde(default)]
    pub baseline_is_frontier: bool,
    #[serde(default)]
    pub baseline_selection_basis: String,
    #[serde(default)]
    pub baseline_fallback_reason: Option<String>,
    #[serde(default)]
    pub evaluated_corpus_full_text_share: f64,
    #[serde(default)]
    pub full_text_scores: TopicScoreSplit,
    #[serde(default)]
    pub abstract_or_low_fidelity_scores: TopicScoreSplit,
    #[serde(default)]
    pub real_data_problem_solving: f64,
    #[serde(default)]
    pub baseline_real_data_problem_solving: f64,
    #[serde(default)]
    pub synthetic_data_problem_solving: f64,
    #[serde(default)]
    pub baseline_synthetic_data_problem_solving: f64,
    #[serde(default)]
    pub real_data_experiment_count: usize,
    #[serde(default)]
    pub synthetic_data_experiment_count: usize,
    #[serde(default)]
    pub domain_competence_verified: bool,
    pub details_json: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DomainTopicCoverageRecord {
    pub domain: String,
    pub topic: String,
    pub discipline: String,
    pub coverage_status: String, // "covered", "thin", "contradicted", "low_fidelity"
    pub full_text_sources_count: usize,
    pub abstract_only_sources_count: usize,
    pub low_fidelity_sources_count: usize,
    pub contradiction_count: usize,
    pub substantive_claims_count: usize,
    pub eval_item_count: usize,
    pub eval_correctness: Option<f64>,
    pub eval_problem_solving: Option<f64>,
    pub eval_citation_accuracy: Option<f64>,
    pub eval_uncertainty_calibration: Option<f64>,
    pub reingestion_requested: bool,
    pub reingestion_source_ids: Vec<String>,
    pub backlog_priority: i64,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainOntologyDiagnostics {
    pub domain: String,
    pub trusted_claims_in_domain: usize,
    pub substantive_claims_in_domain: usize,
    pub raw_specialist_log_claims_in_domain: usize,
    pub unresolved_claims_in_domain: usize,
    pub rejected_claims_in_domain: usize,
    pub research_works_in_domain: usize,
    pub multi_connector_works_in_domain: usize,
    pub works_with_code_in_domain: usize,
    pub research_syntheses_in_domain: usize,
    pub semantic_nodes_in_domain: usize,
    pub empty_property_nodes_in_domain: usize,
    pub total_semantic_edges: usize,
    pub generic_relates_to_edges: usize,
    pub total_experiments: usize,
    pub recorded_failure_experiments: usize,
    pub latest_claim_updated_at: Option<String>,
    pub latest_work_retrieved_at: Option<String>,
    pub outcome_feedback_count: usize,
    pub outcome_feedback_success_count: usize,
}

#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch("
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                revision TEXT,
                retrieved_at TEXT NOT NULL,
                UNIQUE(url)
            );

            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                source_url TEXT NOT NULL,
                proposer TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('unresolved','trusted','rejected','superseded')),
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL REFERENCES claims(id),
                claim_version INTEGER NOT NULL,
                agent TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('endorse','reject')),
                cycle_id INTEGER NOT NULL,
                rationale TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(claim_id, claim_version, cycle_id, agent)
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(memory_type, target_id)
            );

            CREATE TABLE IF NOT EXISTS user_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                answered_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                schema_json TEXT NOT NULL,
                source_code TEXT NOT NULL,
                binary_path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                endorser_1 TEXT,
                endorser_2 TEXT,
                test_passed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                tool_version INTEGER NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT,
                error TEXT,
                exit_code INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                executed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS environment_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                observation TEXT NOT NULL,
                status TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS discussion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                round_num INTEGER NOT NULL,
                agent TEXT NOT NULL,
                public_message TEXT NOT NULL,
                thought TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                node_type TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS semantic_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL REFERENCES semantic_nodes(id),
                target_id TEXT NOT NULL REFERENCES semantic_nodes(id),
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                UNIQUE(source_id, target_id, relation)
            );

            CREATE TABLE IF NOT EXISTS constitution_acknowledgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                digest TEXT NOT NULL,
                context_type TEXT NOT NULL,
                acknowledged_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_reuses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                context TEXT NOT NULL,
                reused_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS curriculum_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                step TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_backlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                discipline TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                hypothesis TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending', 'investigating', 'completed', 'failed')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic)
            );

            CREATE TABLE IF NOT EXISTS research_experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                hypothesis TEXT NOT NULL,
                conditions TEXT NOT NULL,
                attempted_method TEXT NOT NULL,
                observed_outcome TEXT NOT NULL,
                lessons_learned TEXT NOT NULL,
                is_failure INTEGER NOT NULL DEFAULT 0,
                justification_to_retry TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS capability_map (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capability_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK(status IN ('demonstrated', 'struggles', 'missing')),
                description TEXT NOT NULL,
                unlocks_research TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extensions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                hook TEXT NOT NULL,
                version INTEGER NOT NULL,
                protocol_version TEXT NOT NULL,
                description TEXT NOT NULL,
                executable_path TEXT NOT NULL,
                executable_hash TEXT NOT NULL,
                config_json TEXT NOT NULL DEFAULT '{}',
                state TEXT NOT NULL CHECK(state IN ('proposed_finding', 'trusted_finding', 'implemented', 'evaluated', 'active', 'rolled_back')),
                evaluation_results_json TEXT,
                endorser_1 TEXT,
                endorser_2 TEXT,
                failure_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            CREATE TABLE IF NOT EXISTS coordinator_change_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                rationale TEXT NOT NULL,
                proposed_interface_json TEXT NOT NULL,
                experiment_evidence_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('proposed', 'approved', 'implemented', 'rejected')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS specialists (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                task TEXT NOT NULL,
                completion_criteria TEXT NOT NULL,
                max_tokens INTEGER NOT NULL,
                deadline TEXT NOT NULL,
                endorser_1 TEXT NOT NULL,
                endorser_2 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'checkpointed', 'budget_exhausted')),
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS specialist_limit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                limit_val INTEGER NOT NULL,
                reason TEXT NOT NULL,
                change_type TEXT NOT NULL CHECK(change_type IN ('initial', 'increase', 'user_reduction')),
                changed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS specialist_limit_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                current_limit INTEGER NOT NULL,
                requested_limit INTEGER NOT NULL,
                expected_benefit TEXT NOT NULL,
                available_resources TEXT NOT NULL,
                config_revision TEXT NOT NULL,
                votes_explorer INTEGER NOT NULL DEFAULT 0,
                votes_sceptic INTEGER NOT NULL DEFAULT 0,
                votes_builder INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_works (
                canonical_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                doi TEXT,
                arxiv_id TEXT,
                versions_json TEXT NOT NULL,
                published_date TEXT,
                authors_json TEXT NOT NULL,
                primary_connector TEXT NOT NULL,
                corroborating_connectors_json TEXT NOT NULL,
                urls_json TEXT NOT NULL,
                code_urls_json TEXT NOT NULL,
                abstract_excerpt TEXT NOT NULL,
                experimental_details TEXT NOT NULL,
                limitations_excerpt TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                availability_status TEXT NOT NULL,
                duplicate_copies_merged INTEGER NOT NULL DEFAULT 1,
                retrieved_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_syntheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                sub_queries_json TEXT NOT NULL,
                canonical_works_json TEXT NOT NULL,
                competing_findings TEXT NOT NULL,
                limitations_summary TEXT NOT NULL,
                verification_checks_json TEXT NOT NULL,
                experiment_translation TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consultations (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                requested_tool TEXT NOT NULL,
                effective_tool TEXT NOT NULL,
                escalated_from TEXT,
                escalation_reason TEXT,
                depth TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'awaiting_input', 'completed', 'failed', 'cancelled')),
                progress TEXT NOT NULL DEFAULT '',
                input_json TEXT NOT NULL,
                pending_question TEXT,
                clarification_history_json TEXT NOT NULL DEFAULT '[]',
                result_json TEXT,
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS consultation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consultation_id TEXT NOT NULL REFERENCES consultations(id),
                domain TEXT NOT NULL,
                consumer TEXT NOT NULL,
                applied_advice TEXT NOT NULL,
                observed_outcome TEXT NOT NULL,
                succeeded INTEGER NOT NULL DEFAULT 0,
                discrepancies TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'attributable_evaluation_evidence',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                item_count INTEGER NOT NULL,
                triad_correctness REAL NOT NULL,
                baseline_correctness REAL NOT NULL,
                triad_problem_solving REAL NOT NULL,
                baseline_problem_solving REAL NOT NULL,
                triad_calibration_brier REAL NOT NULL,
                baseline_calibration_brier REAL NOT NULL,
                triad_citation_accuracy REAL NOT NULL,
                baseline_citation_accuracy REAL NOT NULL,
                triad_source_diversity REAL NOT NULL DEFAULT 0.0,
                baseline_source_diversity REAL NOT NULL DEFAULT 0.0,
                triad_contradiction_detection REAL NOT NULL DEFAULT 0.0,
                baseline_contradiction_detection REAL NOT NULL DEFAULT 0.0,
                triad_uncertainty_calibration REAL NOT NULL DEFAULT 0.0,
                baseline_uncertainty_calibration REAL NOT NULL DEFAULT 0.0,
                baseline_provider_id TEXT NOT NULL DEFAULT 'local_snap',
                baseline_model_id TEXT NOT NULL DEFAULT 'smollm3-3b',
                baseline_is_frontier INTEGER NOT NULL DEFAULT 0,
                baseline_selection_basis TEXT NOT NULL DEFAULT '',
                baseline_fallback_reason TEXT,
                evaluated_corpus_full_text_share REAL NOT NULL DEFAULT 0.0,
                full_text_scores_json TEXT NOT NULL DEFAULT '{}',
                abstract_or_low_fidelity_scores_json TEXT NOT NULL DEFAULT '{}',
                real_data_problem_solving REAL NOT NULL DEFAULT 0.0,
                baseline_real_data_problem_solving REAL NOT NULL DEFAULT 0.0,
                synthetic_data_problem_solving REAL NOT NULL DEFAULT 0.0,
                baseline_synthetic_data_problem_solving REAL NOT NULL DEFAULT 0.0,
                real_data_experiment_count INTEGER NOT NULL DEFAULT 0,
                synthetic_data_experiment_count INTEGER NOT NULL DEFAULT 0,
                domain_competence_verified INTEGER NOT NULL DEFAULT 0,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS domain_topic_coverage (
                domain TEXT NOT NULL,
                topic TEXT NOT NULL,
                discipline TEXT NOT NULL DEFAULT '',
                coverage_status TEXT NOT NULL CHECK(coverage_status IN ('covered', 'thin', 'contradicted', 'low_fidelity')),
                full_text_sources_count INTEGER NOT NULL DEFAULT 0,
                abstract_only_sources_count INTEGER NOT NULL DEFAULT 0,
                low_fidelity_sources_count INTEGER NOT NULL DEFAULT 0,
                contradiction_count INTEGER NOT NULL DEFAULT 0,
                substantive_claims_count INTEGER NOT NULL DEFAULT 0,
                eval_item_count INTEGER NOT NULL DEFAULT 0,
                eval_correctness REAL,
                eval_problem_solving REAL,
                eval_citation_accuracy REAL,
                eval_uncertainty_calibration REAL,
                reingestion_requested INTEGER NOT NULL DEFAULT 0,
                reingestion_source_ids_json TEXT NOT NULL DEFAULT '[]',
                backlog_priority INTEGER NOT NULL DEFAULT 80,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (domain, topic)
            );

            CREATE TABLE IF NOT EXISTS tracked_sources (
                canonical_id TEXT PRIMARY KEY,
                canonical_url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                normalized_title TEXT NOT NULL DEFAULT '',
                provider_id TEXT,
                doi TEXT,
                arxiv_id TEXT,
                version TEXT NOT NULL DEFAULT 'v1',
                versions_seen_json TEXT NOT NULL DEFAULT '[\"v1\"]',
                content_hash TEXT NOT NULL DEFAULT '',
                source_kind TEXT NOT NULL DEFAULT 'paper',
                domain TEXT NOT NULL DEFAULT 'frontier_ai_research_and_verification',
                observed_urls_json TEXT NOT NULL DEFAULT '[]',
                observed_connectors_json TEXT NOT NULL DEFAULT '[]',
                route_observation_count INTEGER NOT NULL DEFAULT 1,
                discovery_status TEXT NOT NULL DEFAULT 'discovered' CHECK(discovery_status IN ('discovered')),
                discovered_at TEXT NOT NULL,
                ingestion_status TEXT NOT NULL DEFAULT 'pending' CHECK(ingestion_status IN ('pending', 'ingested', 'failed')),
                ingested_at TEXT,
                research_status TEXT NOT NULL DEFAULT 'pending' CHECK(research_status IN ('pending', 'reserved', 'completed', 'failed')),
                reserved_by TEXT,
                reservation_token TEXT,
                reserved_at TEXT,
                completed_at TEXT,
                completed_version TEXT,
                completed_content_hash TEXT,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_failure_reason TEXT,
                last_failed_at TEXT,
                retry_eligible INTEGER NOT NULL DEFAULT 1,
                next_retry_at TEXT,
                revisit_reason TEXT,
                has_contradiction INTEGER NOT NULL DEFAULT 0,
                contradiction_details TEXT,
                investigated_questions_json TEXT NOT NULL DEFAULT '[]',
                ingestion_completeness TEXT NOT NULL DEFAULT 'abstract_only',
                full_text_format TEXT,
                topic TEXT,
                is_orientation_only INTEGER NOT NULL DEFAULT 0,
                section_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_originals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_source_id TEXT NOT NULL REFERENCES tracked_sources(canonical_id),
                version TEXT NOT NULL DEFAULT 'v1',
                connector TEXT NOT NULL,
                content_kind TEXT NOT NULL CHECK(content_kind IN ('full_text', 'abstract', 'orientation')),
                format TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                abstract_text TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                byte_len INTEGER NOT NULL DEFAULT 0,
                harvested_references_json TEXT NOT NULL DEFAULT '[]',
                ingested_at TEXT NOT NULL,
                UNIQUE(canonical_source_id, version, content_kind)
            );

            CREATE TABLE IF NOT EXISTS source_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_source_id TEXT NOT NULL REFERENCES tracked_sources(canonical_id),
                version TEXT NOT NULL DEFAULT 'v1',
                connector TEXT NOT NULL,
                section_path TEXT NOT NULL,
                section_kind TEXT NOT NULL,
                source_location TEXT NOT NULL,
                content TEXT NOT NULL,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS connector_health (
                connector TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT 'metadata',
                status TEXT NOT NULL DEFAULT 'healthy',
                last_successful_fetch_at TEXT,
                last_attempted_at TEXT,
                last_error TEXT,
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_successes INTEGER NOT NULL DEFAULT 0,
                total_failures INTEGER NOT NULL DEFAULT 0,
                discovered_count INTEGER NOT NULL DEFAULT 0,
                ingested_count INTEGER NOT NULL DEFAULT 0,
                full_text_count INTEGER NOT NULL DEFAULT 0,
                abstract_only_count INTEGER NOT NULL DEFAULT 0,
                avg_latency_ms INTEGER NOT NULL DEFAULT 0,
                rate_limit_ms INTEGER NOT NULL DEFAULT 0,
                is_orientation_only INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_corroborations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_id TEXT NOT NULL,
                canonical_source_id TEXT NOT NULL REFERENCES tracked_sources(canonical_id),
                source_kind TEXT NOT NULL DEFAULT 'paper',
                observed_routes_json TEXT NOT NULL DEFAULT '[]',
                observed_connectors_json TEXT NOT NULL DEFAULT '[]',
                route_count INTEGER NOT NULL DEFAULT 1,
                domain TEXT NOT NULL DEFAULT 'frontier_ai_research_and_verification',
                recorded_at TEXT NOT NULL,
                UNIQUE(target_id, canonical_source_id)
            );

            CREATE TABLE IF NOT EXISTS specialist_pending_commits (
                id TEXT PRIMARY KEY,
                specialist_id TEXT NOT NULL,
                specialist_name TEXT NOT NULL DEFAULT '',
                phase TEXT NOT NULL DEFAULT 'phase2_validate_and_commit',
                finding_kind TEXT NOT NULL CHECK(finding_kind IN ('locally_reproduced', 'source_supported')),
                conclusion TEXT NOT NULL,
                assumptions TEXT NOT NULL DEFAULT '',
                negative_results TEXT NOT NULL DEFAULT '',
                is_negative_result INTEGER NOT NULL DEFAULT 0,
                execution_id INTEGER,
                claimed_measurements_json TEXT NOT NULL DEFAULT '{}',
                reproducibility_artifacts_json TEXT NOT NULL DEFAULT '{}',
                source_passage_ids_json TEXT NOT NULL DEFAULT '[]',
                canonical_source_ids_json TEXT NOT NULL DEFAULT '[]',
                independent_source_count INTEGER NOT NULL DEFAULT 0,
                source_fidelity_summary TEXT NOT NULL DEFAULT '',
                validation_status TEXT NOT NULL CHECK(validation_status IN ('pending', 'committed', 'rejected')),
                rejection_reason TEXT,
                committed_claim_id TEXT,
                commit_idempotency_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiment_specifications (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT 'frontier_ai_research_and_verification',
                executable_check_kind TEXT NOT NULL DEFAULT 'simulation',
                research_question TEXT NOT NULL,
                falsifiable_hypothesis TEXT NOT NULL,
                mechanism_under_test TEXT NOT NULL,
                inputs_description TEXT NOT NULL,
                assumptions_json TEXT NOT NULL DEFAULT '[]',
                baseline_description TEXT NOT NULL,
                controls_json TEXT NOT NULL DEFAULT '[]',
                held_out_cases_json TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '[]',
                budget_max_runtime_secs INTEGER NOT NULL DEFAULT 30,
                budget_max_memory_mb INTEGER NOT NULL DEFAULT 512,
                seeds_json TEXT NOT NULL DEFAULT '[]',
                reproducibility_requirements TEXT NOT NULL DEFAULT '',
                expected_outputs_json TEXT NOT NULL DEFAULT '[]',
                known_limitations_json TEXT NOT NULL DEFAULT '[]',
                cited_source_ids_json TEXT NOT NULL DEFAULT '[]',
                cited_passage_ids_json TEXT NOT NULL DEFAULT '[]',
                source_fidelity_status TEXT NOT NULL DEFAULT 'self_contained_computational',
                under_specified INTEGER NOT NULL DEFAULT 0,
                missing_source_details_json TEXT NOT NULL DEFAULT '[]',
                reingestion_requested INTEGER NOT NULL DEFAULT 0,
                builder_agent TEXT NOT NULL DEFAULT 'Builder',
                sceptic_agent TEXT NOT NULL DEFAULT 'Sceptic',
                sceptic_review_status TEXT NOT NULL DEFAULT 'pending',
                sceptic_review_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS frontier_budget_ledger (
                reservation_id TEXT PRIMARY KEY,
                parent_call_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                caller_kind TEXT NOT NULL,
                caller_id TEXT NOT NULL,
                workload_kind TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL DEFAULT 1,
                is_retry INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK(status IN ('reserved', 'completed', 'failed', 'aborted_after_dispatch')),
                reserved_tokens INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_spend_micros INTEGER NOT NULL DEFAULT 0,
                actual_spend_micros INTEGER NOT NULL DEFAULT 0,
                pricing_known INTEGER NOT NULL DEFAULT 0,
                prompt_hash TEXT NOT NULL DEFAULT '',
                created_at_epoch_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_text TEXT
            );

            CREATE TABLE IF NOT EXISTS frontier_response_cache (
                cache_key TEXT PRIMARY KEY,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                workload_kind TEXT NOT NULL,
                response_text TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                hit_count INTEGER NOT NULL DEFAULT 0,
                created_at_epoch_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_hit_at TEXT
            );

            CREATE TABLE IF NOT EXISTS frontier_blocked_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                caller_kind TEXT NOT NULL,
                caller_id TEXT NOT NULL,
                workload_kind TEXT NOT NULL,
                requested_provider TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                exhaustion_reason TEXT NOT NULL,
                exhaustion_action TEXT NOT NULL CHECK(exhaustion_action IN ('queued', 'continued_locally', 'reported_blocked')),
                status TEXT NOT NULL CHECK(status IN ('queued', 'resolved', 'blocked', 'continued_locally')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_evaluations (
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                workload_kind TEXT NOT NULL,
                eval_runs INTEGER NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 0.0,
                verification_pass_rate REAL NOT NULL DEFAULT 0.0,
                calibration_score REAL NOT NULL DEFAULT 0.0,
                avg_latency_ms INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (provider_id, model_id, workload_kind)
            );

            CREATE TABLE IF NOT EXISTS model_provider_discovery (
                provider_id TEXT PRIMARY KEY,
                provider_type TEXT NOT NULL,
                base_url TEXT NOT NULL,
                transport TEXT NOT NULL,
                credential_status TEXT NOT NULL,
                discovered_models_json TEXT NOT NULL DEFAULT '[]',
                health_status TEXT NOT NULL DEFAULT 'unknown',
                last_refreshed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_selection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                caller_kind TEXT NOT NULL,
                caller_id TEXT NOT NULL,
                workload_kind TEXT NOT NULL,
                selected_provider TEXT NOT NULL,
                selected_model TEXT NOT NULL,
                is_frontier INTEGER NOT NULL DEFAULT 0,
                composite_score REAL NOT NULL DEFAULT 0.0,
                ranking_basis TEXT NOT NULL,
                fallback_reason TEXT,
                used_preference_order_fallback INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
        ")?;

        // Ensure domain namespace and separate claim verification columns exist across knowledge tables
        Self::ensure_domain_columns(&conn)?;

        // Backfill tracked_sources from existing research_works if upgrading an existing DB
        Self::backfill_tracked_sources_if_needed(&conn)?;

        // Migration check from older v0.2.0 memories table if it exists
        Self::migrate_legacy_memories_if_present(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn ensure_domain_columns(conn: &Connection) -> Result<()> {
        let default_domain = "frontier_ai_research_and_verification";
        let alter_statements = [
            format!("ALTER TABLE claims ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            "ALTER TABLE claims ADD COLUMN source_support_status TEXT NOT NULL DEFAULT 'single_source'".to_string(),
            "ALTER TABLE claims ADD COLUMN source_corroboration_count INTEGER NOT NULL DEFAULT 1".to_string(),
            "ALTER TABLE claims ADD COLUMN experimental_verification_status TEXT NOT NULL DEFAULT 'unverified'".to_string(),
            "ALTER TABLE claims ADD COLUMN verification_evidence TEXT".to_string(),
            "ALTER TABLE claims ADD COLUMN finding_kind TEXT NOT NULL DEFAULT 'source_supported'".to_string(),
            "ALTER TABLE claims ADD COLUMN source_fidelity_summary TEXT".to_string(),
            "ALTER TABLE claims ADD COLUMN execution_id INTEGER".to_string(),
            "ALTER TABLE claims ADD COLUMN assumptions_text TEXT".to_string(),
            "ALTER TABLE claims ADD COLUMN negative_results_text TEXT".to_string(),
            "ALTER TABLE claims ADD COLUMN reproducibility_artifacts_json TEXT".to_string(),
            format!("ALTER TABLE embeddings ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            format!("ALTER TABLE semantic_nodes ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            format!("ALTER TABLE research_works ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            format!("ALTER TABLE research_syntheses ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            "ALTER TABLE tracked_sources ADD COLUMN ingestion_completeness TEXT NOT NULL DEFAULT 'abstract_only'".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN full_text_format TEXT".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN topic TEXT".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN is_orientation_only INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN section_count INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN equations_preserved INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN tables_found INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN low_fidelity INTEGER NOT NULL DEFAULT 1".to_string(),
            "ALTER TABLE tracked_sources ADD COLUMN low_fidelity_reason TEXT".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN tool_hash TEXT NOT NULL DEFAULT ''".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN seed INTEGER".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN dependencies_json TEXT NOT NULL DEFAULT '[]'".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN artifacts_json TEXT NOT NULL DEFAULT '[]'".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN measured_results_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN specialist_id TEXT".to_string(),
            "ALTER TABLE tool_executions ADD COLUMN immutable_evidence_hash TEXT NOT NULL DEFAULT ''".to_string(),
            "ALTER TABLE specialists ADD COLUMN phase TEXT NOT NULL DEFAULT 'phase1_investigate_and_execute'".to_string(),
            "ALTER TABLE specialists ADD COLUMN plan_json TEXT".to_string(),
            "ALTER TABLE specialists ADD COLUMN execution_ids_json TEXT NOT NULL DEFAULT '[]'".to_string(),
            "ALTER TABLE specialists ADD COLUMN pending_commit_id TEXT".to_string(),
            "ALTER TABLE tools ADD COLUMN experiment_spec_id TEXT".to_string(),
            "ALTER TABLE tools ADD COLUMN sceptic_verification_notes TEXT".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN spec_id TEXT".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN tool_name TEXT".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN execution_id INTEGER".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN baseline_comparison_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN held_out_evaluation_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN measured_results_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN source_fidelity_status TEXT NOT NULL DEFAULT 'no_external_source'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN under_specified INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN missing_source_details_json TEXT NOT NULL DEFAULT '[]'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN outcome_status TEXT NOT NULL DEFAULT 'completed'".to_string(),
            "ALTER TABLE research_experiments ADD COLUMN epistemic_note TEXT NOT NULL DEFAULT 'Numerical agreement supports an experiment; it is not a proof.'".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN baseline_description TEXT NOT NULL DEFAULT ''".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN budget_max_runtime_secs INTEGER NOT NULL DEFAULT 30".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN budget_max_memory_mb INTEGER NOT NULL DEFAULT 512".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN reproducibility_requirements TEXT NOT NULL DEFAULT ''".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN builder_agent TEXT NOT NULL DEFAULT 'Builder'".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN sceptic_agent TEXT NOT NULL DEFAULT 'Sceptic'".to_string(),
            "ALTER TABLE experiment_specifications ADD COLUMN data_kind TEXT NOT NULL DEFAULT 'synthetic_data'".to_string(),
            format!("ALTER TABLE research_experiments ADD COLUMN domain TEXT NOT NULL DEFAULT '{}'", default_domain),
            "ALTER TABLE research_experiments ADD COLUMN data_kind TEXT NOT NULL DEFAULT 'synthetic_data'".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN triad_source_diversity REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_source_diversity REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN triad_contradiction_detection REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_contradiction_detection REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN triad_uncertainty_calibration REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_uncertainty_calibration REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_provider_id TEXT NOT NULL DEFAULT 'local_snap'".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_model_id TEXT NOT NULL DEFAULT 'smollm3-3b'".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_is_frontier INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_selection_basis TEXT NOT NULL DEFAULT ''".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_fallback_reason TEXT".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN evaluated_corpus_full_text_share REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN full_text_scores_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN abstract_or_low_fidelity_scores_json TEXT NOT NULL DEFAULT '{}'".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN real_data_problem_solving REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_real_data_problem_solving REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN synthetic_data_problem_solving REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN baseline_synthetic_data_problem_solving REAL NOT NULL DEFAULT 0.0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN real_data_experiment_count INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN synthetic_data_experiment_count INTEGER NOT NULL DEFAULT 0".to_string(),
            "ALTER TABLE evaluation_runs ADD COLUMN domain_competence_verified INTEGER NOT NULL DEFAULT 0".to_string(),
        ];
        for sql in &alter_statements {
            let _ = conn.execute(sql, []);
        }

        // Backfill data_kind on existing experiment_specifications and research_experiments
        let _ = conn.execute(
            "UPDATE experiment_specifications
             SET data_kind = 'real_data'
             WHERE (source_fidelity_status = 'full_text_verified' AND under_specified = 0)
                OR executable_check_kind IN ('data_reanalysis', 'real_implementation', 'statistical_replication')",
            [],
        );
        let _ = conn.execute(
            "UPDATE research_experiments
             SET data_kind = 'real_data'
             WHERE (source_fidelity_status = 'full_text_verified' AND under_specified = 0)
                OR spec_id IN (SELECT id FROM experiment_specifications WHERE data_kind = 'real_data')",
            [],
        );

        // Retire any legacy superficial verifier tools from earlier runs so they are not selected for experiments
        let _ = conn.execute(
            "UPDATE tools
             SET status = 'retired',
                 sceptic_verification_notes = 'Retired by Sceptic governance: superficial word-counting, string-inspection, or constant-score verifier without ExperimentSpecification'
             WHERE status = 'active'
               AND experiment_spec_id IS NULL
               AND (
                   name IN (
                       'calc_text_metrics',
                       'hash_verifier',
                       'bayesian_claim_calibrator',
                       'citation_graph_scorer',
                       'token_parity_normalizer',
                       'moe_routing_entropy_analyzer',
                       'prm_step_verifier'
                   )
                   OR name LIKE 'cycle_%_research_verifier'
               )",
            [],
        );

        // Enforce host execution evidence immutability at the SQLite level
        let _ = conn.execute_batch("
            CREATE TRIGGER IF NOT EXISTS trg_tool_executions_immutable_update
            BEFORE UPDATE ON tool_executions
            BEGIN
                SELECT RAISE(ABORT, 'tool_executions records are immutable host evidence and cannot be modified');
            END;

            CREATE TRIGGER IF NOT EXISTS trg_tool_executions_immutable_delete
            BEFORE DELETE ON tool_executions
            BEGIN
                SELECT RAISE(ABORT, 'tool_executions records are immutable host evidence and cannot be deleted');
            END;

            CREATE TRIGGER IF NOT EXISTS trg_frontier_budget_ledger_no_delete
            BEFORE DELETE ON frontier_budget_ledger
            BEGIN
                SELECT RAISE(ABORT, 'frontier_budget_ledger accounting history is persistent and cannot be deleted or reset');
            END;
        ");

        // Backfill extraction quality metrics on existing full_text sources from stored `source_sections`
        let _ = conn.execute(
            "UPDATE tracked_sources SET
                section_count = (
                    SELECT COUNT(*) FROM source_sections ss
                    WHERE ss.canonical_source_id = tracked_sources.canonical_id
                      AND ss.section_kind != 'hierarchy'
                ),
                equations_preserved = (
                    SELECT COUNT(*) FROM source_sections ss
                    WHERE ss.canonical_source_id = tracked_sources.canonical_id
                      AND (ss.section_kind = 'equation'
                           OR ss.content LIKE '%[Notation:%'
                           OR ss.content LIKE '%\\begin{equation%'
                           OR ss.content LIKE '%\\begin{align%'
                           OR ss.content LIKE '%$$%')
                ),
                tables_found = (
                    SELECT COUNT(*) FROM source_sections ss
                    WHERE ss.canonical_source_id = tracked_sources.canonical_id
                      AND (ss.section_kind = 'table'
                           OR ss.content LIKE '%\\begin{table%'
                           OR ss.content LIKE '%\\begin{tabular%')
                ),
                low_fidelity = CASE
                    WHEN (SELECT COUNT(*) FROM source_sections ss
                          WHERE ss.canonical_source_id = tracked_sources.canonical_id
                            AND ss.section_kind != 'hierarchy') >= 2 THEN 0
                    ELSE 1
                END
             WHERE ingestion_completeness = 'full_text'",
            [],
        );

        Ok(())
    }

    fn backfill_tracked_sources_if_needed(conn: &Connection) -> Result<()> {
        let tracked_count: i64 = conn
            .query_row("SELECT count(*) FROM tracked_sources", [], |r| r.get(0))
            .unwrap_or(0);
        let works_count: i64 = conn
            .query_row("SELECT count(*) FROM research_works", [], |r| r.get(0))
            .unwrap_or(0);
        if tracked_count == 0 && works_count > 0 {
            let mut stmt = conn.prepare(
                "SELECT canonical_id, title, normalized_title, doi, arxiv_id, versions_json,
                        primary_connector, corroborating_connectors_json, urls_json,
                        abstract_excerpt, duplicate_copies_merged, retrieved_at, domain
                 FROM research_works",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, String>(8)?,
                    r.get::<_, String>(9)?,
                    r.get::<_, i64>(10)?,
                    r.get::<_, String>(11)?,
                    r.get::<_, String>(12).unwrap_or_else(|_| "frontier_ai_research_and_verification".to_string()),
                ))
            })?;
            for item in rows {
                if let Ok((cid, title, norm_title, doi, arxiv_id, ver_json, p_conn, corr_json, urls_json, abs_text, dup_merged, ret_at, dom)) = item {
                    let urls: Vec<String> = serde_json::from_str(&urls_json).unwrap_or_default();
                    let raw_url = urls.first().cloned().unwrap_or_else(|| cid.clone());
                    let versions: Vec<String> = serde_json::from_str(&ver_json).unwrap_or_default();
                    let latest_ver = versions.last().cloned().unwrap_or_else(|| "v1".to_string());
                    let identity = crate::web::canonicalize_source_identity(
                        &raw_url,
                        Some(&title),
                        doi.as_deref(),
                        arxiv_id.as_deref(),
                        Some(&latest_ver),
                        Some(&abs_text),
                        Some(&p_conn),
                    );
                    let versions_seen_str = if versions.is_empty() {
                        serde_json::to_string(&vec![identity.version.clone()]).unwrap_or_else(|_| "[\"v1\"]".to_string())
                    } else {
                        ver_json
                    };
                    let _ = conn.execute(
                        "INSERT OR IGNORE INTO tracked_sources (
                            canonical_id, canonical_url, title, normalized_title, provider_id, doi, arxiv_id,
                            version, versions_seen_json, content_hash, source_kind, domain,
                            observed_urls_json, observed_connectors_json, route_observation_count,
                            discovery_status, discovered_at, ingestion_status, ingested_at,
                            research_status, completed_at, completed_version, completed_content_hash,
                            retry_eligible, updated_at
                        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                                  'discovered', ?16, 'ingested', ?16, 'completed', ?16, ?8, ?10, 0, ?16)",
                        params![
                            cid,
                            identity.canonical_url,
                            title,
                            norm_title,
                            identity.provider_id,
                            doi.or(identity.doi),
                            arxiv_id.or(identity.arxiv_id),
                            identity.version,
                            versions_seen_str,
                            identity.content_hash,
                            identity.source_kind,
                            dom,
                            urls_json,
                            corr_json,
                            dup_merged.max(1),
                            ret_at,
                        ],
                    );
                }
            }
        }
        Ok(())
    }

    fn migrate_legacy_memories_if_present(conn: &Connection) -> Result<()> {
        let has_legacy: bool = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='memories'",
            [],
            |r| r.get(0),
        ).unwrap_or(0) > 0;

        if has_legacy {
            let mut stmt = conn.prepare("SELECT kind, text, source, status, embedding FROM memories")?;
            let legacy_rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })?;

            for item in legacy_rows {
                if let Ok((kind, text, source, status, emb_str)) = item {
                    let claim_id = compute_claim_id(&text);
                    let now = Utc::now().to_rfc3339();
                    let st = if status == "accepted" { "trusted" } else { "unresolved" };

                    conn.execute(
                        "INSERT OR IGNORE INTO claims (id, statement, source_url, proposer, status, version, created_at, updated_at) \
                         VALUES (?1, ?2, ?3, 'migration', ?4, 1, ?5, ?5)",
                        params![claim_id, text, source, st, now],
                    )?;

                    conn.execute(
                        "INSERT OR IGNORE INTO embeddings (memory_type, target_id, text, embedding, created_at) \
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![kind, claim_id, text, emb_str, now],
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn check_or_set_encoder(&self, fingerprint: &str, dim: usize, query_prefix: &str, doc_prefix: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let expected = format!("{}|dim={}|q_pfx={}|doc_pfx={}", fingerprint, dim, query_prefix, doc_prefix);
        let mut stmt = conn.prepare("SELECT value FROM meta WHERE key='encoder'")?;
        let mut rows = stmt.query([])?;
        if let Some(r) = rows.next()? {
            let prev: String = r.get(0)?;
            if prev != expected {
                bail!("Embedding configuration mismatch (current: {}, recorded: {}). Reindex required.", expected, prev);
            }
        } else {
            conn.execute("INSERT INTO meta (key, value) VALUES ('encoder', ?1)", [&expected])?;
        }
        Ok(())
    }

    pub fn add_source(&self, url: &str, title: &str, content: &str, revision: Option<&str>) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO sources (url, title, content, revision, retrieved_at) VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(url) DO UPDATE SET title=excluded.title, content=excluded.content, revision=excluded.revision, retrieved_at=excluded.retrieved_at",
            params![url, title, content, revision, now],
        )?;
        let id: i64 = conn.query_row("SELECT id FROM sources WHERE url=?1", [url], |r| r.get(0))?;
        Ok(id)
    }

    pub fn propose_claim(&self, statement: &str, source_url: &str, proposer: &str) -> Result<String> {
        let claim_id = compute_claim_id(statement);
        let now = Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO claims (id, statement, source_url, proposer, status, version, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, 'unresolved', 1, ?5, ?5)",
            params![claim_id, statement, source_url, proposer, now],
        )?;
        Ok(claim_id)
    }

    pub fn record_vote(
        &self,
        claim_id: &str,
        agent: &str,
        decision: &str,
        cycle_id: i64,
        rationale: Option<&str>,
    ) -> Result<bool> {
        let valid_agents = ["Explorer", "Sceptic", "Builder"];
        if !valid_agents.contains(&agent) {
            bail!("Constitutional voting violation: '{}' is not one of the 3 persistent identities (Explorer, Sceptic, Builder)", agent);
        }

        let conn = self.conn.lock().unwrap();
        let (version, status): (i64, String) = conn.query_row(
            "SELECT version, status FROM claims WHERE id=?1",
            [claim_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).context("Claim not found")?;

        let now = Utc::now().to_rfc3339();
        // Insert or replace vote from this agent for this claim_id + version + cycle
        conn.execute(
            "INSERT OR REPLACE INTO votes (claim_id, claim_version, agent, decision, cycle_id, rationale, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![claim_id, version, agent, decision, cycle_id, rationale, now],
        )?;

        // Check distinct endorsing votes for this cycle and version
        let endorse_count: i64 = conn.query_row(
            "SELECT count(DISTINCT agent) FROM votes WHERE claim_id=?1 AND claim_version=?2 AND cycle_id=?3 AND decision='endorse'",
            params![claim_id, version, cycle_id],
            |r| r.get(0),
        )?;

        if endorse_count >= 2 && status != "trusted" {
            let updated_at = Utc::now().to_rfc3339();
            conn.execute(
                "UPDATE claims SET status='trusted', updated_at=?1 WHERE id=?2",
                params![updated_at, claim_id],
            )?;
            return Ok(true); // Promoted to trusted!
        }
        Ok(false)
    }

    fn map_claim_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Claim> {
        Ok(Claim {
            id: r.get(0)?,
            statement: r.get(1)?,
            source_url: r.get(2)?,
            proposer: r.get(3)?,
            status: r.get(4)?,
            version: r.get(5)?,
            created_at: r.get(6)?,
            updated_at: r.get(7)?,
            source_support_status: r
                .get::<_, String>(8)
                .unwrap_or_else(|_| "single_source".to_string()),
            source_corroboration_count: r.get::<_, i64>(9).unwrap_or(1) as usize,
            experimental_verification_status: r
                .get::<_, String>(10)
                .unwrap_or_else(|_| "unverified".to_string()),
            verification_evidence: r.get(11).unwrap_or(None),
            finding_kind: r
                .get::<_, String>(12)
                .unwrap_or_else(|_| "source_supported".to_string()),
            source_fidelity_summary: r.get(13).unwrap_or(None),
            execution_id: r.get(14).unwrap_or(None),
            assumptions_text: r.get(15).unwrap_or(None),
            negative_results_text: r.get(16).unwrap_or(None),
            reproducibility_artifacts_json: r.get(17).unwrap_or(None),
        })
    }

    pub fn get_claim(&self, claim_id: &str) -> Result<Option<Claim>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, statement, source_url, proposer, status, version, created_at, updated_at, \
                    source_support_status, source_corroboration_count, experimental_verification_status, verification_evidence, \
                    finding_kind, source_fidelity_summary, execution_id, assumptions_text, negative_results_text, reproducibility_artifacts_json \
             FROM claims WHERE id=?1",
        )?;
        let mut rows = stmt.query([claim_id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Self::map_claim_row(r)?))
        } else {
            Ok(None)
        }
    }

    pub fn get_claim_votes(&self, claim_id: &str) -> Result<Vec<VoteRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, claim_id, claim_version, agent, decision, cycle_id, rationale, created_at FROM votes WHERE claim_id=?1 ORDER BY id ASC")?;
        let rows = stmt.query_map([claim_id], |r| {
            Ok(VoteRecord {
                id: r.get(0)?,
                claim_id: r.get(1)?,
                claim_version: r.get(2)?,
                agent: r.get(3)?,
                decision: r.get(4)?,
                cycle_id: r.get(5)?,
                rationale: r.get(6)?,
                created_at: r.get(7)?,
            })
        })?;
        let mut result = Vec::new();
        for item in rows {
            result.push(item?);
        }
        Ok(result)
    }

    pub fn count_trusted_learnings(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM claims WHERE status='trusted'",
            [],
            |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    pub fn list_learnings(&self, limit: usize) -> Result<Vec<Claim>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, statement, source_url, proposer, status, version, created_at, updated_at, \
                    source_support_status, source_corroboration_count, experimental_verification_status, verification_evidence, \
                    finding_kind, source_fidelity_summary, execution_id, assumptions_text, negative_results_text, reproducibility_artifacts_json \
             FROM claims WHERE status='trusted' ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_claim_row)?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn save_embedding(&self, memory_type: &str, target_id: &str, text: &str, embedding: &[f32]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let emb_json = serde_json::to_string(embedding)?;
        conn.execute(
            "INSERT INTO embeddings (memory_type, target_id, text, embedding, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(memory_type, target_id) DO UPDATE SET text=excluded.text, embedding=excluded.embedding, created_at=excluded.created_at",
            params![memory_type, target_id, text, emb_json, now],
        )?;
        Ok(())
    }

    pub fn search_semantic(&self, query_emb: &[f32], limit: usize) -> Result<Vec<(MemoryEmbedding, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT memory_type, target_id, text, embedding FROM embeddings")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;

        let mut scored = Vec::new();
        for row in rows {
            let (mtype, tid, text, emb_str) = row?;
            if let Ok(emb) = serde_json::from_str::<Vec<f32>>(&emb_str) {
                if let Ok(sim) = crate::model::compute_cosine(query_emb, &emb) {
                    scored.push((
                        MemoryEmbedding {
                            memory_type: mtype,
                            target_id: tid,
                            text,
                            embedding: emb,
                        },
                        sim,
                    ));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > limit {
            scored.truncate(limit);
        }
        Ok(scored)
    }

    pub fn record_environment_observation(&self, category: &str, observation: &str, status: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO environment_observations (category, observation, status, observed_at) VALUES (?1, ?2, ?3, ?4)",
            params![category, observation, status, now],
        )?;
        Ok(())
    }

    pub fn get_environment_observations(&self) -> Result<Vec<(String, String, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT category, observation, status, observed_at FROM environment_observations ORDER BY id DESC")?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_discussion_event(
        &self,
        cycle_id: i64,
        round_num: usize,
        agent: &str,
        public_message: &str,
        thought: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO discussion_events (cycle_id, round_num, agent, public_message, thought, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![cycle_id, round_num as i64, agent, public_message, thought, now],
        )?;
        Ok(())
    }

    pub fn add_user_question(&self, agent: &str, question: &str, max_per_day: usize) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let count_today: i64 = conn.query_row(
            "SELECT count(*) FROM user_questions WHERE created_at LIKE ?1",
            [format!("{}%", today)],
            |r| r.get(0),
        )?;
        if count_today as usize >= max_per_day {
            return Ok(false);
        }

        // Deduplication: check if question already exists unanswered
        let exists: i64 = conn.query_row(
            "SELECT count(*) FROM user_questions WHERE question=?1 AND answer IS NULL",
            [question],
            |r| r.get(0),
        )?;
        if exists > 0 {
            return Ok(false);
        }

        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO user_questions (agent, question, created_at) VALUES (?1, ?2, ?3)",
            params![agent, question, now],
        )?;
        Ok(true)
    }

    pub fn list_user_questions(&self) -> Result<Vec<UserQuestionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, agent, question, answer, answered_at, created_at FROM user_questions ORDER BY id ASC")?;
        let rows = stmt.query_map([], |r| {
            Ok(UserQuestionRecord {
                id: r.get(0)?,
                agent: r.get(1)?,
                question: r.get(2)?,
                answer: r.get(3)?,
                answered_at: r.get(4)?,
                created_at: r.get(5)?,
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn answer_user_question(&self, id: i64, answer: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let rows_affected = conn.execute(
            "UPDATE user_questions SET answer=?1, answered_at=?2 WHERE id=?3 AND answer IS NULL",
            params![answer, now, id],
        )?;
        if rows_affected == 0 {
            bail!("Question ID {} not found or already answered", id);
        }
        Ok(())
    }

    pub fn create_task(&self, id: &str, kind: &str, description: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO tasks (id, kind, description, status, created_at) VALUES (?1, ?2, ?3, 'queued', ?4)",
            params![id, kind, description, now],
        )?;
        Ok(())
    }

    pub fn update_task_status(&self, id: &str, status: &str, result: Option<&str>, error: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE tasks SET status=?1, result=?2, error=?3, completed_at=?4 WHERE id=?5",
            params![status, result, error, now, id],
        )?;
        Ok(())
    }

    pub fn get_task(&self, id: &str) -> Result<Option<TaskRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, kind, description, status, result, error, created_at, completed_at FROM tasks WHERE id=?1")?;
        let mut rows = stmt.query([id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(TaskRecord {
                id: r.get(0)?,
                kind: r.get(1)?,
                description: r.get(2)?,
                status: r.get(3)?,
                result: r.get(4)?,
                error: r.get(5)?,
                created_at: r.get(6)?,
                completed_at: r.get(7)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_pending_tasks(&self) -> Result<Vec<TaskRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, kind, description, status, result, error, created_at, completed_at FROM tasks WHERE status='queued' ORDER BY created_at ASC")?;
        let rows = stmt.query_map([], |r| {
            Ok(TaskRecord {
                id: r.get(0)?,
                kind: r.get(1)?,
                description: r.get(2)?,
                status: r.get(3)?,
                result: r.get(4)?,
                error: r.get(5)?,
                created_at: r.get(6)?,
                completed_at: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn register_tool(
        &self,
        name: &str,
        version: i64,
        description: &str,
        schema_json: &str,
        source_code: &str,
        binary_path: &str,
        source_hash: &str,
        endorser_1: Option<&str>,
        endorser_2: Option<&str>,
        test_passed: bool,
    ) -> Result<i64> {
        self.register_tool_with_spec(
            name,
            version,
            description,
            schema_json,
            source_code,
            binary_path,
            source_hash,
            endorser_1,
            endorser_2,
            test_passed,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_tool_with_spec(
        &self,
        name: &str,
        version: i64,
        description: &str,
        schema_json: &str,
        source_code: &str,
        binary_path: &str,
        source_hash: &str,
        endorser_1: Option<&str>,
        endorser_2: Option<&str>,
        test_passed: bool,
        experiment_spec_id: Option<&str>,
        sceptic_verification_notes: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let status = if test_passed && endorser_1.is_some() && endorser_2.is_some() {
            "active"
        } else {
            "draft"
        };
        conn.execute(
            "INSERT INTO tools (name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at, experiment_spec_id, sceptic_verification_notes) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                name,
                version,
                description,
                schema_json,
                source_code,
                binary_path,
                source_hash,
                status,
                endorser_1,
                endorser_2,
                if test_passed { 1 } else { 0 },
                now,
                experiment_spec_id,
                sceptic_verification_notes
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_tools(&self) -> Result<Vec<ToolRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at, experiment_spec_id, sceptic_verification_notes \
             FROM tools ORDER BY name ASC, version DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            let tp: i64 = r.get(11)?;
            Ok(ToolRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                version: r.get(2)?,
                description: r.get(3)?,
                schema_json: r.get(4)?,
                source_code: r.get(5)?,
                binary_path: r.get(6)?,
                source_hash: r.get(7)?,
                status: r.get(8)?,
                endorser_1: r.get(9)?,
                endorser_2: r.get(10)?,
                test_passed: tp == 1,
                created_at: r.get(12)?,
                experiment_spec_id: r.get(13).unwrap_or(None),
                sceptic_verification_notes: r.get(14).unwrap_or(None),
            })
        })?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn get_tool_by_name(&self, name: &str) -> Result<Option<ToolRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at, experiment_spec_id, sceptic_verification_notes \
             FROM tools WHERE name=?1 ORDER BY version DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([name])?;
        if let Some(r) = rows.next()? {
            let tp: i64 = r.get(11)?;
            Ok(Some(ToolRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                version: r.get(2)?,
                description: r.get(3)?,
                schema_json: r.get(4)?,
                source_code: r.get(5)?,
                binary_path: r.get(6)?,
                source_hash: r.get(7)?,
                status: r.get(8)?,
                endorser_1: r.get(9)?,
                endorser_2: r.get(10)?,
                test_passed: tp == 1,
                created_at: r.get(12)?,
                experiment_spec_id: r.get(13).unwrap_or(None),
                sceptic_verification_notes: r.get(14).unwrap_or(None),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn compute_immutable_execution_hash(
        name: &str,
        version: i64,
        tool_hash: &str,
        input_json: &str,
        seed: Option<i64>,
        dependencies_json: &str,
        output_json: Option<&str>,
        error: Option<&str>,
        exit_code: i32,
        duration_ms: i64,
        artifacts_json: &str,
        measured_results_json: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        let payload = format!(
            "tool={}|ver={}|hash={}|in={}|seed={}|deps={}|out={}|err={}|exit={}|ms={}|art={}|meas={}",
            name,
            version,
            tool_hash,
            input_json,
            seed.map(|s| s.to_string()).unwrap_or_else(|| "none".to_string()),
            dependencies_json,
            output_json.unwrap_or(""),
            error.unwrap_or(""),
            exit_code,
            duration_ms,
            artifacts_json,
            measured_results_json
        );
        hasher.update(payload.as_bytes());
        hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect()
    }

    pub fn record_immutable_tool_execution(
        &self,
        name: &str,
        version: i64,
        tool_hash: &str,
        input_json: &str,
        seed: Option<i64>,
        dependencies_json: &str,
        output_json: Option<&str>,
        error: Option<&str>,
        exit_code: i32,
        duration_ms: u64,
        artifacts_json: &str,
        measured_results_json: &str,
        specialist_id: Option<&str>,
    ) -> Result<ToolExecutionRecord> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let dur_i64 = duration_ms as i64;
        let effective_tool_hash = if tool_hash.trim().is_empty() {
            conn.query_row(
                "SELECT source_hash FROM tools WHERE name=?1 AND version=?2 LIMIT 1",
                params![name, version],
                |r| r.get::<_, String>(0),
            )
            .unwrap_or_else(|_| compute_claim_id(&format!("{}@{}", name, version)))
        } else {
            tool_hash.to_string()
        };
        let effective_deps = if dependencies_json.trim().is_empty() {
            r#"["serde@1.0","serde_json@1.0"]"#.to_string()
        } else {
            dependencies_json.to_string()
        };
        let effective_artifacts = if artifacts_json.trim().is_empty() {
            "[]".to_string()
        } else {
            artifacts_json.to_string()
        };
        let effective_measured = if measured_results_json.trim().is_empty() {
            output_json
                .and_then(|o| serde_json::from_str::<serde_json::Value>(o.trim()).ok())
                .map(|v| v.to_string())
                .unwrap_or_else(|| "{}".to_string())
        } else {
            measured_results_json.to_string()
        };

        let immutable_evidence_hash = Self::compute_immutable_execution_hash(
            name,
            version,
            &effective_tool_hash,
            input_json,
            seed,
            &effective_deps,
            output_json,
            error,
            exit_code,
            dur_i64,
            &effective_artifacts,
            &effective_measured,
        );

        conn.execute(
            "INSERT INTO tool_executions (
                tool_name, tool_version, tool_hash, input_json, seed, dependencies_json,
                output_json, error, exit_code, duration_ms, artifacts_json, measured_results_json,
                specialist_id, immutable_evidence_hash, executed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                name,
                version,
                effective_tool_hash,
                input_json,
                seed,
                effective_deps,
                output_json,
                error,
                exit_code,
                dur_i64,
                effective_artifacts,
                effective_measured,
                specialist_id,
                immutable_evidence_hash,
                now
            ],
        )?;
        let exec_id = conn.last_insert_rowid();

        if let Some(spec_id) = specialist_id {
            let existing_ids_str: String = conn
                .query_row(
                    "SELECT COALESCE(execution_ids_json, '[]') FROM specialists WHERE id=?1",
                    [spec_id],
                    |r| r.get(0),
                )
                .unwrap_or_else(|_| "[]".to_string());
            let mut exec_ids: Vec<i64> = serde_json::from_str(&existing_ids_str).unwrap_or_default();
            if !exec_ids.contains(&exec_id) {
                exec_ids.push(exec_id);
            }
            let updated_ids_json = serde_json::to_string(&exec_ids).unwrap_or_else(|_| "[]".to_string());
            let _ = conn.execute(
                "UPDATE specialists SET execution_ids_json=?1, updated_at=?2 WHERE id=?3",
                params![updated_ids_json, now, spec_id],
            );
        }

        Ok(ToolExecutionRecord {
            id: exec_id,
            tool_name: name.to_string(),
            tool_version: version,
            tool_hash: effective_tool_hash,
            input_json: input_json.to_string(),
            seed,
            dependencies_json: effective_deps,
            output_json: output_json.map(|s| s.to_string()),
            error: error.map(|s| s.to_string()),
            exit_code,
            duration_ms: dur_i64,
            artifacts_json: effective_artifacts,
            measured_results_json: effective_measured,
            specialist_id: specialist_id.map(|s| s.to_string()),
            immutable_evidence_hash,
            executed_at: now,
        })
    }

    pub fn record_tool_execution(
        &self,
        name: &str,
        version: i64,
        input_json: &str,
        output_json: Option<&str>,
        error: Option<&str>,
        exit_code: i32,
        duration_ms: u64,
    ) -> Result<()> {
        let _ = self.record_immutable_tool_execution(
            name,
            version,
            "",
            input_json,
            None,
            r#"["serde@1.0","serde_json@1.0"]"#,
            output_json,
            error,
            exit_code,
            duration_ms,
            "[]",
            "{}",
            None,
        )?;
        Ok(())
    }

    pub fn get_tool_execution(&self, id: i64) -> Result<Option<ToolExecutionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, tool_name, tool_version, tool_hash, input_json, seed, dependencies_json,
                    output_json, error, exit_code, duration_ms, artifacts_json, measured_results_json,
                    specialist_id, immutable_evidence_hash, executed_at
             FROM tool_executions WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(ToolExecutionRecord {
                id: r.get(0)?,
                tool_name: r.get(1)?,
                tool_version: r.get(2)?,
                tool_hash: r.get::<_, String>(3).unwrap_or_default(),
                input_json: r.get(4)?,
                seed: r.get(5).unwrap_or(None),
                dependencies_json: r.get::<_, String>(6).unwrap_or_else(|_| "[]".to_string()),
                output_json: r.get(7)?,
                error: r.get(8)?,
                exit_code: r.get(9)?,
                duration_ms: r.get(10)?,
                artifacts_json: r.get::<_, String>(11).unwrap_or_else(|_| "[]".to_string()),
                measured_results_json: r.get::<_, String>(12).unwrap_or_else(|_| "{}".to_string()),
                specialist_id: r.get(13).unwrap_or(None),
                immutable_evidence_hash: r.get::<_, String>(14).unwrap_or_default(),
                executed_at: r.get(15)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn replace_embeddings_atomic(&self, new_encoder_info: &str, items: &[(String, String, String, Vec<f32>)]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM embeddings", [])?;
        let now = Utc::now().to_rfc3339();
        for (mtype, target_id, text, emb) in items {
            let emb_json = serde_json::to_string(emb)?;
            tx.execute(
                "INSERT INTO embeddings (memory_type, target_id, text, embedding, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![mtype, target_id, text, emb_json, now],
            )?;
        }
        tx.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('encoder', ?1)", [new_encoder_info])?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_all_embeddings_raw(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT memory_type, target_id, text FROM embeddings")?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // --- Semantic Graph & Audit Extensions ---

    pub fn upsert_semantic_node(&self, id: &str, label: &str, node_type: &str, properties_json: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO semantic_nodes (id, label, node_type, properties_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET label = excluded.label, properties_json = excluded.properties_json",
            params![id, label, node_type, properties_json, now],
        )?;
        Ok(())
    }

    pub fn add_semantic_edge(&self, source_id: &str, target_id: &str, relation: &str, weight: f64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO semantic_edges (source_id, target_id, relation, weight, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(source_id, target_id, relation) DO UPDATE SET weight = excluded.weight",
            params![source_id, target_id, relation, weight, now],
        )?;
        Ok(())
    }

    pub fn get_semantic_graph_stats(&self) -> Result<(usize, usize)> {
        let conn = self.conn.lock().unwrap();
        let nodes: usize = conn.query_row("SELECT COUNT(*) FROM semantic_nodes", [], |r| r.get(0))?;
        let edges: usize = conn.query_row("SELECT COUNT(*) FROM semantic_edges", [], |r| r.get(0))?;
        Ok((nodes, edges))
    }

    pub fn record_constitution_acknowledgment(&self, agent: &str, digest: &str, context_type: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO constitution_acknowledgments (agent, digest, context_type, acknowledged_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![agent, digest, context_type, now],
        )?;
        Ok(())
    }

    pub fn get_recent_constitution_acknowledgments(&self, limit: usize) -> Result<Vec<(String, String, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT agent, digest, context_type, acknowledged_at FROM constitution_acknowledgments ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_memory_reuse(&self, memory_id: &str, task_id: &str, context: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO memory_reuses (memory_id, task_id, context, reused_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![memory_id, task_id, context, now],
        )?;
        Ok(())
    }

    pub fn list_memory_reuses(&self, limit: usize) -> Result<Vec<(String, String, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT memory_id, task_id, context, reused_at FROM memory_reuses ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_curriculum_evaluation(&self, cycle_id: i64, step: &str, status: &str, details: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO curriculum_evaluations (cycle_id, step, status, details, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![cycle_id, step, status, details, now],
        )?;
        Ok(())
    }

    pub fn list_recent_curriculum_evaluations(&self, limit: usize) -> Result<Vec<(i64, String, String, Option<String>, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT cycle_id, step, status, details, created_at FROM curriculum_evaluations ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_recent_tool_executions(&self, limit: usize) -> Result<Vec<ToolExecutionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, tool_name, tool_version, tool_hash, input_json, seed, dependencies_json,
                    output_json, error, exit_code, duration_ms, artifacts_json, measured_results_json,
                    specialist_id, immutable_evidence_hash, executed_at
             FROM tool_executions ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok(ToolExecutionRecord {
                id: r.get(0)?,
                tool_name: r.get(1)?,
                tool_version: r.get(2)?,
                tool_hash: r.get::<_, String>(3).unwrap_or_default(),
                input_json: r.get(4)?,
                seed: r.get(5).unwrap_or(None),
                dependencies_json: r.get::<_, String>(6).unwrap_or_else(|_| "[]".to_string()),
                output_json: r.get(7)?,
                error: r.get(8)?,
                exit_code: r.get(9)?,
                duration_ms: r.get(10)?,
                artifacts_json: r.get::<_, String>(11).unwrap_or_else(|_| "[]".to_string()),
                measured_results_json: r.get::<_, String>(12).unwrap_or_else(|_| "{}".to_string()),
                specialist_id: r.get(13).unwrap_or(None),
                immutable_evidence_hash: r.get::<_, String>(14).unwrap_or_default(),
                executed_at: r.get(15)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_sources(&self, limit: usize) -> Result<Vec<RawSource>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, url, title, content, revision, retrieved_at FROM sources ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok(RawSource {
                id: r.get(0)?,
                url: r.get(1)?,
                title: r.get(2)?,
                content: r.get(3)?,
                revision: r.get(4)?,
                retrieved_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn upsert_research_backlog(&self, topic: &str, discipline: &str, priority: i64, hypothesis: Option<&str>, status: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO research_backlog (topic, discipline, priority, hypothesis, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(topic) DO UPDATE SET
               discipline=excluded.discipline,
               priority=excluded.priority,
               hypothesis=COALESCE(excluded.hypothesis, research_backlog.hypothesis),
               status=excluded.status,
               updated_at=excluded.updated_at",
            params![topic, discipline, priority, hypothesis, status, now, now],
        )?;
        Ok(())
    }

    pub fn list_research_backlog(&self, limit: usize) -> Result<Vec<ResearchBacklogItem>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, topic, discipline, priority, hypothesis, status, created_at, updated_at
             FROM research_backlog ORDER BY priority DESC, id ASC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok(ResearchBacklogItem {
                id: r.get(0)?,
                topic: r.get(1)?,
                discipline: r.get(2)?,
                priority: r.get(3)?,
                hypothesis: r.get(4)?,
                status: r.get(5)?,
                created_at: r.get(6)?,
                updated_at: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_research_experiment(
        &self,
        cycle_id: i64,
        hypothesis: &str,
        conditions: &str,
        attempted_method: &str,
        observed_outcome: &str,
        lessons_learned: &str,
        is_failure: bool,
        justification_to_retry: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let is_fail_int = if is_failure { 1 } else { 0 };
        let outcome_status = if is_failure { "failed" } else { "completed" };
        let data_kind = if conditions.contains("real_data") || attempted_method.contains("real_data") {
            "real_data"
        } else {
            "synthetic_data"
        };
        conn.execute(
            "INSERT INTO research_experiments (
                cycle_id, domain, data_kind, spec_id, tool_name, execution_id, hypothesis, conditions,
                attempted_method, baseline_comparison_json, held_out_evaluation_json,
                measured_results_json, source_fidelity_status, under_specified,
                missing_source_details_json, observed_outcome, outcome_status,
                lessons_learned, epistemic_note, is_failure, justification_to_retry, created_at
             )
             VALUES (?1, 'frontier_ai_research_and_verification', ?2, NULL, NULL, NULL, ?3, ?4, ?5, '{}', '{}', '{}', 'unspecified', 0, '[]', ?6, ?7, ?8, 'Numerical agreement supports an experiment; it is not a proof.', ?9, ?10, ?11)",
            params![
                cycle_id,
                data_kind,
                hypothesis,
                conditions,
                attempted_method,
                observed_outcome,
                outcome_status,
                lessons_learned,
                is_fail_int,
                justification_to_retry,
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn record_structured_research_experiment(
        &self,
        cycle_id: i64,
        spec_id: Option<&str>,
        tool_name: Option<&str>,
        execution_id: Option<i64>,
        hypothesis: &str,
        conditions: &str,
        attempted_method: &str,
        baseline_comparison_json: &str,
        held_out_evaluation_json: &str,
        measured_results_json: &str,
        source_fidelity_status: &str,
        under_specified: bool,
        missing_source_details: &[String],
        observed_outcome: &str,
        outcome_status: &str,
        lessons_learned: &str,
        is_failure: bool,
        justification_to_retry: Option<&str>,
    ) -> Result<i64> {
        let (exp_domain, exp_data_kind) = if let Some(sid) = spec_id {
            if let Ok(Some(sp)) = self.get_experiment_spec(sid) {
                let dk = if sp.data_kind == "real_data" || sp.data_kind == "synthetic_data" {
                    sp.data_kind
                } else if (source_fidelity_status == "full_text_verified" && !under_specified)
                    || matches!(
                        sp.executable_check_kind.as_str(),
                        "data_reanalysis" | "real_implementation" | "statistical_replication"
                    )
                {
                    "real_data".to_string()
                } else {
                    "synthetic_data".to_string()
                };
                (sp.domain, dk)
            } else {
                (
                    default_spec_domain(),
                    if source_fidelity_status == "full_text_verified" && !under_specified {
                        "real_data".to_string()
                    } else {
                        "synthetic_data".to_string()
                    },
                )
            }
        } else {
            (
                default_spec_domain(),
                if source_fidelity_status == "full_text_verified" && !under_specified {
                    "real_data".to_string()
                } else {
                    "synthetic_data".to_string()
                },
            )
        };

        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let is_fail_int = if is_failure { 1 } else { 0 };
        let under_spec_int = if under_specified { 1 } else { 0 };
        let missing_json =
            serde_json::to_string(missing_source_details).unwrap_or_else(|_| "[]".to_string());
        let epistemic_note =
            "Numerical agreement supports an experiment; it is not a proof.";
        conn.execute(
            "INSERT INTO research_experiments (
                cycle_id, domain, data_kind, spec_id, tool_name, execution_id, hypothesis, conditions,
                attempted_method, baseline_comparison_json, held_out_evaluation_json,
                measured_results_json, source_fidelity_status, under_specified,
                missing_source_details_json, observed_outcome, outcome_status,
                lessons_learned, epistemic_note, is_failure, justification_to_retry, created_at
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22)",
            params![
                cycle_id,
                exp_domain,
                exp_data_kind,
                spec_id,
                tool_name,
                execution_id,
                hypothesis,
                conditions,
                attempted_method,
                baseline_comparison_json,
                held_out_evaluation_json,
                measured_results_json,
                source_fidelity_status,
                under_spec_int,
                missing_json,
                observed_outcome,
                outcome_status,
                lessons_learned,
                epistemic_note,
                is_fail_int,
                justification_to_retry,
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_research_experiments(&self, limit: usize) -> Result<Vec<ResearchExperimentRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, cycle_id, spec_id, tool_name, execution_id, hypothesis, conditions,
                    attempted_method, COALESCE(baseline_comparison_json, '{}'),
                    COALESCE(held_out_evaluation_json, '{}'),
                    COALESCE(measured_results_json, '{}'),
                    COALESCE(source_fidelity_status, 'unspecified'),
                    COALESCE(under_specified, 0),
                    COALESCE(missing_source_details_json, '[]'),
                    observed_outcome, COALESCE(outcome_status, 'completed'),
                    lessons_learned,
                    COALESCE(epistemic_note, 'Numerical agreement supports an experiment; it is not a proof.'),
                    is_failure, justification_to_retry, created_at,
                    COALESCE(domain, 'frontier_ai_research_and_verification'),
                    COALESCE(data_kind, 'synthetic_data')
             FROM research_experiments ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit], |r| {
            let under_spec_int: i64 = r.get(12)?;
            let is_fail_int: i64 = r.get(18)?;
            Ok(ResearchExperimentRecord {
                id: r.get(0)?,
                cycle_id: r.get(1)?,
                domain: r.get(21)?,
                data_kind: r.get(22)?,
                spec_id: r.get(2)?,
                tool_name: r.get(3)?,
                execution_id: r.get(4)?,
                hypothesis: r.get(5)?,
                conditions: r.get(6)?,
                attempted_method: r.get(7)?,
                baseline_comparison_json: r.get(8)?,
                held_out_evaluation_json: r.get(9)?,
                measured_results_json: r.get(10)?,
                source_fidelity_status: r.get(11)?,
                under_specified: under_spec_int != 0,
                missing_source_details_json: r.get(13)?,
                observed_outcome: r.get(14)?,
                outcome_status: r.get(15)?,
                lessons_learned: r.get(16)?,
                epistemic_note: r.get(17)?,
                is_failure: is_fail_int != 0,
                justification_to_retry: r.get(19)?,
                created_at: r.get(20)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_research_experiments_in_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ResearchExperimentRecord>> {
        let all = self.list_research_experiments(limit.max(200))?;
        Ok(all
            .into_iter()
            .filter(|e| e.domain == domain)
            .take(limit)
            .collect())
    }

    /// Validates an `ExperimentSpecification`'s source citations against stored source documents
    /// and passages. Experiments may only cite methods, algorithms, or equations from documents
    /// with full-text extraction (`ingestion_completeness == "full_text"` and `!low_fidelity`).
    /// If any cited source is `abstract_only` or `low_fidelity`, the experiment is flagged
    /// `under_specified = true`, `source_fidelity_status` is set accordingly, and if the caller
    /// did not specify `missing_source_details` then validation rejects the incomplete under-specified
    /// specification or records the missing details and marks the source `reingestion_requested = true`.
    pub fn register_and_validate_experiment_spec(
        &self,
        spec: &mut ExperimentSpecification,
    ) -> Result<ExperimentSpecification> {
        let now = Utc::now().to_rfc3339();
        if spec.id.trim().is_empty() {
            let mut hasher = Sha256::new();
            hasher.update(
                format!(
                    "{}::{}::{}::{}",
                    spec.name.trim(),
                    spec.research_question.trim(),
                    spec.falsifiable_hypothesis.trim(),
                    spec.mechanism_under_test.trim()
                )
                .as_bytes(),
            );
            let hex: String = hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect();
            spec.id = format!("expspec-{}", &hex[..16]);
        }
        if spec.created_at.trim().is_empty() {
            spec.created_at = now.clone();
        }
        spec.updated_at = now.clone();

        // Check cited source passages and canonical source IDs for full-text fidelity
        let mut cited_canonical_ids: Vec<String> = spec
            .cited_source_ids
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        {
            let conn = self.conn.lock().unwrap();
            for pid in &spec.cited_passage_ids {
                if let Ok(sec_cid) = conn.query_row(
                    "SELECT canonical_source_id FROM source_sections WHERE id = ?1",
                    [*pid],
                    |r| r.get::<_, String>(0),
                ) {
                    if !cited_canonical_ids.contains(&sec_cid) {
                        cited_canonical_ids.push(sec_cid);
                    }
                }
            }
        }

        let mut any_abstract_only = false;
        let mut any_low_fidelity = false;
        let mut any_missing_doc = false;
        let mut detected_missing_details: Vec<String> = Vec::new();

        for cid in &cited_canonical_ids {
            if let Some(src) = self.get_tracked_source(cid)? {
                let completeness = src.ingestion_completeness.trim();
                let is_low_fid = src.low_fidelity || completeness == "low_fidelity";

                if completeness != "full_text" {
                    any_abstract_only = true;
                    detected_missing_details.push(format!(
                        "Source '{}' ({}) has ingestion_completeness='{}' (only abstract/metadata available; full-text methods, algorithmic steps, equations, and hyperparameters are missing)",
                        cid, src.canonical_url, completeness
                    ));
                    // Request re-ingestion or higher-fidelity source automatically
                    let _ = self.request_source_reingestion(
                        cid,
                        "Experiment specification cited abstract-only source; full-text methods/equations required",
                    );
                } else if is_low_fid {
                    any_low_fidelity = true;
                    detected_missing_details.push(format!(
                        "Source '{}' ({}) has low_fidelity=true extraction quality (equations, tables, or section structure incomplete)",
                        cid, src.canonical_url
                    ));
                    let _ = self.request_source_reingestion(
                        cid,
                        "Experiment specification cited low-fidelity source; higher-fidelity extraction required",
                    );
                }
            } else {
                any_missing_doc = true;
                detected_missing_details.push(format!(
                    "Source '{}' is not ingested in tracked_sources (full-text methods, algorithms, and equations unavailable)",
                    cid
                ));
            }
        }

        if any_abstract_only || any_low_fidelity || any_missing_doc {
            spec.under_specified = true;
            spec.reingestion_requested = true;
            spec.source_fidelity_status = if any_abstract_only {
                "under_specified_abstract_only".to_string()
            } else if any_low_fidelity {
                "under_specified_low_fidelity".to_string()
            } else {
                "under_specified_untracked_source".to_string()
            };
            for d in detected_missing_details {
                if !spec.missing_source_details.contains(&d) {
                    spec.missing_source_details.push(d);
                }
            }
        } else if !cited_canonical_ids.is_empty() {
            spec.under_specified = false;
            spec.source_fidelity_status = "full_text_verified".to_string();
        } else if spec.source_fidelity_status.trim().is_empty() {
            spec.source_fidelity_status = "self_contained_computational".to_string();
        }

        // Resolve and tag experiment data_kind ("real_data" or "synthetic_data")
        let trimmed_dk = spec.data_kind.trim().to_lowercase();
        if trimmed_dk == "real_data" || trimmed_dk == "synthetic_data" {
            spec.data_kind = trimmed_dk;
        } else if (!cited_canonical_ids.is_empty() && !spec.under_specified)
            || matches!(
                spec.executable_check_kind.as_str(),
                "data_reanalysis" | "real_implementation" | "statistical_replication"
            )
        {
            spec.data_kind = "real_data".to_string();
        } else {
            spec.data_kind = "synthetic_data".to_string();
        }

        let conn = self.conn.lock().unwrap();
        let assumptions_json = serde_json::to_string(&spec.assumptions).unwrap_or_else(|_| "[]".to_string());
        let controls_json = serde_json::to_string(&spec.controls).unwrap_or_else(|_| "[]".to_string());
        let held_out_json = serde_json::to_string(&spec.held_out_cases).unwrap_or_else(|_| "[]".to_string());
        let metrics_json = serde_json::to_string(&spec.metrics).unwrap_or_else(|_| "[]".to_string());
        let seeds_json = serde_json::to_string(&spec.seeds).unwrap_or_else(|_| "[]".to_string());
        let expected_json = serde_json::to_string(&spec.expected_outputs).unwrap_or_else(|_| "[]".to_string());
        let limits_json = serde_json::to_string(&spec.known_limitations).unwrap_or_else(|_| "[]".to_string());
        let cited_srcs_json = serde_json::to_string(&cited_canonical_ids).unwrap_or_else(|_| "[]".to_string());
        let cited_passages_json = serde_json::to_string(&spec.cited_passage_ids).unwrap_or_else(|_| "[]".to_string());
        let missing_json = serde_json::to_string(&spec.missing_source_details).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT INTO experiment_specifications (
                id, name, domain, executable_check_kind, research_question,
                falsifiable_hypothesis, mechanism_under_test, inputs_description,
                assumptions_json, baseline_description, controls_json, held_out_cases_json,
                metrics_json, budget_max_runtime_secs, budget_max_memory_mb, seeds_json,
                reproducibility_requirements, expected_outputs_json, known_limitations_json,
                cited_source_ids_json, cited_passage_ids_json, source_fidelity_status,
                under_specified, missing_source_details_json, reingestion_requested,
                builder_agent, sceptic_agent, sceptic_review_status, sceptic_review_notes,
                created_at, updated_at, data_kind
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32
             )
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                domain = excluded.domain,
                executable_check_kind = excluded.executable_check_kind,
                data_kind = excluded.data_kind,
                research_question = excluded.research_question,
                falsifiable_hypothesis = excluded.falsifiable_hypothesis,
                mechanism_under_test = excluded.mechanism_under_test,
                inputs_description = excluded.inputs_description,
                assumptions_json = excluded.assumptions_json,
                baseline_description = excluded.baseline_description,
                controls_json = excluded.controls_json,
                held_out_cases_json = excluded.held_out_cases_json,
                metrics_json = excluded.metrics_json,
                budget_max_runtime_secs = excluded.budget_max_runtime_secs,
                budget_max_memory_mb = excluded.budget_max_memory_mb,
                seeds_json = excluded.seeds_json,
                reproducibility_requirements = excluded.reproducibility_requirements,
                expected_outputs_json = excluded.expected_outputs_json,
                known_limitations_json = excluded.known_limitations_json,
                cited_source_ids_json = excluded.cited_source_ids_json,
                cited_passage_ids_json = excluded.cited_passage_ids_json,
                source_fidelity_status = excluded.source_fidelity_status,
                under_specified = excluded.under_specified,
                missing_source_details_json = excluded.missing_source_details_json,
                reingestion_requested = excluded.reingestion_requested,
                builder_agent = excluded.builder_agent,
                sceptic_agent = excluded.sceptic_agent,
                sceptic_review_status = excluded.sceptic_review_status,
                sceptic_review_notes = excluded.sceptic_review_notes,
                updated_at = excluded.updated_at",
            params![
                spec.id,
                spec.name,
                spec.domain,
                spec.executable_check_kind,
                spec.research_question,
                spec.falsifiable_hypothesis,
                spec.mechanism_under_test,
                spec.inputs_description,
                assumptions_json,
                spec.baseline_description,
                controls_json,
                held_out_json,
                metrics_json,
                spec.budget_max_runtime_secs as i64,
                spec.budget_max_memory_mb as i64,
                seeds_json,
                spec.reproducibility_requirements,
                expected_json,
                limits_json,
                cited_srcs_json,
                cited_passages_json,
                spec.source_fidelity_status,
                if spec.under_specified { 1 } else { 0 },
                missing_json,
                if spec.reingestion_requested { 1 } else { 0 },
                spec.builder_agent,
                spec.sceptic_agent,
                spec.sceptic_review_status,
                spec.sceptic_review_notes,
                spec.created_at,
                spec.updated_at,
                spec.data_kind,
            ],
        )?;

        Ok(spec.clone())
    }

    pub fn request_source_reingestion(&self, canonical_id: &str, reason: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE tracked_sources
             SET research_status = 'pending',
                 retry_eligible = 1,
                 revisit_reason = ?1,
                 updated_at = ?2
             WHERE canonical_id = ?3",
            params![reason, now, canonical_id],
        )?;
        Ok(())
    }

    pub fn update_experiment_spec_review(
        &self,
        spec_id: &str,
        status: &str,
        notes: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE experiment_specifications
             SET sceptic_review_status = ?1, sceptic_review_notes = ?2, updated_at = ?3
             WHERE id = ?4",
            params![status, notes, now, spec_id],
        )?;
        Ok(())
    }

    fn map_experiment_spec_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ExperimentSpecification> {
        let assumptions_str: String = r.get(8)?;
        let controls_str: String = r.get(10)?;
        let held_out_str: String = r.get(11)?;
        let metrics_str: String = r.get(12)?;
        let seeds_str: String = r.get(15)?;
        let expected_str: String = r.get(17)?;
        let limits_str: String = r.get(18)?;
        let cited_srcs_str: String = r.get(19)?;
        let cited_passages_str: String = r.get(20)?;
        let missing_str: String = r.get(23)?;

        Ok(ExperimentSpecification {
            id: r.get(0)?,
            name: r.get(1)?,
            domain: r.get(2)?,
            executable_check_kind: r.get(3)?,
            data_kind: r
                .get::<_, String>(31)
                .unwrap_or_else(|_| "synthetic_data".to_string()),
            research_question: r.get(4)?,
            falsifiable_hypothesis: r.get(5)?,
            mechanism_under_test: r.get(6)?,
            inputs_description: r.get(7)?,
            assumptions: serde_json::from_str(&assumptions_str).unwrap_or_default(),
            baseline_description: r.get(9)?,
            controls: serde_json::from_str(&controls_str).unwrap_or_default(),
            held_out_cases: serde_json::from_str(&held_out_str).unwrap_or_default(),
            metrics: serde_json::from_str(&metrics_str).unwrap_or_default(),
            budget_max_runtime_secs: r.get::<_, i64>(13)?.max(1) as u64,
            budget_max_memory_mb: r.get::<_, i64>(14)?.max(16) as usize,
            seeds: serde_json::from_str(&seeds_str).unwrap_or_default(),
            reproducibility_requirements: r.get(16)?,
            expected_outputs: serde_json::from_str(&expected_str).unwrap_or_default(),
            known_limitations: serde_json::from_str(&limits_str).unwrap_or_default(),
            cited_source_ids: serde_json::from_str(&cited_srcs_str).unwrap_or_default(),
            cited_passage_ids: serde_json::from_str(&cited_passages_str).unwrap_or_default(),
            source_fidelity_status: r.get(21)?,
            under_specified: r.get::<_, i64>(22)? != 0,
            missing_source_details: serde_json::from_str(&missing_str).unwrap_or_default(),
            reingestion_requested: r.get::<_, i64>(24)? != 0,
            builder_agent: r.get(25)?,
            sceptic_agent: r.get(26)?,
            sceptic_review_status: r.get(27)?,
            sceptic_review_notes: r.get(28)?,
            created_at: r.get(29)?,
            updated_at: r.get(30)?,
        })
    }

    pub fn get_experiment_spec(&self, id: &str) -> Result<Option<ExperimentSpecification>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, domain, executable_check_kind, research_question,
                    falsifiable_hypothesis, mechanism_under_test, inputs_description,
                    assumptions_json, baseline_description, controls_json, held_out_cases_json,
                    metrics_json, budget_max_runtime_secs, budget_max_memory_mb, seeds_json,
                    reproducibility_requirements, expected_outputs_json, known_limitations_json,
                    cited_source_ids_json, cited_passage_ids_json, source_fidelity_status,
                    under_specified, missing_source_details_json, reingestion_requested,
                    builder_agent, sceptic_agent, sceptic_review_status, sceptic_review_notes,
                    created_at, updated_at, COALESCE(data_kind, 'synthetic_data')
             FROM experiment_specifications WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Self::map_experiment_spec_row(r)?))
        } else {
            Ok(None)
        }
    }

    pub fn get_experiment_spec_by_name(&self, name: &str) -> Result<Option<ExperimentSpecification>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, domain, executable_check_kind, research_question,
                    falsifiable_hypothesis, mechanism_under_test, inputs_description,
                    assumptions_json, baseline_description, controls_json, held_out_cases_json,
                    metrics_json, budget_max_runtime_secs, budget_max_memory_mb, seeds_json,
                    reproducibility_requirements, expected_outputs_json, known_limitations_json,
                    cited_source_ids_json, cited_passage_ids_json, source_fidelity_status,
                    under_specified, missing_source_details_json, reingestion_requested,
                    builder_agent, sceptic_agent, sceptic_review_status, sceptic_review_notes,
                    created_at, updated_at, COALESCE(data_kind, 'synthetic_data')
             FROM experiment_specifications WHERE name = ?1 ORDER BY updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([name])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Self::map_experiment_spec_row(r)?))
        } else {
            Ok(None)
        }
    }

    pub fn list_experiment_specs(&self, limit: usize) -> Result<Vec<ExperimentSpecification>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, domain, executable_check_kind, research_question,
                    falsifiable_hypothesis, mechanism_under_test, inputs_description,
                    assumptions_json, baseline_description, controls_json, held_out_cases_json,
                    metrics_json, budget_max_runtime_secs, budget_max_memory_mb, seeds_json,
                    reproducibility_requirements, expected_outputs_json, known_limitations_json,
                    cited_source_ids_json, cited_passage_ids_json, source_fidelity_status,
                    under_specified, missing_source_details_json, reingestion_requested,
                    builder_agent, sceptic_agent, sceptic_review_status, sceptic_review_notes,
                    created_at, updated_at, COALESCE(data_kind, 'synthetic_data')
             FROM experiment_specifications ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_experiment_spec_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn upsert_capability_map(&self, name: &str, status: &str, description: &str, unlocks_research: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO capability_map (capability_name, status, description, unlocks_research, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(capability_name) DO UPDATE SET
               status=excluded.status,
               description=excluded.description,
               unlocks_research=COALESCE(excluded.unlocks_research, capability_map.unlocks_research),
               updated_at=excluded.updated_at",
            params![name, status, description, unlocks_research, now],
        )?;
        Ok(())
    }

    pub fn list_capability_map(&self) -> Result<Vec<CapabilityMapItem>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, capability_name, status, description, unlocks_research, updated_at
             FROM capability_map ORDER BY id ASC"
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(CapabilityMapItem {
                id: r.get(0)?,
                capability_name: r.get(1)?,
                status: r.get(2)?,
                description: r.get(3)?,
                unlocks_research: r.get(4)?,
                updated_at: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // --- EXTENSIONS & COORDINATOR CHANGE TASKS ---

    pub fn register_extension(
        &self,
        name: &str,
        hook: &str,
        version: i64,
        protocol_version: &str,
        description: &str,
        executable_path: &str,
        executable_hash: &str,
        config_json: &str,
        state: &str,
        endorser_1: Option<&str>,
        endorser_2: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO extensions (name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, endorser_1, endorser_2, failure_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 0, ?12, ?12)",
            params![name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, endorser_1, endorser_2, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_extension_by_id(&self, id: i64) -> Result<Option<crate::extensions::ExtensionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, evaluation_results_json, endorser_1, endorser_2, failure_count, created_at, updated_at
             FROM extensions WHERE id=?1"
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(crate::extensions::ExtensionRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                hook: r.get(2)?,
                version: r.get(3)?,
                protocol_version: r.get(4)?,
                description: r.get(5)?,
                executable_path: r.get(6)?,
                executable_hash: r.get(7)?,
                config_json: r.get(8)?,
                state: r.get(9)?,
                evaluation_results_json: r.get(10)?,
                endorser_1: r.get(11)?,
                endorser_2: r.get(12)?,
                failure_count: r.get(13)?,
                created_at: r.get(14)?,
                updated_at: r.get(15)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_extension_by_name(&self, name: &str) -> Result<Option<crate::extensions::ExtensionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, evaluation_results_json, endorser_1, endorser_2, failure_count, created_at, updated_at
             FROM extensions WHERE name=?1 ORDER BY version DESC LIMIT 1"
        )?;
        let mut rows = stmt.query(params![name])?;
        if let Some(r) = rows.next()? {
            Ok(Some(crate::extensions::ExtensionRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                hook: r.get(2)?,
                version: r.get(3)?,
                protocol_version: r.get(4)?,
                description: r.get(5)?,
                executable_path: r.get(6)?,
                executable_hash: r.get(7)?,
                config_json: r.get(8)?,
                state: r.get(9)?,
                evaluation_results_json: r.get(10)?,
                endorser_1: r.get(11)?,
                endorser_2: r.get(12)?,
                failure_count: r.get(13)?,
                created_at: r.get(14)?,
                updated_at: r.get(15)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_extensions(&self) -> Result<Vec<crate::extensions::ExtensionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, evaluation_results_json, endorser_1, endorser_2, failure_count, created_at, updated_at
             FROM extensions ORDER BY hook ASC, version DESC"
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(crate::extensions::ExtensionRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                hook: r.get(2)?,
                version: r.get(3)?,
                protocol_version: r.get(4)?,
                description: r.get(5)?,
                executable_path: r.get(6)?,
                executable_hash: r.get(7)?,
                config_json: r.get(8)?,
                state: r.get(9)?,
                evaluation_results_json: r.get(10)?,
                endorser_1: r.get(11)?,
                endorser_2: r.get(12)?,
                failure_count: r.get(13)?,
                created_at: r.get(14)?,
                updated_at: r.get(15)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_active_extensions(&self) -> Result<Vec<crate::extensions::ExtensionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, hook, version, protocol_version, description, executable_path, executable_hash, config_json, state, evaluation_results_json, endorser_1, endorser_2, failure_count, created_at, updated_at
             FROM extensions WHERE state = 'active' ORDER BY updated_at DESC"
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(crate::extensions::ExtensionRecord {
                id: r.get(0)?,
                name: r.get(1)?,
                hook: r.get(2)?,
                version: r.get(3)?,
                protocol_version: r.get(4)?,
                description: r.get(5)?,
                executable_path: r.get(6)?,
                executable_hash: r.get(7)?,
                config_json: r.get(8)?,
                state: r.get(9)?,
                evaluation_results_json: r.get(10)?,
                endorser_1: r.get(11)?,
                endorser_2: r.get(12)?,
                failure_count: r.get(13)?,
                created_at: r.get(14)?,
                updated_at: r.get(15)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn update_extension_endorsements(&self, id: i64, e1: Option<&str>, e2: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET endorser_1=?1, endorser_2=?2, updated_at=?3 WHERE id=?4",
            params![e1, e2, now, id],
        )?;
        Ok(())
    }

    pub fn update_extension_evaluation(&self, id: i64, eval_json: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET evaluation_results_json=?1, updated_at=?2 WHERE id=?3",
            params![eval_json, now, id],
        )?;
        Ok(())
    }

    pub fn set_extension_state(&self, id: i64, state: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET state=?1, updated_at=?2 WHERE id=?3",
            params![state, now, id],
        )?;
        Ok(())
    }

    pub fn activate_extension(&self, id: i64, _hook: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET state = 'evaluated', updated_at=?1 WHERE name = (SELECT name FROM extensions WHERE id=?2) AND id != ?2 AND state = 'active'",
            params![now, id],
        )?;
        conn.execute(
            "UPDATE extensions SET state = 'active', failure_count=0, updated_at=?1 WHERE id=?2",
            params![now, id],
        )?;
        Ok(())
    }

    pub fn increment_extension_failure(&self, id: i64, _error: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET failure_count = failure_count + 1, updated_at=?1 WHERE id=?2",
            params![now, id],
        )?;
        Ok(())
    }

    pub fn get_extension_failures(&self, id: i64) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT failure_count FROM extensions WHERE id=?1")?;
        let count = stmt.query_row(params![id], |r| r.get(0)).unwrap_or(0);
        Ok(count)
    }

    pub fn create_coordinator_change_task(
        &self,
        title: &str,
        rationale: &str,
        proposed_interface_json: &str,
        experiment_evidence_json: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO coordinator_change_tasks (title, rationale, proposed_interface_json, experiment_evidence_json, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 'proposed', ?5, ?5)",
            params![title, rationale, proposed_interface_json, experiment_evidence_json, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_coordinator_change_tasks(&self) -> Result<Vec<(i64, String, String, String, String, String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, title, rationale, proposed_interface_json, experiment_evidence_json, status, created_at
             FROM coordinator_change_tasks ORDER BY id DESC"
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // --- Specialists & Concurrency Limits ---

    pub fn get_specialist_concurrency_limit(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let limit_opt: Option<usize> = conn.query_row(
            "SELECT limit_val FROM specialist_limit_history ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        ).optional()?;
        Ok(limit_opt.unwrap_or(6))
    }

    pub fn get_specialist_limit_info(&self) -> Result<(usize, Option<String>, Option<String>)> {
        let conn = self.conn.lock().unwrap();
        let current_limit: usize = conn.query_row(
            "SELECT limit_val FROM specialist_limit_history ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        ).optional()?.unwrap_or(6);

        let last_increase: Option<String> = conn.query_row(
            "SELECT changed_at FROM specialist_limit_history WHERE change_type = 'increase' ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        ).optional()?;

        let last_reduction: Option<String> = conn.query_row(
            "SELECT changed_at FROM specialist_limit_history WHERE change_type = 'user_reduction' ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        ).optional()?;

        Ok((current_limit, last_increase, last_reduction))
    }

    pub fn count_running_specialists(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM specialists WHERE status = 'running'",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    pub fn count_queued_specialists(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM specialists WHERE status = 'queued'",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    pub fn count_pending_limit_proposals(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM specialist_limit_proposals WHERE status = 'pending'",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    fn map_specialist_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<crate::specialists::SpecialistRecord> {
        let exec_ids_str: String = r
            .get::<_, String>(15)
            .unwrap_or_else(|_| "[]".to_string());
        let execution_ids: Vec<i64> = serde_json::from_str(&exec_ids_str).unwrap_or_default();
        Ok(crate::specialists::SpecialistRecord {
            id: r.get(0)?,
            name: r.get(1)?,
            role: r.get(2)?,
            task: r.get(3)?,
            completion_criteria: r.get(4)?,
            max_tokens: r.get::<_, i64>(5)? as usize,
            deadline: r.get(6)?,
            endorser_1: r.get(7)?,
            endorser_2: r.get(8)?,
            status: r.get(9)?,
            result: r.get(10)?,
            created_at: r.get(11)?,
            updated_at: r.get(12)?,
            phase: r
                .get::<_, String>(13)
                .unwrap_or_else(|_| "phase1_investigate_and_execute".to_string()),
            plan_json: r.get(14).unwrap_or(None),
            execution_ids,
            pending_commit_id: r.get(16).unwrap_or(None),
        })
    }

    pub fn register_specialist(&self, spec: &crate::specialists::SpecialistRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let phase = if spec.phase.trim().is_empty() {
            "phase1_investigate_and_execute"
        } else {
            spec.phase.as_str()
        };
        let exec_ids_json = serde_json::to_string(&spec.execution_ids).unwrap_or_else(|_| "[]".to_string());
        conn.execute(
            "INSERT INTO specialists (
                id, name, role, task, completion_criteria, max_tokens, deadline,
                endorser_1, endorser_2, status, result, created_at, updated_at,
                phase, plan_json, execution_ids_json, pending_commit_id
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                spec.id,
                spec.name,
                spec.role,
                spec.task,
                spec.completion_criteria,
                spec.max_tokens as i64,
                spec.deadline,
                spec.endorser_1,
                spec.endorser_2,
                spec.status,
                spec.result,
                spec.created_at,
                spec.updated_at,
                phase,
                spec.plan_json,
                exec_ids_json,
                spec.pending_commit_id
            ],
        )?;
        Ok(())
    }

    pub fn update_specialist_status(&self, id: &str, status: &str, result: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let inferred_phase = match status {
            "queued" | "running" => None,
            "completed" => Some("completed"),
            "failed" => Some("failed"),
            "cancelled" => Some("cancelled"),
            "checkpointed" => Some("checkpointed"),
            "budget_exhausted" => Some("budget_exhausted"),
            _ => None,
        };
        if let Some(ph) = inferred_phase {
            conn.execute(
                "UPDATE specialists SET status = ?1, result = ?2, phase = CASE WHEN phase = 'timed_out' THEN 'timed_out' ELSE ?3 END, updated_at = ?4 WHERE id = ?5",
                params![status, result, ph, now, id],
            )?;
        } else {
            conn.execute(
                "UPDATE specialists SET status = ?1, result = ?2, updated_at = ?3 WHERE id = ?4",
                params![status, result, now, id],
            )?;
        }
        Ok(())
    }

    pub fn update_specialist_phase(
        &self,
        id: &str,
        phase: &str,
        status: Option<&str>,
        result: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        if let Some(st) = status {
            conn.execute(
                "UPDATE specialists SET phase = ?1, status = ?2, result = COALESCE(?3, result), updated_at = ?4 WHERE id = ?5",
                params![phase, st, result, now, id],
            )?;
        } else {
            conn.execute(
                "UPDATE specialists SET phase = ?1, result = COALESCE(?2, result), updated_at = ?3 WHERE id = ?4",
                params![phase, result, now, id],
            )?;
        }
        Ok(())
    }

    pub fn get_specialist(&self, id: &str) -> Result<Option<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at,
                    phase, plan_json, execution_ids_json, pending_commit_id
             FROM specialists WHERE id = ?1",
        )?;
        let mut rows = stmt.query([id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Self::map_specialist_row(r)?))
        } else {
            Ok(None)
        }
    }

    pub fn list_running_specialists_newest_first(&self) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at,
                    phase, plan_json, execution_ids_json, pending_commit_id
             FROM specialists WHERE status = 'running' ORDER BY created_at DESC"
        )?;
        let rows = stmt.query_map([], Self::map_specialist_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_queued_specialists(&self, limit: usize) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at,
                    phase, plan_json, execution_ids_json, pending_commit_id
             FROM specialists WHERE status = 'queued' ORDER BY created_at ASC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_specialist_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_specialists(&self, limit: usize) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at,
                    phase, plan_json, execution_ids_json, pending_commit_id
             FROM specialists ORDER BY created_at DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_specialist_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // =========================================================================
    // ENFORCED SPECIALIST STATE MACHINE (PHASE 1 -> PHASE 2 VALIDATE & COMMIT)
    // =========================================================================

    /// Detects whether a statement is merely a plan, hypothesis proposal, or unexecuted prediction.
    /// Plans and predictions stay task records; they never become findings (`claims`).
    pub fn is_plan_or_prediction_only(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return true;
        }
        let lower = trimmed.to_lowercase();
        let plan_prefixes = [
            "i plan to ",
            "we plan to ",
            "i will ",
            "we will ",
            "plan:",
            "planned experiment:",
            "prediction:",
            "predicted outcome:",
            "hypothesis to test:",
            "next step:",
            "proposed plan:",
            "investigation plan:",
            "phase 1 plan:",
            "i predict that ",
            "we predict that ",
            "i propose to run ",
            "we propose to run ",
            "to investigate this, i will",
            "to test this hypothesis, we will",
        ];
        if plan_prefixes.iter().any(|p| lower.starts_with(p)) {
            return true;
        }
        // Check if text contains planning/prediction markers without any completed observation/evidence
        let has_plan_marker = lower.contains("i plan to ")
            || lower.contains("we plan to ")
            || lower.contains("planned experiment:")
            || lower.contains("predicted outcome:")
            || lower.contains("we will design an experiment")
            || lower.contains("i will design a bounded experiment")
            || lower.contains("will be executed in the next");
        let has_completed_marker = lower.contains("measured ")
            || lower.contains("observed ")
            || lower.contains("verified ")
            || lower.contains("executed ")
            || lower.contains("exit_code")
            || lower.contains("section ")
            || lower.contains("theorem ")
            || lower.contains("equation ");
        has_plan_marker && !has_completed_marker
    }

    /// Phase 1 (`phase1_investigate_and_execute`): records a specialist's hypothesis, plan, and
    /// predictions strictly as a task record (`tasks` table + `specialists.plan_json`).
    /// Never creates or promotes a finding (`claims`).
    pub fn record_specialist_phase1_plan(
        &self,
        specialist_id: &str,
        hypothesis: &str,
        planned_steps: &[String],
        predictions: &[String],
    ) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let task_record_id = format!("plan-{}", specialist_id);
        let plan_obj = serde_json::json!({
            "specialist_id": specialist_id,
            "phase": "phase1_investigate_and_execute",
            "hypothesis": hypothesis,
            "planned_steps": planned_steps,
            "predictions": predictions,
            "recorded_at": now,
            "epistemic_status": "task_plan_only_never_finding",
        });
        let plan_json = serde_json::to_string(&plan_obj)?;
        let desc = format!(
            "[Specialist Phase 1 Plan/Prediction ({})] Hypothesis: {} | Steps: {:?} | Predictions: {:?}",
            specialist_id, hypothesis, planned_steps, predictions
        );
        conn.execute(
            "INSERT INTO tasks (id, kind, description, status, result, created_at, completed_at)
             VALUES (?1, 'specialist_plan', ?2, 'completed', ?3, ?4, ?4)
             ON CONFLICT(id) DO UPDATE SET description=excluded.description, result=excluded.result, completed_at=excluded.completed_at",
            params![task_record_id, desc, plan_json, now],
        )?;
        conn.execute(
            "UPDATE specialists SET phase = 'phase1_investigate_and_execute', plan_json = ?1, updated_at = ?2 WHERE id = ?3",
            params![plan_json, now, specialist_id],
        )?;
        Ok(task_record_id)
    }

    /// Stages a pending commit in `specialist_pending_commits` when entering Phase 2 (`phase2_validate_and_commit`).
    pub fn stage_specialist_pending_commit(
        &self,
        proposal: &SpecialistPhase2Proposal,
    ) -> Result<SpecialistPendingCommit> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let spec_name: String = conn
            .query_row(
                "SELECT name FROM specialists WHERE id = ?1",
                [&proposal.specialist_id],
                |r| r.get(0),
            )
            .unwrap_or_else(|_| proposal.specialist_id.clone());

        let mut idemp_hasher = Sha256::new();
        idemp_hasher.update(
            format!(
                "{}::{}::{}::{:?}::{:?}::{:?}",
                proposal.specialist_id,
                proposal.finding_kind,
                proposal.conclusion.trim(),
                proposal.execution_id,
                proposal.source_passage_ids,
                proposal.canonical_source_ids
            )
            .as_bytes(),
        );
        let idemp_key: String = idemp_hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let commit_id = format!("spc-{}", &idemp_key[..16]);

        let claimed_meas_json = serde_json::to_string(&proposal.claimed_measurements)
            .unwrap_or_else(|_| "{}".to_string());
        let passages_json = serde_json::to_string(&proposal.source_passage_ids)
            .unwrap_or_else(|_| "[]".to_string());
        let sources_json = serde_json::to_string(&proposal.canonical_source_ids)
            .unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT INTO specialist_pending_commits (
                id, specialist_id, specialist_name, phase, finding_kind,
                conclusion, assumptions, negative_results, is_negative_result,
                execution_id, claimed_measurements_json, reproducibility_artifacts_json,
                source_passage_ids_json, canonical_source_ids_json, independent_source_count,
                source_fidelity_summary, validation_status, rejection_reason,
                committed_claim_id, commit_idempotency_key, created_at, updated_at
             ) VALUES (?1, ?2, ?3, 'phase2_validate_and_commit', ?4, ?5, ?6, ?7, ?8, ?9, ?10, '{}', ?11, ?12, 0, '', 'pending', NULL, NULL, ?13, ?14, ?14)
             ON CONFLICT(commit_idempotency_key) DO NOTHING",
            params![
                commit_id,
                proposal.specialist_id,
                spec_name,
                proposal.finding_kind,
                proposal.conclusion.trim(),
                proposal.assumptions.trim(),
                proposal.negative_results.trim(),
                if proposal.is_negative_result { 1 } else { 0 },
                proposal.execution_id,
                claimed_meas_json,
                passages_json,
                sources_json,
                idemp_key,
                now
            ],
        )?;

        conn.execute(
            "UPDATE specialists SET phase = 'phase2_validate_and_commit', pending_commit_id = ?1, updated_at = ?2 WHERE id = ?3",
            params![commit_id, now, proposal.specialist_id],
        )?;

        Self::get_pending_commit_by_key_conn(&conn, &idemp_key)?
            .context("Failed to load staged specialist pending commit")
    }

    fn get_pending_commit_by_key_conn(
        conn: &Connection,
        idemp_key: &str,
    ) -> Result<Option<SpecialistPendingCommit>> {
        let mut stmt = conn.prepare(
            "SELECT id, specialist_id, specialist_name, phase, finding_kind,
                    conclusion, assumptions, negative_results, is_negative_result,
                    execution_id, claimed_measurements_json, reproducibility_artifacts_json,
                    source_passage_ids_json, canonical_source_ids_json, independent_source_count,
                    source_fidelity_summary, validation_status, rejection_reason,
                    committed_claim_id, commit_idempotency_key, created_at, updated_at
             FROM specialist_pending_commits WHERE commit_idempotency_key = ?1",
        )?;
        let mut rows = stmt.query([idemp_key])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Self::map_pending_commit_row(r)?))
        } else {
            Ok(None)
        }
    }

    fn map_pending_commit_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<SpecialistPendingCommit> {
        let p_str: String = r.get(12)?;
        let c_str: String = r.get(13)?;
        Ok(SpecialistPendingCommit {
            id: r.get(0)?,
            specialist_id: r.get(1)?,
            specialist_name: r.get(2)?,
            phase: r.get(3)?,
            finding_kind: r.get(4)?,
            conclusion: r.get(5)?,
            assumptions: r.get(6)?,
            negative_results: r.get(7)?,
            is_negative_result: r.get::<_, i64>(8)? != 0,
            execution_id: r.get(9)?,
            claimed_measurements_json: r.get(10)?,
            reproducibility_artifacts_json: r.get(11)?,
            source_passage_ids: serde_json::from_str(&p_str).unwrap_or_default(),
            canonical_source_ids: serde_json::from_str(&c_str).unwrap_or_default(),
            independent_source_count: r.get::<_, i64>(14)?.max(0) as usize,
            source_fidelity_summary: r.get(15)?,
            validation_status: r.get(16)?,
            rejection_reason: r.get(17)?,
            committed_claim_id: r.get(18)?,
            commit_idempotency_key: r.get(19)?,
            created_at: r.get(20)?,
            updated_at: r.get(21)?,
        })
    }

    pub fn list_specialist_pending_commits(
        &self,
        limit: usize,
    ) -> Result<Vec<SpecialistPendingCommit>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, specialist_id, specialist_name, phase, finding_kind,
                    conclusion, assumptions, negative_results, is_negative_result,
                    execution_id, claimed_measurements_json, reproducibility_artifacts_json,
                    source_passage_ids_json, canonical_source_ids_json, independent_source_count,
                    source_fidelity_summary, validation_status, rejection_reason,
                    committed_claim_id, commit_idempotency_key, created_at, updated_at
             FROM specialist_pending_commits ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_pending_commit_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn reject_pending_commits_for_specialist(
        &self,
        specialist_id: &str,
        reason: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE specialist_pending_commits
             SET phase = 'rejected', validation_status = 'rejected', rejection_reason = ?1, updated_at = ?2
             WHERE specialist_id = ?3 AND validation_status = 'pending'",
            params![reason, now, specialist_id],
        )?;
        Ok(())
    }

    /// Validates that any measurements claimed by the specialist match the host's immutable
    /// structured results file (`exec.measured_results_json`).
    /// Numbers that appear only in prose or `stdout` (`exec.output_json`) are NOT validated
    /// and cannot support a claim. If an `ExperimentSpecification` is attached to the tool,
    /// every declared metric `result_key` must exist in `measured_results_json` and claimed keys
    /// must reference the specification's declared metric keys.
    fn verify_claimed_measurements_against_host_output(
        claimed: &serde_json::Value,
        conclusion: &str,
        exec: &ToolExecutionRecord,
        exp_spec: Option<&ExperimentSpecification>,
        is_negative_result: bool,
    ) -> std::result::Result<(), String> {
        let meas_out = exec.measured_results_json.trim();
        let meas_val: serde_json::Value = serde_json::from_str(meas_out).map_err(|e| {
            format!(
                "Structured results file (measured_results_json) for execution #{} is missing or malformed JSON ({}): '{}'",
                exec.id, e, meas_out
            )
        })?;
        let Some(meas_map) = meas_val.as_object() else {
            return Err(format!(
                "Structured results file (measured_results_json) for execution #{} must be a JSON object, got: {}",
                exec.id, meas_out
            ));
        };

        if !is_negative_result && meas_map.is_empty() {
            return Err(format!(
                "Execution #{} has an empty structured results file (measured_results_json = '{{}}'); numbers in stdout or prose are not validated and cannot support a claim",
                exec.id
            ));
        }

        // If the tool has an ExperimentSpecification, verify all declared metric result_keys exist in measured_results_json
        if let Some(spec) = exp_spec {
            for m in &spec.metrics {
                let rk = m.result_key.trim();
                if !rk.is_empty() && !meas_map.contains_key(rk) {
                    return Err(format!(
                        "Structured results file for execution #{} lacks declared metric key '{}' required by experiment specification '{}'",
                        exec.id, rk, spec.id
                    ));
                }
            }
        }

        let Some(claimed_obj) = claimed.as_object() else {
            return Err("claimed_measurements must be a JSON object".to_string());
        };

        if !is_negative_result && claimed_obj.is_empty() {
            return Err(
                "Empirical claim has empty claimed_measurements; numbers that appear only in prose or stdout are not validated and cannot support a claim".to_string(),
            );
        }

        for (k, v) in claimed_obj {
            if let Some(spec) = exp_spec {
                let is_declared = spec.metrics.iter().any(|m| m.result_key.trim() == k.as_str())
                    || matches!(
                        k.as_str(),
                        "exit_code" | "seed" | "baseline_value" | "held_out_generalization_gap"
                    );
                if !is_declared {
                    return Err(format!(
                        "Claimed measurement key '{}' is not a declared metric result_key in experiment specification '{}'",
                        k, spec.id
                    ));
                }
            }

            let Some(host_val) = meas_map.get(k) else {
                let in_stdout = exec
                    .output_json
                    .as_deref()
                    .map(|s| s.contains(k))
                    .unwrap_or(false);
                if in_stdout {
                    return Err(format!(
                        "Measurement '{}' appears only in stdout (output_json) and is missing from structured results file (measured_results_json) in execution #{}; stdout-only numbers cannot support a claim",
                        k, exec.id
                    ));
                }
                return Err(format!(
                    "Invented or unvalidated measurement '{}': key not present in host execution #{} structured results file ({})",
                    k, exec.id, meas_out
                ));
            };

            if let (Some(cf), Some(hf)) = (v.as_f64(), host_val.as_f64()) {
                if (cf - hf).abs() > 1e-4 {
                    return Err(format!(
                        "Fabricated measurement for '{}': specialist claimed {} but host execution #{} structured results recorded {}",
                        k, cf, exec.id, hf
                    ));
                }
                continue;
            }
            if v != host_val {
                return Err(format!(
                    "Fabricated measurement for '{}': specialist claimed {} but host execution #{} structured results recorded {}",
                    k, v, exec.id, host_val
                ));
            }
        }

        // Check if agent conclusion text explicitly states `key=value` for any key in meas_map
        // with a different value, OR states `key=number` for a metric key that is only in prose/stdout!
        let lower_conc = conclusion.to_lowercase();
        for (hk, hv) in meas_map {
            let hk_low = hk.to_lowercase();
            if let Some(host_num) = hv.as_f64() {
                for sep in ["=", ":", " is ", " was "] {
                    let pat = format!("{}{}", hk_low, sep);
                    if let Some(idx) = lower_conc.find(&pat) {
                        let after = lower_conc[idx + pat.len()..].trim_start();
                        let num_prefix: String = after
                            .chars()
                            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                            .collect();
                        if !num_prefix.is_empty() {
                            if let Ok(claimed_num) = num_prefix.parse::<f64>() {
                                if (claimed_num - host_num).abs() > 1e-3 {
                                    return Err(format!(
                                        "Agent text invented measurement for '{}': text claimed {} but host execution #{} structured results recorded {}",
                                        hk, claimed_num, exec.id, host_num
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also detect `metric_name=123.45` patterns in conclusion prose where `metric_name` is NOT in `meas_map`
        let ignored_prose_keys = [
            "seed", "budget", "step", "exec_id", "id", "b", "t", "n", "k", "exit_code", "version",
        ];
        for token in conclusion.split_whitespace() {
            let clean = token.trim_matches(|c: char| {
                matches!(c, ',' | '.' | ';' | '(' | ')' | '[' | ']' | '"' | '\'')
            });
            if let Some((lhs, rhs)) = clean.split_once('=') {
                let key_trim = lhs.trim();
                let key_low = key_trim.to_lowercase();
                if !key_low.is_empty()
                    && key_low.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                    && !ignored_prose_keys.contains(&key_low.as_str())
                {
                    if rhs.parse::<f64>().is_ok() {
                        let exists_in_meas = meas_map
                            .keys()
                            .any(|mk| mk.eq_ignore_ascii_case(key_trim));
                        if !exists_in_meas {
                            return Err(format!(
                                "Prose-only or stdout-only number '{}={}' in conclusion is not present in structured results file (measured_results_json) and cannot support a claim",
                                key_trim, rhs
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Host-enforced Phase 2 (`phase2_validate_and_commit`):
    /// - Rejects plans and predictions (they stay task records and never become findings).
    /// - Validates every empirical (`locally_reproduced`) claim against an immutable host
    ///   `ToolExecutionRecord` and its outputs; rejects missing/fabricated run references or
    ///   invented measurements.
    /// - Validates every literature (`source_supported`) finding against stored source passages,
    ///   recording the independent source count and extraction fidelity (`full-text`,
    ///   `low-fidelity`, `abstract-only`) of each source, keeping them distinct from locally
    ///   reproduced results.
    /// - Applies the two-agent trust rule (`endorser_1`, `endorser_2`) ONLY after host validation
    ///   succeeds. Agreement never fabricates evidence.
    pub fn validate_and_commit_specialist_finding(
        &self,
        proposal: &SpecialistPhase2Proposal,
    ) -> Result<SpecialistCommitOutcome> {
        // Stage pending commit first (idempotent)
        let staged = self.stage_specialist_pending_commit(proposal)?;
        if staged.validation_status == "committed" {
            let trusted = staged
                .committed_claim_id
                .as_deref()
                .and_then(|cid| self.get_claim(cid).ok().flatten())
                .map(|c| c.status == "trusted")
                .unwrap_or(false);
            return Ok(SpecialistCommitOutcome {
                commit_id: staged.id,
                validation_status: "committed".to_string(),
                rejection_reason: None,
                committed_claim_id: staged.committed_claim_id,
                trusted,
                finding_kind: staged.finding_kind,
                independent_source_count: staged.independent_source_count,
                source_fidelity_summary: staged.source_fidelity_summary,
                deduplicated_existing: true,
            });
        }

        let spec = self
            .get_specialist(&proposal.specialist_id)?
            .context("Specialist not found for Phase 2 commit")?;

        // Helper closure to reject a staged commit and record the rejection reason
        let reject_commit = |reason: &str| -> Result<SpecialistCommitOutcome> {
            let conn = self.conn.lock().unwrap();
            let now = Utc::now().to_rfc3339();
            conn.execute(
                "UPDATE specialist_pending_commits
                 SET phase = 'rejected', validation_status = 'rejected', rejection_reason = ?1, updated_at = ?2
                 WHERE id = ?3",
                params![reason, now, staged.id],
            )?;
            Ok(SpecialistCommitOutcome {
                commit_id: staged.id.clone(),
                validation_status: "rejected".to_string(),
                rejection_reason: Some(reason.to_string()),
                committed_claim_id: None,
                trusted: false,
                finding_kind: proposal.finding_kind.clone(),
                independent_source_count: 0,
                source_fidelity_summary: String::new(),
                deduplicated_existing: false,
            })
        };

        // 1. Check specialist lifecycle state (cancelled / timed_out specialists cannot commit findings)
        if spec.status == "cancelled" || spec.phase == "cancelled" || spec.phase == "timed_out" {
            return reject_commit(&format!(
                "Specialist '{}' is in state status='{}', phase='{}'; cancelled or timed-out runs cannot commit findings",
                spec.id, spec.status, spec.phase
            ));
        }

        // 2. Enforce Phase 1 vs Phase 2 boundary: plans and predictions stay task records; they never become findings!
        if Self::is_plan_or_prediction_only(&proposal.conclusion) {
            let _ = self.record_specialist_phase1_plan(
                &spec.id,
                &proposal.conclusion,
                &[proposal.conclusion.clone()],
                &[],
            );
            return reject_commit(
                "Plans and predictions stay task records; they never become findings",
            );
        }

        let effective_domain = if proposal.domain.trim().is_empty() {
            "frontier_ai_research_and_verification"
        } else {
            proposal.domain.trim()
        };

        match proposal.finding_kind.as_str() {
            "locally_reproduced" => {
                // Every empirical claim MUST reference a completed execution record in `tool_executions`
                let Some(exec_id) = proposal.execution_id else {
                    return reject_commit(
                        "Empirical (locally_reproduced) claim missing completed execution record reference (execution_id is None)",
                    );
                };
                let Some(exec) = self.get_tool_execution(exec_id)? else {
                    return reject_commit(&format!(
                        "Fabricated execution reference: execution_id={} does not exist in host tool_executions",
                        exec_id
                    ));
                };

                // Verify immutable host evidence hash
                if !exec.immutable_evidence_hash.is_empty() {
                    let expected_hash = Self::compute_immutable_execution_hash(
                        &exec.tool_name,
                        exec.tool_version,
                        &exec.tool_hash,
                        &exec.input_json,
                        exec.seed,
                        &exec.dependencies_json,
                        exec.output_json.as_deref(),
                        exec.error.as_deref(),
                        exec.exit_code,
                        exec.duration_ms,
                        &exec.artifacts_json,
                        &exec.measured_results_json,
                    );
                    if expected_hash != exec.immutable_evidence_hash {
                        return reject_commit(&format!(
                            "Immutable execution evidence hash mismatch for execution_id={}",
                            exec_id
                        ));
                    }
                }

                let tool_exp_spec: Option<ExperimentSpecification> = self
                    .get_tool_by_name(&exec.tool_name)?
                    .and_then(|t| t.experiment_spec_id)
                    .and_then(|sid| self.get_experiment_spec(&sid).ok().flatten())
                    .or_else(|| {
                        self.get_experiment_spec_by_name(&exec.tool_name)
                            .ok()
                            .flatten()
                    });

                let run_failed = exec.exit_code != 0 || exec.error.is_some();
                if run_failed && !proposal.is_negative_result {
                    return reject_commit(&format!(
                        "Execution record #{} failed (exit_code={}, error={:?}); failed runs cannot be committed as positive empirical findings",
                        exec.id, exec.exit_code, exec.error
                    ));
                }

                if !run_failed {
                    if let Err(meas_err) = Self::verify_claimed_measurements_against_host_output(
                        &proposal.claimed_measurements,
                        &proposal.conclusion,
                        &exec,
                        tool_exp_spec.as_ref(),
                        proposal.is_negative_result,
                    ) {
                        return reject_commit(&meas_err);
                    }
                } else {
                    // For negative results from a failed run, ensure no fabricated positive measurements are claimed
                    if let Some(obj) = proposal.claimed_measurements.as_object() {
                        if !obj.is_empty() {
                            if let Err(meas_err) =
                                Self::verify_claimed_measurements_against_host_output(
                                    &proposal.claimed_measurements,
                                    &proposal.conclusion,
                                    &exec,
                                    tool_exp_spec.as_ref(),
                                    true,
                                )
                            {
                                return reject_commit(&meas_err);
                            }
                        }
                    }
                }

                let repro_artifacts = serde_json::json!({
                    "execution_id": exec.id,
                    "tool_name": exec.tool_name,
                    "tool_version": exec.tool_version,
                    "tool_hash": exec.tool_hash,
                    "input": exec.input_json,
                    "seed": exec.seed,
                    "dependencies": serde_json::from_str::<serde_json::Value>(&exec.dependencies_json).unwrap_or(serde_json::json!([])),
                    "exit_code": exec.exit_code,
                    "duration_ms": exec.duration_ms,
                    "artifacts": serde_json::from_str::<serde_json::Value>(&exec.artifacts_json).unwrap_or(serde_json::json!([])),
                    "measured_results": serde_json::from_str::<serde_json::Value>(&exec.measured_results_json).unwrap_or(serde_json::json!({})),
                    "error": exec.error,
                    "immutable_evidence_hash": exec.immutable_evidence_hash,
                });
                let repro_json = serde_json::to_string(&repro_artifacts)?;

                let formatted_statement = if proposal.is_negative_result {
                    format!(
                        "[locally-reproduced | negative-result | exec_id={} | tool={}@{} (hash={}) | exit_code={}]: {} (Assumptions: {} | Negative Result: {})",
                        exec.id,
                        exec.tool_name,
                        exec.tool_version,
                        &exec.tool_hash[..exec.tool_hash.len().min(12)],
                        exec.exit_code,
                        proposal.conclusion.trim(),
                        proposal.assumptions.trim(),
                        proposal.negative_results.trim()
                    )
                } else {
                    format!(
                        "[locally-reproduced | verified | exec_id={} | tool={}@{} (hash={}) | exit_code=0]: {} (Assumptions: {})",
                        exec.id,
                        exec.tool_name,
                        exec.tool_version,
                        &exec.tool_hash[..exec.tool_hash.len().min(12)],
                        proposal.conclusion.trim(),
                        proposal.assumptions.trim()
                    )
                };

                let source_url = format!("triad:execution:{}", exec.id);
                let claim_id = self.propose_claim_in_domain(
                    &formatted_statement,
                    &source_url,
                    &spec.endorser_1,
                    effective_domain,
                )?;

                let exp_status = if proposal.is_negative_result {
                    "failed"
                } else {
                    "verified"
                };
                {
                    let conn = self.conn.lock().unwrap();
                    let now = Utc::now().to_rfc3339();
                    conn.execute(
                        "UPDATE claims SET
                            finding_kind = 'locally_reproduced',
                            source_support_status = 'unsupported',
                            source_corroboration_count = 0,
                            experimental_verification_status = ?1,
                            verification_evidence = ?2,
                            execution_id = ?3,
                            assumptions_text = ?4,
                            negative_results_text = ?5,
                            reproducibility_artifacts_json = ?2,
                            updated_at = ?6
                         WHERE id = ?7",
                        params![
                            exp_status,
                            repro_json,
                            exec.id,
                            proposal.assumptions.trim(),
                            proposal.negative_results.trim(),
                            now,
                            claim_id
                        ],
                    )?;
                }

                // Apply the two-agent trust rule AFTER host validation succeeds
                let _ = self.record_vote(
                    &claim_id,
                    &spec.endorser_1,
                    "endorse",
                    proposal.cycle_id.max(1),
                    Some("Host-validated execution evidence & reproducibility artifacts"),
                )?;
                let trusted = self.record_vote(
                    &claim_id,
                    &spec.endorser_2,
                    "endorse",
                    proposal.cycle_id.max(1),
                    Some("Second permanent agent endorsement after Phase 2 host validation"),
                )?;

                {
                    let conn = self.conn.lock().unwrap();
                    let now = Utc::now().to_rfc3339();
                    conn.execute(
                        "UPDATE specialist_pending_commits SET
                            phase = 'committed',
                            reproducibility_artifacts_json = ?1,
                            validation_status = 'committed',
                            committed_claim_id = ?2,
                            updated_at = ?3
                         WHERE id = ?4",
                        params![repro_json, claim_id, now, staged.id],
                    )?;
                }

                Ok(SpecialistCommitOutcome {
                    commit_id: staged.id,
                    validation_status: "committed".to_string(),
                    rejection_reason: None,
                    committed_claim_id: Some(claim_id),
                    trusted,
                    finding_kind: "locally_reproduced".to_string(),
                    independent_source_count: 0,
                    source_fidelity_summary: "locally_reproduced (host execution evidence)".to_string(),
                    deduplicated_existing: false,
                })
            }

            "source_supported" => {
                // Literature findings MUST reference real stored source passages / sources
                if proposal.source_passage_ids.is_empty() && proposal.canonical_source_ids.is_empty() {
                    return reject_commit(
                        "Literature (source_supported) finding must reference at least one stored source passage or canonical source",
                    );
                }

                let mut resolved_source_ids: Vec<String> = Vec::new();
                let mut passage_locations: Vec<String> = Vec::new();

                {
                    let conn = self.conn.lock().unwrap();
                    for pid in &proposal.source_passage_ids {
                        let row: Option<(String, String, String)> = conn
                            .query_row(
                                "SELECT canonical_source_id, section_path, source_location FROM source_sections WHERE id = ?1",
                                [pid],
                                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                            )
                            .optional()?;
                        let Some((cid, sec_path, loc)) = row else {
                            drop(conn);
                            return reject_commit(&format!(
                                "Fabricated source passage reference: passage_id={} does not exist in source_sections",
                                pid
                            ));
                        };
                        if !resolved_source_ids.contains(&cid) {
                            resolved_source_ids.push(cid);
                        }
                        passage_locations.push(format!("{}#{}", loc, sec_path));
                    }
                }

                for raw_cid in &proposal.canonical_source_ids {
                    if let Some(tracked) = self.get_tracked_source(raw_cid)? {
                        if !resolved_source_ids.contains(&tracked.canonical_id) {
                            resolved_source_ids.push(tracked.canonical_id);
                        }
                    } else {
                        return reject_commit(&format!(
                            "Fabricated source reference: '{}' does not exist in tracked_sources",
                            raw_cid
                        ));
                    }
                }

                // Inspect each independent canonical source and compute its extraction fidelity:
                // "full-text", "low-fidelity", or "abstract-only"
                let mut full_text_count = 0usize;
                let mut low_fidelity_count = 0usize;
                let mut abstract_only_count = 0usize;
                let mut per_source_labels = Vec::new();
                let mut primary_url = String::new();
                let mut valid_independent_sources = Vec::new();

                for cid in &resolved_source_ids {
                    let Some(src) = self.get_tracked_source(cid)? else {
                        return reject_commit(&format!(
                            "Tracked source '{}' missing during fidelity verification",
                            cid
                        ));
                    };
                    if src.is_orientation_only
                        || src.ingestion_completeness == "orientation_only"
                        || src.canonical_id.starts_with("wikipedia:")
                    {
                        return reject_commit(&format!(
                            "Orientation-only source '{}' cannot be used as primary evidence for a literature finding",
                            src.canonical_id
                        ));
                    }
                    if primary_url.is_empty() {
                        primary_url = src.canonical_url.clone();
                    }
                    let fidelity_label = if src.ingestion_completeness == "full_text" {
                        if src.low_fidelity {
                            low_fidelity_count += 1;
                            "low-fidelity"
                        } else {
                            full_text_count += 1;
                            "full-text"
                        }
                    } else {
                        abstract_only_count += 1;
                        "abstract-only"
                    };
                    per_source_labels.push(format!("{}={}", src.canonical_id, fidelity_label));
                    valid_independent_sources.push(src);
                }

                if valid_independent_sources.is_empty() {
                    return reject_commit(
                        "No valid non-orientation sources found for source_supported literature finding",
                    );
                }

                let independent_count = valid_independent_sources.len();
                let fidelity_summary = format!(
                    "full-text: {}, low-fidelity: {}, abstract-only: {} [{}]",
                    full_text_count,
                    low_fidelity_count,
                    abstract_only_count,
                    per_source_labels.join(", ")
                );
                let passages_tag = if passage_locations.is_empty() {
                    resolved_source_ids.join(", ")
                } else {
                    passage_locations.join("; ")
                };

                let formatted_statement = format!(
                    "[source-supported | independent_sources={} | fidelity: {} | passages: {}]: {} (Assumptions: {})",
                    independent_count,
                    fidelity_summary,
                    passages_tag,
                    proposal.conclusion.trim(),
                    proposal.assumptions.trim()
                );

                let claim_id = self.propose_claim_in_domain(
                    &formatted_statement,
                    &primary_url,
                    &spec.endorser_1,
                    effective_domain,
                )?;

                for src in &valid_independent_sources {
                    let _ = self.record_source_corroboration(
                        &claim_id,
                        &src.canonical_url,
                        Some(&src.title),
                        src.doi.as_deref(),
                        src.arxiv_id.as_deref(),
                        Some(&src.version),
                        None,
                        src.observed_connectors.first().map(|s| s.as_str()),
                        effective_domain,
                    );
                }

                let support_status = if independent_count >= 2 {
                    "corroborated"
                } else {
                    "single_source"
                };

                {
                    let conn = self.conn.lock().unwrap();
                    let now = Utc::now().to_rfc3339();
                    conn.execute(
                        "UPDATE claims SET
                            finding_kind = 'source_supported',
                            source_support_status = ?1,
                            source_corroboration_count = ?2,
                            source_fidelity_summary = ?3,
                            experimental_verification_status = 'unverified',
                            execution_id = NULL,
                            assumptions_text = ?4,
                            negative_results_text = ?5,
                            updated_at = ?6
                         WHERE id = ?7",
                        params![
                            support_status,
                            independent_count as i64,
                            fidelity_summary,
                            proposal.assumptions.trim(),
                            proposal.negative_results.trim(),
                            now,
                            claim_id
                        ],
                    )?;
                }

                // Apply the two-agent trust rule AFTER host validation succeeds
                let _ = self.record_vote(
                    &claim_id,
                    &spec.endorser_1,
                    "endorse",
                    proposal.cycle_id.max(1),
                    Some("Host-validated source-supported literature finding"),
                )?;
                let trusted = self.record_vote(
                    &claim_id,
                    &spec.endorser_2,
                    "endorse",
                    proposal.cycle_id.max(1),
                    Some("Second permanent agent endorsement after source passage & fidelity verification"),
                )?;

                {
                    let conn = self.conn.lock().unwrap();
                    let now = Utc::now().to_rfc3339();
                    let cids_json = serde_json::to_string(&resolved_source_ids)
                        .unwrap_or_else(|_| "[]".to_string());
                    conn.execute(
                        "UPDATE specialist_pending_commits SET
                            phase = 'committed',
                            canonical_source_ids_json = ?1,
                            independent_source_count = ?2,
                            source_fidelity_summary = ?3,
                            validation_status = 'committed',
                            committed_claim_id = ?4,
                            updated_at = ?5
                         WHERE id = ?6",
                        params![
                            cids_json,
                            independent_count as i64,
                            fidelity_summary,
                            claim_id,
                            now,
                            staged.id
                        ],
                    )?;
                }

                Ok(SpecialistCommitOutcome {
                    commit_id: staged.id,
                    validation_status: "committed".to_string(),
                    rejection_reason: None,
                    committed_claim_id: Some(claim_id),
                    trusted,
                    finding_kind: "source_supported".to_string(),
                    independent_source_count: independent_count,
                    source_fidelity_summary: fidelity_summary,
                    deduplicated_existing: false,
                })
            }

            other => reject_commit(&format!(
                "Invalid finding_kind '{}': must be 'locally_reproduced' or 'source_supported'",
                other
            )),
        }
    }

    pub fn record_user_specialist_reduction(&self, new_limit: usize) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO specialist_limit_history (limit_val, reason, change_type, changed_at)
             VALUES (?1, 'User CLI limit reduction', 'user_reduction', ?2)",
            params![new_limit as i64, now],
        )?;
        Ok(())
    }

    pub fn reject_all_pending_limit_proposals(&self, _reason: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE specialist_limit_proposals SET status = 'rejected' WHERE status = 'pending'",
            [],
        )?;
        Ok(())
    }

    pub fn record_limit_increase_proposal(
        &self,
        current_limit: usize,
        requested_limit: usize,
        expected_benefit: &str,
        available_resources: &str,
        config_revision: &str,
        explorer: bool,
        sceptic: bool,
        builder: bool,
        status: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO specialist_limit_proposals (current_limit, requested_limit, expected_benefit, available_resources, config_revision, votes_explorer, votes_sceptic, votes_builder, status, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                current_limit as i64, requested_limit as i64, expected_benefit, available_resources,
                config_revision, if explorer { 1 } else { 0 }, if sceptic { 1 } else { 0 },
                if builder { 1 } else { 0 }, status, now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn apply_specialist_limit_increase(&self, new_limit: usize) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO specialist_limit_history (limit_val, reason, change_type, changed_at)
             VALUES (?1, 'Unanimous permanent agent vote approved increase', 'increase', ?2)",
            params![new_limit as i64, now],
        )?;
        Ok(())
    }

    pub fn get_node_degree(&self, node_id: &str) -> usize {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM semantic_edges WHERE source_id = ?1 OR target_id = ?1",
            [node_id],
            |r| r.get::<_, usize>(0),
        ).unwrap_or(0)
    }

    pub fn upsert_research_work(&self, work: &crate::web::ResolvedResearchWork) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let versions_json = serde_json::to_string(&work.versions).unwrap_or_else(|_| "[]".to_string());
        let authors_json = serde_json::to_string(&work.authors).unwrap_or_else(|_| "[]".to_string());
        let corr_json = serde_json::to_string(&work.corroborating_connectors).unwrap_or_else(|_| "[]".to_string());
        let urls_json = serde_json::to_string(&work.urls).unwrap_or_else(|_| "[]".to_string());
        let code_json = serde_json::to_string(&work.code_urls).unwrap_or_else(|_| "[]".to_string());
        let cit_json = serde_json::to_string(&work.citations).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT INTO research_works (
                canonical_id, title, normalized_title, doi, arxiv_id, versions_json, published_date,
                authors_json, primary_connector, corroborating_connectors_json, urls_json, code_urls_json,
                abstract_excerpt, experimental_details, limitations_excerpt, citations_json,
                availability_status, duplicate_copies_merged, retrieved_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)
            ON CONFLICT(canonical_id) DO UPDATE SET
                doi = COALESCE(excluded.doi, research_works.doi),
                arxiv_id = COALESCE(excluded.arxiv_id, research_works.arxiv_id),
                versions_json = excluded.versions_json,
                corroborating_connectors_json = excluded.corroborating_connectors_json,
                urls_json = excluded.urls_json,
                code_urls_json = excluded.code_urls_json,
                abstract_excerpt = CASE WHEN length(excluded.abstract_excerpt) > length(research_works.abstract_excerpt) THEN excluded.abstract_excerpt ELSE research_works.abstract_excerpt END,
                experimental_details = CASE WHEN length(excluded.experimental_details) > length(research_works.experimental_details) THEN excluded.experimental_details ELSE research_works.experimental_details END,
                limitations_excerpt = CASE WHEN length(excluded.limitations_excerpt) > length(research_works.limitations_excerpt) THEN excluded.limitations_excerpt ELSE research_works.limitations_excerpt END,
                duplicate_copies_merged = excluded.duplicate_copies_merged,
                retrieved_at = excluded.retrieved_at",
            params![
                work.canonical_id,
                work.title,
                work.normalized_title,
                work.doi,
                work.arxiv_id,
                versions_json,
                work.published_date,
                authors_json,
                work.primary_connector,
                corr_json,
                urls_json,
                code_json,
                work.abstract_excerpt,
                work.experimental_details,
                work.limitations_excerpt,
                cit_json,
                work.availability_status,
                work.duplicate_copies_merged as i64,
                work.retrieved_at,
            ],
        )?;
        Ok(())
    }

    pub fn list_research_works(&self, limit: usize) -> Result<Vec<crate::web::ResolvedResearchWork>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT canonical_id, title, normalized_title, doi, arxiv_id, versions_json, published_date,
                    authors_json, primary_connector, corroborating_connectors_json, urls_json, code_urls_json,
                    abstract_excerpt, experimental_details, limitations_excerpt, citations_json,
                    availability_status, duplicate_copies_merged, retrieved_at
             FROM research_works ORDER BY retrieved_at DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            let v_str: String = r.get(5)?;
            let a_str: String = r.get(7)?;
            let c_str: String = r.get(9)?;
            let u_str: String = r.get(10)?;
            let cu_str: String = r.get(11)?;
            let cit_str: String = r.get(15)?;
            let urls: Vec<String> = serde_json::from_str(&u_str).unwrap_or_default();
            let primary_url = urls.first().cloned().unwrap_or_default();
            let title: String = r.get(1)?;
            let doi: Option<String> = r.get(3)?;
            let arxiv_id: Option<String> = r.get(4)?;
            let versions: Vec<String> = serde_json::from_str(&v_str).unwrap_or_default();
            let primary_connector: String = r.get(8)?;
            let abstract_excerpt: String = r.get(12)?;
            let identity = crate::web::canonicalize_source_identity(
                &primary_url,
                Some(&title),
                doi.as_deref(),
                arxiv_id.as_deref(),
                versions.last().map(|s| s.as_str()),
                Some(&abstract_excerpt),
                Some(&primary_connector),
            );
            Ok(crate::web::ResolvedResearchWork {
                canonical_id: r.get(0)?,
                title,
                normalized_title: r.get(2)?,
                provider_id: identity.provider_id,
                doi,
                arxiv_id,
                versions,
                published_date: r.get(6)?,
                authors: serde_json::from_str(&a_str).unwrap_or_default(),
                primary_connector,
                corroborating_connectors: serde_json::from_str(&c_str).unwrap_or_default(),
                canonical_url: identity.canonical_url,
                urls,
                code_urls: serde_json::from_str(&cu_str).unwrap_or_default(),
                abstract_excerpt,
                experimental_details: r.get(13)?,
                limitations_excerpt: r.get(14)?,
                citations: serde_json::from_str(&cit_str).unwrap_or_default(),
                availability_status: r.get(16)?,
                content_hash: identity.content_hash,
                source_kind: identity.source_kind,
                duplicate_copies_merged: r.get::<_, i64>(17)? as usize,
                independent_corroboration_count: 1,
                retrieved_at: r.get(18)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_research_synthesis(
        &self,
        question: &str,
        sub_queries: &[String],
        canonical_works: &[String],
        competing_findings: &str,
        limitations_summary: &str,
        verification_checks: &[String],
        experiment_translation: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let sq_json = serde_json::to_string(sub_queries).unwrap_or_else(|_| "[]".to_string());
        let cw_json = serde_json::to_string(canonical_works).unwrap_or_else(|_| "[]".to_string());
        let vc_json = serde_json::to_string(verification_checks).unwrap_or_else(|_| "[]".to_string());

        conn.execute(
            "INSERT INTO research_syntheses (
                question, sub_queries_json, canonical_works_json, competing_findings,
                limitations_summary, verification_checks_json, experiment_translation, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                question,
                sq_json,
                cw_json,
                competing_findings,
                limitations_summary,
                vc_json,
                experiment_translation,
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_research_syntheses(&self, limit: usize) -> Result<Vec<ResearchSynthesisRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, question, sub_queries_json, canonical_works_json, competing_findings,
                    limitations_summary, verification_checks_json, experiment_translation, created_at
             FROM research_syntheses ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(ResearchSynthesisRecord {
                id: r.get(0)?,
                question: r.get(1)?,
                sub_queries_json: r.get(2)?,
                canonical_works_json: r.get(3)?,
                competing_findings: r.get(4)?,
                limitations_summary: r.get(5)?,
                verification_checks_json: r.get(6)?,
                experiment_translation: r.get(7)?,
                created_at: r.get(8)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn propose_claim_in_domain(
        &self,
        statement: &str,
        source_url: &str,
        proposer: &str,
        domain: &str,
    ) -> Result<String> {
        let claim_id = compute_claim_id(&format!("{}::{}", domain.trim(), statement.trim()));
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        let existing: Option<(String, i64)> = conn
            .query_row(
                "SELECT status, version FROM claims WHERE id=?1",
                [&claim_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;

        if let Some((status, version)) = existing {
            if status == "rejected" {
                conn.execute(
                    "UPDATE claims SET status='unresolved', version=?1, source_url=?2, proposer=?3, domain=?4, updated_at=?5 WHERE id=?6",
                    params![version + 1, source_url, proposer, domain, now, claim_id],
                )?;
            }
        } else {
            conn.execute(
                "INSERT INTO claims (id, statement, source_url, proposer, status, version, created_at, updated_at, domain) \
                 VALUES (?1, ?2, ?3, ?4, 'unresolved', 1, ?5, ?6, ?7)",
                params![claim_id, statement.trim(), source_url, proposer, now, now, domain],
            )?;
        }
        Ok(claim_id)
    }

    pub fn save_embedding_in_domain(
        &self,
        memory_type: &str,
        target_id: &str,
        text: &str,
        embedding: &[f32],
        domain: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let emb_json = serde_json::to_string(embedding)?;
        conn.execute(
            "INSERT INTO embeddings (memory_type, target_id, text, embedding, created_at, domain) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(memory_type, target_id) DO UPDATE SET text=excluded.text, embedding=excluded.embedding, created_at=excluded.created_at, domain=excluded.domain",
            params![memory_type, target_id, text, emb_json, now, domain],
        )?;
        Ok(())
    }

    pub fn search_semantic_in_domain(
        &self,
        domain: &str,
        query_emb: &[f32],
        limit: usize,
    ) -> Result<Vec<(MemoryEmbedding, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT memory_type, target_id, text, embedding FROM embeddings WHERE domain = ?1",
        )?;
        let rows = stmt.query_map([domain], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;

        let mut scored = Vec::new();
        for row in rows {
            let (mtype, tid, text, emb_str) = row?;
            let trimmed = text.trim_start();
            // Filter out raw specialist turn logs and raw JSON dumps so retrieval returns substantive domain propositions
            if trimmed.starts_with("Specialist [")
                || trimmed.starts_with('{')
                || trimmed.starts_with("```")
            {
                continue;
            }
            if let Ok(emb) = serde_json::from_str::<Vec<f32>>(&emb_str) {
                if let Ok(sim) = crate::model::compute_cosine(query_emb, &emb) {
                    scored.push((
                        MemoryEmbedding {
                            memory_type: mtype,
                            target_id: tid,
                            text,
                            embedding: emb,
                        },
                        sim,
                    ));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > limit {
            scored.truncate(limit);
        }
        Ok(scored)
    }

    pub fn list_substantive_learnings_in_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<Claim>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, statement, source_url, proposer, status, version, created_at, updated_at, \
                    source_support_status, source_corroboration_count, experimental_verification_status, verification_evidence, \
                    finding_kind, source_fidelity_summary, execution_id, assumptions_text, negative_results_text, reproducibility_artifacts_json \
             FROM claims \
             WHERE status = 'trusted' AND domain = ?1 \
               AND statement NOT LIKE 'Specialist [%' \
               AND statement NOT LIKE '{%' \
               AND statement NOT LIKE '```%' \
             ORDER BY updated_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, limit as i64], Self::map_claim_row)?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r?);
        }
        Ok(list)
    }

    pub fn search_claims_lexical_in_domain(
        &self,
        domain: &str,
        keywords: &[String],
        limit: usize,
    ) -> Result<Vec<Claim>> {
        let all = self.list_substantive_learnings_in_domain(domain, 3000)?;
        if keywords.is_empty() {
            return Ok(all.into_iter().take(limit).collect());
        }
        let lowered_kw: Vec<String> = keywords
            .iter()
            .map(|k| k.trim().to_lowercase())
            .filter(|k| k.len() >= 2)
            .collect();
        let mut scored: Vec<(usize, Claim)> = all
            .into_iter()
            .filter_map(|c| {
                let s_low = c.statement.to_lowercase();
                let u_low = c.source_url.to_lowercase();
                let hits = lowered_kw
                    .iter()
                    .filter(|kw| s_low.contains(kw.as_str()) || u_low.contains(kw.as_str()))
                    .count();
                if hits > 0 {
                    Some((hits, c))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(scored.into_iter().take(limit).map(|(_, c)| c).collect())
    }

    pub fn list_research_works_in_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<crate::web::ResolvedResearchWork>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT canonical_id, title, normalized_title, doi, arxiv_id, versions_json, published_date,
                    authors_json, primary_connector, corroborating_connectors_json, urls_json, code_urls_json,
                    abstract_excerpt, experimental_details, limitations_excerpt, citations_json,
                    availability_status, duplicate_copies_merged, retrieved_at
             FROM research_works WHERE domain = ?1 ORDER BY retrieved_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, limit as i64], |r| {
            let v_str: String = r.get(5)?;
            let a_str: String = r.get(7)?;
            let c_str: String = r.get(9)?;
            let u_str: String = r.get(10)?;
            let cu_str: String = r.get(11)?;
            let cit_str: String = r.get(15)?;
            let urls: Vec<String> = serde_json::from_str(&u_str).unwrap_or_default();
            let primary_url = urls.first().cloned().unwrap_or_default();
            let title: String = r.get(1)?;
            let doi: Option<String> = r.get(3)?;
            let arxiv_id: Option<String> = r.get(4)?;
            let versions: Vec<String> = serde_json::from_str(&v_str).unwrap_or_default();
            let primary_connector: String = r.get(8)?;
            let abstract_excerpt: String = r.get(12)?;
            let identity = crate::web::canonicalize_source_identity(
                &primary_url,
                Some(&title),
                doi.as_deref(),
                arxiv_id.as_deref(),
                versions.last().map(|s| s.as_str()),
                Some(&abstract_excerpt),
                Some(&primary_connector),
            );
            Ok(crate::web::ResolvedResearchWork {
                canonical_id: r.get(0)?,
                title,
                normalized_title: r.get(2)?,
                provider_id: identity.provider_id,
                doi,
                arxiv_id,
                versions,
                published_date: r.get(6)?,
                authors: serde_json::from_str(&a_str).unwrap_or_default(),
                primary_connector,
                corroborating_connectors: serde_json::from_str(&c_str).unwrap_or_default(),
                canonical_url: identity.canonical_url,
                urls,
                code_urls: serde_json::from_str(&cu_str).unwrap_or_default(),
                abstract_excerpt,
                experimental_details: r.get(13)?,
                limitations_excerpt: r.get(14)?,
                citations: serde_json::from_str(&cit_str).unwrap_or_default(),
                availability_status: r.get(16)?,
                content_hash: identity.content_hash,
                source_kind: identity.source_kind,
                duplicate_copies_merged: r.get::<_, i64>(17)? as usize,
                independent_corroboration_count: 1,
                retrieved_at: r.get(18)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_domain_ontology_diagnostics(&self, domain: &str) -> Result<DomainOntologyDiagnostics> {
        let conn = self.conn.lock().unwrap();
        let trusted_claims_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM claims WHERE status='trusted' AND domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let raw_specialist_log_claims_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM claims WHERE status='trusted' AND domain=?1 \
             AND (statement LIKE 'Specialist [%' OR statement LIKE '{%' OR statement LIKE '```%')",
            [domain],
            |r| r.get(0),
        )?;
        let substantive_claims_in_domain =
            (trusted_claims_in_domain - raw_specialist_log_claims_in_domain).max(0) as usize;
        let unresolved_claims_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM claims WHERE status='unresolved' AND domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let rejected_claims_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM claims WHERE status='rejected' AND domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let research_works_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM research_works WHERE domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let multi_connector_works_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM research_works WHERE domain=?1 AND corroborating_connectors_json LIKE '%,%'",
            [domain],
            |r| r.get(0),
        )?;
        let works_with_code_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM research_works WHERE domain=?1 AND code_urls_json != '[]'",
            [domain],
            |r| r.get(0),
        )?;
        let research_syntheses_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM research_syntheses WHERE domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let semantic_nodes_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM semantic_nodes WHERE domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let empty_property_nodes_in_domain: i64 = conn.query_row(
            "SELECT count(*) FROM semantic_nodes WHERE domain=?1 AND (properties_json='{}' OR properties_json='')",
            [domain],
            |r| r.get(0),
        )?;
        let total_semantic_edges: i64 =
            conn.query_row("SELECT count(*) FROM semantic_edges", [], |r| r.get(0))?;
        let generic_relates_to_edges: i64 = conn.query_row(
            "SELECT count(*) FROM semantic_edges WHERE relation='relates_to'",
            [],
            |r| r.get(0),
        )?;
        let total_experiments: i64 =
            conn.query_row("SELECT count(*) FROM research_experiments", [], |r| r.get(0))?;
        let recorded_failure_experiments: i64 = conn.query_row(
            "SELECT count(*) FROM research_experiments WHERE is_failure=1",
            [],
            |r| r.get(0),
        )?;
        let latest_claim_updated_at: Option<String> = conn
            .query_row(
                "SELECT MAX(updated_at) FROM claims WHERE domain=?1 AND status='trusted'",
                [domain],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        let latest_work_retrieved_at: Option<String> = conn
            .query_row(
                "SELECT MAX(retrieved_at) FROM research_works WHERE domain=?1",
                [domain],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        let outcome_feedback_count: i64 = conn.query_row(
            "SELECT count(*) FROM consultation_feedback WHERE domain=?1",
            [domain],
            |r| r.get(0),
        )?;
        let outcome_feedback_success_count: i64 = conn.query_row(
            "SELECT count(*) FROM consultation_feedback WHERE domain=?1 AND succeeded=1",
            [domain],
            |r| r.get(0),
        )?;

        Ok(DomainOntologyDiagnostics {
            domain: domain.to_string(),
            trusted_claims_in_domain: trusted_claims_in_domain as usize,
            substantive_claims_in_domain,
            raw_specialist_log_claims_in_domain: raw_specialist_log_claims_in_domain as usize,
            unresolved_claims_in_domain: unresolved_claims_in_domain as usize,
            rejected_claims_in_domain: rejected_claims_in_domain as usize,
            research_works_in_domain: research_works_in_domain as usize,
            multi_connector_works_in_domain: multi_connector_works_in_domain as usize,
            works_with_code_in_domain: works_with_code_in_domain as usize,
            research_syntheses_in_domain: research_syntheses_in_domain as usize,
            semantic_nodes_in_domain: semantic_nodes_in_domain as usize,
            empty_property_nodes_in_domain: empty_property_nodes_in_domain as usize,
            total_semantic_edges: total_semantic_edges as usize,
            generic_relates_to_edges: generic_relates_to_edges as usize,
            total_experiments: total_experiments as usize,
            recorded_failure_experiments: recorded_failure_experiments as usize,
            latest_claim_updated_at,
            latest_work_retrieved_at,
            outcome_feedback_count: outcome_feedback_count as usize,
            outcome_feedback_success_count: outcome_feedback_success_count as usize,
        })
    }

    // --- Consultation Audit Trail & Async Lifecycle ---

    pub fn create_consultation(
        &self,
        id: &str,
        domain: &str,
        requested_tool: &str,
        effective_tool: &str,
        escalated_from: Option<&str>,
        escalation_reason: Option<&str>,
        depth: &str,
        state: &str,
        progress: &str,
        input_json: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO consultations (
                id, domain, requested_tool, effective_tool, escalated_from, escalation_reason,
                depth, state, progress, input_json, pending_question, clarification_history_json,
                result_json, artifacts_json, error, created_at, updated_at, completed_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL, '[]', NULL, '[]', NULL, ?11, ?12, NULL)",
            params![
                id,
                domain,
                requested_tool,
                effective_tool,
                escalated_from,
                escalation_reason,
                depth,
                state,
                progress,
                input_json,
                now,
                now
            ],
        )?;
        Ok(())
    }

    pub fn update_consultation_progress(
        &self,
        id: &str,
        state: &str,
        progress: &str,
        artifacts_json: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        if let Some(art) = artifacts_json {
            conn.execute(
                "UPDATE consultations SET state=?1, progress=?2, artifacts_json=?3, updated_at=?4 WHERE id=?5",
                params![state, progress, art, now, id],
            )?;
        } else {
            conn.execute(
                "UPDATE consultations SET state=?1, progress=?2, updated_at=?3 WHERE id=?4",
                params![state, progress, now, id],
            )?;
        }
        Ok(())
    }

    pub fn set_consultation_awaiting_input(
        &self,
        id: &str,
        question: &str,
        progress: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE consultations SET state='awaiting_input', pending_question=?1, progress=?2, updated_at=?3 WHERE id=?4",
            params![question, progress, now, id],
        )?;
        Ok(())
    }

    pub fn record_consultation_clarification(
        &self,
        id: &str,
        answer: &str,
    ) -> Result<Option<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let row: Option<(Option<String>, String)> = conn
            .query_row(
                "SELECT pending_question, clarification_history_json FROM consultations WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((pending_q, hist_str)) = row else {
            return Ok(None);
        };
        let question = pending_q.unwrap_or_else(|| "Clarification provided".to_string());
        let mut hist: Vec<serde_json::Value> =
            serde_json::from_str(&hist_str).unwrap_or_default();
        let now = Utc::now().to_rfc3339();
        hist.push(serde_json::json!({
            "question": question,
            "answer": answer,
            "answered_at": now,
        }));
        let new_hist = serde_json::to_string(&hist)?;
        conn.execute(
            "UPDATE consultations SET state='running', pending_question=NULL, clarification_history_json=?1, progress='Clarification received; resuming consultation', updated_at=?2 WHERE id=?3",
            params![new_hist, now, id],
        )?;
        Ok(Some((question, new_hist)))
    }

    pub fn complete_consultation(
        &self,
        id: &str,
        result_json: &str,
        artifacts_json: &str,
        progress: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE consultations SET state='completed', pending_question=NULL, result_json=?1, artifacts_json=?2, progress=?3, updated_at=?4, completed_at=?5 WHERE id=?6",
            params![result_json, artifacts_json, progress, now, now, id],
        )?;
        Ok(())
    }

    pub fn fail_consultation(&self, id: &str, error: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE consultations SET state='failed', error=?1, progress=?2, updated_at=?3, completed_at=?4 WHERE id=?5",
            params![error, format!("Failed: {}", error), now, now, id],
        )?;
        Ok(())
    }

    pub fn cancel_consultation(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let changed = conn.execute(
            "UPDATE consultations SET state='cancelled', pending_question=NULL, progress='Cancelled by caller', updated_at=?1, completed_at=?2 \
             WHERE id=?3 AND state IN ('queued','running','awaiting_input')",
            params![now, now, id],
        )?;
        Ok(changed > 0)
    }

    pub fn get_consultation(&self, id: &str) -> Result<Option<ConsultationRecord>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, domain, requested_tool, effective_tool, escalated_from, escalation_reason,
                    depth, state, progress, input_json, pending_question, clarification_history_json,
                    result_json, artifacts_json, error, created_at, updated_at, completed_at
             FROM consultations WHERE id=?1",
            [id],
            |r| {
                Ok(ConsultationRecord {
                    id: r.get(0)?,
                    domain: r.get(1)?,
                    requested_tool: r.get(2)?,
                    effective_tool: r.get(3)?,
                    escalated_from: r.get(4)?,
                    escalation_reason: r.get(5)?,
                    depth: r.get(6)?,
                    state: r.get(7)?,
                    progress: r.get(8)?,
                    input_json: r.get(9)?,
                    pending_question: r.get(10)?,
                    clarification_history_json: r.get(11)?,
                    result_json: r.get(12)?,
                    artifacts_json: r.get(13)?,
                    error: r.get(14)?,
                    created_at: r.get(15)?,
                    updated_at: r.get(16)?,
                    completed_at: r.get(17)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn list_consultations(&self, limit: usize) -> Result<Vec<ConsultationRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, domain, requested_tool, effective_tool, escalated_from, escalation_reason,
                    depth, state, progress, input_json, pending_question, clarification_history_json,
                    result_json, artifacts_json, error, created_at, updated_at, completed_at
             FROM consultations ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(ConsultationRecord {
                id: r.get(0)?,
                domain: r.get(1)?,
                requested_tool: r.get(2)?,
                effective_tool: r.get(3)?,
                escalated_from: r.get(4)?,
                escalation_reason: r.get(5)?,
                depth: r.get(6)?,
                state: r.get(7)?,
                progress: r.get(8)?,
                input_json: r.get(9)?,
                pending_question: r.get(10)?,
                clarification_history_json: r.get(11)?,
                result_json: r.get(12)?,
                artifacts_json: r.get(13)?,
                error: r.get(14)?,
                created_at: r.get(15)?,
                updated_at: r.get(16)?,
                completed_at: r.get(17)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Records outcome feedback from a consultation consumer.
    /// IMPORTANT: Outcome feedback is stored strictly as attributable evaluation evidence
    /// (`status = 'attributable_evaluation_evidence'`) and NEVER creates or promotes a trusted claim.
    pub fn record_outcome_feedback(
        &self,
        consultation_id: &str,
        domain: &str,
        consumer: &str,
        applied_advice: &str,
        observed_outcome: &str,
        succeeded: bool,
        discrepancies: &str,
        metrics_json: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO consultation_feedback (
                consultation_id, domain, consumer, applied_advice, observed_outcome,
                succeeded, discrepancies, metrics_json, status, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'attributable_evaluation_evidence', ?9)",
            params![
                consultation_id,
                domain,
                consumer,
                applied_advice,
                observed_outcome,
                if succeeded { 1 } else { 0 },
                discrepancies,
                metrics_json,
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_outcome_feedback_for_consultation(
        &self,
        consultation_id: &str,
    ) -> Result<Vec<ConsultationFeedbackRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, consultation_id, domain, consumer, applied_advice, observed_outcome,
                    succeeded, discrepancies, metrics_json, status, created_at
             FROM consultation_feedback WHERE consultation_id=?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([consultation_id], |r| {
            Ok(ConsultationFeedbackRecord {
                id: r.get(0)?,
                consultation_id: r.get(1)?,
                domain: r.get(2)?,
                consumer: r.get(3)?,
                applied_advice: r.get(4)?,
                observed_outcome: r.get(5)?,
                succeeded: r.get::<_, i64>(6)? != 0,
                discrepancies: r.get(7)?,
                metrics_json: r.get(8)?,
                status: r.get(9)?,
                created_at: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_outcome_feedback_in_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<ConsultationFeedbackRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, consultation_id, domain, consumer, applied_advice, observed_outcome,
                    succeeded, discrepancies, metrics_json, status, created_at
             FROM consultation_feedback WHERE domain=?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, limit as i64], |r| {
            Ok(ConsultationFeedbackRecord {
                id: r.get(0)?,
                consultation_id: r.get(1)?,
                domain: r.get(2)?,
                consumer: r.get(3)?,
                applied_advice: r.get(4)?,
                observed_outcome: r.get(5)?,
                succeeded: r.get::<_, i64>(6)? != 0,
                discrepancies: r.get(7)?,
                metrics_json: r.get(8)?,
                status: r.get(9)?,
                created_at: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    fn map_evaluation_run_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<EvaluationRunRecord> {
        let ft_json: String = r
            .get::<_, String>(24)
            .unwrap_or_else(|_| "{}".to_string());
        let abs_json: String = r
            .get::<_, String>(25)
            .unwrap_or_else(|_| "{}".to_string());
        let full_text_scores: TopicScoreSplit =
            serde_json::from_str(&ft_json).unwrap_or_default();
        let abstract_or_low_fidelity_scores: TopicScoreSplit =
            serde_json::from_str(&abs_json).unwrap_or_default();

        Ok(EvaluationRunRecord {
            id: r.get(0)?,
            domain: r.get(1)?,
            benchmark_name: r.get(2)?,
            item_count: r.get::<_, i64>(3)? as usize,
            triad_correctness: r.get(4)?,
            baseline_correctness: r.get(5)?,
            triad_problem_solving: r.get(6)?,
            baseline_problem_solving: r.get(7)?,
            triad_calibration_brier: r.get(8)?,
            baseline_calibration_brier: r.get(9)?,
            triad_citation_accuracy: r.get(10)?,
            baseline_citation_accuracy: r.get(11)?,
            details_json: r.get(12)?,
            created_at: r.get(13)?,
            triad_source_diversity: r.get::<_, f64>(14).unwrap_or(0.0),
            baseline_source_diversity: r.get::<_, f64>(15).unwrap_or(0.0),
            triad_contradiction_detection: r.get::<_, f64>(16).unwrap_or(0.0),
            baseline_contradiction_detection: r.get::<_, f64>(17).unwrap_or(0.0),
            triad_uncertainty_calibration: r.get::<_, f64>(18).unwrap_or(0.0),
            baseline_uncertainty_calibration: r.get::<_, f64>(19).unwrap_or(0.0),
            baseline_provider_id: r
                .get::<_, String>(20)
                .unwrap_or_else(|_| "local_snap".to_string()),
            baseline_model_id: r
                .get::<_, String>(21)
                .unwrap_or_else(|_| "smollm3-3b".to_string()),
            baseline_is_frontier: r.get::<_, i64>(22).unwrap_or(0) != 0,
            baseline_selection_basis: r.get::<_, String>(23).unwrap_or_default(),
            baseline_fallback_reason: r.get::<_, Option<String>>(26).unwrap_or(None),
            evaluated_corpus_full_text_share: r.get::<_, f64>(27).unwrap_or(0.0),
            full_text_scores,
            abstract_or_low_fidelity_scores,
            real_data_problem_solving: r.get::<_, f64>(28).unwrap_or(0.0),
            baseline_real_data_problem_solving: r.get::<_, f64>(29).unwrap_or(0.0),
            synthetic_data_problem_solving: r.get::<_, f64>(30).unwrap_or(0.0),
            baseline_synthetic_data_problem_solving: r.get::<_, f64>(31).unwrap_or(0.0),
            real_data_experiment_count: r.get::<_, i64>(32).unwrap_or(0).max(0) as usize,
            synthetic_data_experiment_count: r.get::<_, i64>(33).unwrap_or(0).max(0) as usize,
            domain_competence_verified: r.get::<_, i64>(34).unwrap_or(0) != 0,
        })
    }

    pub fn record_evaluation_run(&self, run: &EvaluationRunRecord) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = if run.created_at.trim().is_empty() {
            Utc::now().to_rfc3339()
        } else {
            run.created_at.clone()
        };
        let ft_json =
            serde_json::to_string(&run.full_text_scores).unwrap_or_else(|_| "{}".to_string());
        let abs_json = serde_json::to_string(&run.abstract_or_low_fidelity_scores)
            .unwrap_or_else(|_| "{}".to_string());

        conn.execute(
            "INSERT INTO evaluation_runs (
                domain, benchmark_name, item_count, triad_correctness, baseline_correctness,
                triad_problem_solving, baseline_problem_solving, triad_calibration_brier,
                baseline_calibration_brier, triad_citation_accuracy, baseline_citation_accuracy,
                details_json, created_at,
                triad_source_diversity, baseline_source_diversity,
                triad_contradiction_detection, baseline_contradiction_detection,
                triad_uncertainty_calibration, baseline_uncertainty_calibration,
                baseline_provider_id, baseline_model_id, baseline_is_frontier,
                baseline_selection_basis, full_text_scores_json, abstract_or_low_fidelity_scores_json,
                baseline_fallback_reason, evaluated_corpus_full_text_share,
                real_data_problem_solving, baseline_real_data_problem_solving,
                synthetic_data_problem_solving, baseline_synthetic_data_problem_solving,
                real_data_experiment_count, synthetic_data_experiment_count,
                domain_competence_verified
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25,
                ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34
            )",
            params![
                run.domain,
                run.benchmark_name,
                run.item_count as i64,
                run.triad_correctness,
                run.baseline_correctness,
                run.triad_problem_solving,
                run.baseline_problem_solving,
                run.triad_calibration_brier,
                run.baseline_calibration_brier,
                run.triad_citation_accuracy,
                run.baseline_citation_accuracy,
                run.details_json,
                now,
                run.triad_source_diversity,
                run.baseline_source_diversity,
                run.triad_contradiction_detection,
                run.baseline_contradiction_detection,
                run.triad_uncertainty_calibration,
                run.baseline_uncertainty_calibration,
                run.baseline_provider_id,
                run.baseline_model_id,
                if run.baseline_is_frontier { 1 } else { 0 },
                run.baseline_selection_basis,
                ft_json,
                abs_json,
                run.baseline_fallback_reason,
                run.evaluated_corpus_full_text_share,
                run.real_data_problem_solving,
                run.baseline_real_data_problem_solving,
                run.synthetic_data_problem_solving,
                run.baseline_synthetic_data_problem_solving,
                run.real_data_experiment_count as i64,
                run.synthetic_data_experiment_count as i64,
                if run.domain_competence_verified { 1 } else { 0 },
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_latest_evaluation_run(&self, domain: &str) -> Result<Option<EvaluationRunRecord>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, domain, benchmark_name, item_count, triad_correctness, baseline_correctness,
                    triad_problem_solving, baseline_problem_solving, triad_calibration_brier,
                    baseline_calibration_brier, triad_citation_accuracy, baseline_citation_accuracy,
                    details_json, created_at,
                    COALESCE(triad_source_diversity, 0.0), COALESCE(baseline_source_diversity, 0.0),
                    COALESCE(triad_contradiction_detection, 0.0), COALESCE(baseline_contradiction_detection, 0.0),
                    COALESCE(triad_uncertainty_calibration, 0.0), COALESCE(baseline_uncertainty_calibration, 0.0),
                    COALESCE(baseline_provider_id, 'local_snap'), COALESCE(baseline_model_id, 'smollm3-3b'),
                    COALESCE(baseline_is_frontier, 0), COALESCE(baseline_selection_basis, ''),
                    COALESCE(full_text_scores_json, '{}'), COALESCE(abstract_or_low_fidelity_scores_json, '{}'),
                    baseline_fallback_reason, COALESCE(evaluated_corpus_full_text_share, 0.0),
                    COALESCE(real_data_problem_solving, 0.0), COALESCE(baseline_real_data_problem_solving, 0.0),
                    COALESCE(synthetic_data_problem_solving, 0.0), COALESCE(baseline_synthetic_data_problem_solving, 0.0),
                    COALESCE(real_data_experiment_count, 0), COALESCE(synthetic_data_experiment_count, 0),
                    COALESCE(domain_competence_verified, 0)
             FROM evaluation_runs WHERE domain=?1 ORDER BY id DESC LIMIT 1",
            [domain],
            Self::map_evaluation_run_row,
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn list_evaluation_runs_in_domain(
        &self,
        domain: &str,
        limit: usize,
    ) -> Result<Vec<EvaluationRunRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, domain, benchmark_name, item_count, triad_correctness, baseline_correctness,
                    triad_problem_solving, baseline_problem_solving, triad_calibration_brier,
                    baseline_calibration_brier, triad_citation_accuracy, baseline_citation_accuracy,
                    details_json, created_at,
                    COALESCE(triad_source_diversity, 0.0), COALESCE(baseline_source_diversity, 0.0),
                    COALESCE(triad_contradiction_detection, 0.0), COALESCE(baseline_contradiction_detection, 0.0),
                    COALESCE(triad_uncertainty_calibration, 0.0), COALESCE(baseline_uncertainty_calibration, 0.0),
                    COALESCE(baseline_provider_id, 'local_snap'), COALESCE(baseline_model_id, 'smollm3-3b'),
                    COALESCE(baseline_is_frontier, 0), COALESCE(baseline_selection_basis, ''),
                    COALESCE(full_text_scores_json, '{}'), COALESCE(abstract_or_low_fidelity_scores_json, '{}'),
                    baseline_fallback_reason, COALESCE(evaluated_corpus_full_text_share, 0.0),
                    COALESCE(real_data_problem_solving, 0.0), COALESCE(baseline_real_data_problem_solving, 0.0),
                    COALESCE(synthetic_data_problem_solving, 0.0), COALESCE(baseline_synthetic_data_problem_solving, 0.0),
                    COALESCE(real_data_experiment_count, 0), COALESCE(synthetic_data_experiment_count, 0),
                    COALESCE(domain_competence_verified, 0)
             FROM evaluation_runs WHERE domain=?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, limit as i64], Self::map_evaluation_run_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Computes the evaluated corpus full-text share (`full_text_non_low_fidelity / total_primary_sources`)
    /// and the real-data vs synthetic-data experiment breakdown for `domain`.
    pub fn compute_domain_corpus_and_experiment_stats(
        &self,
        domain: &str,
    ) -> Result<(f64, usize, usize, usize, usize, usize, f64, f64)> {
        let conn = self.conn.lock().unwrap();
        let total_primary_sources: i64 = conn
            .query_row(
                "SELECT count(*) FROM tracked_sources WHERE domain = ?1 AND is_orientation_only = 0",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let full_text_verified_sources: i64 = conn
            .query_row(
                "SELECT count(*) FROM tracked_sources
                 WHERE domain = ?1 AND is_orientation_only = 0
                   AND ingestion_completeness = 'full_text' AND low_fidelity = 0",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let abstract_only_sources: i64 = conn
            .query_row(
                "SELECT count(*) FROM tracked_sources
                 WHERE domain = ?1 AND is_orientation_only = 0
                   AND ingestion_completeness != 'full_text'",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let low_fidelity_sources: i64 = conn
            .query_row(
                "SELECT count(*) FROM tracked_sources
                 WHERE domain = ?1 AND is_orientation_only = 0
                   AND ingestion_completeness = 'full_text' AND low_fidelity = 1",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let full_text_share = if total_primary_sources > 0 {
            ((full_text_verified_sources as f64) / (total_primary_sources as f64) * 1000.0).round()
                / 1000.0
        } else {
            0.0
        };

        let real_exp_total: i64 = conn
            .query_row(
                "SELECT count(*) FROM research_experiments WHERE domain = ?1 AND data_kind = 'real_data'",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let real_exp_success: i64 = conn
            .query_row(
                "SELECT count(*) FROM research_experiments
                 WHERE domain = ?1 AND data_kind = 'real_data' AND is_failure = 0",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let synth_exp_total: i64 = conn
            .query_row(
                "SELECT count(*) FROM research_experiments WHERE domain = ?1 AND data_kind = 'synthetic_data'",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let synth_exp_success: i64 = conn
            .query_row(
                "SELECT count(*) FROM research_experiments
                 WHERE domain = ?1 AND data_kind = 'synthetic_data' AND is_failure = 0",
                [domain],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let real_exp_rate = if real_exp_total > 0 {
            ((real_exp_success as f64) / (real_exp_total as f64) * 1000.0).round() / 1000.0
        } else {
            0.0
        };
        let synth_exp_rate = if synth_exp_total > 0 {
            ((synth_exp_success as f64) / (synth_exp_total as f64) * 1000.0).round() / 1000.0
        } else {
            0.0
        };

        Ok((
            full_text_share,
            total_primary_sources.max(0) as usize,
            full_text_verified_sources.max(0) as usize,
            abstract_only_sources.max(0) as usize,
            low_fidelity_sources.max(0) as usize,
            real_exp_total.max(0) as usize,
            real_exp_rate,
            synth_exp_rate,
        ))
    }

    /// Evaluates per-topic coverage in `domain_cfg.name`, marks each topic as
    /// `"covered"`, `"thin"`, `"contradicted"`, or `"low_fidelity"`, links evaluation scores
    /// to coverage, feeds `"thin"`, `"contradicted"`, and `"low_fidelity"` topics into
    /// `research_backlog` prioritization, and requests re-ingestion on abstract-only or
    /// low-fidelity sources.
    pub fn sync_and_prioritize_domain_topic_coverage(
        &self,
        domain_cfg: &crate::config::DomainSection,
        eval_topic_metrics: Option<
            &std::collections::HashMap<String, (usize, f64, f64, f64, f64)>,
        >,
    ) -> Result<Vec<DomainTopicCoverageRecord>> {
        let domain = domain_cfg.name.as_str();
        let all_sources = self
            .list_tracked_sources(500)?
            .into_iter()
            .filter(|s| s.domain == domain && !s.is_orientation_only)
            .collect::<Vec<_>>();
        let all_claims = self.list_substantive_learnings_in_domain(domain, 3000)?;

        // Collect all candidate topics: configured topics + evaluation case topics + tracked source topics
        let mut topic_list: Vec<(String, String, i64, Option<String>)> = Vec::new();
        for t in &domain_cfg.topics {
            if !topic_list
                .iter()
                .any(|(existing, _, _, _)| existing.eq_ignore_ascii_case(&t.topic))
            {
                topic_list.push((
                    t.topic.clone(),
                    t.discipline.clone(),
                    t.priority,
                    t.hypothesis.clone(),
                ));
            }
        }
        for ec in &domain_cfg.evaluation_cases {
            let et = ec.topic.trim();
            if !et.is_empty()
                && !topic_list
                    .iter()
                    .any(|(existing, _, _, _)| existing.eq_ignore_ascii_case(et))
            {
                topic_list.push((et.to_string(), domain_cfg.display_name.clone(), 85, None));
            }
        }
        if let Some(map) = eval_topic_metrics {
            for k in map.keys() {
                let kt = k.trim();
                if !kt.is_empty()
                    && !topic_list
                        .iter()
                        .any(|(existing, _, _, _)| existing.eq_ignore_ascii_case(kt))
                {
                    topic_list.push((kt.to_string(), domain_cfg.display_name.clone(), 85, None));
                }
            }
        }
        for s in &all_sources {
            if let Some(ref st) = s.topic {
                let st_trim = st.trim();
                if !st_trim.is_empty()
                    && !topic_list
                        .iter()
                        .any(|(existing, _, _, _)| existing.eq_ignore_ascii_case(st_trim))
                {
                    topic_list.push((
                        st_trim.to_string(),
                        domain_cfg.display_name.clone(),
                        80,
                        None,
                    ));
                }
            }
        }

        let stop_words = [
            "and", "the", "for", "with", "from", "into", "over", "under", "via",
        ];
        let extract_topic_tokens = |topic_str: &str| -> Vec<String> {
            topic_str
                .split(|c: char| !c.is_alphanumeric() && c != '-')
                .map(|w| w.trim().to_lowercase())
                .filter(|w| w.len() >= 4 && !stop_words.contains(&w.as_str()))
                .collect()
        };

        let mut records = Vec::new();
        let now = Utc::now().to_rfc3339();

        for (topic_name, discipline, base_priority, seed_hyp) in topic_list {
            let tokens = extract_topic_tokens(&topic_name);
            let matching_sources: Vec<&TrackedSourceRecord> = all_sources
                .iter()
                .filter(|s| {
                    if let Some(ref st) = s.topic {
                        if st.eq_ignore_ascii_case(&topic_name)
                            || st.to_lowercase().contains(&topic_name.to_lowercase())
                            || topic_name.to_lowercase().contains(&st.to_lowercase())
                        {
                            return true;
                        }
                    }
                    let title_low = s.title.to_lowercase();
                    tokens.iter().filter(|tk| title_low.contains(tk.as_str())).count() >= 2
                })
                .collect();

            let mut full_text_count = 0usize;
            let mut abstract_only_count = 0usize;
            let mut low_fidelity_count = 0usize;
            let mut contradiction_count = 0usize;
            let mut reingestion_ids = Vec::new();

            for s in &matching_sources {
                if s.has_contradiction {
                    contradiction_count += 1;
                }
                if s.ingestion_completeness == "full_text" && !s.low_fidelity {
                    full_text_count += 1;
                } else if s.low_fidelity || s.ingestion_completeness == "low_fidelity" {
                    low_fidelity_count += 1;
                    if s.ingestion_completeness != "full_text" {
                        abstract_only_count += 1;
                    }
                    reingestion_ids.push(s.canonical_id.clone());
                } else {
                    abstract_only_count += 1;
                    reingestion_ids.push(s.canonical_id.clone());
                }
            }

            let matching_claims = all_claims
                .iter()
                .filter(|c| {
                    let cl = c.statement.to_lowercase();
                    tokens.iter().filter(|tk| cl.contains(tk.as_str())).count() >= 2
                })
                .count();

            // Determine coverage_status: "covered", "thin", "contradicted", or "low_fidelity"
            let coverage_status = if contradiction_count > 0 {
                "contradicted".to_string()
            } else if full_text_count > 0 {
                "covered".to_string()
            } else if low_fidelity_count > 0 {
                "low_fidelity".to_string()
            } else {
                "thin".to_string()
            };

            let eval_entry = eval_topic_metrics.and_then(|m| {
                m.get(&topic_name).copied().or_else(|| {
                    m.iter()
                        .find(|(k, _)| {
                            k.eq_ignore_ascii_case(&topic_name)
                                || k.to_lowercase().contains(&topic_name.to_lowercase())
                                || topic_name.to_lowercase().contains(&k.to_lowercase())
                        })
                        .map(|(_, v)| *v)
                })
            });

            let (eval_cnt, eval_corr, eval_ps, eval_cit, eval_cal) = match eval_entry {
                Some((cnt, corr, ps, cit, cal)) => {
                    (cnt, Some(corr), Some(ps), Some(cit), Some(cal))
                }
                None => (0, None, None, None, None),
            };

            // Feed thin, contradicted, and low-fidelity topics into research prioritization & re-ingestion requests
            let needs_prioritization = matches!(
                coverage_status.as_str(),
                "thin" | "contradicted" | "low_fidelity"
            );
            let boosted_priority = match coverage_status.as_str() {
                "contradicted" => 99,
                "low_fidelity" => 98,
                "thin" => 97,
                _ => base_priority,
            };

            let mut reingestion_requested = false;
            if needs_prioritization {
                for cid in &reingestion_ids {
                    let _ = self.request_source_reingestion(
                        cid,
                        &format!(
                            "reingestion_requested: topic '{}' marked '{}' during domain coverage & evaluation linking",
                            topic_name, coverage_status
                        ),
                    );
                    reingestion_requested = true;
                }
                let hyp_str = seed_hyp.unwrap_or_else(|| {
                    format!(
                        "Prioritized {} topic '{}' (full_text={}, abstract_only={}, low_fidelity={}, contradictions={}): acquire full-text sources and verify claims",
                        coverage_status, topic_name, full_text_count, abstract_only_count, low_fidelity_count, contradiction_count
                    )
                });
                let _ = self.upsert_research_backlog(
                    &topic_name,
                    &discipline,
                    boosted_priority,
                    Some(&hyp_str),
                    "pending",
                );
            }

            let rec = DomainTopicCoverageRecord {
                domain: domain.to_string(),
                topic: topic_name.clone(),
                discipline: discipline.clone(),
                coverage_status: coverage_status.clone(),
                full_text_sources_count: full_text_count,
                abstract_only_sources_count: abstract_only_count,
                low_fidelity_sources_count: low_fidelity_count,
                contradiction_count,
                substantive_claims_count: matching_claims,
                eval_item_count: eval_cnt,
                eval_correctness: eval_corr,
                eval_problem_solving: eval_ps,
                eval_citation_accuracy: eval_cit,
                eval_uncertainty_calibration: eval_cal,
                reingestion_requested,
                reingestion_source_ids: reingestion_ids.clone(),
                backlog_priority: boosted_priority,
                updated_at: now.clone(),
            };

            {
                let conn = self.conn.lock().unwrap();
                let reing_json =
                    serde_json::to_string(&reingestion_ids).unwrap_or_else(|_| "[]".to_string());
                conn.execute(
                    "INSERT INTO domain_topic_coverage (
                        domain, topic, discipline, coverage_status,
                        full_text_sources_count, abstract_only_sources_count, low_fidelity_sources_count,
                        contradiction_count, substantive_claims_count,
                        eval_item_count, eval_correctness, eval_problem_solving,
                        eval_citation_accuracy, eval_uncertainty_calibration,
                        reingestion_requested, reingestion_source_ids_json, backlog_priority, updated_at
                    ) VALUES (
                        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18
                    )
                    ON CONFLICT(domain, topic) DO UPDATE SET
                        discipline = excluded.discipline,
                        coverage_status = excluded.coverage_status,
                        full_text_sources_count = excluded.full_text_sources_count,
                        abstract_only_sources_count = excluded.abstract_only_sources_count,
                        low_fidelity_sources_count = excluded.low_fidelity_sources_count,
                        contradiction_count = excluded.contradiction_count,
                        substantive_claims_count = excluded.substantive_claims_count,
                        eval_item_count = CASE WHEN excluded.eval_item_count > 0 THEN excluded.eval_item_count ELSE domain_topic_coverage.eval_item_count END,
                        eval_correctness = COALESCE(excluded.eval_correctness, domain_topic_coverage.eval_correctness),
                        eval_problem_solving = COALESCE(excluded.eval_problem_solving, domain_topic_coverage.eval_problem_solving),
                        eval_citation_accuracy = COALESCE(excluded.eval_citation_accuracy, domain_topic_coverage.eval_citation_accuracy),
                        eval_uncertainty_calibration = COALESCE(excluded.eval_uncertainty_calibration, domain_topic_coverage.eval_uncertainty_calibration),
                        reingestion_requested = excluded.reingestion_requested,
                        reingestion_source_ids_json = excluded.reingestion_source_ids_json,
                        backlog_priority = excluded.backlog_priority,
                        updated_at = excluded.updated_at",
                    params![
                        rec.domain,
                        rec.topic,
                        rec.discipline,
                        rec.coverage_status,
                        rec.full_text_sources_count as i64,
                        rec.abstract_only_sources_count as i64,
                        rec.low_fidelity_sources_count as i64,
                        rec.contradiction_count as i64,
                        rec.substantive_claims_count as i64,
                        rec.eval_item_count as i64,
                        rec.eval_correctness,
                        rec.eval_problem_solving,
                        rec.eval_citation_accuracy,
                        rec.eval_uncertainty_calibration,
                        if rec.reingestion_requested { 1 } else { 0 },
                        reing_json,
                        rec.backlog_priority,
                        rec.updated_at,
                    ],
                )?;
            }

            records.push(rec);
        }

        Ok(records)
    }

    pub fn list_domain_topic_coverage(
        &self,
        domain: &str,
    ) -> Result<Vec<DomainTopicCoverageRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT domain, topic, discipline, coverage_status,
                    full_text_sources_count, abstract_only_sources_count, low_fidelity_sources_count,
                    contradiction_count, substantive_claims_count,
                    eval_item_count, eval_correctness, eval_problem_solving,
                    eval_citation_accuracy, eval_uncertainty_calibration,
                    reingestion_requested, reingestion_source_ids_json, backlog_priority, updated_at
             FROM domain_topic_coverage WHERE domain = ?1 ORDER BY backlog_priority DESC, topic ASC",
        )?;
        let rows = stmt.query_map([domain], |r| {
            let reing_str: String = r.get(15)?;
            Ok(DomainTopicCoverageRecord {
                domain: r.get(0)?,
                topic: r.get(1)?,
                discipline: r.get(2)?,
                coverage_status: r.get(3)?,
                full_text_sources_count: r.get::<_, i64>(4)?.max(0) as usize,
                abstract_only_sources_count: r.get::<_, i64>(5)?.max(0) as usize,
                low_fidelity_sources_count: r.get::<_, i64>(6)?.max(0) as usize,
                contradiction_count: r.get::<_, i64>(7)?.max(0) as usize,
                substantive_claims_count: r.get::<_, i64>(8)?.max(0) as usize,
                eval_item_count: r.get::<_, i64>(9)?.max(0) as usize,
                eval_correctness: r.get(10)?,
                eval_problem_solving: r.get(11)?,
                eval_citation_accuracy: r.get(12)?,
                eval_uncertainty_calibration: r.get(13)?,
                reingestion_requested: r.get::<_, i64>(14)? != 0,
                reingestion_source_ids: serde_json::from_str(&reing_str).unwrap_or_default(),
                backlog_priority: r.get(16)?,
                updated_at: r.get(17)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // =========================================================================
    // SEPARATE CLAIM ATTRIBUTES: TRUST, SOURCE SUPPORT, EXPERIMENTAL VERIFICATION
    // =========================================================================

    pub fn update_claim_source_support(
        &self,
        claim_id: &str,
        support_status: &str,
        corroboration_count: usize,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE claims SET source_support_status=?1, source_corroboration_count=?2, updated_at=?3 WHERE id=?4",
            params![support_status, corroboration_count as i64, now, claim_id],
        )?;
        Ok(())
    }

    pub fn record_claim_experimental_verification(
        &self,
        claim_id: &str,
        verified: bool,
        evidence: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let ver_status = if verified { "verified" } else { "failed" };
        conn.execute(
            "UPDATE claims SET experimental_verification_status=?1, verification_evidence=?2, updated_at=?3 WHERE id=?4",
            params![ver_status, evidence, now, claim_id],
        )?;
        Ok(())
    }

    // =========================================================================
    // PERSISTENT, VERSION-AWARE SOURCE TRACKING & CORROBORATION
    // =========================================================================

    fn map_tracked_source_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<TrackedSourceRecord> {
        let versions_str: String = r.get(8)?;
        let urls_str: String = r.get(12)?;
        let conns_str: String = r.get(13)?;
        let questions_str: String = r.get(34)?;
        Ok(TrackedSourceRecord {
            canonical_id: r.get(0)?,
            canonical_url: r.get(1)?,
            title: r.get(2)?,
            normalized_title: r.get(3)?,
            provider_id: r.get(4)?,
            doi: r.get(5)?,
            arxiv_id: r.get(6)?,
            version: r.get(7)?,
            versions_seen: serde_json::from_str(&versions_str).unwrap_or_default(),
            content_hash: r.get(9)?,
            source_kind: r.get(10)?,
            domain: r.get(11)?,
            observed_urls: serde_json::from_str(&urls_str).unwrap_or_default(),
            observed_connectors: serde_json::from_str(&conns_str).unwrap_or_default(),
            route_observation_count: r.get::<_, i64>(14)? as usize,
            discovery_status: r.get(15)?,
            discovered_at: r.get(16)?,
            ingestion_status: r.get(17)?,
            ingested_at: r.get(18)?,
            research_status: r.get(19)?,
            reserved_by: r.get(20)?,
            reservation_token: r.get(21)?,
            reserved_at: r.get(22)?,
            completed_at: r.get(23)?,
            completed_version: r.get(24)?,
            completed_content_hash: r.get(25)?,
            failure_count: r.get::<_, i64>(26)? as usize,
            last_failure_reason: r.get(27)?,
            last_failed_at: r.get(28)?,
            retry_eligible: r.get::<_, i64>(29)? != 0,
            next_retry_at: r.get(30)?,
            revisit_reason: r.get(31)?,
            has_contradiction: r.get::<_, i64>(32)? != 0,
            contradiction_details: r.get(33)?,
            investigated_questions: serde_json::from_str(&questions_str).unwrap_or_default(),
            updated_at: r.get(35)?,
            ingestion_completeness: r.get::<_, String>(36).unwrap_or_else(|_| "abstract_only".to_string()),
            full_text_format: r.get::<_, Option<String>>(37).unwrap_or(None),
            topic: r.get::<_, Option<String>>(38).unwrap_or(None),
            is_orientation_only: r.get::<_, i64>(39).unwrap_or(0) != 0,
            section_count: r.get::<_, i64>(40).unwrap_or(0).max(0) as usize,
            equations_preserved: r.get::<_, i64>(41).unwrap_or(0).max(0) as usize,
            tables_found: r.get::<_, i64>(42).unwrap_or(0).max(0) as usize,
            low_fidelity: r.get::<_, i64>(43).unwrap_or(1) != 0,
            low_fidelity_reason: r.get::<_, Option<String>>(44).unwrap_or(None),
        })
    }

    fn query_tracked_source_by_id_conn(
        conn: &Connection,
        canonical_id: &str,
    ) -> Result<Option<TrackedSourceRecord>> {
        let sql = "SELECT canonical_id, canonical_url, title, normalized_title, provider_id, doi, arxiv_id,
                          version, versions_seen_json, content_hash, source_kind, domain,
                          observed_urls_json, observed_connectors_json, route_observation_count,
                          discovery_status, discovered_at, ingestion_status, ingested_at,
                          research_status, reserved_by, reservation_token, reserved_at,
                          completed_at, completed_version, completed_content_hash,
                          failure_count, last_failure_reason, last_failed_at,
                          retry_eligible, next_retry_at, revisit_reason,
                          has_contradiction, contradiction_details, investigated_questions_json, updated_at,
                          ingestion_completeness, full_text_format, topic, is_orientation_only, section_count,
                          equations_preserved, tables_found, low_fidelity, low_fidelity_reason
                   FROM tracked_sources WHERE canonical_id = ?1";
        conn.query_row(sql, [canonical_id], Self::map_tracked_source_row)
            .optional()
            .map_err(Into::into)
    }

    fn resolve_existing_canonical_id_conn(
        conn: &Connection,
        identity: &crate::web::CanonicalSourceIdentity,
        norm_title: &str,
    ) -> Result<Option<String>> {
        // 1. Direct canonical_id or canonical_url match
        if let Some(cid) = conn
            .query_row(
                "SELECT canonical_id FROM tracked_sources WHERE canonical_id = ?1 OR canonical_url = ?2 LIMIT 1",
                params![identity.canonical_id, identity.canonical_url],
                |r| r.get::<_, String>(0),
            )
            .optional()?
        {
            return Ok(Some(cid));
        }

        // 2. Match by arXiv ID
        if let Some(ref ax) = identity.arxiv_id {
            if let Some(cid) = conn
                .query_row(
                    "SELECT canonical_id FROM tracked_sources WHERE arxiv_id = ?1 LIMIT 1",
                    [ax],
                    |r| r.get::<_, String>(0),
                )
                .optional()?
            {
                return Ok(Some(cid));
            }
        }

        // 3. Match by DOI
        if let Some(ref d) = identity.doi {
            if let Some(cid) = conn
                .query_row(
                    "SELECT canonical_id FROM tracked_sources WHERE doi = ?1 LIMIT 1",
                    [d],
                    |r| r.get::<_, String>(0),
                )
                .optional()?
            {
                return Ok(Some(cid));
            }
        }

        // 4. Match by provider_id
        if let Some(ref pid) = identity.provider_id {
            if let Some(cid) = conn
                .query_row(
                    "SELECT canonical_id FROM tracked_sources WHERE provider_id = ?1 LIMIT 1",
                    [pid],
                    |r| r.get::<_, String>(0),
                )
                .optional()?
            {
                return Ok(Some(cid));
            }
        }

        // 5. Match by normalized_title for same source_kind (e.g. same paper across arXiv, OpenAlex, Crossref)
        if !norm_title.is_empty() && norm_title.len() >= 6 {
            if let Some(cid) = conn
                .query_row(
                    "SELECT canonical_id FROM tracked_sources WHERE source_kind = ?1 AND normalized_title = ?2 LIMIT 1",
                    params![identity.source_kind, norm_title],
                    |r| r.get::<_, String>(0),
                )
                .optional()?
            {
                return Ok(Some(cid));
            }
        }

        Ok(None)
    }

    pub fn discover_source(
        &self,
        raw_url: &str,
        title: Option<&str>,
        doi: Option<&str>,
        arxiv_id: Option<&str>,
        explicit_version: Option<&str>,
        content: Option<&str>,
        connector: Option<&str>,
        domain: &str,
    ) -> Result<TrackedSourceRecord> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rec = Self::discover_source_in_tx(
            &tx,
            raw_url,
            title,
            doi,
            arxiv_id,
            explicit_version,
            content,
            connector,
            domain,
        )?;
        tx.commit()?;
        Ok(rec)
    }

    fn discover_source_in_tx(
        conn: &Connection,
        raw_url: &str,
        title: Option<&str>,
        doi: Option<&str>,
        arxiv_id: Option<&str>,
        explicit_version: Option<&str>,
        content: Option<&str>,
        connector: Option<&str>,
        domain: &str,
    ) -> Result<TrackedSourceRecord> {
        let identity = crate::web::canonicalize_source_identity(
            raw_url,
            title,
            doi,
            arxiv_id,
            explicit_version,
            content,
            connector,
        );
        let title_str = title
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .unwrap_or(&identity.canonical_url);
        let norm_title = crate::web::normalize_title_for_dedup(title_str);
        let now = Utc::now().to_rfc3339();

        let existing_cid = Self::resolve_existing_canonical_id_conn(conn, &identity, &norm_title)?;
        if let Some(cid) = existing_cid {
            let mut existing = Self::query_tracked_source_by_id_conn(conn, &cid)?
                .context("Resolved canonical_id missing in tracked_sources")?;

            let mut new_route_observed = false;
            let raw_trimmed = raw_url.trim().to_string();
            if !raw_trimmed.is_empty() && !existing.observed_urls.contains(&raw_trimmed) {
                existing.observed_urls.push(raw_trimmed);
                new_route_observed = true;
            }
            if !existing.observed_urls.contains(&identity.canonical_url) {
                existing.observed_urls.push(identity.canonical_url.clone());
            }
            if let Some(conn_name) = connector.map(|c| c.trim()).filter(|c| !c.is_empty()) {
                let c_str = conn_name.to_string();
                if !existing.observed_connectors.contains(&c_str) {
                    existing.observed_connectors.push(c_str);
                    new_route_observed = true;
                }
            }
            if new_route_observed {
                existing.route_observation_count += 1;
            }

            if existing.doi.is_none() && identity.doi.is_some() {
                existing.doi = identity.doi.clone();
            }
            if existing.arxiv_id.is_none() && identity.arxiv_id.is_some() {
                existing.arxiv_id = identity.arxiv_id.clone();
            }
            if existing.provider_id.is_none() && identity.provider_id.is_some() {
                existing.provider_id = identity.provider_id.clone();
            }
            if (existing.title.is_empty() || existing.title.starts_with("http"))
                && !title_str.starts_with("http")
            {
                existing.title = title_str.to_string();
                existing.normalized_title = norm_title;
            }

            // Version-aware check: determine if this observation introduces a new document version
            let incoming_ver_explicit = explicit_version.is_some()
                || raw_url.to_lowercase().contains(&format!("v{}", identity.version.trim_start_matches('v')));
            let is_new_version = !existing.versions_seen.contains(&identity.version)
                || (existing.research_status == "completed"
                    && incoming_ver_explicit
                    && existing.completed_version.as_deref() != Some(identity.version.as_str()));

            if !existing.versions_seen.contains(&identity.version) {
                existing.versions_seen.push(identity.version.clone());
            }
            if is_new_version {
                existing.version = identity.version.clone();
                existing.content_hash = identity.content_hash.clone();
                if existing.research_status == "completed" {
                    let prev_ver = existing
                        .completed_version
                        .clone()
                        .unwrap_or_else(|| "v1".to_string());
                    existing.revisit_reason = Some(format!(
                        "new_version:{}->{}",
                        prev_ver, identity.version
                    ));
                    existing.retry_eligible = true;
                }
            } else if content.map(|c| !c.trim().is_empty()).unwrap_or(false)
                && existing.research_status != "completed"
            {
                existing.content_hash = identity.content_hash.clone();
            }

            if content.map(|c| !c.trim().is_empty()).unwrap_or(false)
                && existing.ingestion_status == "pending"
            {
                existing.ingestion_status = "ingested".to_string();
                existing.ingested_at = Some(now.clone());
            }

            let versions_json = serde_json::to_string(&existing.versions_seen)?;
            let urls_json = serde_json::to_string(&existing.observed_urls)?;
            let conns_json = serde_json::to_string(&existing.observed_connectors)?;
            conn.execute(
                "UPDATE tracked_sources SET
                    title = ?1,
                    normalized_title = ?2,
                    provider_id = ?3,
                    doi = ?4,
                    arxiv_id = ?5,
                    version = ?6,
                    versions_seen_json = ?7,
                    content_hash = ?8,
                    observed_urls_json = ?9,
                    observed_connectors_json = ?10,
                    route_observation_count = ?11,
                    ingestion_status = ?12,
                    ingested_at = ?13,
                    retry_eligible = ?14,
                    revisit_reason = ?15,
                    updated_at = ?16
                 WHERE canonical_id = ?17",
                params![
                    existing.title,
                    existing.normalized_title,
                    existing.provider_id,
                    existing.doi,
                    existing.arxiv_id,
                    existing.version,
                    versions_json,
                    existing.content_hash,
                    urls_json,
                    conns_json,
                    existing.route_observation_count as i64,
                    existing.ingestion_status,
                    existing.ingested_at,
                    if existing.retry_eligible { 1 } else { 0 },
                    existing.revisit_reason,
                    now,
                    existing.canonical_id,
                ],
            )?;
            existing.updated_at = now;
            return Ok(existing);
        }

        // New canonical source discovery
        let mut observed_urls = Vec::new();
        if !raw_url.trim().is_empty() {
            observed_urls.push(raw_url.trim().to_string());
        }
        if !observed_urls.contains(&identity.canonical_url) {
            observed_urls.push(identity.canonical_url.clone());
        }
        let mut observed_connectors = Vec::new();
        if let Some(c) = connector.map(|s| s.trim()).filter(|s| !s.is_empty()) {
            observed_connectors.push(c.to_string());
        }
        let has_content = content.map(|c| !c.trim().is_empty()).unwrap_or(false);
        let ingestion_status = if has_content { "ingested" } else { "pending" };
        let ingested_at = if has_content { Some(now.clone()) } else { None };
        let versions_seen = vec![identity.version.clone()];
        let versions_json = serde_json::to_string(&versions_seen)?;
        let urls_json = serde_json::to_string(&observed_urls)?;
        let conns_json = serde_json::to_string(&observed_connectors)?;

        conn.execute(
            "INSERT INTO tracked_sources (
                canonical_id, canonical_url, title, normalized_title, provider_id, doi, arxiv_id,
                version, versions_seen_json, content_hash, source_kind, domain,
                observed_urls_json, observed_connectors_json, route_observation_count,
                discovery_status, discovered_at, ingestion_status, ingested_at,
                research_status, retry_eligible, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 1,
                       'discovered', ?15, ?16, ?17, 'pending', 1, ?15)",
            params![
                identity.canonical_id,
                identity.canonical_url,
                title_str,
                norm_title,
                identity.provider_id,
                identity.doi,
                identity.arxiv_id,
                identity.version,
                versions_json,
                identity.content_hash,
                identity.source_kind,
                domain,
                urls_json,
                conns_json,
                now,
                ingestion_status,
                ingested_at,
            ],
        )?;

        Self::query_tracked_source_by_id_conn(conn, &identity.canonical_id)?
            .context("Failed to load newly inserted tracked_source")
    }

    pub fn get_tracked_source(&self, canonical_id_or_url: &str) -> Result<Option<TrackedSourceRecord>> {
        let conn = self.conn.lock().unwrap();
        if let Some(rec) = Self::query_tracked_source_by_id_conn(&conn, canonical_id_or_url)? {
            return Ok(Some(rec));
        }
        let identity = crate::web::canonicalize_source_identity(
            canonical_id_or_url,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        if let Some(cid) = Self::resolve_existing_canonical_id_conn(&conn, &identity, "")? {
            return Self::query_tracked_source_by_id_conn(&conn, &cid);
        }
        Ok(None)
    }

    pub fn mark_source_ingested(
        &self,
        canonical_id_or_url: &str,
        title: Option<&str>,
        content: &str,
        version: Option<&str>,
    ) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rec = Self::discover_source_in_tx(
            &tx,
            canonical_id_or_url,
            title,
            None,
            None,
            version,
            Some(content),
            None,
            "frontier_ai_research_and_verification",
        )?;
        let now = Utc::now().to_rfc3339();
        let ver_to_hash = version.unwrap_or(&rec.version);
        let new_hash = crate::web::compute_content_hash(&format!("{}::{}", ver_to_hash, content));
        tx.execute(
            "UPDATE tracked_sources SET ingestion_status='ingested', ingested_at=?1, content_hash=?2, updated_at=?1 WHERE canonical_id=?3",
            params![now, new_hash, rec.canonical_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn mark_source_ingestion_failed(
        &self,
        canonical_id_or_url: &str,
        reason: &str,
        max_retries: usize,
        retry_backoff_seconds: u64,
    ) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rec = Self::discover_source_in_tx(
            &tx,
            canonical_id_or_url,
            None,
            None,
            None,
            None,
            None,
            None,
            "frontier_ai_research_and_verification",
        )?;
        let now = Utc::now().to_rfc3339();
        let new_failures = rec.failure_count + 1;
        let eligible = new_failures < max_retries.max(1);
        let next_retry = if eligible {
            Some((Utc::now() + chrono::Duration::seconds(retry_backoff_seconds as i64)).to_rfc3339())
        } else {
            None
        };
        tx.execute(
            "UPDATE tracked_sources SET
                ingestion_status='failed',
                failure_count=?1,
                last_failure_reason=?2,
                last_failed_at=?3,
                retry_eligible=?4,
                next_retry_at=?5,
                updated_at=?3
             WHERE canonical_id=?6",
            params![
                new_failures as i64,
                reason,
                now,
                if eligible { 1 } else { 0 },
                next_retry,
                rec.canonical_id,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Evaluates whether a tracked source is eligible for research investigation right now.
    /// Completed investigations are skipped unless a new version, new question, contradiction,
    /// or configured refresh interval justifies a revisit.
    pub fn evaluate_source_eligibility(
        rec: &TrackedSourceRecord,
        question: Option<&str>,
        refresh_interval_seconds: u64,
        lease_seconds: u64,
    ) -> (bool, Option<String>) {
        let now = Utc::now();

        if rec.research_status == "reserved" {
            if let Some(ref res_at) = rec.reserved_at {
                if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(res_at) {
                    let age = (now - parsed.with_timezone(&Utc)).num_seconds();
                    if age >= 0 && (age as u64) < lease_seconds.max(1) {
                        return (false, None);
                    }
                    return (true, Some("expired_reservation_lease".to_string()));
                }
            }
            return (false, None);
        }

        if rec.research_status == "pending" {
            return (true, Some("initial_investigation".to_string()));
        }

        if rec.research_status == "failed" {
            if !rec.retry_eligible {
                return (false, None);
            }
            if let Some(ref next_at) = rec.next_retry_at {
                if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(next_at) {
                    if now < parsed.with_timezone(&Utc) {
                        return (false, None);
                    }
                }
            }
            return (true, Some("retry_eligible_failure".to_string()));
        }

        // research_status == "completed":
        // Check the 4 revisit conditions:
        // 1. New version
        if rec.completed_version.as_deref() != Some(rec.version.as_str()) {
            return (
                true,
                Some(format!(
                    "new_version:{}->{}",
                    rec.completed_version.as_deref().unwrap_or("none"),
                    rec.version
                )),
            );
        }
        if let Some(ref reason) = rec.revisit_reason {
            if reason.starts_with("new_version:") {
                return (true, Some(reason.clone()));
            }
        }

        // 2. Contradiction
        if rec.has_contradiction {
            return (true, Some("contradiction".to_string()));
        }

        // 3. New question
        if let Some(q) = question.map(crate::web::normalize_title_for_dedup).filter(|s| !s.is_empty()) {
            if !rec.investigated_questions.contains(&q) {
                return (true, Some(format!("new_question:{}", q)));
            }
        }

        // 4. Configured refresh interval
        if refresh_interval_seconds > 0 {
            if let Some(ref comp_at) = rec.completed_at {
                if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(comp_at) {
                    let elapsed = (now - parsed.with_timezone(&Utc)).num_seconds();
                    if elapsed >= (refresh_interval_seconds as i64) {
                        return (true, Some("refresh_interval_elapsed".to_string()));
                    }
                }
            }
        }

        (false, None)
    }

    /// Atomically reserves a research investigation for `agent_id`.
    /// Concurrent agents attempting to reserve the same document (even via different equivalent URLs
    /// such as abstract, HTML, PDF, or download) never claim the same investigation.
    pub fn try_reserve_source_investigation(
        &self,
        raw_url_or_id: &str,
        agent_id: &str,
        title: Option<&str>,
        doi: Option<&str>,
        arxiv_id: Option<&str>,
        explicit_version: Option<&str>,
        content: Option<&str>,
        connector: Option<&str>,
        domain: &str,
        question: Option<&str>,
        refresh_interval_seconds: u64,
        lease_seconds: u64,
    ) -> Result<Option<TrackedSourceRecord>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        let rec = Self::discover_source_in_tx(
            &tx,
            raw_url_or_id,
            title,
            doi,
            arxiv_id,
            explicit_version,
            content,
            connector,
            domain,
        )?;

        let (eligible, reason) = Self::evaluate_source_eligibility(
            &rec,
            question,
            refresh_interval_seconds,
            lease_seconds,
        );
        if !eligible {
            tx.commit()?;
            return Ok(None);
        }

        let now = Utc::now().to_rfc3339();
        let token = format!("res-{}-{}-{}", rec.canonical_id, agent_id, Utc::now().timestamp_nanos_opt().unwrap_or(0));
        let changed = tx.execute(
            "UPDATE tracked_sources SET
                research_status = 'reserved',
                reserved_by = ?1,
                reservation_token = ?2,
                reserved_at = ?3,
                revisit_reason = ?4,
                updated_at = ?3
             WHERE canonical_id = ?5",
            params![agent_id, token, now, reason, rec.canonical_id],
        )?;
        if changed == 0 {
            tx.commit()?;
            return Ok(None);
        }

        let updated = Self::query_tracked_source_by_id_conn(&tx, &rec.canonical_id)?;
        tx.commit()?;
        Ok(updated)
    }

    pub fn complete_source_investigation(
        &self,
        canonical_id_or_url: &str,
        agent_id: &str,
        version_completed: Option<&str>,
        content_hash_completed: Option<&str>,
        question_investigated: Option<&str>,
    ) -> Result<TrackedSourceRecord> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        let rec = if let Some(r) = Self::query_tracked_source_by_id_conn(&tx, canonical_id_or_url)? {
            r
        } else {
            Self::discover_source_in_tx(
                &tx,
                canonical_id_or_url,
                None,
                None,
                None,
                version_completed,
                None,
                None,
                "frontier_ai_research_and_verification",
            )?
        };

        let now = Utc::now().to_rfc3339();
        let ver = version_completed
            .map(|v| v.to_string())
            .unwrap_or_else(|| rec.version.clone());
        let chash = content_hash_completed
            .map(|h| h.to_string())
            .unwrap_or_else(|| rec.content_hash.clone());
        let mut questions = rec.investigated_questions.clone();
        if let Some(q) = question_investigated
            .map(crate::web::normalize_title_for_dedup)
            .filter(|s| !s.is_empty())
        {
            if !questions.contains(&q) {
                questions.push(q);
            }
        }
        let mut versions_seen = rec.versions_seen.clone();
        if !versions_seen.contains(&ver) {
            versions_seen.push(ver.clone());
        }
        let q_json = serde_json::to_string(&questions)?;
        let v_json = serde_json::to_string(&versions_seen)?;

        tx.execute(
            "UPDATE tracked_sources SET
                version = ?1,
                versions_seen_json = ?2,
                content_hash = ?3,
                ingestion_status = 'ingested',
                ingested_at = COALESCE(ingested_at, ?4),
                research_status = 'completed',
                reserved_by = NULL,
                reservation_token = NULL,
                reserved_at = NULL,
                completed_at = ?4,
                completed_version = ?1,
                completed_content_hash = ?3,
                retry_eligible = 0,
                next_retry_at = NULL,
                revisit_reason = NULL,
                has_contradiction = 0,
                investigated_questions_json = ?5,
                updated_at = ?4
             WHERE canonical_id = ?6 AND (reserved_by IS NULL OR reserved_by = ?7 OR research_status != 'reserved')",
            params![ver, v_json, chash, now, q_json, rec.canonical_id, agent_id],
        )?;

        let out = Self::query_tracked_source_by_id_conn(&tx, &rec.canonical_id)?
            .context("Tracked source missing after completion")?;
        tx.commit()?;
        Ok(out)
    }

    pub fn fail_source_investigation(
        &self,
        canonical_id_or_url: &str,
        reason: &str,
        max_retries: usize,
        retry_backoff_seconds: u64,
    ) -> Result<TrackedSourceRecord> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        let rec = if let Some(r) = Self::query_tracked_source_by_id_conn(&tx, canonical_id_or_url)? {
            r
        } else {
            Self::discover_source_in_tx(
                &tx,
                canonical_id_or_url,
                None,
                None,
                None,
                None,
                None,
                None,
                "frontier_ai_research_and_verification",
            )?
        };

        let now = Utc::now().to_rfc3339();
        let new_failures = rec.failure_count + 1;
        let eligible = new_failures < max_retries.max(1);
        let next_retry = if eligible {
            Some((Utc::now() + chrono::Duration::seconds(retry_backoff_seconds as i64)).to_rfc3339())
        } else {
            None
        };

        tx.execute(
            "UPDATE tracked_sources SET
                research_status = 'failed',
                reserved_by = NULL,
                reservation_token = NULL,
                reserved_at = NULL,
                failure_count = ?1,
                last_failure_reason = ?2,
                last_failed_at = ?3,
                retry_eligible = ?4,
                next_retry_at = ?5,
                updated_at = ?3
             WHERE canonical_id = ?6",
            params![
                new_failures as i64,
                reason,
                now,
                if eligible { 1 } else { 0 },
                next_retry,
                rec.canonical_id,
            ],
        )?;

        let out = Self::query_tracked_source_by_id_conn(&tx, &rec.canonical_id)?
            .context("Tracked source missing after failure update")?;
        tx.commit()?;
        Ok(out)
    }

    pub fn flag_source_contradiction(
        &self,
        canonical_id_or_url: &str,
        contradiction_details: &str,
    ) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rec = if let Some(r) = Self::query_tracked_source_by_id_conn(&tx, canonical_id_or_url)? {
            r
        } else {
            Self::discover_source_in_tx(
                &tx,
                canonical_id_or_url,
                None,
                None,
                None,
                None,
                None,
                None,
                "frontier_ai_research_and_verification",
            )?
        };
        let now = Utc::now().to_rfc3339();
        tx.execute(
            "UPDATE tracked_sources SET
                has_contradiction = 1,
                contradiction_details = ?1,
                retry_eligible = 1,
                revisit_reason = 'contradiction',
                updated_at = ?2
             WHERE canonical_id = ?3",
            params![contradiction_details, now, rec.canonical_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Records source corroboration for a target (claim ID, research question, or synthesis).
    /// Copies of one paper reached through multiple routes (abstract, HTML, PDF, download, or
    /// different repository connectors) resolve to a single `canonical_source_id` and count ONCE;
    /// independent papers, datasets, or implementations count separately.
    pub fn record_source_corroboration(
        &self,
        target_id: &str,
        raw_url: &str,
        title: Option<&str>,
        doi: Option<&str>,
        arxiv_id: Option<&str>,
        version: Option<&str>,
        content: Option<&str>,
        connector: Option<&str>,
        domain: &str,
    ) -> Result<CorroborationSummary> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        let tracked = Self::discover_source_in_tx(
            &tx,
            raw_url,
            title,
            doi,
            arxiv_id,
            version,
            content,
            connector,
            domain,
        )?;
        let now = Utc::now().to_rfc3339();

        // Orientation-only sources (e.g. Wikipedia and domain wikis) are strictly for discovery
        // and reference harvesting, NEVER counted as primary corroboration evidence.
        let is_wiki_or_orientation = tracked.is_orientation_only
            || connector
                .map(|c| c.eq_ignore_ascii_case("wikipedia"))
                .unwrap_or(false)
            || raw_url.to_lowercase().contains("wikipedia.org/wiki/");
        if is_wiki_or_orientation {
            let _ = tx.execute(
                "UPDATE tracked_sources SET is_orientation_only = 1, ingestion_completeness = 'orientation_only', updated_at = ?1 WHERE canonical_id = ?2",
                params![now, tracked.canonical_id],
            );
            let summary = Self::compute_corroboration_summary_conn(&tx, target_id)?;
            tx.commit()?;
            return Ok(summary);
        }

        let existing_row: Option<(i64, String, String, i64)> = tx
            .query_row(
                "SELECT id, observed_routes_json, observed_connectors_json, route_count
                 FROM source_corroborations
                 WHERE target_id = ?1 AND canonical_source_id = ?2",
                params![target_id, tracked.canonical_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;

        if let Some((row_id, routes_str, conns_str, route_cnt)) = existing_row {
            let mut routes: Vec<String> = serde_json::from_str(&routes_str).unwrap_or_default();
            let mut conns: Vec<String> = serde_json::from_str(&conns_str).unwrap_or_default();
            let mut inc = 0i64;
            let raw_t = raw_url.trim().to_string();
            if !raw_t.is_empty() && !routes.contains(&raw_t) {
                routes.push(raw_t);
                inc = 1;
            }
            if let Some(c) = connector.map(|s| s.trim()).filter(|s| !s.is_empty()) {
                let cs = c.to_string();
                if !conns.contains(&cs) {
                    conns.push(cs);
                    inc = 1;
                }
            }
            let r_json = serde_json::to_string(&routes)?;
            let c_json = serde_json::to_string(&conns)?;
            tx.execute(
                "UPDATE source_corroborations SET observed_routes_json=?1, observed_connectors_json=?2, route_count=?3, recorded_at=?4 WHERE id=?5",
                params![r_json, c_json, route_cnt + inc, now, row_id],
            )?;
        } else {
            let routes = if raw_url.trim().is_empty() {
                vec![tracked.canonical_url.clone()]
            } else {
                vec![raw_url.trim().to_string()]
            };
            let conns: Vec<String> = connector
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .map(|c| vec![c.to_string()])
                .unwrap_or_default();
            let r_json = serde_json::to_string(&routes)?;
            let c_json = serde_json::to_string(&conns)?;
            tx.execute(
                "INSERT INTO source_corroborations (
                    target_id, canonical_source_id, source_kind, observed_routes_json,
                    observed_connectors_json, route_count, domain, recorded_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6, ?7)",
                params![
                    target_id,
                    tracked.canonical_id,
                    tracked.source_kind,
                    r_json,
                    c_json,
                    domain,
                    now,
                ],
            )?;
        }

        let summary = Self::compute_corroboration_summary_conn(&tx, target_id)?;

        // If target_id corresponds to a claim in `claims`, update its separate source_support attributes
        let support_status = if summary.total_independent_sources >= 2 {
            "corroborated"
        } else if summary.total_independent_sources == 1 {
            "single_source"
        } else {
            "unsupported"
        };
        let _ = tx.execute(
            "UPDATE claims SET source_support_status=?1, source_corroboration_count=?2, updated_at=?3 WHERE id=?4",
            params![
                support_status,
                summary.total_independent_sources as i64,
                now,
                target_id,
            ],
        );

        tx.commit()?;
        Ok(summary)
    }

    fn compute_corroboration_summary_conn(
        conn: &Connection,
        target_id: &str,
    ) -> Result<CorroborationSummary> {
        let mut stmt = conn.prepare(
            "SELECT canonical_source_id, source_kind, route_count
             FROM source_corroborations
             WHERE target_id = ?1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map([target_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })?;

        let mut canonical_source_ids = Vec::new();
        let mut independent_papers = 0usize;
        let mut independent_datasets = 0usize;
        let mut independent_implementations = 0usize;
        let mut total_route_observations = 0usize;

        for item in rows {
            let (cid, kind, rc) = item?;
            canonical_source_ids.push(cid);
            total_route_observations += (rc.max(1)) as usize;
            match kind.as_str() {
                "dataset" => independent_datasets += 1,
                "implementation" => independent_implementations += 1,
                _ => independent_papers += 1,
            }
        }

        let total_independent_sources =
            independent_papers + independent_datasets + independent_implementations;

        Ok(CorroborationSummary {
            target_id: target_id.to_string(),
            total_independent_sources,
            independent_papers,
            independent_datasets,
            independent_implementations,
            total_route_observations,
            canonical_source_ids,
        })
    }

    pub fn get_corroboration_summary(&self, target_id: &str) -> Result<CorroborationSummary> {
        let conn = self.conn.lock().unwrap();
        Self::compute_corroboration_summary_conn(&conn, target_id)
    }

    pub fn list_tracked_sources(&self, limit: usize) -> Result<Vec<TrackedSourceRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT canonical_id, canonical_url, title, normalized_title, provider_id, doi, arxiv_id,
                    version, versions_seen_json, content_hash, source_kind, domain,
                    observed_urls_json, observed_connectors_json, route_observation_count,
                    discovery_status, discovered_at, ingestion_status, ingested_at,
                    research_status, reserved_by, reservation_token, reserved_at,
                    completed_at, completed_version, completed_content_hash,
                    failure_count, last_failure_reason, last_failed_at,
                    retry_eligible, next_retry_at, revisit_reason,
                    has_contradiction, contradiction_details, investigated_questions_json, updated_at,
                    ingestion_completeness, full_text_format, topic, is_orientation_only, section_count,
                    equations_preserved, tables_found, low_fidelity, low_fidelity_reason
             FROM tracked_sources ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_tracked_source_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Records connector health and fetch telemetry across adapters.
    pub fn record_connector_telemetry(
        &self,
        telemetry: &[crate::web::ConnectorFetchTelemetry],
    ) -> Result<()> {
        if telemetry.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = Utc::now().to_rfc3339();
        for t in telemetry {
            let existing: Option<(i64, i64, i64, i64, Option<String>)> = tx
                .query_row(
                    "SELECT total_requests, total_successes, total_failures, avg_latency_ms, last_successful_fetch_at
                     FROM connector_health WHERE connector = ?1",
                    [&t.connector_name],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                )
                .optional()?;

            let (prev_req, prev_succ, prev_fail, prev_lat, prev_last_succ) =
                existing.unwrap_or((0, 0, 0, 0, None));
            let new_req = prev_req + 1;
            let new_succ = prev_succ + if t.succeeded { 1 } else { 0 };
            let new_fail = prev_fail + if t.succeeded { 0 } else { 1 };
            let new_avg_lat = if prev_req <= 0 {
                t.latency_ms as i64
            } else {
                ((prev_lat * prev_req) + (t.latency_ms as i64)) / new_req.max(1)
            };
            let last_succ = if t.succeeded {
                Some(now.clone())
            } else {
                prev_last_succ
            };
            let status = if t.succeeded {
                "healthy"
            } else if t
                .error
                .as_deref()
                .map(|e| e.contains("429") || e.to_lowercase().contains("rate limit"))
                .unwrap_or(false)
            {
                "rate_limited"
            } else if new_succ > 0 {
                "degraded"
            } else {
                "unreachable"
            };
            let is_orient = t.category == "orientation_only" || t.connector_name.eq_ignore_ascii_case("wikipedia");

            tx.execute(
                "INSERT INTO connector_health (
                    connector, category, status, last_successful_fetch_at, last_attempted_at,
                    last_error, total_requests, total_successes, total_failures,
                    discovered_count, ingested_count, full_text_count, abstract_only_count,
                    avg_latency_ms, rate_limit_ms, is_orientation_only, updated_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0, 0, 0, ?11, ?12, ?13, ?14)
                 ON CONFLICT(connector) DO UPDATE SET
                    category = excluded.category,
                    status = excluded.status,
                    last_successful_fetch_at = COALESCE(excluded.last_successful_fetch_at, connector_health.last_successful_fetch_at),
                    last_attempted_at = excluded.last_attempted_at,
                    last_error = excluded.last_error,
                    total_requests = excluded.total_requests,
                    total_successes = excluded.total_successes,
                    total_failures = excluded.total_failures,
                    discovered_count = connector_health.discovered_count + ?10,
                    avg_latency_ms = excluded.avg_latency_ms,
                    rate_limit_ms = excluded.rate_limit_ms,
                    is_orientation_only = excluded.is_orientation_only,
                    updated_at = excluded.updated_at",
                params![
                    t.connector_name,
                    t.category,
                    status,
                    last_succ,
                    now,
                    t.error,
                    new_req,
                    new_succ,
                    new_fail,
                    t.items_discovered as i64,
                    new_avg_lat,
                    t.rate_limit_ms as i64,
                    if is_orient { 1 } else { 0 },
                    now,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Stores an original document (full text or abstract) and its structured section chunks.
    /// Enforces the invariant that abstracts are stored strictly as `'abstract_only'` and
    /// are NEVER presented as full-text ingestion or allowed to overwrite an existing `'full_text'` record.
    pub fn store_ingestion_payload(
        &self,
        work: &crate::web::ResolvedResearchWork,
        payload: &crate::web::FullTextIngestionPayload,
        topic: Option<&str>,
        domain: &str,
    ) -> Result<TrackedSourceRecord> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let now = Utc::now().to_rfc3339();

        let primary_url = work
            .urls
            .first()
            .map(|s| s.as_str())
            .unwrap_or(&work.canonical_url);
        let latest_ver = work.versions.last().map(|s| s.as_str());
        let rec = Self::discover_source_in_tx(
            &tx,
            primary_url,
            Some(&work.title),
            work.doi.as_deref(),
            work.arxiv_id.as_deref(),
            latest_ver,
            Some(&payload.raw_original),
            Some(&payload.connector),
            domain,
        )?;

        let incoming_completeness = if payload.is_orientation_only {
            "orientation_only"
        } else if payload.ingestion_completeness == "full_text" {
            "full_text"
        } else if payload.ingestion_completeness == "metadata_only" {
            "metadata_only"
        } else {
            "abstract_only"
        };

        let content_kind = if incoming_completeness == "full_text" {
            "full_text"
        } else if incoming_completeness == "orientation_only" {
            "orientation"
        } else {
            "abstract"
        };

        let refs_json = serde_json::to_string(&payload.harvested_references).unwrap_or_else(|_| "[]".to_string());
        let payload_hash = crate::web::compute_content_hash(&payload.raw_original);
        let payload_bytes = payload.raw_original.len();
        tx.execute(
            "INSERT INTO source_originals (
                canonical_source_id, version, connector, content_kind, format,
                raw_content, abstract_text, content_hash, byte_len, harvested_references_json, ingested_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(canonical_source_id, version, content_kind) DO UPDATE SET
                connector = excluded.connector,
                format = excluded.format,
                raw_content = excluded.raw_content,
                abstract_text = excluded.abstract_text,
                content_hash = excluded.content_hash,
                byte_len = excluded.byte_len,
                harvested_references_json = excluded.harvested_references_json,
                ingested_at = excluded.ingested_at",
            params![
                rec.canonical_id,
                rec.version,
                payload.connector,
                content_kind,
                payload.format,
                payload.raw_original,
                payload.abstract_text,
                payload_hash,
                payload_bytes as i64,
                refs_json,
                now,
            ],
        )?;

        // Store structured sections when full text is extracted (or non-empty structured chunks)
        let mut stored_sections_count = rec.section_count;
        if incoming_completeness == "full_text" && !payload.sections.is_empty() {
            tx.execute(
                "DELETE FROM source_sections WHERE canonical_source_id = ?1 AND version = ?2",
                params![rec.canonical_id, rec.version],
            )?;
            for sec in &payload.sections {
                tx.execute(
                    "INSERT INTO source_sections (
                        canonical_source_id, version, connector, section_path, section_kind,
                        source_location, content, token_estimate, created_at
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![
                        rec.canonical_id,
                        rec.version,
                        payload.connector,
                        sec.section_path,
                        sec.section_kind,
                        sec.source_location,
                        sec.content,
                        sec.token_estimate as i64,
                        now,
                    ],
                )?;
            }
            stored_sections_count = payload.sections.len();
        }

        // Never downgrade a source that already has 'full_text' for the same version to 'abstract_only'
        let keep_existing_full_text =
            rec.ingestion_completeness == "full_text" && incoming_completeness != "full_text";
        let effective_completeness = if keep_existing_full_text {
            "full_text"
        } else {
            incoming_completeness
        };
        let effective_format = if keep_existing_full_text {
            rec.full_text_format.clone().unwrap_or_else(|| payload.format.clone())
        } else {
            payload.format.clone()
        };

        let effective_sections_found = if keep_existing_full_text {
            rec.section_count
        } else if payload.extraction_quality.sections_found > 0 {
            payload.extraction_quality.sections_found
        } else {
            stored_sections_count
        };
        let effective_equations = if keep_existing_full_text {
            rec.equations_preserved
        } else {
            payload.extraction_quality.equations_preserved
        };
        let effective_tables = if keep_existing_full_text {
            rec.tables_found
        } else {
            payload.extraction_quality.tables_found
        };
        let effective_low_fidelity = if keep_existing_full_text {
            rec.low_fidelity
        } else {
            payload.extraction_quality.low_fidelity
        };
        let effective_low_fidelity_reason = if keep_existing_full_text {
            rec.low_fidelity_reason.clone()
        } else {
            payload.extraction_quality.low_fidelity_reason.clone()
        };

        let clean_topic = topic
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .or(rec.topic.as_deref());

        tx.execute(
            "UPDATE tracked_sources SET
                ingestion_status = 'ingested',
                ingested_at = COALESCE(ingested_at, ?1),
                ingestion_completeness = ?2,
                full_text_format = ?3,
                topic = COALESCE(?4, topic),
                is_orientation_only = ?5,
                section_count = ?6,
                equations_preserved = ?7,
                tables_found = ?8,
                low_fidelity = ?9,
                low_fidelity_reason = ?10,
                updated_at = ?1
             WHERE canonical_id = ?11",
            params![
                now,
                effective_completeness,
                effective_format,
                clean_topic,
                if payload.is_orientation_only || rec.is_orientation_only { 1 } else { 0 },
                effective_sections_found as i64,
                effective_equations as i64,
                effective_tables as i64,
                if effective_low_fidelity { 1 } else { 0 },
                effective_low_fidelity_reason,
                rec.canonical_id,
            ],
        )?;

        // Update connector_health ingestion counters
        let is_ft = if incoming_completeness == "full_text" { 1i64 } else { 0i64 };
        let is_abs = if incoming_completeness == "abstract_only" { 1i64 } else { 0i64 };
        let _ = tx.execute(
            "INSERT INTO connector_health (
                connector, category, status, last_successful_fetch_at, last_attempted_at,
                total_requests, total_successes, total_failures, discovered_count,
                ingested_count, full_text_count, abstract_only_count, avg_latency_ms,
                rate_limit_ms, is_orientation_only, updated_at
             ) VALUES (?1, 'preprints', 'healthy', ?2, ?2, 1, 1, 0, 1, 1, ?3, ?4, 0, 0, ?5, ?2)
             ON CONFLICT(connector) DO UPDATE SET
                ingested_count = connector_health.ingested_count + 1,
                full_text_count = connector_health.full_text_count + ?3,
                abstract_only_count = connector_health.abstract_only_count + ?4,
                updated_at = ?2",
            params![
                payload.connector,
                now,
                is_ft,
                is_abs,
                if payload.is_orientation_only { 1 } else { 0 },
            ],
        )?;

        let updated = Self::query_tracked_source_by_id_conn(&tx, &rec.canonical_id)?
            .context("Tracked source missing after storing ingestion payload")?;
        tx.commit()?;
        Ok(updated)
    }

    /// Retrieves the most relevant structured full-text sections for a research query
    /// within `max_tokens`, preserving section hierarchy, equations, theorems, algorithms,
    /// experiments, limitations, and exact source locations while NEVER injecting whole papers.
    pub fn retrieve_relevant_source_sections(
        &self,
        query: &str,
        canonical_ids: &[String],
        max_tokens: usize,
    ) -> Result<Vec<StoredSourceSection>> {
        let conn = self.conn.lock().unwrap();
        let mut all_sections = Vec::new();

        if canonical_ids.is_empty() {
            let mut stmt = conn.prepare(
                "SELECT id, canonical_source_id, version, connector, section_path, section_kind,
                        source_location, content, token_estimate, created_at
                 FROM source_sections ORDER BY id DESC LIMIT 200",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok(StoredSourceSection {
                    id: r.get(0)?,
                    canonical_source_id: r.get(1)?,
                    version: r.get(2)?,
                    connector: r.get(3)?,
                    section_path: r.get(4)?,
                    section_kind: r.get(5)?,
                    source_location: r.get(6)?,
                    content: r.get(7)?,
                    token_estimate: r.get::<_, i64>(8)?.max(1) as usize,
                    created_at: r.get(9)?,
                })
            })?;
            for r in rows {
                all_sections.push(r?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, canonical_source_id, version, connector, section_path, section_kind,
                        source_location, content, token_estimate, created_at
                 FROM source_sections WHERE canonical_source_id = ?1 ORDER BY id ASC",
            )?;
            for cid in canonical_ids {
                let rows = stmt.query_map([cid], |r| {
                    Ok(StoredSourceSection {
                        id: r.get(0)?,
                        canonical_source_id: r.get(1)?,
                        version: r.get(2)?,
                        connector: r.get(3)?,
                        section_path: r.get(4)?,
                        section_kind: r.get(5)?,
                        source_location: r.get(6)?,
                        content: r.get(7)?,
                        token_estimate: r.get::<_, i64>(8)?.max(1) as usize,
                        created_at: r.get(9)?,
                    })
                })?;
                for r in rows {
                    all_sections.push(r?);
                }
            }
        }

        if all_sections.is_empty() {
            return Ok(Vec::new());
        }

        let keywords: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(|w| w.to_lowercase())
            .collect();

        let mut scored: Vec<(i64, StoredSourceSection)> = all_sections
            .into_iter()
            .map(|sec| {
                let text_low = format!("{} {}", sec.section_path, sec.content).to_lowercase();
                let kw_hits = keywords
                    .iter()
                    .filter(|kw| text_low.contains(kw.as_str()))
                    .count() as i64;
                let kind_bonus = match sec.section_kind.as_str() {
                    "theorem" | "equation" | "algorithm" => 6,
                    "method" | "experiment" | "assumption" => 5,
                    "limitation" => 4,
                    "hierarchy" => 2,
                    "reference" => 1,
                    _ => 1,
                };
                (kw_hits * 4 + kind_bonus, sec)
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));

        let budget = max_tokens.max(120);
        let mut used_tokens = 0usize;
        let mut selected = Vec::new();
        for (_score, mut sec) in scored {
            let est = sec.token_estimate.max(sec.content.len() / 4).max(1);
            if used_tokens + est > budget {
                let remaining_tokens = budget.saturating_sub(used_tokens);
                if remaining_tokens >= 60 && selected.is_empty() {
                    let max_chars = remaining_tokens * 4;
                    sec.content = sec.content.chars().take(max_chars).collect();
                    sec.token_estimate = remaining_tokens;
                    selected.push(sec);
                }
                continue;
            }
            used_tokens += est;
            selected.push(sec);
        }

        Ok(selected)
    }

    pub fn list_connector_health_conn(conn: &Connection) -> Result<Vec<ConnectorHealthRecord>> {
        let mut stmt = conn.prepare(
            "SELECT connector, category, status, last_successful_fetch_at, last_attempted_at,
                    last_error, total_requests, total_successes, total_failures,
                    discovered_count, ingested_count, full_text_count, abstract_only_count,
                    avg_latency_ms, rate_limit_ms, is_orientation_only
             FROM connector_health ORDER BY connector ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ConnectorHealthRecord {
                connector: r.get(0)?,
                category: r.get(1)?,
                status: r.get(2)?,
                last_successful_fetch_at: r.get(3)?,
                last_attempted_at: r.get(4)?,
                last_error: r.get(5)?,
                total_requests: r.get::<_, i64>(6)?.max(0) as usize,
                total_successes: r.get::<_, i64>(7)?.max(0) as usize,
                total_failures: r.get::<_, i64>(8)?.max(0) as usize,
                discovered_count: r.get::<_, i64>(9)?.max(0) as usize,
                ingested_count: r.get::<_, i64>(10)?.max(0) as usize,
                full_text_count: r.get::<_, i64>(11)?.max(0) as usize,
                abstract_only_count: r.get::<_, i64>(12)?.max(0) as usize,
                avg_latency_ms: r.get::<_, i64>(13)?.max(0) as u64,
                rate_limit_ms: r.get::<_, i64>(14)?.max(0) as u64,
                is_orientation_only: r.get::<_, i64>(15).unwrap_or(0) != 0,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_connector_health(&self) -> Result<Vec<ConnectorHealthRecord>> {
        let conn = self.conn.lock().unwrap();
        Self::list_connector_health_conn(&conn)
    }

    pub fn list_topic_coverage_conn(conn: &Connection) -> Result<Vec<TopicCoverageSummary>> {
        let mut stmt = conn.prepare(
            "SELECT COALESCE(NULLIF(topic, ''), domain),
                    COUNT(*),
                    SUM(CASE WHEN ingestion_status = 'ingested' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN ingestion_completeness = 'full_text' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN ingestion_completeness = 'abstract_only' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN is_orientation_only = 1 OR ingestion_completeness = 'orientation_only' THEN 1 ELSE 0 END)
             FROM tracked_sources
             GROUP BY COALESCE(NULLIF(topic, ''), domain)
             ORDER BY COUNT(*) DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(TopicCoverageSummary {
                topic: r.get(0)?,
                discovered_count: r.get::<_, i64>(1)?.max(0) as usize,
                ingested_count: r.get::<_, i64>(2)?.max(0) as usize,
                full_text_count: r.get::<_, i64>(3)?.max(0) as usize,
                abstract_only_count: r.get::<_, i64>(4)?.max(0) as usize,
                orientation_only_count: r.get::<_, i64>(5)?.max(0) as usize,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_source_tracking_diagnostics(&self, limit: usize) -> Result<SourceTrackingDiagnostics> {
        let conn = self.conn.lock().unwrap();
        let total_tracked_sources: i64 =
            conn.query_row("SELECT count(*) FROM tracked_sources", [], |r| r.get(0))?;
        let discovered_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE discovery_status='discovered'",
            [],
            |r| r.get(0),
        )?;
        let ingested_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE ingestion_status='ingested'",
            [],
            |r| r.get(0),
        )?;
        let full_text_ingested_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE ingestion_completeness='full_text'",
            [],
            |r| r.get(0),
        )?;
        let abstract_only_ingested_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE ingestion_status='ingested' AND ingestion_completeness='abstract_only'",
            [],
            |r| r.get(0),
        )?;
        let orientation_only_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE is_orientation_only=1 OR ingestion_completeness='orientation_only'",
            [],
            |r| r.get(0),
        )?;
        let total_stored_sections: i64 = conn
            .query_row("SELECT count(*) FROM source_sections", [], |r| r.get(0))
            .unwrap_or(0);
        let total_equations_preserved: i64 = conn
            .query_row("SELECT COALESCE(SUM(equations_preserved), 0) FROM tracked_sources", [], |r| r.get(0))
            .unwrap_or(0);
        let total_tables_found: i64 = conn
            .query_row("SELECT COALESCE(SUM(tables_found), 0) FROM tracked_sources", [], |r| r.get(0))
            .unwrap_or(0);
        let low_fidelity_count: i64 = conn
            .query_row("SELECT count(*) FROM tracked_sources WHERE low_fidelity=1", [], |r| r.get(0))
            .unwrap_or(0);
        let pending_research_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE research_status='pending'",
            [],
            |r| r.get(0),
        )?;
        let reserved_research_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE research_status='reserved'",
            [],
            |r| r.get(0),
        )?;
        let completed_research_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE research_status='completed'",
            [],
            |r| r.get(0),
        )?;
        let failed_research_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE research_status='failed'",
            [],
            |r| r.get(0),
        )?;
        let retry_eligible_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE retry_eligible=1",
            [],
            |r| r.get(0),
        )?;
        let contradiction_flagged_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE has_contradiction=1",
            [],
            |r| r.get(0),
        )?;
        let independent_papers_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE source_kind='paper' AND is_orientation_only=0",
            [],
            |r| r.get(0),
        )?;
        let independent_datasets_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE source_kind='dataset'",
            [],
            |r| r.get(0),
        )?;
        let independent_implementations_count: i64 = conn.query_row(
            "SELECT count(*) FROM tracked_sources WHERE source_kind='implementation'",
            [],
            |r| r.get(0),
        )?;
        let total_route_observations: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(route_observation_count), 0) FROM tracked_sources",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let duplicate_route_observations_deduped =
            (total_route_observations - total_tracked_sources).max(0) as usize;

        let connector_health = Self::list_connector_health_conn(&conn).unwrap_or_default();
        let topic_coverage = Self::list_topic_coverage_conn(&conn).unwrap_or_default();

        let mut stmt = conn.prepare(
            "SELECT canonical_id, canonical_url, title, normalized_title, provider_id, doi, arxiv_id,
                    version, versions_seen_json, content_hash, source_kind, domain,
                    observed_urls_json, observed_connectors_json, route_observation_count,
                    discovery_status, discovered_at, ingestion_status, ingested_at,
                    research_status, reserved_by, reservation_token, reserved_at,
                    completed_at, completed_version, completed_content_hash,
                    failure_count, last_failure_reason, last_failed_at,
                    retry_eligible, next_retry_at, revisit_reason,
                    has_contradiction, contradiction_details, investigated_questions_json, updated_at,
                    ingestion_completeness, full_text_format, topic, is_orientation_only, section_count,
                    equations_preserved, tables_found, low_fidelity, low_fidelity_reason
             FROM tracked_sources ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::map_tracked_source_row)?;
        let mut recent_sources = Vec::new();
        for r in rows {
            recent_sources.push(r?);
        }

        Ok(SourceTrackingDiagnostics {
            total_tracked_sources: total_tracked_sources as usize,
            discovered_count: discovered_count as usize,
            ingested_count: ingested_count as usize,
            full_text_ingested_count: full_text_ingested_count as usize,
            abstract_only_ingested_count: abstract_only_ingested_count as usize,
            orientation_only_count: orientation_only_count as usize,
            total_stored_sections: total_stored_sections as usize,
            total_equations_preserved: total_equations_preserved.max(0) as usize,
            total_tables_found: total_tables_found.max(0) as usize,
            low_fidelity_count: low_fidelity_count.max(0) as usize,
            pending_research_count: pending_research_count as usize,
            reserved_research_count: reserved_research_count as usize,
            completed_research_count: completed_research_count as usize,
            failed_research_count: failed_research_count as usize,
            retry_eligible_count: retry_eligible_count as usize,
            contradiction_flagged_count: contradiction_flagged_count as usize,
            independent_papers_count: independent_papers_count as usize,
            independent_datasets_count: independent_datasets_count as usize,
            independent_implementations_count: independent_implementations_count as usize,
            total_route_observations: total_route_observations as usize,
            duplicate_route_observations_deduped,
            connector_health,
            topic_coverage,
            recent_sources,
        })
    }

    // =========================================================================
    // MODEL PROVIDERS, SELECTION, EVALUATIONS, AND FRONTIER BUDGETS
    // =========================================================================

    /// Atomically checks all frontier budget quotas (concurrency, rolling 24h calls, per-task calls,
    /// retries, rolling 24h tokens, per-task tokens, rolling 24h spend with conservative unknown
    /// pricing, and anti-spending-loop consecutive failure limits) and inserts a 'reserved' ledger
    /// entry inside a SQLite `BEGIN IMMEDIATE` transaction before dispatch.
    ///
    /// Retries consume quota (`max_calls_per_24h`, tokens, spend).
    /// Restarts and parallel requests cannot bypass limits.
    pub fn try_reserve_frontier_attempt(
        &self,
        req: &FrontierBudgetReservationRequest,
        budget: &crate::config::FrontierBudgetConfig,
    ) -> Result<std::result::Result<FrontierBudgetLedgerRecord, String>> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        let now = Utc::now();
        let now_ms = now.timestamp_millis();
        let now_rfc = now.to_rfc3339();
        let cutoff_24h_ms = now_ms - 86_400_000;
        let stale_lease_cutoff_ms = now_ms - 300_000; // 5-minute lease for crashed in-flight calls

        // Reclaim only the active concurrency slot of any crashed in-flight reservation older than 5 minutes,
        // while keeping its attempt/token/spend consumption in the persistent ledger!
        tx.execute(
            "UPDATE frontier_budget_ledger
             SET status = 'aborted_after_dispatch',
                 completed_at = ?1,
                 error_text = 'Reservation lease expired across crash/restart; quota remains consumed'
             WHERE status = 'reserved' AND created_at_epoch_ms < ?2",
            params![now_rfc, stale_lease_cutoff_ms],
        )?;

        // 1. Check anti-spending-loop guard (consecutive failures for same task & prompt_hash)
        if !req.prompt_hash.is_empty() && budget.max_consecutive_failures_before_halt > 0 {
            let recent_prompt_failures: i64 = tx.query_row(
                "SELECT count(*) FROM frontier_budget_ledger
                 WHERE task_id = ?1 AND prompt_hash = ?2 AND status IN ('failed', 'aborted_after_dispatch')",
                params![req.task_id, req.prompt_hash],
                |r| r.get(0),
            )?;
            if recent_prompt_failures as usize >= budget.max_consecutive_failures_before_halt {
                return Ok(Err(format!(
                    "spending_loop_halted: task '{}' reached {} consecutive failures for identical prompt hash",
                    req.task_id, recent_prompt_failures
                )));
            }
        }

        // 2. Check active concurrency limit (default 1)
        let active_concurrency: i64 = tx.query_row(
            "SELECT count(*) FROM frontier_budget_ledger WHERE status = 'reserved'",
            [],
            |r| r.get(0),
        )?;
        if active_concurrency as usize >= budget.max_concurrency {
            return Ok(Err(format!(
                "concurrency_limit_exceeded: active_concurrency={} >= max_concurrency={}",
                active_concurrency, budget.max_concurrency
            )));
        }

        // 3. Check retry limit vs per-task call limit
        if req.is_retry {
            let retries_for_parent: i64 = tx.query_row(
                "SELECT count(*) FROM frontier_budget_ledger WHERE parent_call_id = ?1 AND is_retry = 1",
                params![req.parent_call_id],
                |r| r.get(0),
            )?;
            if retries_for_parent as usize >= budget.max_retries
                || req.attempt_number > budget.max_retries + 1
            {
                return Ok(Err(format!(
                    "retry_limit_exceeded: retries_used={} >= max_retries={} for call '{}'",
                    retries_for_parent, budget.max_retries, req.parent_call_id
                )));
            }
        } else {
            let task_initial_calls: i64 = tx.query_row(
                "SELECT count(*) FROM frontier_budget_ledger WHERE task_id = ?1 AND is_retry = 0",
                params![req.task_id],
                |r| r.get(0),
            )?;
            if task_initial_calls as usize >= budget.max_calls_per_task {
                return Ok(Err(format!(
                    "task_call_limit_exceeded: task '{}' used {}/{} frontier calls",
                    req.task_id, task_initial_calls, budget.max_calls_per_task
                )));
            }
        }

        // 4. Check rolling 24h call limit (Default: 3 attempts per 24h; retries consume quota!)
        let calls_24h: i64 = tx.query_row(
            "SELECT count(*) FROM frontier_budget_ledger WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        if calls_24h as usize >= budget.max_calls_per_24h {
            return Ok(Err(format!(
                "calls_per_24h_exceeded: used {}/{} frontier attempts in rolling 24h",
                calls_24h, budget.max_calls_per_24h
            )));
        }

        // 5. Check token limits (rolling 24h and per-task)
        let used_tokens_24h: i64 = tx.query_row(
            "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN reserved_tokens ELSE MAX(total_tokens, reserved_tokens) END), 0)
             FROM frontier_budget_ledger
             WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        if (used_tokens_24h as usize) + req.reserved_tokens > budget.max_tokens_per_24h {
            return Ok(Err(format!(
                "tokens_per_24h_exceeded: used={} + requested={} > max_tokens_per_24h={}",
                used_tokens_24h, req.reserved_tokens, budget.max_tokens_per_24h
            )));
        }

        let used_tokens_task: i64 = tx.query_row(
            "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN reserved_tokens ELSE MAX(total_tokens, reserved_tokens) END), 0)
             FROM frontier_budget_ledger
             WHERE task_id = ?1",
            params![req.task_id],
            |r| r.get(0),
        )?;
        if (used_tokens_task as usize) + req.reserved_tokens > budget.max_tokens_per_task {
            return Ok(Err(format!(
                "tokens_per_task_exceeded: task '{}' used={} + requested={} > max_tokens_per_task={}",
                req.task_id, used_tokens_task, req.reserved_tokens, budget.max_tokens_per_task
            )));
        }

        // 6. Enforce conservative unknown-price rule: unknown billing is never free
        let effective_estimated_spend_micros = if req.pricing_known && req.estimated_spend_micros > 0 {
            req.estimated_spend_micros
        } else {
            let token_est_micros = ((req.reserved_tokens as u64)
                .saturating_mul(budget.conservative_unknown_output_micros_per_1m))
                / 1_000_000;
            req.estimated_spend_micros
                .max(budget.conservative_unknown_call_micros)
                .max(token_est_micros)
                .max(1)
        };

        let used_spend_micros_24h: i64 = tx.query_row(
            "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN estimated_spend_micros ELSE MAX(actual_spend_micros, estimated_spend_micros) END), 0)
             FROM frontier_budget_ledger
             WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        if (used_spend_micros_24h.max(0) as u64) + effective_estimated_spend_micros
            > budget.max_spend_micros_per_24h
        {
            return Ok(Err(format!(
                "spend_per_24h_exceeded: used_micros={} + estimated_micros={} > max_spend_micros_per_24h={}",
                used_spend_micros_24h,
                effective_estimated_spend_micros,
                budget.max_spend_micros_per_24h
            )));
        }

        // 7. Atomically insert reservation before dispatch
        let seq: i64 = tx.query_row("SELECT count(*) + 1 FROM frontier_budget_ledger", [], |r| {
            r.get(0)
        })?;
        let reservation_id = format!("fres-{}-{}-a{}", now_ms, seq, req.attempt_number);

        tx.execute(
            "INSERT INTO frontier_budget_ledger (
                reservation_id, parent_call_id, task_id, caller_kind, caller_id,
                workload_kind, provider_id, model_id, attempt_number, is_retry,
                status, reserved_tokens, prompt_tokens, completion_tokens, total_tokens,
                estimated_spend_micros, actual_spend_micros, pricing_known, prompt_hash,
                created_at_epoch_ms, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'reserved', ?11, 0, 0, 0, ?12, 0, ?13, ?14, ?15, ?16)",
            params![
                reservation_id,
                req.parent_call_id,
                req.task_id,
                req.caller_kind,
                req.caller_id,
                req.workload_kind,
                req.provider_id,
                req.model_id,
                req.attempt_number as i64,
                if req.is_retry { 1i64 } else { 0i64 },
                req.reserved_tokens as i64,
                effective_estimated_spend_micros as i64,
                if req.pricing_known { 1i64 } else { 0i64 },
                req.prompt_hash,
                now_ms,
                now_rfc,
            ],
        )?;

        tx.commit()?;

        Ok(Ok(FrontierBudgetLedgerRecord {
            reservation_id,
            parent_call_id: req.parent_call_id.clone(),
            task_id: req.task_id.clone(),
            caller_kind: req.caller_kind.clone(),
            caller_id: req.caller_id.clone(),
            workload_kind: req.workload_kind.clone(),
            provider_id: req.provider_id.clone(),
            model_id: req.model_id.clone(),
            attempt_number: req.attempt_number,
            is_retry: req.is_retry,
            status: "reserved".to_string(),
            reserved_tokens: req.reserved_tokens,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            estimated_spend_micros: effective_estimated_spend_micros,
            actual_spend_micros: 0,
            pricing_known: req.pricing_known,
            prompt_hash: req.prompt_hash.clone(),
            created_at_epoch_ms: now_ms,
            created_at: now_rfc,
            completed_at: None,
            error_text: None,
        }))
    }

    /// Finalizes an in-flight frontier reservation, releasing its concurrency slot while preserving
    /// its call/token/spend consumption in the persistent ledger.
    pub fn finalize_frontier_reservation(
        &self,
        reservation_id: &str,
        succeeded: bool,
        prompt_tokens: usize,
        completion_tokens: usize,
        actual_spend_micros: u64,
        error_text: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now_rfc = Utc::now().to_rfc3339();
        let total_tokens = prompt_tokens + completion_tokens;
        let status = if succeeded { "completed" } else { "failed" };

        conn.execute(
            "UPDATE frontier_budget_ledger
             SET status = ?1,
                 prompt_tokens = ?2,
                 completion_tokens = ?3,
                 total_tokens = ?4,
                 actual_spend_micros = CASE WHEN pricing_known = 1 AND ?5 > 0 THEN ?5 ELSE MAX(?5, estimated_spend_micros) END,
                 completed_at = ?6,
                 error_text = ?7
             WHERE reservation_id = ?8",
            params![
                status,
                prompt_tokens as i64,
                completion_tokens as i64,
                total_tokens as i64,
                actual_spend_micros as i64,
                now_rfc,
                error_text,
                reservation_id
            ],
        )?;
        Ok(())
    }

    pub fn get_cached_frontier_response(&self, cache_key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let found: Option<String> = conn
            .query_row(
                "SELECT response_text FROM frontier_response_cache WHERE cache_key = ?1",
                params![cache_key],
                |r| r.get(0),
            )
            .optional()?;
        if found.is_some() {
            let now_rfc = Utc::now().to_rfc3339();
            let _ = conn.execute(
                "UPDATE frontier_response_cache
                 SET hit_count = hit_count + 1, last_hit_at = ?1
                 WHERE cache_key = ?2",
                params![now_rfc, cache_key],
            );
        }
        Ok(found)
    }

    pub fn put_cached_frontier_response(
        &self,
        cache_key: &str,
        provider_id: &str,
        model_id: &str,
        workload_kind: &str,
        response_text: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now();
        conn.execute(
            "INSERT INTO frontier_response_cache (
                cache_key, provider_id, model_id, workload_kind, response_text,
                prompt_tokens, completion_tokens, hit_count, created_at_epoch_ms, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, ?8, ?9)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_text = excluded.response_text,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens",
            params![
                cache_key,
                provider_id,
                model_id,
                workload_kind,
                response_text,
                prompt_tokens as i64,
                completion_tokens as i64,
                now.timestamp_millis(),
                now.to_rfc3339()
            ],
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_frontier_blocked_task(
        &self,
        task_id: &str,
        caller_kind: &str,
        caller_id: &str,
        workload_kind: &str,
        requested_provider: &str,
        requested_model: &str,
        exhaustion_reason: &str,
        exhaustion_action: &str,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let status = match exhaustion_action {
            "queued" => "queued",
            "continued_locally" => "continued_locally",
            _ => "blocked",
        };
        conn.execute(
            "INSERT INTO frontier_blocked_tasks (
                task_id, caller_kind, caller_id, workload_kind, requested_provider,
                requested_model, exhaustion_reason, exhaustion_action, status, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)",
            params![
                task_id,
                caller_kind,
                caller_id,
                workload_kind,
                requested_provider,
                requested_model,
                exhaustion_reason,
                exhaustion_action,
                status,
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_frontier_blocked_tasks(&self, limit: usize) -> Result<Vec<FrontierBlockedTaskRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, task_id, caller_kind, caller_id, workload_kind, requested_provider,
                    requested_model, exhaustion_reason, exhaustion_action, status, created_at, updated_at
             FROM frontier_blocked_tasks
             ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(FrontierBlockedTaskRecord {
                id: r.get(0)?,
                task_id: r.get(1)?,
                caller_kind: r.get(2)?,
                caller_id: r.get(3)?,
                workload_kind: r.get(4)?,
                requested_provider: r.get(5)?,
                requested_model: r.get(6)?,
                exhaustion_reason: r.get(7)?,
                exhaustion_action: r.get(8)?,
                status: r.get(9)?,
                created_at: r.get(10)?,
                updated_at: r.get(11)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn get_frontier_budget_summary(
        &self,
        budget: &crate::config::FrontierBudgetConfig,
        ledger_limit: usize,
    ) -> Result<FrontierBudgetSummary> {
        let conn = self.conn.lock().unwrap();
        let now_ms = Utc::now().timestamp_millis();
        let cutoff_24h_ms = now_ms - 86_400_000;

        let used_calls_24h: i64 = conn.query_row(
            "SELECT count(*) FROM frontier_budget_ledger WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        let active_concurrency: i64 = conn.query_row(
            "SELECT count(*) FROM frontier_budget_ledger WHERE status = 'reserved'",
            [],
            |r| r.get(0),
        )?;
        let used_tokens_24h: i64 = conn.query_row(
            "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN reserved_tokens ELSE MAX(total_tokens, reserved_tokens) END), 0)
             FROM frontier_budget_ledger WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        let used_spend_micros_24h: i64 = conn.query_row(
            "SELECT COALESCE(SUM(CASE WHEN status = 'reserved' THEN estimated_spend_micros ELSE MAX(actual_spend_micros, estimated_spend_micros) END), 0)
             FROM frontier_budget_ledger WHERE created_at_epoch_ms >= ?1",
            params![cutoff_24h_ms],
            |r| r.get(0),
        )?;
        let cache_hits_total: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(hit_count), 0) FROM frontier_response_cache",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let total_historical_ledger_entries: i64 = conn
            .query_row("SELECT count(*) FROM frontier_budget_ledger", [], |r| r.get(0))
            .unwrap_or(0);

        let mut stmt = conn.prepare(
            "SELECT reservation_id, parent_call_id, task_id, caller_kind, caller_id,
                    workload_kind, provider_id, model_id, attempt_number, is_retry,
                    status, reserved_tokens, prompt_tokens, completion_tokens, total_tokens,
                    estimated_spend_micros, actual_spend_micros, pricing_known, prompt_hash,
                    created_at_epoch_ms, created_at, completed_at, error_text
             FROM frontier_budget_ledger
             ORDER BY created_at_epoch_ms DESC, rowid DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([ledger_limit as i64], |r| {
            Ok(FrontierBudgetLedgerRecord {
                reservation_id: r.get(0)?,
                parent_call_id: r.get(1)?,
                task_id: r.get(2)?,
                caller_kind: r.get(3)?,
                caller_id: r.get(4)?,
                workload_kind: r.get(5)?,
                provider_id: r.get(6)?,
                model_id: r.get(7)?,
                attempt_number: r.get::<_, i64>(8)? as usize,
                is_retry: r.get::<_, i64>(9)? != 0,
                status: r.get(10)?,
                reserved_tokens: r.get::<_, i64>(11)? as usize,
                prompt_tokens: r.get::<_, i64>(12)? as usize,
                completion_tokens: r.get::<_, i64>(13)? as usize,
                total_tokens: r.get::<_, i64>(14)? as usize,
                estimated_spend_micros: r.get::<_, i64>(15)?.max(0) as u64,
                actual_spend_micros: r.get::<_, i64>(16)?.max(0) as u64,
                pricing_known: r.get::<_, i64>(17)? != 0,
                prompt_hash: r.get(18)?,
                created_at_epoch_ms: r.get(19)?,
                created_at: r.get(20)?,
                completed_at: r.get(21)?,
                error_text: r.get(22)?,
            })
        })?;
        let mut recent_ledger = Vec::new();
        for r in rows {
            recent_ledger.push(r?);
        }

        drop(stmt);
        drop(conn);
        let blocked_tasks = self.list_frontier_blocked_tasks(ledger_limit)?;

        let used_c = used_calls_24h.max(0) as usize;
        let used_t = used_tokens_24h.max(0) as usize;
        let used_s = used_spend_micros_24h.max(0) as u64;

        Ok(FrontierBudgetSummary {
            max_calls_per_24h: budget.max_calls_per_24h,
            used_calls_24h: used_c,
            remaining_calls_24h: budget.max_calls_per_24h.saturating_sub(used_c),
            max_calls_per_task: budget.max_calls_per_task,
            max_concurrency: budget.max_concurrency,
            active_concurrency: active_concurrency.max(0) as usize,
            max_retries: budget.max_retries,
            max_tokens_per_24h: budget.max_tokens_per_24h,
            used_tokens_24h: used_t,
            remaining_tokens_24h: budget.max_tokens_per_24h.saturating_sub(used_t),
            max_tokens_per_task: budget.max_tokens_per_task,
            max_spend_micros_per_24h: budget.max_spend_micros_per_24h,
            used_spend_micros_24h: used_s,
            remaining_spend_micros_24h: budget.max_spend_micros_per_24h.saturating_sub(used_s),
            exhaustion_policy: budget.exhaustion_policy.clone(),
            cache_hits_total: cache_hits_total.max(0) as usize,
            total_historical_ledger_entries: total_historical_ledger_entries.max(0) as usize,
            recent_ledger,
            blocked_tasks,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_model_evaluation(
        &self,
        provider_id: &str,
        model_id: &str,
        workload_kind: &str,
        succeeded: bool,
        verification_passed: bool,
        calibration_score: f64,
        latency_ms: u64,
        notes: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let existing: Option<(i64, f64, f64, f64, i64)> = conn
            .query_row(
                "SELECT eval_runs, success_rate, verification_pass_rate, calibration_score, avg_latency_ms
                 FROM model_evaluations
                 WHERE provider_id = ?1 AND model_id = ?2 AND workload_kind = ?3",
                params![provider_id, model_id, workload_kind],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .optional()?;

        let s_val = if succeeded { 1.0 } else { 0.0 };
        let v_val = if verification_passed { 1.0 } else { 0.0 };

        if let Some((runs, prev_s, prev_v, prev_c, prev_lat)) = existing {
            let new_runs = runs + 1;
            let n_f = new_runs as f64;
            let r_f = runs as f64;
            let new_s = (prev_s * r_f + s_val) / n_f;
            let new_v = (prev_v * r_f + v_val) / n_f;
            let new_c = (prev_c * r_f + calibration_score) / n_f;
            let new_lat = ((prev_lat as u64) * (runs as u64) + latency_ms) / (new_runs as u64);
            conn.execute(
                "UPDATE model_evaluations
                 SET eval_runs = ?1,
                     success_rate = ?2,
                     verification_pass_rate = ?3,
                     calibration_score = ?4,
                     avg_latency_ms = ?5,
                     notes = ?6,
                     updated_at = ?7
                 WHERE provider_id = ?8 AND model_id = ?9 AND workload_kind = ?10",
                params![
                    new_runs,
                    new_s,
                    new_v,
                    new_c,
                    new_lat as i64,
                    notes,
                    now,
                    provider_id,
                    model_id,
                    workload_kind
                ],
            )?;
        } else {
            conn.execute(
                "INSERT INTO model_evaluations (
                    provider_id, model_id, workload_kind, eval_runs, success_rate,
                    verification_pass_rate, calibration_score, avg_latency_ms, notes, updated_at
                ) VALUES (?1, ?2, ?3, 1, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    provider_id,
                    model_id,
                    workload_kind,
                    s_val,
                    v_val,
                    calibration_score,
                    latency_ms as i64,
                    notes,
                    now
                ],
            )?;
        }
        Ok(())
    }

    pub fn list_model_evaluations(&self) -> Result<Vec<ModelEvaluationRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT provider_id, model_id, workload_kind, eval_runs, success_rate,
                    verification_pass_rate, calibration_score, avg_latency_ms, notes, updated_at
             FROM model_evaluations
             ORDER BY verification_pass_rate DESC, success_rate DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ModelEvaluationRecord {
                provider_id: r.get(0)?,
                model_id: r.get(1)?,
                workload_kind: r.get(2)?,
                eval_runs: r.get::<_, i64>(3)? as usize,
                success_rate: r.get(4)?,
                verification_pass_rate: r.get(5)?,
                calibration_score: r.get(6)?,
                avg_latency_ms: r.get::<_, i64>(7)?.max(0) as u64,
                notes: r.get(8)?,
                updated_at: r.get(9)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn upsert_provider_discovery(&self, rec: &ModelProviderDiscoveryRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let models_json =
            serde_json::to_string(&rec.discovered_models).unwrap_or_else(|_| "[]".to_string());
        conn.execute(
            "INSERT INTO model_provider_discovery (
                provider_id, provider_type, base_url, transport, credential_status,
                discovered_models_json, health_status, last_refreshed_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(provider_id) DO UPDATE SET
                provider_type = excluded.provider_type,
                base_url = excluded.base_url,
                transport = excluded.transport,
                credential_status = excluded.credential_status,
                discovered_models_json = excluded.discovered_models_json,
                health_status = excluded.health_status,
                last_refreshed_at = excluded.last_refreshed_at",
            params![
                rec.provider_id,
                rec.provider_type,
                rec.base_url,
                rec.transport,
                rec.credential_status,
                models_json,
                rec.health_status,
                rec.last_refreshed_at
            ],
        )?;
        Ok(())
    }

    pub fn list_provider_discoveries(&self) -> Result<Vec<ModelProviderDiscoveryRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT provider_id, provider_type, base_url, transport, credential_status,
                    discovered_models_json, health_status, last_refreshed_at
             FROM model_provider_discovery
             ORDER BY provider_id ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            let models_json: String = r.get(5)?;
            let discovered_models: Vec<String> =
                serde_json::from_str(&models_json).unwrap_or_default();
            Ok(ModelProviderDiscoveryRecord {
                provider_id: r.get(0)?,
                provider_type: r.get(1)?,
                base_url: r.get(2)?,
                transport: r.get(3)?,
                credential_status: r.get(4)?,
                discovered_models,
                health_status: r.get(6)?,
                last_refreshed_at: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn record_model_selection(&self, rec: &ModelSelectionRecord) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let now = if rec.created_at.is_empty() {
            Utc::now().to_rfc3339()
        } else {
            rec.created_at.clone()
        };
        conn.execute(
            "INSERT INTO model_selection_log (
                task_id, caller_kind, caller_id, workload_kind, selected_provider,
                selected_model, is_frontier, composite_score, ranking_basis,
                fallback_reason, used_preference_order_fallback, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                rec.task_id,
                rec.caller_kind,
                rec.caller_id,
                rec.workload_kind,
                rec.selected_provider,
                rec.selected_model,
                if rec.is_frontier { 1i64 } else { 0i64 },
                rec.composite_score,
                rec.ranking_basis,
                rec.fallback_reason,
                if rec.used_preference_order_fallback {
                    1i64
                } else {
                    0i64
                },
                now
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_recent_model_selections(&self, limit: usize) -> Result<Vec<ModelSelectionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, task_id, caller_kind, caller_id, workload_kind, selected_provider,
                    selected_model, is_frontier, composite_score, ranking_basis,
                    fallback_reason, used_preference_order_fallback, created_at
             FROM model_selection_log
             ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(ModelSelectionRecord {
                id: r.get(0)?,
                task_id: r.get(1)?,
                caller_kind: r.get(2)?,
                caller_id: r.get(3)?,
                workload_kind: r.get(4)?,
                selected_provider: r.get(5)?,
                selected_model: r.get(6)?,
                is_frontier: r.get::<_, i64>(7)? != 0,
                composite_score: r.get(8)?,
                ranking_basis: r.get(9)?,
                fallback_reason: r.get(10)?,
                used_preference_order_fallback: r.get::<_, i64>(11)? != 0,
                created_at: r.get(12)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

pub fn compute_claim_id(statement: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(statement.trim().as_bytes());
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecutionAdapter;

    #[test]
    fn test_store_votes_and_consensus() {
        let store = Store::open(Path::new(":memory:")).unwrap();
        let claim_id = store.propose_claim("Earth orbits the Sun", "https://wiki/Earth", "Explorer").unwrap();

        // 1st endorsement: not trusted yet
        let promoted1 = store.record_vote(&claim_id, "Explorer", "endorse", 1, Some("Confirmed")).unwrap();
        assert!(!promoted1);
        let c1 = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(c1.status, "unresolved");

        // Duplicate endorsement by same agent does not count twice
        let dup = store.record_vote(&claim_id, "Explorer", "endorse", 1, Some("Again")).unwrap();
        assert!(!dup);

        // Dissent from Sceptic in same cycle
        store.record_vote(&claim_id, "Sceptic", "reject", 1, Some("Needs proof")).unwrap();
        let c2 = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(c2.status, "unresolved");

        // 2nd distinct agent Builder endorses in same cycle -> promotes to trusted!
        let promoted2 = store.record_vote(&claim_id, "Builder", "endorse", 1, Some("Agreed")).unwrap();
        assert!(promoted2);
        let c3 = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(c3.status, "trusted");

        // Check votes history has 3 votes recorded
        let votes = store.get_claim_votes(&claim_id).unwrap();
        assert_eq!(votes.len(), 3);
    }

    #[test]
    fn test_user_questions_bounded() {
        let store = Store::open(Path::new(":memory:")).unwrap();
        assert!(store.add_user_question("Explorer", "Q1?", 2).unwrap());
        assert!(store.add_user_question("Builder", "Q2?", 2).unwrap());
        // 3rd question today should fail
        assert!(!store.add_user_question("Sceptic", "Q3?", 2).unwrap());
    }

    #[test]
    fn test_domain_namespace_isolation() {
        let store = Store::open(Path::new(":memory:")).unwrap();
        let id_a = store
            .propose_claim_in_domain(
                "HDDL 2.1 defines hierarchical task network temporal semantics.",
                "https://arxiv.org/abs/2306.07353",
                "Explorer",
                "frontier_ai_research_and_verification",
            )
            .unwrap();
        store.record_vote(&id_a, "Explorer", "endorse", 1, None).unwrap();
        store.record_vote(&id_a, "Builder", "endorse", 1, None).unwrap();

        let id_b = store
            .propose_claim_in_domain(
                "Cardiology guideline on beta blocker titration.",
                "https://example.org/cardio",
                "Explorer",
                "clinical_cardiology",
            )
            .unwrap();
        store.record_vote(&id_b, "Explorer", "endorse", 1, None).unwrap();
        store.record_vote(&id_b, "Sceptic", "endorse", 1, None).unwrap();

        let ai_claims = store
            .list_substantive_learnings_in_domain("frontier_ai_research_and_verification", 10)
            .unwrap();
        assert_eq!(ai_claims.len(), 1);
        assert!(ai_claims[0].statement.contains("HDDL 2.1"));

        let cardio_claims = store
            .list_substantive_learnings_in_domain("clinical_cardiology", 10)
            .unwrap();
        assert_eq!(cardio_claims.len(), 1);
        assert!(cardio_claims[0].statement.contains("Cardiology"));

        let empty_domain = store
            .list_substantive_learnings_in_domain("quantum_chemistry", 10)
            .unwrap();
        assert!(empty_domain.is_empty());
    }

    #[test]
    fn test_consultation_lifecycle_and_feedback_never_promotes_claim() {
        let store = Store::open(Path::new(":memory:")).unwrap();
        store
            .create_consultation(
                "cons-1",
                "frontier_ai_research_and_verification",
                "consult_expert",
                "solve_problem",
                Some("consult_expert"),
                Some("Requires executable verification"),
                "deep",
                "running",
                "Initialized",
                "{}",
            )
            .unwrap();
        store
            .set_consultation_awaiting_input(
                "cons-1",
                "What is your target latency budget in ms?",
                "Awaiting clarification on latency budget",
            )
            .unwrap();
        let c = store.get_consultation("cons-1").unwrap().unwrap();
        assert_eq!(c.state, "awaiting_input");
        assert_eq!(
            c.pending_question.as_deref(),
            Some("What is your target latency budget in ms?")
        );

        store
            .record_consultation_clarification("cons-1", "50ms p99 on 16GB VRAM")
            .unwrap();
        let c2 = store.get_consultation("cons-1").unwrap().unwrap();
        assert_eq!(c2.state, "running");
        assert!(c2.pending_question.is_none());

        store
            .complete_consultation("cons-1", "{\"conclusion\":\"ok\"}", "[]", "Completed")
            .unwrap();
        let c3 = store.get_consultation("cons-1").unwrap().unwrap();
        assert_eq!(c3.state, "completed");

        let claims_before = store.list_learnings(100).unwrap().len();
        store
            .record_outcome_feedback(
                "cons-1",
                "frontier_ai_research_and_verification",
                "agent-client",
                "Applied leverage sampling",
                "Reduced query count by 38%",
                true,
                "None",
                "{\"queries_saved_pct\": 38.0}",
            )
            .unwrap();
        let claims_after = store.list_learnings(100).unwrap().len();
        assert_eq!(claims_before, claims_after);
        let fb = store.list_outcome_feedback_for_consultation("cons-1").unwrap();
        assert_eq!(fb.len(), 1);
        assert_eq!(fb[0].status, "attributable_evaluation_evidence");
    }

    #[test]
    fn test_equivalent_urls_dedupe_across_agents_and_restarts() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_source_test.db");

        // Phase 1: Explorer discovers and atomically reserves via abstract URL;
        // Sceptic and Builder concurrently attempt to reserve via PDF, HTML, and download URLs.
        {
            let store = Store::open(&db_path).unwrap();

            let res_explorer = store
                .try_reserve_source_investigation(
                    "https://arxiv.org/abs/2305.10601v1",
                    "Explorer",
                    Some("Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
                    None,
                    Some("2305.10601v1"),
                    Some("v1"),
                    Some("Abstract content v1"),
                    Some("arxiv"),
                    "frontier_ai_research_and_verification",
                    Some("How does Tree of Thoughts scale search?"),
                    86400,
                    1800,
                )
                .unwrap();
            assert!(
                res_explorer.is_some(),
                "First agent (Explorer) should atomically reserve the investigation"
            );
            let reserved = res_explorer.unwrap();
            assert_eq!(reserved.canonical_id, "arxiv:2305.10601");
            assert_eq!(reserved.research_status, "reserved");
            assert_eq!(reserved.reserved_by.as_deref(), Some("Explorer"));

            // Concurrent agent Sceptic tries to claim the same paper via PDF download URL
            let res_sceptic = store
                .try_reserve_source_investigation(
                    "https://arxiv.org/pdf/2305.10601v1.pdf?download=1",
                    "Sceptic",
                    Some("Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
                    None,
                    None,
                    None,
                    None,
                    Some("arxiv"),
                    "frontier_ai_research_and_verification",
                    Some("How does Tree of Thoughts scale search?"),
                    86400,
                    1800,
                )
                .unwrap();
            assert!(
                res_sceptic.is_none(),
                "Concurrent agent (Sceptic) must never claim the same reserved investigation via PDF URL"
            );

            // Concurrent agent Builder tries to claim the same paper via HTML URL
            let res_builder = store
                .try_reserve_source_investigation(
                    "https://arxiv.org/html/2305.10601v1",
                    "Builder",
                    Some("Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
                    Some("10.48550/arXiv.2305.10601"),
                    None,
                    None,
                    None,
                    Some("semantic_scholar"),
                    "frontier_ai_research_and_verification",
                    Some("How does Tree of Thoughts scale search?"),
                    86400,
                    1800,
                )
                .unwrap();
            assert!(
                res_builder.is_none(),
                "Concurrent agent (Builder) must never claim the same reserved investigation via HTML URL"
            );

            // Complete the investigation as Explorer
            let completed = store
                .complete_source_investigation(
                    "https://arxiv.org/e-print/2305.10601v1",
                    "Explorer",
                    Some("v1"),
                    None,
                    Some("How does Tree of Thoughts scale search?"),
                )
                .unwrap();
            assert_eq!(completed.research_status, "completed");
            assert_eq!(completed.completed_version.as_deref(), Some("v1"));
        }

        // Phase 2: Reopen Store from disk (simulated restart) and verify deduplication & skip state persist
        {
            let store_restarted = Store::open(&db_path).unwrap();
            let all_sources = store_restarted.list_tracked_sources(50).unwrap();
            assert_eq!(
                all_sources.len(),
                1,
                "All equivalent URLs (abs, pdf, html, e-print) must dedupe to a single persisted source across restarts"
            );
            let tracked = &all_sources[0];
            assert_eq!(tracked.canonical_id, "arxiv:2305.10601");
            assert_eq!(tracked.research_status, "completed");
            assert_eq!(tracked.completed_version.as_deref(), Some("v1"));
            assert!(tracked.observed_urls.len() >= 3);

            // Attempting to reserve the completed v1 paper for the same question after restart must be skipped
            let res_after_restart = store_restarted
                .try_reserve_source_investigation(
                    "https://ar5iv.labs.arxiv.org/html/2305.10601v1",
                    "Sceptic",
                    Some("Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
                    None,
                    None,
                    Some("v1"),
                    None,
                    Some("openalex"),
                    "frontier_ai_research_and_verification",
                    Some("How does Tree of Thoughts scale search?"),
                    86400,
                    1800,
                )
                .unwrap();
            assert!(
                res_after_restart.is_none(),
                "Completed investigation must be skipped across restarts when version and question are unchanged"
            );
        }
    }

    #[test]
    fn test_new_document_versions_stay_eligible() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_version_test.db");
        let store = Store::open(&db_path).unwrap();

        // 1. Discover, reserve, and complete v1
        let reserved_v1 = store
            .try_reserve_source_investigation(
                "https://arxiv.org/abs/2401.99999v1",
                "Explorer",
                Some("Scaling Verification Laws"),
                None,
                Some("2401.99999v1"),
                Some("v1"),
                Some("Initial v1 abstract"),
                Some("arxiv"),
                "frontier_ai_research_and_verification",
                Some("Scaling verification laws"),
                86400,
                1800,
            )
            .unwrap()
            .expect("v1 should be reserved");
        assert_eq!(reserved_v1.version, "v1");

        store
            .complete_source_investigation(
                &reserved_v1.canonical_id,
                "Explorer",
                Some("v1"),
                None,
                Some("Scaling verification laws"),
            )
            .unwrap();

        // 2. Re-encountering v1 (even via PDF URL) is skipped
        let skip_v1 = store
            .try_reserve_source_investigation(
                "https://arxiv.org/pdf/2401.99999v1.pdf",
                "Builder",
                Some("Scaling Verification Laws"),
                None,
                Some("2401.99999v1"),
                Some("v1"),
                Some("Initial v1 abstract"),
                Some("arxiv"),
                "frontier_ai_research_and_verification",
                Some("Scaling verification laws"),
                86400,
                1800,
            )
            .unwrap();
        assert!(skip_v1.is_none(), "Same version v1 must be skipped");

        // 3. Discovering v2 makes the document eligible again!
        let reserved_v2 = store
            .try_reserve_source_investigation(
                "https://arxiv.org/pdf/2401.99999v2.pdf",
                "Builder",
                Some("Scaling Verification Laws"),
                None,
                Some("2401.99999v2"),
                Some("v2"),
                Some("Updated v2 abstract with new theorem"),
                Some("arxiv"),
                "frontier_ai_research_and_verification",
                Some("Scaling verification laws"),
                86400,
                1800,
            )
            .unwrap();
        assert!(
            reserved_v2.is_some(),
            "New document version v2 must stay eligible for investigation"
        );
        let rec_v2 = reserved_v2.unwrap();
        assert_eq!(rec_v2.canonical_id, "arxiv:2401.99999");
        assert_eq!(rec_v2.version, "v2");
        assert_eq!(
            rec_v2.revisit_reason.as_deref(),
            Some("new_version:v1->v2")
        );

        // Complete v2
        store
            .complete_source_investigation(
                &rec_v2.canonical_id,
                "Builder",
                Some("v2"),
                None,
                Some("Scaling verification laws"),
            )
            .unwrap();

        // 4. Also verify that contradiction or new question justifies revisit
        let reserved_new_q = store
            .try_reserve_source_investigation(
                "https://arxiv.org/abs/2401.99999v2",
                "Sceptic",
                Some("Scaling Verification Laws"),
                None,
                None,
                Some("v2"),
                None,
                Some("arxiv"),
                "frontier_ai_research_and_verification",
                Some("What are the failure modes of process reward models under distribution shift?"),
                86400,
                1800,
            )
            .unwrap();
        assert!(
            reserved_new_q.is_some(),
            "A new question on a completed source must justify a revisit"
        );
    }

    #[test]
    fn test_one_paper_three_routes_counts_once_two_independent_papers_count_twice() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_corroboration_test.db");
        let store = Store::open(&db_path).unwrap();

        let claim_id = store
            .propose_claim(
                "Process reward supervision outperforms outcome-only supervision on multi-step math.",
                "https://arxiv.org/abs/2305.20050",
                "Explorer",
            )
            .unwrap();

        // Paper 1 reached through THREE different routes (arXiv abstract, arXiv PDF download, OpenAlex with arXiv DOI)
        store
            .record_source_corroboration(
                &claim_id,
                "https://arxiv.org/abs/2305.20050v1",
                Some("Let's Verify Step by Step"),
                None,
                Some("2305.20050v1"),
                Some("v1"),
                Some("We train process reward models on PRM800K."),
                Some("arxiv"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        store
            .record_source_corroboration(
                &claim_id,
                "https://arxiv.org/pdf/2305.20050v1.pdf?download=1",
                Some("Let's Verify Step by Step"),
                None,
                None,
                Some("v1"),
                Some("We train process reward models on PRM800K."),
                Some("semantic_scholar"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        let summary_one_paper = store
            .record_source_corroboration(
                &claim_id,
                "https://openalex.org/W4378901234",
                Some("Let's Verify Step by Step!"),
                Some("https://doi.org/10.48550/arXiv.2305.20050"),
                None,
                Some("v1"),
                Some("OpenAlex mirror of Let's Verify Step by Step."),
                Some("openalex"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        assert_eq!(
            summary_one_paper.independent_papers, 1,
            "One paper reached through three routes must count ONCE"
        );
        assert_eq!(summary_one_paper.total_independent_sources, 1);
        assert_eq!(
            summary_one_paper.total_route_observations, 3,
            "All three route observations must still be tracked for diagnostics"
        );

        let claim_after_one = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(claim_after_one.source_support_status, "single_source");
        assert_eq!(claim_after_one.source_corroboration_count, 1);

        // Paper 2: An independent second paper on the same claim
        let summary_two_papers = store
            .record_source_corroboration(
                &claim_id,
                "https://arxiv.org/abs/2312.08935v1",
                Some("Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"),
                Some("10.48550/arXiv.2312.08935"),
                Some("2312.08935v1"),
                Some("v1"),
                Some("Independent confirmation of step-level process verification."),
                Some("arxiv"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        assert_eq!(
            summary_two_papers.independent_papers, 2,
            "Two independent papers must count TWICE"
        );
        assert_eq!(summary_two_papers.total_independent_sources, 2);

        let claim_after_two = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(claim_after_two.source_support_status, "corroborated");
        assert_eq!(claim_after_two.source_corroboration_count, 2);
        assert_eq!(
            claim_after_two.status, "unresolved",
            "Trust attribute remains separate from source support attribute"
        );
        assert_eq!(
            claim_after_two.experimental_verification_status, "unverified",
            "Experimental verification attribute remains separate from source support and trust"
        );

        // Also verify independent dataset and implementation count separately
        let summary_with_dataset_and_code = store
            .record_source_corroboration(
                &claim_id,
                "https://huggingface.co/datasets/openai/prm800k",
                Some("PRM800K Step Supervision Dataset"),
                None,
                None,
                Some("v1"),
                Some("800K step-level labels"),
                Some("huggingface"),
                "frontier_ai_research_and_verification",
            )
            .and_then(|_| {
                store.record_source_corroboration(
                    &claim_id,
                    "https://github.com/openai/prm800k",
                    Some("PRM800K Reference Implementation"),
                    None,
                    None,
                    Some("v1"),
                    Some("Evaluation repository"),
                    Some("github"),
                    "frontier_ai_research_and_verification",
                )
            })
            .unwrap();

        assert_eq!(summary_with_dataset_and_code.independent_papers, 2);
        assert_eq!(summary_with_dataset_and_code.independent_datasets, 1);
        assert_eq!(summary_with_dataset_and_code.independent_implementations, 1);
        assert_eq!(summary_with_dataset_and_code.total_independent_sources, 4);

        // Verify Experimental Verification and Trust are recorded independently
        store
            .record_claim_experimental_verification(
                &claim_id,
                true,
                "SOLVER_VERIFIED: PRM step accuracy +11.4% over ORM",
            )
            .unwrap();
        let claim_verified = store.get_claim(&claim_id).unwrap().unwrap();
        assert_eq!(claim_verified.status, "unresolved");
        assert_eq!(claim_verified.source_support_status, "corroborated");
        assert_eq!(claim_verified.experimental_verification_status, "verified");
    }

    #[tokio::test]
    async fn test_full_text_section_evidence_reaches_research_context() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_full_text_sections.db");
        let store = Arc::new(Store::open(&db_path).unwrap());

        // 1. Construct realistic LaTeXML / arXiv HTML with section hierarchy, method, algorithm,
        // equation (preserving <math alttext="..."> notation), theorem, assumption, experiment,
        // limitation, and bibliography references.
        let arxiv_html = r#"
        <html><body>
        <section id="S1" class="ltx_section">
            <h2 class="ltx_title_section">1 Introduction and Problem Formulation</h2>
            <p class="ltx_p">We investigate test-time compute scaling with process reward verification across multi-step proofs.</p>
        </section>
        <section id="S2" class="ltx_section">
            <h2 class="ltx_title_section">2 Proposed Method and Search Algorithm</h2>
            <p class="ltx_p">Our verifier scores each intermediate reasoning step before beam expansion.</p>
            <div id="alg1" class="ltx_listing">Algorithm 1: Step-Verified MCTS Rollout with Process Reward Pruning and backtracking threshold tau.</div>
            <div id="Thm1" class="ltx_theorem">Theorem 1 (Sub-optimality Bound): Under Assumption 1, step-level pruning bounds cumulative regret by O(sqrt(T log K)).</div>
            <div id="Assump1" class="ltx_theorem">Assumption 1 (Bounded Calibration Error): Step verifier calibration error satisfies epsilon_step &lt;= 0.05 across all depth levels.</div>
            <table id="Ex1" class="ltx_equation"><tr><td><math alttext="\hat{V}(s_t) = \prod_{k=1}^{t} \sigma(r_\theta(s_k, a_k))">V_hat</math></td></tr></table>
        </section>
        <section id="S3" class="ltx_section">
            <h2 class="ltx_title_section">3 Experimental Evaluation and Ablations</h2>
            <p class="ltx_p">On MATH500 and GSM8K benchmarks under strict token parity, our method improves accuracy from 74.2% to 86.9%.</p>
        </section>
        <section id="S4" class="ltx_section">
            <h2 class="ltx_title_section">4 Limitations and Failure Modes</h2>
            <p class="ltx_p">Limitation: Latency increases by 1.8x when verifier batch size exceeds 64 on long-horizon proofs.</p>
        </section>
        <section id="bib" class="ltx_bibliography">
            <li id="bib.bib1" class="ltx_bibitem">Lightman et al., Let's Verify Step by Step, arXiv:2305.20050, doi:10.48550/arXiv.2305.20050</li>
        </section>
        </body></html>
        "#;

        let extracted_sections = crate::web::extract_structured_sections_from_html(
            arxiv_html,
            "https://arxiv.org/html/2504.13171v1",
        );
        assert!(
            extracted_sections.len() >= 6,
            "Must extract structured sections (method, algorithm, theorem, equation, experiment, limitation)"
        );
        assert!(
            extracted_sections.iter().any(|s| s.section_kind == "equation" && s.content.contains(r"\hat{V}(s_t)")),
            "Mathematical notation from <math alttext=...> must be preserved"
        );
        assert!(
            extracted_sections.iter().any(|s| s.section_kind == "theorem" && s.source_location.contains("#Thm1")),
            "Exact source location anchor (#Thm1) must be preserved"
        );

        let raw_paper = crate::web::ResearchSourcePaper {
            title: "Step-Verified MCTS for Test-Time Compute Scaling".to_string(),
            doi: Some("10.48550/arXiv.2504.13171".to_string()),
            arxiv_id: Some("2504.13171v1".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2025-04-18T00:00:00Z".to_string()),
            authors: vec!["Alice Researcher".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2504.13171v1".to_string(),
            code_url: Some("https://github.com/example/step-mcts".to_string()),
            abstract_or_excerpt: "Abstract summary of step-verified MCTS.".to_string(),
            experimental_details: Some("MATH500 accuracy 86.9%".to_string()),
            limitations_excerpt: Some("Latency scales with batch size".to_string()),
            citations: vec!["2305.20050".to_string()],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-04-18T00:00:00Z".to_string(),
        };
        let resolved = crate::web::deduplicate_and_resolve_provenance(vec![raw_paper]);
        let work = &resolved[0];

        let quality = crate::web::compute_extraction_quality(&extracted_sections, true, arxiv_html.len());
        let payload = crate::web::FullTextIngestionPayload {
            canonical_id: work.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "html".to_string(),
            raw_original: arxiv_html.to_string(),
            abstract_text: work.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: extracted_sections,
            harvested_references: vec!["2305.20050".to_string()],
            extraction_quality: quality,
        };

        let tracked = store
            .store_ingestion_payload(
                work,
                &payload,
                Some("Test-Time Compute Scaling"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();
        assert_eq!(tracked.ingestion_completeness, "full_text");
        assert_eq!(tracked.full_text_format.as_deref(), Some("html"));
        assert!(tracked.section_count >= 6);
        assert!(tracked.equations_preserved >= 1);
        assert!(!tracked.low_fidelity);

        // 2. Verify that an abstract-only payload is stored strictly as 'abstract_only' and NEVER
        // presented as full-text or allowed to downgrade an already-ingested full_text source.
        let abstract_only_payload = crate::web::FullTextIngestionPayload {
            canonical_id: work.canonical_id.clone(),
            connector: "openalex".to_string(),
            ingestion_completeness: "abstract_only".to_string(),
            format: "abstract".to_string(),
            raw_original: work.abstract_excerpt.clone(),
            abstract_text: work.abstract_excerpt.clone(),
            is_full_text: false,
            is_orientation_only: false,
            sections: Vec::new(),
            harvested_references: Vec::new(),
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 0,
                equations_preserved: 0,
                tables_found: 0,
                low_fidelity: true,
                low_fidelity_reason: Some("abstract_or_orientation_only".to_string()),
            },
        };
        let tracked_after_abs = store
            .store_ingestion_payload(
                work,
                &abstract_only_payload,
                Some("Test-Time Compute Scaling"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();
        assert_eq!(
            tracked_after_abs.ingestion_completeness, "full_text",
            "Abstract-only observation must never overwrite or downgrade full-text ingestion"
        );

        // 3. Build Coordinator and verify full-text section evidence reaches the composed research context
        // within token bounds without injecting the raw HTML document.
        let cfg = crate::config::AppConfig::parse_and_validate(crate::config::DEFAULT_CONFIG_TOML).unwrap();
        let logger = crate::logging::Logger::new(
            tmp_dir.path().to_path_buf(),
            10,
            2,
        );
        let m_client = crate::model::ModelClient::new(
            &cfg.models.chat.base_url,
            &cfg.models.chat.model,
            cfg.models.chat.max_tokens,
            cfg.models.chat.no_think,
            cfg.models.chat.concurrency,
            &cfg.models.embedding.base_url,
            &cfg.models.embedding.model,
            &cfg.models.embedding.query_prefix,
            &cfg.models.embedding.document_prefix,
        );
        let web = crate::web::WebExplorer::new(&cfg.web.wikipedia_api, cfg.web.request_timeout_seconds);
        let adapter: Arc<dyn crate::execution::ExecutionAdapter> =
            Arc::new(crate::execution::AntigravityAdapter::probe("bwrap"));
        let (constitution, constitution_raw, constitution_digest) =
            crate::constitution::load_and_verify_constitution().unwrap();
        let coordinator = crate::coordinator::Coordinator::new(
            cfg,
            constitution,
            constitution_raw,
            constitution_digest,
            (*store).clone(),
            m_client,
            web,
            adapter,
            logger,
            tmp_dir.path().to_path_buf(),
        );

        let (sections, context_block) = coordinator
            .compose_full_text_section_evidence_context(
                "How does step-verified MCTS bound regret and what equation scores intermediate steps on MATH500?",
                &[work.canonical_id.clone()],
                600,
            )
            .unwrap();
        assert!(!sections.is_empty(), "Full-text sections must be retrieved for research context");
        assert!(
            context_block.contains("[Full-Text Structured Section Evidence"),
            "Research context must contain the structured full-text section evidence header"
        );
        assert!(
            context_block.contains(r"\hat{V}(s_t)") || context_block.contains("Theorem 1") || context_block.contains("MATH500"),
            "Research context must include extracted full-text equations/theorems/experiments"
        );
        assert!(
            context_block.contains("https://arxiv.org/html/2504.13171v1#"),
            "Research context must preserve exact section source locations"
        );
        assert!(
            !context_block.contains("<html><body>"),
            "Whole raw paper HTML must never be injected into research context"
        );
    }

    #[tokio::test]
    async fn test_disabling_connector_removes_it_from_discovery_without_touching_stored_knowledge() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_disable_connector.db");
        let store = Store::open(&db_path).unwrap();

        // 1. Populate stored knowledge (trusted claim, research_work, tracked_source, full-text section)
        // originating from 'arxiv' and 'github'.
        let claim_id = store
            .propose_claim(
                "Process reward verifiers improve MATH500 accuracy under token parity.",
                "https://arxiv.org/abs/2504.13171v1",
                "Explorer",
            )
            .unwrap();
        store
            .record_vote(&claim_id, "Explorer", "endorse", 1, Some("Verified"))
            .unwrap();
        store
            .record_vote(&claim_id, "Sceptic", "endorse", 1, Some("Corroborated"))
            .unwrap();

        let raw_paper = crate::web::ResearchSourcePaper {
            title: "Step-Verified MCTS for Test-Time Compute Scaling".to_string(),
            doi: Some("10.48550/arXiv.2504.13171".to_string()),
            arxiv_id: Some("2504.13171v1".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2025-04-18T00:00:00Z".to_string()),
            authors: vec!["Alice Researcher".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2504.13171v1".to_string(),
            code_url: None,
            abstract_or_excerpt: "Step-verified MCTS abstract.".to_string(),
            experimental_details: None,
            limitations_excerpt: None,
            citations: Vec::new(),
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-04-18T00:00:00Z".to_string(),
        };
        let resolved = crate::web::deduplicate_and_resolve_provenance(vec![raw_paper]);
        let work = &resolved[0];
        store.upsert_research_work(work).unwrap();

        let payload = crate::web::FullTextIngestionPayload {
            canonical_id: work.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "latex".to_string(),
            raw_original: r"\section{Method}\begin{theorem}Regret is O(sqrt(T)).\end{theorem}".to_string(),
            abstract_text: work.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: crate::web::extract_structured_sections_from_latex(
                r"\section{Method}\begin{theorem}Regret is O(sqrt(T)).\end{theorem}",
                "arxiv:2504.13171v1",
            ),
            harvested_references: Vec::new(),
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 0,
                tables_found: 0,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work,
                &payload,
                Some("Test-Time Compute"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        // Confirm stored knowledge before disabling connector
        assert_eq!(store.list_learnings(10).unwrap().len(), 1);
        assert_eq!(store.list_research_works(10).unwrap().len(), 1);
        assert_eq!(store.list_tracked_sources(10).unwrap().len(), 1);

        // 2. Disable 'arxiv' (and all other connectors) in DomainSection config
        let mut domain_cfg = crate::config::DomainSection::default();
        assert!(
            domain_cfg.is_connector_enabled("arxiv"),
            "arxiv should be enabled by default"
        );
        for conn in &mut domain_cfg.connectors {
            conn.enabled = false;
        }
        assert!(
            !domain_cfg.is_connector_enabled("arxiv"),
            "Disabling arxiv in domain config must mark it disabled"
        );
        assert!(
            !domain_cfg
                .effective_enabled_connectors()
                .iter()
                .any(|c| c.name == "arxiv"),
            "Disabled connector must be removed from effective_enabled_connectors()"
        );

        let web = crate::web::WebExplorer::new("https://en.wikipedia.org/api/rest_v1", 5);
        let (discovered, logs, telemetry) = web
            .search_multi_source_for_domain("test-time compute scaling", &domain_cfg, 2)
            .await;
        assert!(
            discovered.is_empty(),
            "Disabled connectors must not be queried during discovery"
        );
        assert!(
            !logs.iter().any(|l| l.contains("arxiv:")),
            "Disabled arxiv connector must not appear in discovery logs"
        );
        assert!(
            !telemetry.iter().any(|t| t.connector_name == "arxiv"),
            "Disabled arxiv connector must not emit fetch telemetry"
        );

        // 3. Verify that disabling the connector did NOT touch any stored knowledge!
        let stored_claims = store.list_learnings(10).unwrap();
        assert_eq!(
            stored_claims.len(),
            1,
            "Trusted claims from disabled connector must remain untouched"
        );
        assert_eq!(stored_claims[0].id, claim_id);

        let stored_works = store.list_research_works(10).unwrap();
        assert_eq!(
            stored_works.len(),
            1,
            "Stored research_works from disabled connector must remain untouched"
        );
        assert_eq!(stored_works[0].canonical_id, "arxiv:2504.13171");

        let stored_sources = store.list_tracked_sources(10).unwrap();
        assert_eq!(
            stored_sources.len(),
            1,
            "Tracked sources from disabled connector must remain untouched"
        );
        assert_eq!(stored_sources[0].ingestion_completeness, "full_text");

        let stored_sections = store
            .retrieve_relevant_source_sections("Regret theorem", &["arxiv:2504.13171".to_string()], 500)
            .unwrap();
        assert!(
            !stored_sections.is_empty(),
            "Stored full-text sections from disabled connector must remain retrievable"
        );
    }

    #[tokio::test]
    async fn test_plans_and_fabricated_run_references_cannot_become_findings() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_specialist_state_machine.db");
        let store = Arc::new(Store::open(&db_path).unwrap());

        // 1. Register a specialist endorsed by two permanent agents (Explorer + Builder).
        // Even with two permanent-agent endorsements, plans and fabricated run references must
        // NEVER become findings ("agreement never fabricates evidence").
        let spec_id = "spec-enforced-phase-001";
        let now = Utc::now().to_rfc3339();
        let spec_record = crate::specialists::SpecialistRecord {
            id: spec_id.to_string(),
            name: "VerifierRolloutSpecialist".to_string(),
            role: "step_verifier_calibration".to_string(),
            task: "Audit step-level PRM calibration vs beam rollout accuracy".to_string(),
            completion_criteria: "Compare step-level PRM rollout accuracy against outcome-only baseline".to_string(),
            max_tokens: 4000,
            deadline: now.clone(),
            endorser_1: "Explorer".to_string(),
            endorser_2: "Builder".to_string(),
            status: "running".to_string(),
            result: None,
            created_at: now.clone(),
            updated_at: now,
            phase: "phase1_investigate_and_execute".to_string(),
            plan_json: None,
            execution_ids: vec![],
            pending_commit_id: None,
        };
        store.register_specialist(&spec_record).unwrap();

        // 2. Phase 1: Record an investigation plan & prediction.
        // Verify it is stored as a task record (kind = "specialist_plan") and phase is phase1_investigate_and_execute.
        let task_id = store
            .record_specialist_phase1_plan(
                spec_id,
                "Step-level PRM pruning reduces cumulative regret on MATH500",
                &["Run prm_rollout_verifier on seed 42 with budget 2048".to_string()],
                &["Predict accuracy will increase by 14.5%".to_string()],
            )
            .unwrap();
        assert_eq!(task_id, format!("plan-{}", spec_id));
        let spec_after_phase1 = store.get_specialist(spec_id).unwrap().unwrap();
        assert_eq!(spec_after_phase1.phase, "phase1_investigate_and_execute");
        assert!(spec_after_phase1.plan_json.is_some());
        let count_all_claims = || -> i64 {
            store
                .conn
                .lock()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM claims", [], |r| r.get::<_, i64>(0))
                .unwrap()
        };
        assert_eq!(
            count_all_claims(),
            0,
            "Phase 1 plans and predictions must stay task records and never become findings"
        );

        // 3. Attempt to commit a plan/prediction directly through Phase 2 validate_and_commit_specialist_finding.
        let plan_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "We plan to evaluate step-level MCTS pruning and predict accuracy will improve to 91.0%.".to_string(),
            assumptions: "Step calibration error <= 0.05".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: None,
            claimed_measurements: serde_json::json!({}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let plan_outcome = store
            .validate_and_commit_specialist_finding(&plan_proposal)
            .unwrap();
        assert_eq!(plan_outcome.validation_status, "rejected");
        assert!(plan_outcome.committed_claim_id.is_none());
        let plan_rej = plan_outcome.rejection_reason.unwrap_or_default();
        assert!(
            plan_rej.contains("Plans and predictions stay task records"),
            "Unexpected rejection reason: {}",
            plan_rej
        );
        assert_eq!(count_all_claims(), 0);

        // 4. Attempt to commit an empirical claim referencing a fabricated execution_id (999999) that does not exist.
        let fabricated_run_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Measured rollout on seed 42 achieved accuracy=0.875 with step verifier pruning.".to_string(),
            assumptions: "Fixed token budget B=2048".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: Some(999999),
            claimed_measurements: serde_json::json!({"accuracy": 0.875}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let fabricated_run_outcome = store
            .validate_and_commit_specialist_finding(&fabricated_run_proposal)
            .unwrap();
        assert_eq!(fabricated_run_outcome.validation_status, "rejected");
        assert!(fabricated_run_outcome.committed_claim_id.is_none());
        let fab_run_rej = fabricated_run_outcome.rejection_reason.unwrap_or_default();
        assert!(
            fab_run_rej.contains("does not exist in host tool_executions"),
            "Unexpected rejection reason: {}",
            fab_run_rej
        );
        assert_eq!(count_all_claims(), 0);

        // 5. Record a real host-executed run with immutable execution evidence.
        let real_exec = store
            .record_immutable_tool_execution(
                "prm_rollout_verifier",
                1,
                "sha256:8f9d2a1b4c3e5f6a7b8c9d0e1f2a3b4c",
                r#"{"benchmark":"MATH500","budget":2048,"seed":42}"#,
                Some(42),
                r#"["python3>=3.11","sympy==1.12"]"#,
                Some(r#"{"status":"ok","accuracy":0.875,"regret_bound":0.042,"tokens_used":2048}"#),
                None,
                0,
                184,
                r#"["artifacts/rollout_seed42.json"]"#,
                r#"{"accuracy":0.875,"regret_bound":0.042,"tokens_used":2048}"#,
                Some(spec_id),
            )
            .unwrap();
        let real_exec_id = real_exec.id;
        let exec_rec = store.get_tool_execution(real_exec_id).unwrap().unwrap();
        assert_eq!(exec_rec.seed, Some(42));
        assert_eq!(exec_rec.exit_code, 0);
        assert!(!exec_rec.immutable_evidence_hash.is_empty());

        // Verify SQLite immutability triggers forbid mutating or deleting host execution evidence.
        {
            let conn = store.conn.lock().unwrap();
            let update_attempt = conn.execute(
                "UPDATE tool_executions SET output_json = '{\"accuracy\":0.999}' WHERE id = ?1",
                rusqlite::params![real_exec_id],
            );
            assert!(
                update_attempt.is_err(),
                "tool_executions must be immutable once recorded by host"
            );
            let delete_attempt = conn.execute(
                "DELETE FROM tool_executions WHERE id = ?1",
                rusqlite::params![real_exec_id],
            );
            assert!(
                delete_attempt.is_err(),
                "tool_executions rows must not be deletable"
            );
        }

        // 6. Attempt to commit an empirical claim referencing the real execution_id,
        // but inventing a fabricated measurement (accuracy: 0.965) contradicting the host's output (0.875)!
        let fabricated_measurement_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Measured rollout on seed 42 achieved accuracy=0.965 with step verifier pruning.".to_string(),
            assumptions: "Fixed token budget B=2048, seed 42".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: Some(real_exec_id),
            claimed_measurements: serde_json::json!({"accuracy": 0.965}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let fabricated_meas_outcome = store
            .validate_and_commit_specialist_finding(&fabricated_measurement_proposal)
            .unwrap();
        assert_eq!(fabricated_meas_outcome.validation_status, "rejected");
        assert!(fabricated_meas_outcome.committed_claim_id.is_none());
        let fab_meas_rej = fabricated_meas_outcome.rejection_reason.unwrap_or_default();
        assert!(
            fab_meas_rej.contains("0.965"),
            "Host validator must catch fabricated measurement 0.965: {}",
            fab_meas_rej
        );
        assert_eq!(count_all_claims(), 0);

        // 7. Commit a truthful Phase 2 locally_reproduced finding whose numbers match the host execution record.
        let valid_empirical_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Measured rollout on seed 42 achieved accuracy=0.875 and regret_bound=0.042 under budget 2048.".to_string(),
            assumptions: "Step verifier calibration error <= 0.05 under budget 2048".to_string(),
            negative_results: "No gain observed when verifier batch size exceeds 64".to_string(),
            is_negative_result: false,
            execution_id: Some(real_exec_id),
            claimed_measurements: serde_json::json!({"accuracy": 0.875, "regret_bound": 0.042}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let valid_outcome = store
            .validate_and_commit_specialist_finding(&valid_empirical_proposal)
            .unwrap();
        assert_eq!(valid_outcome.validation_status, "committed");
        assert_eq!(valid_outcome.finding_kind, "locally_reproduced");
        assert!(valid_outcome.trusted);
        let emp_claim_id = valid_outcome.committed_claim_id.clone().unwrap();
        let emp_claim = store.get_claim(&emp_claim_id).unwrap().unwrap();
        assert_eq!(emp_claim.finding_kind, "locally_reproduced");
        assert_eq!(emp_claim.execution_id, Some(real_exec_id));
        assert_eq!(emp_claim.experimental_verification_status, "verified");
        assert_eq!(emp_claim.source_support_status, "unsupported");
        assert_eq!(emp_claim.status, "trusted");

        // Verify idempotency across restart/re-submission: repeating the same commit does NOT duplicate claims.
        let dup_outcome = store
            .validate_and_commit_specialist_finding(&valid_empirical_proposal)
            .unwrap();
        assert!(dup_outcome.deduplicated_existing);
        assert_eq!(
            dup_outcome.committed_claim_id.as_deref(),
            Some(emp_claim_id.as_str())
        );
        assert_eq!(store.list_learnings(50).unwrap().len(), 1);

        // 8. Also verify literature findings (source_supported) reference stored source passages,
        // record independent source counts and extraction fidelity of each source (full-text, low-fidelity, abstract-only),
        // and remain distinct from locally_reproduced results.
        let raw_p1 = crate::web::ResearchSourcePaper {
            title: "Let's Verify Step by Step".to_string(),
            doi: Some("10.48550/arXiv.2305.20050".to_string()),
            arxiv_id: Some("2305.20050v2".to_string()),
            version: Some("v2".to_string()),
            published_date: Some("2023-05-31T00:00:00Z".to_string()),
            authors: vec!["Hunter Lightman".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2305.20050v2".to_string(),
            code_url: None,
            abstract_or_excerpt: "Section 3 proves step-level process supervision outperforms outcome supervision on MATH.".to_string(),
            experimental_details: Some("78.2% accuracy on MATH".to_string()),
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-04-18T00:00:00Z".to_string(),
        };
        let w1 = &crate::web::deduplicate_and_resolve_provenance(vec![raw_p1])[0];
        let payload1 = crate::web::FullTextIngestionPayload {
            canonical_id: w1.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "html".to_string(),
            raw_original: "<section id=\"S3\">Process reward models achieve 78.2% accuracy.</section>".to_string(),
            abstract_text: w1.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "3 > Process Supervision Evaluation".to_string(),
                section_kind: "experiment".to_string(),
                content: "Process reward models achieve 78.2% accuracy on representative MATH subset.".to_string(),
                source_location: "https://arxiv.org/html/2305.20050v2#S3".to_string(),
                token_estimate: 20,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 1,
                tables_found: 0,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                w1,
                &payload1,
                Some("step_verifier_calibration"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        // Second independent source: low-fidelity PDF extraction
        let raw_p2 = crate::web::ResearchSourcePaper {
            title: "Verifier Scaling Laws".to_string(),
            doi: Some("10.48550/arXiv.2402.01234".to_string()),
            arxiv_id: Some("2402.01234v1".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2024-02-02T00:00:00Z".to_string()),
            authors: vec!["Bob Researcher".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2402.01234v1".to_string(),
            code_url: None,
            abstract_or_excerpt: "Noisy PDF text with verifier scaling measurements.".to_string(),
            experimental_details: None,
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-04-18T00:00:00Z".to_string(),
        };
        let w2 = &crate::web::deduplicate_and_resolve_provenance(vec![raw_p2])[0];
        let payload2 = crate::web::FullTextIngestionPayload {
            canonical_id: w2.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "pdf".to_string(),
            raw_original: "Noisy PDF text with verifier scaling measurements.".to_string(),
            abstract_text: w2.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "1 > Overview".to_string(),
                section_kind: "section".to_string(),
                content: "Noisy PDF text with verifier scaling measurements.".to_string(),
                source_location: "https://arxiv.org/pdf/2402.01234v1#page=1".to_string(),
                token_estimate: 12,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 0,
                tables_found: 0,
                low_fidelity: true,
                low_fidelity_reason: Some("low_fidelity pdf stream".to_string()),
            },
        };
        store
            .store_ingestion_payload(
                w2,
                &payload2,
                Some("step_verifier_calibration"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        // Third independent source: abstract-only metadata
        let raw_p3 = crate::web::ResearchSourcePaper {
            title: "Independent Benchmark of Step Verifiers".to_string(),
            doi: Some("10.1145/3600000.3600001".to_string()),
            arxiv_id: None,
            version: Some("v1".to_string()),
            published_date: Some("2024-06-01T00:00:00Z".to_string()),
            authors: vec!["Carol Reviewer".to_string()],
            source_connector: "crossref".to_string(),
            url: "https://doi.org/10.1145/3600000.3600001".to_string(),
            code_url: None,
            abstract_or_excerpt: "Abstract confirming step verifier gains across independent reasoning tasks.".to_string(),
            experimental_details: None,
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-04-18T00:00:00Z".to_string(),
        };
        let w3 = &crate::web::deduplicate_and_resolve_provenance(vec![raw_p3])[0];
        let payload3 = crate::web::FullTextIngestionPayload {
            canonical_id: w3.canonical_id.clone(),
            connector: "crossref".to_string(),
            ingestion_completeness: "abstract_only".to_string(),
            format: "abstract".to_string(),
            raw_original: w3.abstract_excerpt.clone(),
            abstract_text: w3.abstract_excerpt.clone(),
            is_full_text: false,
            is_orientation_only: false,
            sections: vec![],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality::default(),
        };
        store
            .store_ingestion_payload(
                w3,
                &payload3,
                Some("step_verifier_calibration"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        let lit_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "source_supported".to_string(),
            conclusion: "Section 3 demonstrates process reward supervision consistently improves multi-step mathematical reasoning accuracy over outcome-only reward models.".to_string(),
            assumptions: "Step labels are aligned with valid intermediate derivations".to_string(),
            negative_results: "Annotation cost scales linearly with reasoning depth".to_string(),
            is_negative_result: false,
            execution_id: None,
            claimed_measurements: serde_json::json!({}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![
                "arxiv:2305.20050".to_string(),
                "arxiv:2402.01234".to_string(),
                "doi:10.1145/3600000.3600001".to_string(),
            ],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let lit_outcome = store
            .validate_and_commit_specialist_finding(&lit_proposal)
            .unwrap();
        assert_eq!(lit_outcome.validation_status, "committed");
        assert_eq!(lit_outcome.finding_kind, "source_supported");
        assert_eq!(lit_outcome.independent_source_count, 3);
        assert!(
            lit_outcome
                .source_fidelity_summary
                .starts_with("full-text: 1, low-fidelity: 1, abstract-only: 1"),
            "Unexpected source_fidelity_summary: {}",
            lit_outcome.source_fidelity_summary
        );
        let lit_claim = store
            .get_claim(&lit_outcome.committed_claim_id.unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(lit_claim.finding_kind, "source_supported");
        assert_eq!(lit_claim.source_support_status, "corroborated");
        assert_eq!(
            lit_claim.experimental_verification_status, "unverified",
            "Literature findings must stay distinct from locally reproduced results"
        );
    }

    #[tokio::test]
    async fn test_failed_experiments_keep_accurate_outcomes() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_failed_experiments.db");
        let store = Arc::new(Store::open(&db_path).unwrap());

        let spec_id = "spec-failed-exp-001";
        let now = Utc::now().to_rfc3339();
        let spec_record = crate::specialists::SpecialistRecord {
            id: spec_id.to_string(),
            name: "NegativeResultSpecialist".to_string(),
            role: "unbounded_step_temperature".to_string(),
            task: "Test whether high step temperature T=5.0 preserves verifier calibration".to_string(),
            completion_criteria: "Measure ECE at T=5.0 across 10 steps".to_string(),
            max_tokens: 4000,
            deadline: now.clone(),
            endorser_1: "Builder".to_string(),
            endorser_2: "Sceptic".to_string(),
            status: "running".to_string(),
            result: None,
            created_at: now.clone(),
            updated_at: now,
            phase: "phase1_investigate_and_execute".to_string(),
            plan_json: None,
            execution_ids: vec![],
            pending_commit_id: None,
        };
        store.register_specialist(&spec_record).unwrap();

        // 1. Record a failed tool execution (exit_code = 2) with immutable host evidence.
        let failed_exec = store
            .record_immutable_tool_execution(
                "step_calibration_experiment",
                1,
                "sha256:de12fa34bc567890de12fa34bc567890",
                r#"{"temperature":5.0,"seed":1337}"#,
                Some(1337),
                r#"["python3>=3.11"]"#,
                Some(r#"{"status":"failed","ece":0.412,"divergence_step":4,"exit_code":2}"#),
                Some("calibration divergence: ECE 0.412 exceeded stability threshold 0.10 at step 4"),
                2,
                95,
                r#"["artifacts/failed_temp5_seed1337.log"]"#,
                r#"{"ece":0.412,"divergence_step":4,"exit_code":2}"#,
                Some(spec_id),
            )
            .unwrap();
        let failed_exec_id = failed_exec.id;

        // Record the experiment outcome in research_experiments as well
        let exp_id = store
            .record_research_experiment(
                1,
                "High step temperature T=5.0 preserves calibration",
                r#"{"temperature":5.0,"seed":1337}"#,
                "step_calibration_experiment",
                "exit_code=2: calibration divergence at step 4 (ECE=0.412)",
                "Unbounded step temperature > 2.0 destabilizes PRM logits",
                true,
                Some("Lower temperature to T <= 1.2"),
            )
            .unwrap();
        assert!(exp_id > 0);

        let exps = store.list_research_experiments(10).unwrap();
        assert_eq!(exps.len(), 1);
        assert!(exps[0].is_failure, "Failed experiment must retain is_failure=true");
        assert!(exps[0].observed_outcome.contains("exit_code=2"));

        // 2. Verify that attempting to commit a POSITIVE empirical finding from the failed execution
        // is rejected by host validation (agent text cannot spin a failed run into a positive finding).
        let spin_positive_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Observed high step temperature 5.0 on seed 1337 maintained calibration with ece=0.412 at step 4.".to_string(),
            assumptions: "Temperature T=5.0, seed 1337".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: Some(failed_exec_id),
            claimed_measurements: serde_json::json!({"ece": 0.412}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let spin_outcome = store
            .validate_and_commit_specialist_finding(&spin_positive_proposal)
            .unwrap();
        assert_eq!(
            spin_outcome.validation_status, "rejected",
            "Failed execution (exit_code=2) must never be committed as a positive finding"
        );
        assert!(spin_outcome.committed_claim_id.is_none());

        // 3. Verify that committing an accurate NEGATIVE result referencing the failed execution succeeds
        // and preserves the failed experimental verification status ("failed") and negative_results_text.
        let honest_negative_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Observed negative result: step temperature 5.0 on seed 1337 failed with exit_code=2 and ece=0.412 at divergence_step=4.".to_string(),
            assumptions: "Temperature T=5.0, seed 1337".to_string(),
            negative_results: "Calibration diverged at step 4 (ECE 0.412 > 0.10) with exit code 2".to_string(),
            is_negative_result: true,
            execution_id: Some(failed_exec_id),
            claimed_measurements: serde_json::json!({"ece": 0.412, "divergence_step": 4, "exit_code": 2}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let neg_outcome = store
            .validate_and_commit_specialist_finding(&honest_negative_proposal)
            .unwrap();
        assert_eq!(neg_outcome.validation_status, "committed");
        let neg_claim_id = neg_outcome.committed_claim_id.unwrap();
        let neg_claim = store.get_claim(&neg_claim_id).unwrap().unwrap();
        assert_eq!(neg_claim.finding_kind, "locally_reproduced");
        assert_eq!(neg_claim.execution_id, Some(failed_exec_id));
        assert_eq!(
            neg_claim.experimental_verification_status, "failed",
            "Negative result from failed execution must keep experimental_verification_status='failed'"
        );
        assert!(
            neg_claim
                .negative_results_text
                .as_deref()
                .unwrap_or_default()
                .contains("ECE 0.412")
        );
        assert!(neg_claim.statement.contains("negative-result"));
        assert!(neg_claim.statement.contains("exit_code=2"));
    }

    fn sample_complete_experiment_spec(name: &str) -> ExperimentSpecification {
        ExperimentSpecification {
            id: String::new(),
            name: name.to_string(),
            domain: "frontier_ai_research_and_verification".to_string(),
            executable_check_kind: "simulation".to_string(),
            data_kind: "synthetic_data".to_string(),
            research_question: "Does process-verified rollout pruning reduce mean squared calibration error compared to unpruned best-of-N search under equal token budget?".to_string(),
            falsifiable_hypothesis: "Under fixed token budget B=1024, process-verified pruning achieves candidate_mse <= 0.85 * baseline_mse on both training and held-out problem instances.".to_string(),
            mechanism_under_test: "Step-level process-reward branch pruning vs uniform outcome-only best-of-N rollout selection".to_string(),
            inputs_description: "JSON input with seed (u64) and trials (usize)".to_string(),
            assumptions: vec![
                "Total token budget B=1024 is matched across baseline and candidate".to_string(),
                "Step error hazards are stationary across seeds".to_string(),
            ],
            baseline_description: "Outcome-only unpruned best-of-N rollout baseline under identical token budget B=1024".to_string(),
            controls: vec![
                "Matched total token budget B=1024".to_string(),
                "Paired deterministic PRNG seed across baseline and candidate".to_string(),
            ],
            held_out_cases: vec![
                "Out-of-distribution high-depth reasoning chains (depth=12, seeds 500..532)".to_string(),
            ],
            metrics: vec![
                ExperimentMetricSpec {
                    name: "Baseline MSE".to_string(),
                    result_key: "baseline_mse".to_string(),
                    unit: "mse".to_string(),
                    direction: "lower_is_better".to_string(),
                    success_criterion: "<= 0.25".to_string(),
                },
                ExperimentMetricSpec {
                    name: "Candidate Pruned MSE".to_string(),
                    result_key: "candidate_mse".to_string(),
                    unit: "mse".to_string(),
                    direction: "lower_is_better".to_string(),
                    success_criterion: "<= 0.85 * baseline_mse".to_string(),
                },
                ExperimentMetricSpec {
                    name: "Held-Out Generalization MSE".to_string(),
                    result_key: "held_out_mse".to_string(),
                    unit: "mse".to_string(),
                    direction: "lower_is_better".to_string(),
                    success_criterion: "<= 0.18".to_string(),
                },
            ],
            budget_max_runtime_secs: 20,
            budget_max_memory_mb: 512,
            seeds: vec![42, 1337],
            reproducibility_requirements: "Deterministic PRNG seeded by `seed`, writing structured JSON metrics to TRIAD_RESULTS_FILE (`results.json`).".to_string(),
            expected_outputs: vec![
                "results.json with baseline_mse, candidate_mse, held_out_mse".to_string(),
            ],
            known_limitations: vec![
                "Synthetic hazard rate model; numerical agreement supports an experiment; it is not a proof.".to_string(),
            ],
            cited_source_ids: vec![],
            cited_passage_ids: vec![],
            source_fidelity_status: "self_contained_computational".to_string(),
            under_specified: false,
            missing_source_details: vec![],
            reingestion_requested: false,
            builder_agent: "Builder".to_string(),
            sceptic_agent: "Sceptic".to_string(),
            sceptic_review_status: "pending".to_string(),
            sceptic_review_notes: String::new(),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[tokio::test]
    async fn test_superficial_verifier_tools_fail_review() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_superficial_review.db");
        let ws_dir = tmp_dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();
        let store = Store::open(&db_path).unwrap();
        let adapter = Arc::new(crate::execution::BubblewrapAdapter::probe());
        let tm = crate::tools::ToolManager::new(store.clone(), ws_dir, adapter);

        let spec = sample_complete_experiment_spec("superficial_test");

        // 1. Tool that counts words (`split_whitespace().count()`)
        let word_counter = crate::tools::ToolProposal {
            name: "word_count_verifier".to_string(),
            description: "Counts words in hypothesis text".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
fn main() {
    let text = "hypothesis text";
    let verified_tokens = text.split_whitespace().count();
    println!("{{\"verified_tokens\": {}}}", verified_tokens);
}
"#.to_string(),
            test_code: String::new(),
            experiment_spec: Some(spec.clone()),
        };
        let rev1 = crate::tools::ToolManager::review_research_tool(&word_counter, &spec, None);
        assert!(!rev1.approved, "Word-counting tool must fail Sceptic review");
        assert!(
            rev1.rejection_reasons
                .iter()
                .any(|r| r.contains("counts words")),
            "Expected word-counting rejection reason, got: {:?}",
            rev1.rejection_reasons
        );
        assert!(tm
            .build_and_register_tool(&word_counter, "Builder", "Sceptic")
            .await
            .is_err());

        // 2. Tool that inspects strings/substrings (`contains("arxiv")`, `contains("doi")`)
        let string_inspector = crate::tools::ToolProposal {
            name: "citation_string_inspector".to_string(),
            description: "Checks if string contains arxiv or doi".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
fn main() {
    let s = "arxiv:2305.20050 doi:10.1145";
    let ok = s.contains("arxiv") && s.contains("doi");
    println!("{{\"ok\": {}}}", ok);
}
"#.to_string(),
            test_code: String::new(),
            experiment_spec: Some(spec.clone()),
        };
        let rev2 = crate::tools::ToolManager::review_research_tool(&string_inspector, &spec, None);
        assert!(!rev2.approved, "String-inspecting tool must fail Sceptic review");
        assert!(
            rev2.rejection_reasons
                .iter()
                .any(|r| r.contains("inspects strings")),
            "Expected string-inspection rejection reason, got: {:?}",
            rev2.rejection_reasons
        );

        // 3. Tool that prints a predefined conclusion
        let predefined_printer = crate::tools::ToolProposal {
            name: "predefined_conclusion_tool".to_string(),
            description: "Prints predefined conclusion".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
fn main() {
    println!("HYPOTHESIS CONFIRMED");
}
"#.to_string(),
            test_code: String::new(),
            experiment_spec: Some(spec.clone()),
        };
        let rev3 = crate::tools::ToolManager::review_research_tool(&predefined_printer, &spec, None);
        assert!(!rev3.approved, "Predefined-conclusion tool must fail Sceptic review");
        assert!(
            rev3.rejection_reasons
                .iter()
                .any(|r| r.contains("predefined conclusion")),
            "Expected predefined-conclusion rejection reason, got: {:?}",
            rev3.rejection_reasons
        );

        // 4. Tool that emits an arbitrary hardcoded score (`routing_entropy_bits: 2.85`)
        let arbitrary_scorer = crate::tools::ToolProposal {
            name: "arbitrary_score_emitter".to_string(),
            description: "Emits hardcoded score".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
fn main() {
    let out = "{\"routing_entropy_bits\": 2.85, \"step_consistency_score\": 0.92}";
    println!("{}", out);
}
"#.to_string(),
            test_code: String::new(),
            experiment_spec: Some(spec.clone()),
        };
        let rev4 = crate::tools::ToolManager::review_research_tool(&arbitrary_scorer, &spec, None);
        assert!(!rev4.approved, "Arbitrary-score tool must fail Sceptic review");
        assert!(
            rev4.rejection_reasons
                .iter()
                .any(|r| r.contains("arbitrary or hardcoded constant score")),
            "Expected arbitrary-score rejection reason, got: {:?}",
            rev4.rejection_reasons
        );

        assert!(
            store.list_tools().unwrap().is_empty(),
            "No superficial verifier tool may be registered"
        );
    }

    #[tokio::test]
    async fn test_substantive_experiment_yields_traceable_measurements_linked_to_execution_record() {
        let adapter = crate::execution::BubblewrapAdapter::probe();
        if !adapter.is_available() {
            return;
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_substantive_exp.db");
        let ws_dir = tmp_dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();
        let store = Store::open(&db_path).unwrap();
        let tm = crate::tools::ToolManager::new(store.clone(), ws_dir, Arc::new(adapter));

        // 1. Ingest a full-text source so the experiment cites a verified full-text document
        let raw_paper = crate::web::ResearchSourcePaper {
            title: "Process-Verified Tree Search Under Fixed Compute Budgets".to_string(),
            doi: Some("10.48550/arXiv.2406.11111".to_string()),
            arxiv_id: Some("2406.11111v1".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2024-06-15T00:00:00Z".to_string()),
            authors: vec!["Alice Verifier".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2406.11111v1".to_string(),
            code_url: None,
            abstract_or_excerpt: "Full-text derivation of process-reward branch pruning.".to_string(),
            experimental_details: Some("B=1024 token budget comparison".to_string()),
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work = &crate::web::deduplicate_and_resolve_provenance(vec![raw_paper])[0];
        let payload = crate::web::FullTextIngestionPayload {
            canonical_id: work.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "html".to_string(),
            raw_original: "<section id=\"S2\">Algorithm 1: Prune rollout branch when step log-odds < -0.5.</section>".to_string(),
            abstract_text: work.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "2 > Method & Algorithm".to_string(),
                section_kind: "method".to_string(),
                content: "Algorithm 1: Prune rollout branch when step log-odds < -0.5 under budget B=1024.".to_string(),
                source_location: "https://arxiv.org/html/2406.11111v1#S2".to_string(),
                token_estimate: 24,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 2,
                tables_found: 1,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work,
                &payload,
                Some("prm_pruning"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        // 2. Build a complete ExperimentSpecification citing the full-text source
        let mut spec = sample_complete_experiment_spec("prm_mse_simulator");
        spec.cited_source_ids = vec![work.canonical_id.clone()];

        let proposal = crate::tools::ToolProposal {
            name: "prm_mse_simulator".to_string(),
            description: "Simulates process-reward pruning MSE vs outcome-only baseline across training and held-out seeds".to_string(),
            schema_json: r#"{"type":"object","properties":{"seed":{"type":"integer"},"trials":{"type":"integer"}}}"#.to_string(),
            rust_source: r#"
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_trials")]
    trials: usize,
}
fn default_seed() -> u64 { 42 }
fn default_trials() -> usize { 64 }

#[derive(Serialize)]
struct Results {
    baseline_mse: f64,
    candidate_mse: f64,
    held_out_mse: f64,
    baseline_comparison: serde_json::Value,
    held_out_evaluation: serde_json::Value,
}

fn simulate(seed: u64, trials: usize) -> Results {
    let mut state = if seed == 0 { 42 } else { seed };
    let n = trials.max(16);
    let mut base_err_sq = 0.0_f64;
    let mut cand_err_sq = 0.0_f64;
    let mut held_out_err_sq = 0.0_f64;
    for i in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        let target = 0.80 - 0.30 * ((i as f64) / (n as f64));
        let base_pred = (target + 0.38 * (u - 0.25)).clamp(0.0, 1.0);
        let cand_pred = (target + 0.16 * (u - 0.25)).clamp(0.0, 1.0);
        let ho_pred = (target + 0.19 * (u - 0.25)).clamp(0.0, 1.0);
        base_err_sq += (base_pred - target).powi(2);
        cand_err_sq += (cand_pred - target).powi(2);
        held_out_err_sq += (ho_pred - target).powi(2);
    }
    let baseline_mse = ((base_err_sq / n as f64) * 10000.0).round() / 10000.0;
    let candidate_mse = ((cand_err_sq / n as f64) * 10000.0).round() / 10000.0;
    let held_out_mse = ((held_out_err_sq / n as f64) * 10000.0).round() / 10000.0;
    Results {
        baseline_mse,
        candidate_mse,
        held_out_mse,
        baseline_comparison: serde_json::json!({
            "baseline_mse": baseline_mse,
            "candidate_mse": candidate_mse,
            "relative_reduction": ((baseline_mse - candidate_mse) / baseline_mse.max(1e-6))
        }),
        held_out_evaluation: serde_json::json!({
            "held_out_mse": held_out_mse,
            "generalization_gap": (held_out_mse - candidate_mse).abs()
        }),
    }
}

fn main() {
    let mut s = String::new();
    let _ = std::io::stdin().read_to_string(&mut s);
    let input: Input = serde_json::from_str(&s).unwrap_or(Input { seed: 42, trials: 64 });
    let res = simulate(input.seed, input.trials);
    let json_out = serde_json::to_string_pretty(&res).unwrap();
    let results_path = std::env::var("TRIAD_RESULTS_FILE").unwrap_or_else(|_| "results.json".to_string());
    std::fs::write(&results_path, &json_out).expect("Failed to write results.json");
    println!("{}", json_out);
}
"#.to_string(),
            test_code: r#"
    #[test]
    fn test_sim() {
        let r = simulate(42, 64);
        assert!(r.candidate_mse < r.baseline_mse);
        assert!(r.held_out_mse < r.baseline_mse);
    }
"#.to_string(),
            experiment_spec: Some(spec.clone()),
        };

        let (tool_id, validated_spec, review) = tm
            .build_and_register_research_tool(&proposal, spec, "Builder", "Sceptic", None)
            .await
            .expect("Substantive research tool must pass Sceptic review and compile");
        assert!(tool_id > 0);
        assert!(review.approved);
        assert!(!validated_spec.under_specified);
        assert_eq!(validated_spec.source_fidelity_status, "full_text_verified");

        // 3. Execute the experiment and verify traceable measurements linked to the execution record
        let outcome = tm
            .execute_research_experiment(
                1,
                "prm_mse_simulator",
                r#"{"seed":42,"trials":64}"#,
                Some(42),
                None,
                None,
            )
            .await
            .expect("Substantive experiment execution must succeed");

        assert_eq!(outcome.execution_record.exit_code, 0);
        assert_eq!(outcome.experiment_record.outcome_status, "completed");
        assert_eq!(
            outcome.experiment_record.execution_id,
            Some(outcome.execution_record.id)
        );
        assert_eq!(
            outcome.experiment_record.spec_id.as_deref(),
            Some(validated_spec.id.as_str())
        );

        let b_mse = outcome.measured_results["baseline_mse"].as_f64().unwrap();
        let c_mse = outcome.measured_results["candidate_mse"].as_f64().unwrap();
        let h_mse = outcome.measured_results["held_out_mse"].as_f64().unwrap();
        assert!(c_mse < b_mse);
        assert!(h_mse < b_mse);

        // 4. Commit a Phase 2 specialist finding referencing this execution record and declared metric keys
        let now = Utc::now().to_rfc3339();
        let spec_agent_id = "spec-substantive-001";
        store
            .register_specialist(&crate::specialists::SpecialistRecord {
                id: spec_agent_id.to_string(),
                name: "PRMExperimentSpecialist".to_string(),
                role: "prm_pruning_experimenter".to_string(),
                task: "Run prm_mse_simulator and report MSE against baseline".to_string(),
                completion_criteria: "Verify candidate_mse < baseline_mse".to_string(),
                max_tokens: 2048,
                deadline: now.clone(),
                endorser_1: "Builder".to_string(),
                endorser_2: "Sceptic".to_string(),
                status: "running".to_string(),
                result: None,
                created_at: now.clone(),
                updated_at: now,
                phase: "phase1_investigate_and_execute".to_string(),
                plan_json: None,
                execution_ids: vec![outcome.execution_record.id],
                pending_commit_id: None,
            })
            .unwrap();

        let phase2_prop = SpecialistPhase2Proposal {
            specialist_id: spec_agent_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: format!(
                "Measured prm_mse_simulator on seed 42: candidate_mse={} vs baseline_mse={} and held_out_mse={}.",
                c_mse, b_mse, h_mse
            ),
            assumptions: "Matched token budget B=1024, seed 42".to_string(),
            negative_results: "Numerical agreement supports an experiment; it is not a proof.".to_string(),
            is_negative_result: false,
            execution_id: Some(outcome.execution_record.id),
            claimed_measurements: serde_json::json!({
                "baseline_mse": b_mse,
                "candidate_mse": c_mse,
                "held_out_mse": h_mse
            }),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let commit_res = store
            .validate_and_commit_specialist_finding(&phase2_prop)
            .unwrap();
        assert_eq!(commit_res.validation_status, "committed");
        assert!(commit_res.trusted);
        let claim = store
            .get_claim(&commit_res.committed_claim_id.unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(claim.execution_id, Some(outcome.execution_record.id));
        assert_eq!(claim.experimental_verification_status, "verified");
    }

    #[test]
    fn test_experiment_citing_abstract_only_source_is_flagged_under_specified() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_abstract_only_spec.db");
        let store = Store::open(&db_path).unwrap();

        // 1. Ingest an abstract-only source into tracked_sources
        let raw_abstract_paper = crate::web::ResearchSourcePaper {
            title: "Opaque Verifier Architecture Without Full Text".to_string(),
            doi: Some("10.1145/9999999.8888888".to_string()),
            arxiv_id: None,
            version: Some("v1".to_string()),
            published_date: Some("2024-09-01T00:00:00Z".to_string()),
            authors: vec!["Dave Abstract".to_string()],
            source_connector: "crossref".to_string(),
            url: "https://doi.org/10.1145/9999999.8888888".to_string(),
            code_url: None,
            abstract_or_excerpt: "We propose a step verifier that improves accuracy by 12%, but equations and hyperparameters are only in the full paper.".to_string(),
            experimental_details: None,
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work = &crate::web::deduplicate_and_resolve_provenance(vec![raw_abstract_paper])[0];
        let abstract_payload = crate::web::FullTextIngestionPayload {
            canonical_id: work.canonical_id.clone(),
            connector: "crossref".to_string(),
            ingestion_completeness: "abstract_only".to_string(),
            format: "abstract".to_string(),
            raw_original: work.abstract_excerpt.clone(),
            abstract_text: work.abstract_excerpt.clone(),
            is_full_text: false,
            is_orientation_only: false,
            sections: vec![],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality::default(),
        };
        store
            .store_ingestion_payload(
                work,
                &abstract_payload,
                Some("opaque_verifier"),
                "frontier_ai_research_and_verification",
            )
            .unwrap();

        // 2. Register an ExperimentSpecification citing the abstract-only source
        let mut spec = sample_complete_experiment_spec("opaque_verifier_replication");
        spec.cited_source_ids = vec![work.canonical_id.clone()];
        spec.under_specified = false;
        spec.missing_source_details.clear();

        let validated = store
            .register_and_validate_experiment_spec(&mut spec)
            .unwrap();

        assert!(
            validated.under_specified,
            "Experiment citing an abstract-only source MUST be flagged under_specified"
        );
        assert_eq!(
            validated.source_fidelity_status,
            "under_specified_abstract_only"
        );
        assert!(
            !validated.missing_source_details.is_empty(),
            "Specification must explicitly state which method/algorithm/equation details are missing"
        );
        assert!(
            validated.missing_source_details[0].contains("abstract_only"),
            "Missing details must mention abstract_only status: {:?}",
            validated.missing_source_details
        );
        assert!(
            validated.reingestion_requested,
            "Re-ingestion or higher-fidelity source must be requested"
        );

        // Verify the tracked source in SQLite was marked for re-ingestion
        let tracked = store
            .get_tracked_source(&work.canonical_id)
            .unwrap()
            .unwrap();
        assert!(tracked.retry_eligible);
        assert!(
            tracked
                .revisit_reason
                .as_deref()
                .unwrap_or_default()
                .contains("abstract-only"),
            "Expected revisit_reason to mention abstract-only, got: {:?}",
            tracked.revisit_reason
        );
    }

    #[tokio::test]
    async fn test_tool_lacking_declared_metric_key_rejected_and_prose_only_number_cannot_support_claim() {
        let adapter = crate::execution::BubblewrapAdapter::probe();
        if !adapter.is_available() {
            return;
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_missing_key_and_prose.db");
        let ws_dir = tmp_dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();
        let store = Store::open(&db_path).unwrap();
        let tm = crate::tools::ToolManager::new(store.clone(), ws_dir, Arc::new(adapter));

        // Part A: Build a tool whose source references the metric keys in comments/stdout,
        // but whose `results.json` file omits the declared key `"held_out_mse"`.
        let spec = sample_complete_experiment_spec("incomplete_results_tool");
        let incomplete_proposal = crate::tools::ToolProposal {
            name: "incomplete_results_tool".to_string(),
            description: "Writes baseline_mse and candidate_mse to results.json, but omits held_out_mse from results.json and prints held_out_mse only to stdout".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
use serde::Serialize;

#[derive(Serialize)]
struct PartialResults {
    baseline_mse: f64,
    candidate_mse: f64,
}

fn main() {
    let mut acc = 0.0_f64;
    for i in 1..=32_usize {
        acc += (i as f64) * 0.002;
    }
    let partial = PartialResults {
        baseline_mse: 0.2200,
        candidate_mse: (acc * 0.1).min(0.1500),
    };
    let results_path = std::env::var("TRIAD_RESULTS_FILE").unwrap_or_else(|_| "results.json".to_string());
    std::fs::write(&results_path, serde_json::to_string(&partial).unwrap()).unwrap();
    // Print held_out_mse ONLY to stdout, omitting it from results.json!
    println!("{{\"baseline_mse\": 0.22, \"candidate_mse\": 0.1056, \"held_out_mse\": 0.1180}}");
}
"#.to_string(),
            test_code: r#"
    #[test]
    fn test_partial() {
        assert!(0.1056 < 0.2200);
    }
"#.to_string(),
            experiment_spec: Some(spec.clone()),
        };

        let (tool_id, _, _) = tm
            .build_and_register_research_tool(&incomplete_proposal, spec, "Builder", "Sceptic", None)
            .await
            .unwrap();
        assert!(tool_id > 0);

        // Execute the experiment: host MUST reject the run because `results.json` lacks declared key `held_out_mse`!
        let run_res = tm
            .execute_research_experiment(
                1,
                "incomplete_results_tool",
                "{}",
                Some(42),
                None,
                None,
            )
            .await;
        assert!(
            run_res.is_err(),
            "Host must reject tool whose results.json lacks declared metric key held_out_mse"
        );
        let err_msg = run_res.unwrap_err().to_string();
        assert!(
            err_msg.contains("RESULTS_FILE_MISSING_DECLARED_KEYS")
                && err_msg.contains("held_out_mse"),
            "Expected RESULTS_FILE_MISSING_DECLARED_KEYS for held_out_mse, got: {}",
            err_msg
        );

        // Part B: Verify that a number appearing ONLY in prose or ONLY in stdout cannot support a Phase 2 empirical claim!
        let now = Utc::now().to_rfc3339();
        let spec_agent_id = "spec-prose-only-001";
        store
            .register_specialist(&crate::specialists::SpecialistRecord {
                id: spec_agent_id.to_string(),
                name: "ProseNumberSpecialist".to_string(),
                role: "prose_auditor".to_string(),
                task: "Test prose-only and stdout-only numbers".to_string(),
                completion_criteria: "Validate strict results.json grounding".to_string(),
                max_tokens: 2048,
                deadline: now.clone(),
                endorser_1: "Builder".to_string(),
                endorser_2: "Sceptic".to_string(),
                status: "running".to_string(),
                result: None,
                created_at: now.clone(),
                updated_at: now,
                phase: "phase1_investigate_and_execute".to_string(),
                plan_json: None,
                execution_ids: vec![],
                pending_commit_id: None,
            })
            .unwrap();

        // Record an execution where `stdout` (`output_json`) contains `stdout_only_score: 0.99`,
        // while `results.json` (`measured_results_json`) only contains `{"verified_metric": 0.75}`.
        let valid_exec = store
            .record_immutable_tool_execution(
                "strict_json_tool",
                1,
                "sha256:11223344556677889900aabbccddeeff",
                r#"{"seed":42}"#,
                Some(42),
                r#"["serde_json@1.0"]"#,
                Some(r#"{"verified_metric":0.75,"stdout_only_score":0.99}"#),
                None,
                0,
                50,
                r#"["results.json"]"#,
                r#"{"verified_metric":0.75}"#,
                Some(spec_agent_id),
            )
            .unwrap();

        // 1) Claiming `stdout_only_score` in `claimed_measurements` must be rejected because it is not in `measured_results_json`
        let stdout_claim_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_agent_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Measured stdout_only_score=0.99 on seed 42.".to_string(),
            assumptions: "Seed 42".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: Some(valid_exec.id),
            claimed_measurements: serde_json::json!({"stdout_only_score": 0.99}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let out_stdout = store
            .validate_and_commit_specialist_finding(&stdout_claim_proposal)
            .unwrap();
        assert_eq!(
            out_stdout.validation_status, "rejected",
            "Number present only in stdout must not support a claim"
        );
        assert!(
            out_stdout
                .rejection_reason
                .as_deref()
                .unwrap_or_default()
                .contains("stdout"),
            "Expected stdout rejection reason, got: {:?}",
            out_stdout.rejection_reason
        );

        // 2) Stating a prose-only number (`prose_only_f1=0.985`) in `conclusion` while only `verified_metric` is in `results.json` must be rejected!
        let prose_claim_proposal = SpecialistPhase2Proposal {
            specialist_id: spec_agent_id.to_string(),
            finding_kind: "locally_reproduced".to_string(),
            conclusion: "Measured verified_metric=0.75 and prose_only_f1=0.985 on seed 42.".to_string(),
            assumptions: "Seed 42".to_string(),
            negative_results: String::new(),
            is_negative_result: false,
            execution_id: Some(valid_exec.id),
            claimed_measurements: serde_json::json!({"verified_metric": 0.75}),
            source_passage_ids: vec![],
            canonical_source_ids: vec![],
            cycle_id: 1,
            domain: "frontier_ai_research_and_verification".to_string(),
        };
        let out_prose = store
            .validate_and_commit_specialist_finding(&prose_claim_proposal)
            .unwrap();
        assert_eq!(
            out_prose.validation_status, "rejected",
            "Prose-only number in conclusion must be rejected"
        );
        assert!(
            out_prose
                .rejection_reason
                .as_deref()
                .unwrap_or_default()
                .contains("prose_only_f1=0.985"),
            "Expected prose-only rejection reason for prose_only_f1=0.985, got: {:?}",
            out_prose.rejection_reason
        );
    }

    #[tokio::test]
    async fn test_tool_sleeping_past_timeout_is_killed_leaves_no_child_running_and_records_timeout() {
        let adapter = crate::execution::BubblewrapAdapter::probe();
        if !adapter.is_available() {
            return;
        }

        let tmp_dir = tempfile::tempdir().unwrap();
        let db_path = tmp_dir.path().join("triad_timeout_killpg.db");
        let ws_dir = tmp_dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();
        let store = Store::open(&db_path).unwrap();
        let tm = crate::tools::ToolManager::new(store.clone(), ws_dir.clone(), Arc::new(adapter));

        let mut spec = sample_complete_experiment_spec("runaway_sleeper_tool");
        spec.budget_max_runtime_secs = 2;

        // This tool spawns a child `sleep 60` process inside the sandbox and sleeps for 60s itself.
        let sleeper_proposal = crate::tools::ToolProposal {
            name: "runaway_sleeper_tool".to_string(),
            description: "Spawns a background sleep child process and sleeps past its timeout".to_string(),
            schema_json: "{}".to_string(),
            rust_source: r#"
use serde::Serialize;

#[derive(Serialize)]
struct Metrics {
    baseline_mse: f64,
    candidate_mse: f64,
    held_out_mse: f64,
}

fn compute_preliminary(n: usize) -> Metrics {
    let mut s = 0.0_f64;
    for i in 0..n {
        s += (i as f64) * 0.001;
    }
    Metrics {
        baseline_mse: 0.20,
        candidate_mse: (0.12 + s * 0.0001).min(0.18),
        held_out_mse: 0.14,
    }
}

fn main() {
    let mut stdin_buf = String::new();
    let _ = std::io::Read::read_to_string(&mut std::io::stdin(), &mut stdin_buf);
    if stdin_buf.contains("sleep") {
        // Spawn a child process inside the sandbox process group and sleep past timeout
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("Failed to spawn sleep child");
        std::thread::sleep(std::time::Duration::from_secs(60));
        let _ = child.wait();
    }
    let m = compute_preliminary(16);
    let results_path = std::env::var("TRIAD_RESULTS_FILE").unwrap_or_else(|_| "results.json".to_string());
    std::fs::write(&results_path, serde_json::to_string(&m).unwrap()).unwrap();
}
"#.to_string(),
            test_code: r#"
    #[test]
    fn test_prelim() {
        let m = compute_preliminary(16);
        assert!(m.candidate_mse < m.baseline_mse);
    }
"#.to_string(),
            experiment_spec: Some(spec.clone()),
        };

        let (tool_id, _, _) = tm
            .build_and_register_research_tool(&sleeper_proposal, spec, "Builder", "Sceptic", None)
            .await
            .expect("runaway_sleeper_tool should compile and register");
        assert!(tool_id > 0);

        let start = std::time::Instant::now();
        let outcome = tm
            .execute_research_experiment(
                1,
                "runaway_sleeper_tool",
                r#"{"mode":"sleep"}"#,
                Some(42),
                None,
                Some(std::time::Duration::from_secs(2)),
            )
            .await
            .expect("Timeout outcome must be recorded cleanly");
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "Timeout must terminate runaway tool promptly, took {:?}",
            elapsed
        );
        assert_eq!(
            outcome.execution_record.exit_code, -124,
            "Timed-out execution must record exit_code = -124"
        );
        assert!(
            outcome
                .execution_record
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("SANDBOX_TIMED_OUT"),
            "Execution record error must record SANDBOX_TIMED_OUT: {:?}",
            outcome.execution_record.error
        );
        assert_eq!(outcome.experiment_record.outcome_status, "timed_out");
        assert!(outcome.experiment_record.is_failure);

        // Confirm zero active sandbox process groups remain for this workspace
        let remaining_groups =
            crate::execution::terminate_all_active_sandbox_process_groups(Some(&ws_dir));
        assert_eq!(
            remaining_groups, 0,
            "No sandbox process group may remain running after timeout termination"
        );

        // Also check /proc to confirm no residual `runaway_sleeper_tool` process is alive on the host
        let mut live_runaway_pids = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for entry in entries.flatten() {
                if let Some(pid_str) = entry.file_name().to_str() {
                    if let Ok(pid) = pid_str.parse::<i32>() {
                        let cmdline_path = entry.path().join("cmdline");
                        if let Ok(cmdline) = std::fs::read_to_string(cmdline_path) {
                            if cmdline.contains("runaway_sleeper_tool") {
                                live_runaway_pids.push(pid);
                            }
                        }
                    }
                }
            }
        }
        assert!(
            live_runaway_pids.is_empty(),
            "Found residual runaway_sleeper_tool processes still running after timeout: {:?}",
            live_runaway_pids
        );
    }

    #[tokio::test]
    async fn test_concurrent_frontier_calls_retries_and_restarts_cannot_exceed_quotas() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_frontier_quota.db");
        let log_dir = dir.path().join("logs");
        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir.clone(), 5, 2);

        // Spin up a local mock frontier HTTP server that:
        // - Tracks active concurrent requests and records peak concurrency
        // - Fails the very first request with HTTP 500 to trigger 1 retry (which must consume quota)
        // - Succeeds on subsequent requests
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let total_dispatched = Arc::new(AtomicUsize::new(0));
        let active_in_flight = Arc::new(AtomicUsize::new(0));
        let peak_concurrency = Arc::new(AtomicUsize::new(0));

        let td_clone = total_dispatched.clone();
        let act_clone = active_in_flight.clone();
        let peak_clone = peak_concurrency.clone();

        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let td = td_clone.clone();
                let act = act_clone.clone();
                let peak = peak_clone.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let _ = stream.read(&mut buf).await;
                    let cur = act.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(cur, Ordering::SeqCst);
                    let call_idx = td.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                    act.fetch_sub(1, Ordering::SeqCst);

                    if call_idx == 0 {
                        // First attempt fails -> forces 1 retry, which must also consume 24h quota
                        let body = r#"{"error":{"message":"transient upstream 500"}}"#;
                        let resp = format!(
                            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes()).await;
                    } else {
                        let body = r#"{"choices":[{"message":{"content":"{\"thought\":\"ok\",\"public_message\":\"frontier response\",\"proposed_claims\":[]}"}}],"usage":{"prompt_tokens":120,"completion_tokens":80,"total_tokens":200}}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = stream.write_all(resp.as_bytes()).await;
                    }
                });
            }
        });

        let cred_env = "TRIAD_TEST_FRONTIER_QUOTA_KEY";
        std::env::set_var(cred_env, "sk-test-secret-do-not-log-99999");

        let mut models_cfg =
            crate::config::AppConfig::parse_and_validate(crate::config::DEFAULT_CONFIG_TOML)
                .unwrap()
                .models;
        // Enforce defaults: 3 attempts per 24h, 1 per task, concurrency 1, one retry, report on exhaustion
        models_cfg.frontier_budget.max_calls_per_24h = 3;
        models_cfg.frontier_budget.max_calls_per_task = 1;
        models_cfg.frontier_budget.max_concurrency = 1;
        models_cfg.frontier_budget.max_retries = 1;
        models_cfg.frontier_budget.exhaustion_policy = "report".to_string();
        models_cfg.frontier_budget.cache_enabled = true;
        models_cfg.providers = vec![crate::config::ModelProviderConfig {
            id: "mock_frontier".to_string(),
            provider_type: "openai_compatible".to_string(),
            enabled: true,
            base_url: format!("http://{}", addr),
            transport: "openai_compatible".to_string(),
            credential_env: Some(cred_env.to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec!["mock-frontier-v1".to_string()],
            fallback_provider: None,
            is_frontier: true,
        }];
        models_cfg.catalog = vec![crate::config::ModelCapabilityConfig {
            provider_id: "mock_frontier".to_string(),
            model_id: "mock-frontier-v1".to_string(),
            context_window_tokens: 128_000,
            max_output_tokens: 8192,
            supports_structured_json: true,
            supports_code_generation: true,
            supports_reasoning_trace: true,
            supports_tool_use: true,
            is_frontier: true,
            input_price_micros_per_1m: None, // Unknown pricing must be billed conservatively, never free!
            output_price_micros_per_1m: None,
            reasoning_score_bps: 950,
            coding_score_bps: 950,
            synthesis_score_bps: 950,
            critique_score_bps: 950,
        }];
        models_cfg.selection.preference_order = vec!["mock_frontier:mock-frontier-v1".to_string()];

        let client = crate::model::ModelClient::from_models_config(
            &models_cfg,
            Some(store.clone()),
            Some(logger.clone()),
        );

        // 1. Task 1 dispatches: attempt #1 fails with HTTP 500, retry attempt #2 succeeds.
        // Because retries consume quota, Task 1 uses 2 of the 3 rolling-24h attempts!
        let ctx1 = crate::model::WorkloadSelectionContext {
            task_id: "task_alpha".to_string(),
            caller_kind: "agent".to_string(),
            caller_id: "Explorer".to_string(),
            workload: crate::model::WorkloadKind::HardSynthesis,
            required_context_tokens: 2048,
            require_structured_json: true,
            explicit_override: None,
        };
        let msgs1 = vec![crate::model::ChatMessage {
            role: "user".to_string(),
            content: "Synthesize theorem alpha".to_string(),
        }];
        let res1 = client
            .complete_chat_for_workload(&ctx1, "System prompt", &msgs1)
            .await
            .expect("Task alpha should succeed on its 1 permitted retry");
        assert!(res1.contains("frontier response"));

        let summary_after_task1 = store
            .get_frontier_budget_summary(&models_cfg.frontier_budget, 10)
            .unwrap();
        assert_eq!(
            summary_after_task1.used_calls_24h, 2,
            "Initial attempt + 1 retry must consume 2 attempts of the 3/24h quota"
        );
        assert_eq!(summary_after_task1.remaining_calls_24h, 1);
        // Unknown pricing must have charged conservative floor (> 0 micros, never free)
        assert!(
            summary_after_task1.used_spend_micros_24h
                >= models_cfg.frontier_budget.conservative_unknown_call_micros,
            "Unknown pricing must be billed conservatively and never treated as free"
        );

        // 2. Calling again with identical prompt hits the response cache without consuming remaining quota
        let cached_res = client
            .complete_chat_for_workload(&ctx1, "System prompt", &msgs1)
            .await
            .expect("Identical prompt must return cached frontier response");
        assert_eq!(cached_res, res1);
        let summary_after_cache = store
            .get_frontier_budget_summary(&models_cfg.frontier_budget, 10)
            .unwrap();
        assert_eq!(
            summary_after_cache.used_calls_24h, 2,
            "Cached frontier response must not consume additional quota"
        );
        assert!(summary_after_cache.cache_hits_total >= 1);

        // 3. A second distinct call on the SAME task (`task_alpha`) must be blocked by `max_calls_per_task = 1`
        let msgs1_second = vec![crate::model::ChatMessage {
            role: "user".to_string(),
            content: "Second distinct call on task_alpha".to_string(),
        }];
        let err_same_task = client
            .complete_chat_for_workload(&ctx1, "System prompt", &msgs1_second)
            .await
            .expect_err("Second call on task_alpha must be rejected by per-task limit");
        assert!(
            err_same_task.to_string().contains("task_call_limit_exceeded"),
            "Expected task_call_limit_exceeded rejection, got: {}",
            err_same_task
        );

        // 4. Launch 5 concurrent tasks (`task_beta_0` .. `task_beta_4`) in parallel across agents, specialists, and MCP consultations.
        // Only 1 call remains in the rolling 24h budget (2/3 used -> max 1 more can succeed).
        let mut handles = Vec::new();
        for i in 0..5 {
            let c = client.clone();
            handles.push(tokio::spawn(async move {
                let ctx = crate::model::WorkloadSelectionContext {
                    task_id: format!("task_beta_{}", i),
                    caller_kind: if i % 2 == 0 {
                        "specialist".to_string()
                    } else {
                        "mcp_consultation".to_string()
                    },
                    caller_id: format!("Worker_{}", i),
                    workload: crate::model::WorkloadKind::ComplexRustGeneration,
                    required_context_tokens: 2048,
                    require_structured_json: true,
                    explicit_override: None,
                };
                let msgs = vec![crate::model::ChatMessage {
                    role: "user".to_string(),
                    content: format!("Generate complex Rust solver #{}", i),
                }];
                c.complete_chat_for_workload(&ctx, "System prompt", &msgs).await
            }));
        }

        let mut successes = 0;
        let mut quota_rejections = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => successes += 1,
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        msg.contains("FRONTIER_BUDGET_EXHAUSTED"),
                        "Unexpected error: {}",
                        msg
                    );
                    quota_rejections += 1;
                }
            }
        }
        assert_eq!(
            successes, 1,
            "Exactly 1 of the 5 concurrent tasks may succeed given 1 remaining call in 24h"
        );
        assert_eq!(quota_rejections, 4);
        assert_eq!(
            peak_concurrency.load(Ordering::SeqCst),
            1,
            "Max concurrency of 1 must never be exceeded under parallel load"
        );
        assert_eq!(
            total_dispatched.load(Ordering::SeqCst),
            3,
            "Total upstream HTTP dispatches across retries and parallel tasks must never exceed 3"
        );

        // 5. Simulate process restart: drop Store & ModelClient, reopen from the same SQLite file,
        // and verify that a post-restart call cannot bypass the exhausted 3/3 rolling 24h quota.
        drop(client);
        drop(store);

        let store_restarted = Store::open(&db_path).unwrap();
        // Verify ledger deletion protection trigger prevents silent counter resets
        {
            let conn = store_restarted.conn.lock().unwrap();
            let del_res = conn.execute("DELETE FROM frontier_budget_ledger", []);
            assert!(
                del_res.is_err(),
                "SQLite trigger must forbid deleting or resetting frontier_budget_ledger"
            );
        }

        let client_restarted = crate::model::ModelClient::from_models_config(
            &models_cfg,
            Some(store_restarted.clone()),
            Some(logger.clone()),
        );
        let ctx_after_restart = crate::model::WorkloadSelectionContext {
            task_id: "task_post_restart".to_string(),
            caller_kind: "background".to_string(),
            caller_id: "Builder".to_string(),
            workload: crate::model::WorkloadKind::Debugging,
            required_context_tokens: 2048,
            require_structured_json: true,
            explicit_override: None,
        };
        let msgs_restart = vec![crate::model::ChatMessage {
            role: "user".to_string(),
            content: "Post-restart call attempt".to_string(),
        }];
        let err_restart = client_restarted
            .complete_chat_for_workload(&ctx_after_restart, "System prompt", &msgs_restart)
            .await
            .expect_err("Restart must not reset or bypass the rolling 24h quota");
        assert!(
            err_restart.to_string().contains("calls_per_24h_exceeded"),
            "Expected 24h quota exhaustion after restart, got: {}",
            err_restart
        );

        let final_summary = store_restarted
            .get_frontier_budget_summary(&models_cfg.frontier_budget, 20)
            .unwrap();
        assert_eq!(final_summary.used_calls_24h, 3);
        assert_eq!(final_summary.remaining_calls_24h, 0);
        assert!(!final_summary.blocked_tasks.is_empty());
    }

    #[tokio::test]
    async fn test_provider_selection_and_fallback_respect_capabilities_and_configuration() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_provider_selection.db");
        let log_dir = dir.path().join("logs");
        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir.clone(), 5, 2);

        // 1. Verify marketing product labels and raw secrets in credential_env are rejected by config validation
        let mut bad_cfg =
            crate::config::AppConfig::parse_and_validate(crate::config::DEFAULT_CONFIG_TOML)
                .unwrap();
        bad_cfg.models.providers[1].permitted_models = vec!["ChatGPT".to_string()];
        assert!(
            bad_cfg.validate().is_err(),
            "Marketing product label 'ChatGPT' must be rejected in favor of real API model identifiers"
        );

        let mut bad_secret_cfg =
            crate::config::AppConfig::parse_and_validate(crate::config::DEFAULT_CONFIG_TOML)
                .unwrap();
        bad_secret_cfg.models.providers[1].credential_env =
            Some("sk-proj-1234567890abcdefghijklmnop".to_string());
        assert!(
            bad_secret_cfg.validate().is_err(),
            "Raw API key in credential_env must be rejected"
        );

        // 2. Verify secret redaction never leaks credential values
        let secret_env = "TRIAD_TEST_ANTHROPIC_SECRET_ENV";
        let secret_val = "sk-ant-secret-key-val-88888888";
        std::env::set_var(secret_env, secret_val);
        let dummy_prov = crate::config::ModelProviderConfig {
            id: "anthropic".to_string(),
            provider_type: "anthropic".to_string(),
            enabled: true,
            base_url: "https://api.anthropic.com/v1".to_string(),
            transport: "anthropic_messages".to_string(),
            credential_env: Some(secret_env.to_string()),
            model_policy: "escalate_only".to_string(),
            permitted_models: vec!["claude-sonnet-4-20250514".to_string()],
            fallback_provider: None,
            is_frontier: true,
        };
        let redacted = crate::model::redact_secrets(
            &format!("Header x-api-key: {} failed with Bearer {}", secret_val, secret_val),
            &[dummy_prov],
        );
        assert!(
            !redacted.contains(secret_val),
            "Redacted output must never contain the raw secret: {}",
            redacted
        );
        assert!(redacted.contains("[REDACTED_SECRET]"));

        // 3. Spin up mock servers for native Anthropic Messages (/v1/messages) and native Gemini (/v1beta/models/...:generateContent)
        let anthropic_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let anthropic_addr = anthropic_listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = anthropic_listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let req_str = String::from_utf8_lossy(&buf[..n]).to_string();
                assert!(
                    req_str.contains("POST /v1/messages"),
                    "Anthropic native adapter must hit /v1/messages"
                );
                assert!(
                    req_str.to_lowercase().contains("x-api-key:")
                        && req_str.to_lowercase().contains("anthropic-version: 2023-06-01"),
                    "Anthropic native adapter must send x-api-key and anthropic-version headers"
                );
                let body = r#"{"content":[{"type":"text","text":"anthropic_native_ok"}],"usage":{"input_tokens":50,"output_tokens":25}}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        });

        let gemini_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let gemini_addr = gemini_listener.local_addr().unwrap();
        let gemini_env = "TRIAD_TEST_GEMINI_SECRET_ENV";
        std::env::set_var(gemini_env, "AIzaSyTestGeminiKey7777777");
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = gemini_listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                let req_str = String::from_utf8_lossy(&buf[..n]).to_string();
                assert!(
                    req_str.contains(":generateContent"),
                    "Gemini native adapter must hit :generateContent"
                );
                assert!(
                    req_str.to_lowercase().contains("x-goog-api-key:"),
                    "Gemini native adapter must send x-goog-api-key header"
                );
                let body = r#"{"candidates":[{"content":{"parts":[{"text":"gemini_native_ok"}]}}],"usageMetadata":{"promptTokenCount":40,"candidatesTokenCount":20,"totalTokenCount":60}}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
            }
        });

        let mut models_cfg =
            crate::config::AppConfig::parse_and_validate(crate::config::DEFAULT_CONFIG_TOML)
                .unwrap()
                .models;
        models_cfg.frontier_budget.max_calls_per_24h = 10;
        models_cfg.frontier_budget.max_calls_per_task = 2;
        models_cfg.selection.min_eval_runs_for_empirical_ranking = 2;
        models_cfg.selection.preference_order = vec![
            "anthropic:claude-sonnet-4-20250514".to_string(),
            "gemini:gemini-2.5-pro".to_string(),
            "local_snap:smollm3-3b".to_string(),
        ];
        models_cfg.providers = vec![
            crate::config::ModelProviderConfig {
                id: "local_snap".to_string(),
                provider_type: "local_snap".to_string(),
                enabled: true,
                base_url: models_cfg.chat.base_url.clone(),
                transport: "openai_compatible".to_string(),
                credential_env: None,
                model_policy: "all".to_string(),
                permitted_models: vec!["smollm3-3b".to_string()],
                fallback_provider: None,
                is_frontier: false,
            },
            // Provider with insufficient context window & codegen disabled for ComplexRustGeneration
            crate::config::ModelProviderConfig {
                id: "small_ctx_provider".to_string(),
                provider_type: "openai".to_string(),
                enabled: true,
                base_url: format!("http://{}", anthropic_addr),
                transport: "openai_compatible".to_string(),
                credential_env: Some(secret_env.to_string()),
                model_policy: "escalate_only".to_string(),
                permitted_models: vec!["gpt-4.1-mini-short".to_string()],
                fallback_provider: Some("anthropic".to_string()),
                is_frontier: true,
            },
            crate::config::ModelProviderConfig {
                id: "anthropic".to_string(),
                provider_type: "anthropic".to_string(),
                enabled: true,
                base_url: format!("http://{}/v1", anthropic_addr),
                transport: "anthropic_messages".to_string(),
                credential_env: Some(secret_env.to_string()),
                model_policy: "escalate_only".to_string(),
                permitted_models: vec!["claude-sonnet-4-20250514".to_string()],
                fallback_provider: Some("gemini".to_string()),
                is_frontier: true,
            },
            crate::config::ModelProviderConfig {
                id: "gemini".to_string(),
                provider_type: "gemini".to_string(),
                enabled: true,
                base_url: format!("http://{}/v1beta", gemini_addr),
                transport: "gemini_native".to_string(),
                credential_env: Some(gemini_env.to_string()),
                model_policy: "escalate_only".to_string(),
                permitted_models: vec!["gemini-2.5-pro".to_string()],
                fallback_provider: Some("local_snap".to_string()),
                is_frontier: true,
            },
        ];
        models_cfg.catalog = vec![
            crate::config::ModelCapabilityConfig {
                provider_id: "local_snap".to_string(),
                model_id: "smollm3-3b".to_string(),
                context_window_tokens: 8192,
                max_output_tokens: 1024,
                supports_structured_json: true,
                supports_code_generation: true,
                supports_reasoning_trace: false,
                supports_tool_use: true,
                is_frontier: false,
                input_price_micros_per_1m: Some(0),
                output_price_micros_per_1m: Some(0),
                reasoning_score_bps: 620,
                coding_score_bps: 600,
                synthesis_score_bps: 640,
                critique_score_bps: 630,
            },
            crate::config::ModelCapabilityConfig {
                provider_id: "small_ctx_provider".to_string(),
                model_id: "gpt-4.1-mini-short".to_string(),
                context_window_tokens: 2048,
                max_output_tokens: 512,
                supports_structured_json: true,
                supports_code_generation: false, // Ineligible for code generation!
                supports_reasoning_trace: false,
                supports_tool_use: false,
                is_frontier: true,
                input_price_micros_per_1m: Some(100_000),
                output_price_micros_per_1m: Some(200_000),
                reasoning_score_bps: 800,
                coding_score_bps: 800,
                synthesis_score_bps: 800,
                critique_score_bps: 800,
            },
            crate::config::ModelCapabilityConfig {
                provider_id: "anthropic".to_string(),
                model_id: "claude-sonnet-4-20250514".to_string(),
                context_window_tokens: 200_000,
                max_output_tokens: 8192,
                supports_structured_json: true,
                supports_code_generation: true,
                supports_reasoning_trace: true,
                supports_tool_use: true,
                is_frontier: true,
                input_price_micros_per_1m: Some(3_000_000),
                output_price_micros_per_1m: Some(15_000_000),
                reasoning_score_bps: 950,
                coding_score_bps: 950,
                synthesis_score_bps: 950,
                critique_score_bps: 950,
            },
            crate::config::ModelCapabilityConfig {
                provider_id: "gemini".to_string(),
                model_id: "gemini-2.5-pro".to_string(),
                context_window_tokens: 1_000_000,
                max_output_tokens: 8192,
                supports_structured_json: true,
                supports_code_generation: true,
                supports_reasoning_trace: true,
                supports_tool_use: true,
                is_frontier: true,
                input_price_micros_per_1m: Some(1_250_000),
                output_price_micros_per_1m: Some(10_000_000),
                reasoning_score_bps: 945,
                coding_score_bps: 945,
                synthesis_score_bps: 950,
                critique_score_bps: 940,
            },
        ];

        let mut client = crate::model::ModelClient::from_models_config(
            &models_cfg,
            Some(store.clone()),
            Some(logger.clone()),
        );

        // A. Local workloads (`Routing`, `Summarization`, `CheapWork`) must stay on local model
        for wk in [
            crate::model::WorkloadKind::Routing,
            crate::model::WorkloadKind::Summarization,
            crate::model::WorkloadKind::CheapWork,
        ] {
            let dec = client
                .select_model_for_workload(&crate::model::WorkloadSelectionContext {
                    task_id: format!("local_{}", wk.as_str()),
                    caller_kind: "agent".to_string(),
                    caller_id: "Explorer".to_string(),
                    workload: wk,
                    required_context_tokens: 1024,
                    require_structured_json: false,
                    explicit_override: None,
                })
                .await
                .unwrap();
            assert_eq!(dec.provider_id, "local_snap");
            assert_eq!(dec.model_id, "smollm3-3b");
            assert!(!dec.is_frontier);
        }

        // B. Escalated workload (`ComplexRustGeneration` requiring code_generation and 16k context)
        // when empirical evaluation evidence is thin (0 evaluations recorded < min 2):
        // - Rejects `small_ctx_provider:gpt-4.1-mini-short` due to capabilities (`context_window` and `supports_code_generation=false`)
        // - Falls back to configured `preference_order` (`anthropic:claude-sonnet-4-20250514`) and explicitly says so!
        let thin_ctx = crate::model::WorkloadSelectionContext {
            task_id: "task_thin_eval".to_string(),
            caller_kind: "specialist".to_string(),
            caller_id: "SolverSpecialist".to_string(),
            workload: crate::model::WorkloadKind::ComplexRustGeneration,
            required_context_tokens: 16_000,
            require_structured_json: true,
            explicit_override: None,
        };
        let dec_thin = client.select_model_for_workload(&thin_ctx).await.unwrap();
        assert_eq!(dec_thin.provider_id, "anthropic");
        assert_eq!(dec_thin.model_id, "claude-sonnet-4-20250514");
        assert!(
            dec_thin.used_preference_order_fallback,
            "Must flag thin evaluation evidence fallback"
        );
        assert!(
            dec_thin
                .ranking_basis
                .contains("thin_evaluation_evidence_fallback_to_configured_preference_order"),
            "Selection basis must explicitly state thin evidence fallback: {}",
            dec_thin.ranking_basis
        );
        assert!(
            dec_thin
                .fallback_reason
                .as_deref()
                .unwrap_or_default()
                .contains("small_ctx_provider:gpt-4.1-mini-short"),
            "Fallback reason must report rejected candidates: {:?}",
            dec_thin.fallback_reason
        );

        // Dispatch through native Anthropic adapter and verify response
        let ant_resp = client
            .complete_chat_for_workload(
                &thin_ctx,
                "System",
                &[crate::model::ChatMessage {
                    role: "user".to_string(),
                    content: "Test anthropic native".to_string(),
                }],
            )
            .await
            .unwrap();
        assert_eq!(ant_resp, "anthropic_native_ok");

        // C. Record empirical evaluations showing `gemini:gemini-2.5-pro` outperforms `anthropic:claude-sonnet-4-20250514`
        // on `complex_rust_generation` (>= 2 evaluations each). Selection must now pick `gemini:gemini-2.5-pro` by empirical score!
        for _ in 0..2 {
            store
                .record_model_evaluation(
                    "anthropic",
                    "claude-sonnet-4-20250514",
                    "complex_rust_generation",
                    true,
                    false,
                    0.50,
                    400,
                    "partial verification",
                )
                .unwrap();
            store
                .record_model_evaluation(
                    "gemini",
                    "gemini-2.5-pro",
                    "complex_rust_generation",
                    true,
                    true,
                    0.98,
                    350,
                    "full verification passed",
                )
                .unwrap();
        }

        let emp_ctx = crate::model::WorkloadSelectionContext {
            task_id: "task_empirical_eval".to_string(),
            caller_kind: "specialist".to_string(),
            caller_id: "SolverSpecialist".to_string(),
            workload: crate::model::WorkloadKind::ComplexRustGeneration,
            required_context_tokens: 16_000,
            require_structured_json: true,
            explicit_override: None,
        };
        let dec_emp = client.select_model_for_workload(&emp_ctx).await.unwrap();
        assert_eq!(dec_emp.provider_id, "gemini");
        assert_eq!(dec_emp.model_id, "gemini-2.5-pro");
        assert!(
            !dec_emp.used_preference_order_fallback,
            "Empirical evaluation evidence is now sufficient (>= 2 evals)"
        );
        assert!(
            dec_emp
                .ranking_basis
                .contains("empirical_evaluation_and_capability_ranking"),
            "Basis must reflect empirical evaluation ranking: {}",
            dec_emp.ranking_basis
        );

        // Dispatch through native Gemini adapter and verify response
        let gem_resp = client
            .complete_chat_for_workload(
                &emp_ctx,
                "System",
                &[crate::model::ChatMessage {
                    role: "user".to_string(),
                    content: "Test gemini native".to_string(),
                }],
            )
            .await
            .unwrap();
        assert_eq!(gem_resp, "gemini_native_ok");

        // D. Explicit override (`models.selection.explicit_override`)
        models_cfg.selection.explicit_override = Some("anthropic:claude-sonnet-4-20250514".to_string());
        client.reload_models_config(models_cfg.clone()).await;
        let dec_override = client.select_model_for_workload(&emp_ctx).await.unwrap();
        assert_eq!(dec_override.provider_id, "anthropic");
        assert_eq!(dec_override.model_id, "claude-sonnet-4-20250514");
        assert!(dec_override.ranking_basis.contains("explicit_override"));

        // E. When frontier credentials are unavailable for escalated providers, selection falls back
        // to local_snap (`smollm3-3b`) and logs the fallback reason.
        models_cfg.selection.explicit_override = None;
        for p in &mut models_cfg.providers {
            if p.is_frontier {
                p.credential_env = Some("TRIAD_UNSET_CREDENTIAL_ENV_VAR_XYZ".to_string());
            }
        }
        client.reload_models_config(models_cfg.clone()).await;

        let dec_fallback = client
            .select_model_for_workload(&crate::model::WorkloadSelectionContext {
                task_id: "task_missing_creds_fallback".to_string(),
                caller_kind: "agent".to_string(),
                caller_id: "Sceptic".to_string(),
                workload: crate::model::WorkloadKind::CriticalReview,
                required_context_tokens: 2048,
                require_structured_json: true,
                explicit_override: None,
            })
            .await
            .unwrap();
        assert_eq!(dec_fallback.provider_id, "local_snap");
        assert_eq!(dec_fallback.model_id, "smollm3-3b");
        assert!(
            dec_fallback
                .fallback_reason
                .as_deref()
                .unwrap_or_default()
                .contains("missing credential env"),
            "Expected missing credential env in fallback_reason, got: {:?}",
            dec_fallback.fallback_reason
        );
    }

    #[tokio::test]
    async fn test_bounded_live_smoke_within_quota_system() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_live_smoke.db");
        let log_dir = dir.path().join("logs");
        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir.clone(), 5, 2);

        let (mut cfg, _, _) = crate::config::ensure_canonical_config().unwrap();
        // Enforce strict bounded quota for the live smoke test: max 1 call per task, max 3 per 24h, concurrency 1
        cfg.models.frontier_budget.max_calls_per_24h = 3;
        cfg.models.frontier_budget.max_calls_per_task = 1;
        cfg.models.frontier_budget.max_concurrency = 1;
        cfg.models.frontier_budget.max_retries = 1;
        cfg.models.frontier_budget.exhaustion_policy = "continue_locally".to_string();

        let client = crate::model::ModelClient::from_models_config(
            &cfg.models,
            Some(store.clone()),
            Some(logger.clone()),
        );

        // 1. Refresh live provider discovery
        let discovered = client.refresh_provider_discovery(true).await;
        assert!(
            !discovered.is_empty(),
            "Provider discovery must record status for configured providers"
        );

        // 2. Verify dedicated embedding endpoint (`nomic-embed-text-v1-5`) works independently of chat model selection
        let emb = client
            .embed_text("bounded live smoke test embedding verification", false)
            .await
            .expect("Live dedicated Nomic embedding endpoint must succeed");
        assert!(
            emb.len() >= 128,
            "Expected valid embedding vector from dedicated embedding endpoint, got len={}",
            emb.len()
        );

        // 3. Execute one bounded live call inside the quota system (`HardSynthesis`).
        // If any external frontier provider credential env var is configured in the environment, it will
        // reserve 1 frontier attempt in `frontier_budget_ledger` and call that provider; otherwise it
        // selects the configured local snap fallback (`local_snap:smollm3-3b`) and logs the basis & missing-credential fallback reason.
        let smoke_ctx = crate::model::WorkloadSelectionContext {
            task_id: "live_smoke_task_1".to_string(),
            caller_kind: "agent".to_string(),
            caller_id: "Explorer".to_string(),
            workload: crate::model::WorkloadKind::HardSynthesis,
            required_context_tokens: 1024,
            require_structured_json: false,
            explicit_override: None,
        };
        let msgs = vec![crate::model::ChatMessage {
            role: "user".to_string(),
            content: "Reply with 'LIVE_SMOKE_OK' in one short sentence.".to_string(),
        }];
        let resp = client
            .complete_chat_for_workload(&smoke_ctx, "You are a concise assistant.", &msgs)
            .await
            .expect("Bounded live smoke call must succeed within the quota system");
        assert!(!resp.trim().is_empty());

        let selections = store.list_recent_model_selections(5).unwrap();
        assert!(
            !selections.is_empty(),
            "Model selection decision and ranking basis must be persisted"
        );
        let summary = store
            .get_frontier_budget_summary(&cfg.models.frontier_budget, 5)
            .unwrap();
        assert!(
            summary.used_calls_24h <= 1,
            "Bounded live smoke test must use at most 1 frontier call"
        );
    }

    #[tokio::test]
    async fn test_same_workflow_runs_under_two_different_domain_configs() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_two_domains.db");
        let log_dir = dir.path().join("logs");
        let ws_dir = dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();

        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir.clone(), 5, 2);
        let (mut cfg, _, _) = crate::config::ensure_canonical_config().unwrap();

        // Domain 1: Frontier AI Research & Verification
        let eval_path_1 = dir.path().join("eval_domain_1.json");
        std::fs::write(
            &eval_path_1,
            r#"[
                {
                    "id": "ai_q1",
                    "source_origin": "arXiv:2305.20050 (Lightman et al., 2023)",
                    "author": "external_domain_config",
                    "case_kind": "question",
                    "topic": "Process Reward Models & Test-Time Compute",
                    "category": "Process Reward Models",
                    "in_triad_strong_coverage": true,
                    "question": "According to Lightman et al. (arXiv:2305.20050), why does process supervision outperform outcome supervision on multi-step mathematical reasoning?",
                    "expected_keywords": ["step", "credit assignment", "process"],
                    "expected_citation_markers": ["2305.20050"],
                    "requires_solver": false,
                    "expects_contradiction_detection": false,
                    "data_kind": "real_data",
                    "ground_truth_summary": "Process supervision provides step-level credit assignment and localizes the exact step where an error occurs."
                }
            ]"#,
        )
        .unwrap();

        cfg.domain.name = "frontier_ai_research_and_verification".to_string();
        cfg.domain.display_name = "Frontier AI Research & Verification".to_string();
        cfg.domain.evaluation_set_path = eval_path_1.display().to_string();
        cfg.domain.topics = vec![crate::config::DomainTopicSeed {
            topic: "Process Reward Models & Test-Time Compute".to_string(),
            discipline: "Machine Learning & Reasoning".to_string(),
            priority: 100,
            hypothesis: Some("Step-level process supervision improves credit assignment.".to_string()),
        }];

        // Seed full-text source and trusted claim in Domain 1
        let raw_paper_d1 = crate::web::ResearchSourcePaper {
            title: "Let's Verify Step by Step".to_string(),
            doi: None,
            arxiv_id: Some("2305.20050".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2023-05-31T00:00:00Z".to_string()),
            authors: vec!["Hunter Lightman".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2305.20050".to_string(),
            code_url: None,
            abstract_or_excerpt: "Process supervision provides step-level credit assignment and localizes the exact step where an error occurs.".to_string(),
            experimental_details: Some("Evaluated on MATH dataset.".to_string()),
            limitations_excerpt: Some("Higher annotation cost.".to_string()),
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work_d1 = &crate::web::deduplicate_and_resolve_provenance(vec![raw_paper_d1])[0];
        store.upsert_research_work(work_d1).unwrap();
        let payload_d1 = crate::web::FullTextIngestionPayload {
            canonical_id: work_d1.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "latex_tar_gz".to_string(),
            raw_original: "\\section{Introduction}\nProcess supervision provides step-level credit assignment.".to_string(),
            abstract_text: work_d1.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "1 Introduction".to_string(),
                section_kind: "method".to_string(),
                content: "Process supervision provides step-level credit assignment and localizes errors.".to_string(),
                source_location: "https://arxiv.org/abs/2305.20050#sec1".to_string(),
                token_estimate: 20,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 2,
                tables_found: 1,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work_d1,
                &payload_d1,
                Some("Process Reward Models & Test-Time Compute"),
                &cfg.domain.name,
            )
            .unwrap();
        let c1_id = store
            .propose_claim_in_domain(
                "According to Lightman et al. (arXiv:2305.20050), process supervision outperforms outcome supervision because it provides step-level credit assignment and localizes reasoning errors [source: arxiv:2305.20050, openalex:W230520050].",
                "https://arxiv.org/abs/2305.20050",
                "Explorer",
                &cfg.domain.name,
            )
            .unwrap();
        store.record_vote(&c1_id, "Explorer", "endorse", 1, Some("Verified")).unwrap();
        store.record_vote(&c1_id, "Sceptic", "endorse", 1, Some("Verified")).unwrap();

        let m_client = crate::model::ModelClient::from_models_config(
            &cfg.models,
            Some(store.clone()),
            Some(logger.clone()),
        );
        let web = crate::web::WebExplorer::new(&cfg.web.wikipedia_api, cfg.web.request_timeout_seconds);
        let adapter: Arc<dyn crate::execution::ExecutionAdapter> =
            Arc::new(crate::execution::BubblewrapAdapter::probe());
        let (constitution, constitution_raw, constitution_digest) =
            crate::constitution::load_and_verify_constitution().unwrap();
        let coordinator = Arc::new(crate::coordinator::Coordinator::new(
            cfg.clone(),
            constitution,
            constitution_raw,
            constitution_digest,
            store.clone(),
            m_client,
            web,
            adapter,
            logger,
            ws_dir.clone(),
        ));
        let engine = Arc::new(crate::consultation::ConsultationEngine::new(coordinator.clone()));
        let runner = crate::evaluation::EvaluationRunner::new(engine.clone());

        let eval_run_1 = runner.run_domain_evaluation(None).await.unwrap();
        assert_eq!(eval_run_1.domain, "frontier_ai_research_and_verification");
        assert_eq!(eval_run_1.item_count, 1);
        assert!(eval_run_1.domain_competence_verified);
        let profile_1 = engine.expert_profile().await.unwrap();
        assert_eq!(
            profile_1["domain"]["name"].as_str().unwrap(),
            "frontier_ai_research_and_verification"
        );

        // Domain 2: Computational Systems Biology & Gene Regulatory Networks (same coordinator code, different DomainSection config)
        let eval_path_2 = dir.path().join("eval_domain_2.json");
        std::fs::write(
            &eval_path_2,
            r#"[
                {
                    "id": "bio_q1",
                    "source_origin": "doi:10.1016/j.jtbi.2020.110234 (Novak, 2020)",
                    "author": "external_domain_config",
                    "case_kind": "question",
                    "topic": "Michaelis-Menten Enzyme Kinetics & Hill Cooperativity",
                    "category": "Enzyme Kinetics",
                    "in_triad_strong_coverage": true,
                    "question": "In enzyme kinetics (doi:10.1016/j.jtbi.2020.110234), how does a Hill coefficient n > 1 affect substrate binding cooperativity compared to standard Michaelis-Menten kinetics?",
                    "expected_keywords": ["sigmoidal", "cooperativity", "positive"],
                    "expected_citation_markers": ["10.1016/j.jtbi.2020.110234"],
                    "requires_solver": false,
                    "expects_contradiction_detection": false,
                    "data_kind": "real_data",
                    "ground_truth_summary": "A Hill coefficient n > 1 produces sigmoidal velocity curves reflecting positive cooperativity in multi-site substrate binding."
                }
            ]"#,
        )
        .unwrap();

        {
            let mut locked_cfg = coordinator.config.lock().await;
            locked_cfg.domain.name = "computational_systems_biology".to_string();
            locked_cfg.domain.display_name =
                "Computational Systems Biology & Gene Regulatory Networks".to_string();
            locked_cfg.domain.scope = vec![
                "Enzyme kinetics, gene regulatory networks, stochastic biochemical reaction networks".to_string(),
            ];
            locked_cfg.domain.evaluation_set_path = eval_path_2.display().to_string();
            locked_cfg.domain.topics = vec![crate::config::DomainTopicSeed {
                topic: "Michaelis-Menten Enzyme Kinetics & Hill Cooperativity".to_string(),
                discipline: "Systems Biology".to_string(),
                priority: 100,
                hypothesis: Some("Hill coefficient n > 1 yields sigmoidal positive cooperativity.".to_string()),
            }];
        }

        let raw_paper_d2 = crate::web::ResearchSourcePaper {
            title: "Hill Cooperativity in Biochemical Reaction Networks".to_string(),
            doi: Some("10.1016/j.jtbi.2020.110234".to_string()),
            arxiv_id: None,
            version: Some("v1".to_string()),
            published_date: Some("2020-03-01T00:00:00Z".to_string()),
            authors: vec!["A. Novák".to_string()],
            source_connector: "pubmed".to_string(),
            url: "https://doi.org/10.1016/j.jtbi.2020.110234".to_string(),
            code_url: None,
            abstract_or_excerpt: "A Hill coefficient n > 1 produces sigmoidal velocity curves reflecting positive cooperativity in multi-site substrate binding.".to_string(),
            experimental_details: Some("ODE bifurcation analysis.".to_string()),
            limitations_excerpt: Some("Steady-state approximation.".to_string()),
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work_d2 = &crate::web::deduplicate_and_resolve_provenance(vec![raw_paper_d2])[0];
        store.upsert_research_work(work_d2).unwrap();
        let payload_d2 = crate::web::FullTextIngestionPayload {
            canonical_id: work_d2.canonical_id.clone(),
            connector: "pubmed".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "pdf".to_string(),
            raw_original: "1 Introduction\nHill coefficient n > 1 produces sigmoidal velocity curves and positive cooperativity.".to_string(),
            abstract_text: work_d2.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "1 Introduction".to_string(),
                section_kind: "method".to_string(),
                content: "Hill coefficient n > 1 produces sigmoidal velocity curves and positive cooperativity.".to_string(),
                source_location: "https://doi.org/10.1016/j.jtbi.2020.110234#sec1".to_string(),
                token_estimate: 20,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 2,
                tables_found: 1,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work_d2,
                &payload_d2,
                Some("Michaelis-Menten Enzyme Kinetics & Hill Cooperativity"),
                "computational_systems_biology",
            )
            .unwrap();
        let c2_id = store
            .propose_claim_in_domain(
                "In enzyme kinetics (doi:10.1016/j.jtbi.2020.110234), a Hill coefficient n > 1 produces a sigmoidal reaction velocity curve and positive cooperativity across binding sites [source: pubmed:110234, crossref:10.1016/j.jtbi.2020.110234].",
                "https://doi.org/10.1016/j.jtbi.2020.110234",
                "Explorer",
                "computational_systems_biology",
            )
            .unwrap();
        store.record_vote(&c2_id, "Explorer", "endorse", 2, Some("Verified")).unwrap();
        store.record_vote(&c2_id, "Sceptic", "endorse", 2, Some("Verified")).unwrap();

        let eval_run_2 = runner.run_domain_evaluation(None).await.unwrap();
        assert_eq!(eval_run_2.domain, "computational_systems_biology");
        assert_eq!(eval_run_2.item_count, 1);
        assert!(eval_run_2.domain_competence_verified);

        let profile_2 = engine.expert_profile().await.unwrap();
        assert_eq!(
            profile_2["domain"]["name"].as_str().unwrap(),
            "computational_systems_biology"
        );
        // Verify strict domain isolation between the two domains
        let cov_d1 = store
            .list_domain_topic_coverage("frontier_ai_research_and_verification")
            .unwrap();
        let cov_d2 = store
            .list_domain_topic_coverage("computational_systems_biology")
            .unwrap();
        assert_eq!(cov_d1.len(), 1);
        assert_eq!(cov_d2.len(), 1);
        assert_ne!(cov_d1[0].topic, cov_d2[0].topic);
    }

    #[tokio::test]
    async fn test_evaluation_runs_against_configured_set_and_rejects_triad_authored_set() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_eval_authorship.db");
        let log_dir = dir.path().join("logs");
        let ws_dir = dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();

        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir, 5, 2);
        let (cfg, _, _) = crate::config::ensure_canonical_config().unwrap();

        // 1. Verify config validation rejects Triad-authored inline evaluation cases
        let mut bad_cfg = cfg.clone();
        bad_cfg.domain.evaluation_cases = vec![crate::config::DomainEvaluationCase {
            id: "self_q1".to_string(),
            source_origin: "authored by explorer".to_string(),
            author: Some("Explorer".to_string()),
            case_kind: "question".to_string(),
            topic: "Test Topic".to_string(),
            category: "Test".to_string(),
            in_triad_strong_coverage: false,
            question: "What did Explorer just invent?".to_string(),
            objective: String::new(),
            constraints: vec![],
            expected_keywords: vec!["answer".to_string()],
            expected_citation_markers: vec!["arxiv:1234.5678".to_string()],
            requires_solver: false,
            expects_contradiction_detection: false,
            data_kind: Some("real_data".to_string()),
            ground_truth_summary: "Known rubric summary".to_string(),
        }];
        let cfg_err = bad_cfg.validate().expect_err("Config must reject Explorer-authored evaluation case");
        assert!(
            cfg_err.to_string().contains("Triad never authors its own evaluation set"),
            "Expected Triad self-authoring rejection, got: {}",
            cfg_err
        );

        // 2. Verify EvaluationRunner rejects a benchmark JSON file authored by Triad or its agents
        let m_client = crate::model::ModelClient::from_models_config(
            &cfg.models,
            Some(store.clone()),
            Some(logger.clone()),
        );
        let web = crate::web::WebExplorer::new(&cfg.web.wikipedia_api, cfg.web.request_timeout_seconds);
        let adapter: Arc<dyn crate::execution::ExecutionAdapter> =
            Arc::new(crate::execution::BubblewrapAdapter::probe());
        let (constitution, constitution_raw, constitution_digest) =
            crate::constitution::load_and_verify_constitution().unwrap();
        let coordinator = Arc::new(crate::coordinator::Coordinator::new(
            cfg.clone(),
            constitution,
            constitution_raw,
            constitution_digest,
            store.clone(),
            m_client,
            web,
            adapter,
            logger,
            ws_dir,
        ));
        let engine = Arc::new(crate::consultation::ConsultationEngine::new(coordinator.clone()));
        let runner = crate::evaluation::EvaluationRunner::new(engine);

        for forbidden_author in [
            "triad",
            "Explorer",
            "Sceptic",
            "Builder",
            "authored by specialist",
            "self_authored",
        ] {
            let bad_bench_path = dir.path().join(format!("bad_bench_{}.json", forbidden_author.replace([':', ' '], "_")));
            std::fs::write(
                &bad_bench_path,
                serde_json::json!([
                    {
                        "id": "item_1",
                        "source_origin": forbidden_author,
                        "author": forbidden_author,
                        "case_kind": "question",
                        "topic": "Process Reward Models",
                        "category": "PRM",
                        "in_triad_strong_coverage": true,
                        "question": "Self-generated question?",
                        "expected_keywords": ["step"],
                        "expected_citation_markers": ["2305.20050"],
                        "requires_solver": false,
                        "ground_truth_summary": "Self-generated answer"
                    }
                ])
                .to_string(),
            )
            .unwrap();

            let err = runner
                .run_domain_evaluation(Some(&bad_bench_path))
                .await
                .expect_err("EvaluationRunner must reject Triad-authored evaluation set");
            assert!(
                err.to_string()
                    .contains("Triad never authors its own evaluation set"),
                "Expected Triad self-authoring rejection for author '{}', got: {}",
                forbidden_author,
                err
            );
        }

        // 3. Verify EvaluationRunner succeeds against a valid external domain-configured evaluation set
        // and records the exact plain-LLM baseline model Triad selected under current configuration and credentials.
        let valid_bench_path = dir.path().join("valid_external_bench.json");
        std::fs::write(
            &valid_bench_path,
            r#"[
                {
                    "id": "ext_q1",
                    "source_origin": "arXiv:2305.20050 (Lightman et al., 2023)",
                    "author": "external_domain_config",
                    "case_kind": "question",
                    "topic": "Process Reward Models & Test-Time Compute",
                    "category": "Process Reward Models",
                    "in_triad_strong_coverage": true,
                    "question": "Why does process supervision (arXiv:2305.20050) improve multi-step reasoning?",
                    "expected_keywords": ["step", "credit assignment"],
                    "expected_citation_markers": ["2305.20050"],
                    "requires_solver": false,
                    "expects_contradiction_detection": false,
                    "data_kind": "real_data",
                    "ground_truth_summary": "Step-level credit assignment localizes reasoning errors."
                }
            ]"#,
        )
        .unwrap();

        {
            let mut locked_cfg = coordinator.config.lock().await;
            locked_cfg.domain.evaluation_set_path = valid_bench_path.display().to_string();
        }

        let valid_record = runner.run_domain_evaluation(None).await.unwrap();
        assert_eq!(valid_record.item_count, 1);
        assert!(
            !valid_record.baseline_provider_id.is_empty()
                && !valid_record.baseline_model_id.is_empty(),
            "Must record which baseline model Triad selected under current configuration and credentials"
        );
        assert!(!valid_record.baseline_selection_basis.is_empty());
    }

    #[tokio::test]
    async fn test_scores_reported_separately_for_full_text_and_abstract_only_topics_and_real_vs_synthetic_experiments() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("triad_split_scores.db");
        let log_dir = dir.path().join("logs");
        let ws_dir = dir.path().join("workspace");
        std::fs::create_dir_all(&ws_dir).unwrap();

        let store = Store::open(&db_path).unwrap();
        let logger = crate::logging::Logger::new(log_dir, 5, 2);
        let (mut cfg, _, _) = crate::config::ensure_canonical_config().unwrap();
        cfg.domain.name = "frontier_ai_research_and_verification".to_string();
        cfg.domain.topics = vec![
            crate::config::DomainTopicSeed {
                topic: "Process Reward Models & Test-Time Compute".to_string(),
                discipline: "Reasoning".to_string(),
                priority: 100,
                hypothesis: Some("Step-level PRM pruning outperforms outcome-only best-of-N.".to_string()),
            },
            crate::config::DomainTopicSeed {
                topic: "Abstract-Only Speculative Decoding Topic".to_string(),
                discipline: "Inference".to_string(),
                priority: 80,
                hypothesis: Some("Draft-model acceptance rate governs speedup.".to_string()),
            },
            crate::config::DomainTopicSeed {
                topic: "Contradicted Quantization Topic".to_string(),
                discipline: "Systems".to_string(),
                priority: 70,
                hypothesis: Some("Outlier channels degrade sub-4-bit quantization.".to_string()),
            },
            crate::config::DomainTopicSeed {
                topic: "Thin Uninvestigated Topic".to_string(),
                discipline: "Theory".to_string(),
                priority: 60,
                hypothesis: Some("Uninvestigated hypothesis.".to_string()),
            },
        ];

        // 1. Seed Topic 1 as FULL-TEXT verified (`covered`)
        let raw_ft = crate::web::ResearchSourcePaper {
            title: "Let's Verify Step by Step".to_string(),
            doi: None,
            arxiv_id: Some("2305.20050".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2023-05-31T00:00:00Z".to_string()),
            authors: vec!["Hunter Lightman".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2305.20050".to_string(),
            code_url: Some("https://github.com/openai/prm800k".to_string()),
            abstract_or_excerpt: "Process supervision provides step-level credit assignment and localizes errors.".to_string(),
            experimental_details: Some("Evaluated on MATH.".to_string()),
            limitations_excerpt: Some("Annotation cost.".to_string()),
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work_ft = &crate::web::deduplicate_and_resolve_provenance(vec![raw_ft])[0];
        store.upsert_research_work(work_ft).unwrap();
        let payload_ft = crate::web::FullTextIngestionPayload {
            canonical_id: work_ft.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "latex_tar_gz".to_string(),
            raw_original: "\\section{Method}\nProcess supervision provides step-level credit assignment.".to_string(),
            abstract_text: work_ft.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "1 Method".to_string(),
                section_kind: "method".to_string(),
                content: "Process supervision provides step-level credit assignment and localizes reasoning errors.".to_string(),
                source_location: "https://arxiv.org/abs/2305.20050#sec1".to_string(),
                token_estimate: 25,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 3,
                tables_found: 2,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work_ft,
                &payload_ft,
                Some("Process Reward Models & Test-Time Compute"),
                &cfg.domain.name,
            )
            .unwrap();
        let cid_ft = store
            .propose_claim_in_domain(
                "Process Reward Models & Test-Time Compute: Lightman et al. (arXiv:2305.20050) show step-level credit assignment localizes errors and outperforms outcome supervision [source: arxiv:2305.20050, openalex:W1, semantic_scholar:S1].",
                "https://arxiv.org/abs/2305.20050",
                "Explorer",
                &cfg.domain.name,
            )
            .unwrap();
        store.record_vote(&cid_ft, "Explorer", "endorse", 1, Some("ok")).unwrap();
        store.record_vote(&cid_ft, "Sceptic", "endorse", 1, Some("ok")).unwrap();

        // 2. Seed Topic 2 as ABSTRACT-ONLY (`low_fidelity`)
        let raw_abs = crate::web::ResearchSourcePaper {
            title: "Fast Inference from Transformers via Speculative Decoding".to_string(),
            doi: None,
            arxiv_id: Some("2211.17192".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2022-11-30T00:00:00Z".to_string()),
            authors: vec!["Yaniv Leviathan".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2211.17192".to_string(),
            code_url: None,
            abstract_or_excerpt: "Speculative decoding uses a draft model acceptance rate to accelerate sampling without changing the target distribution.".to_string(),
            experimental_details: Some("Abstract only.".to_string()),
            limitations_excerpt: Some("Draft overhead.".to_string()),
            citations: vec![],
            availability_status: "metadata_only".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work_abs = &crate::web::deduplicate_and_resolve_provenance(vec![raw_abs])[0];
        store.upsert_research_work(work_abs).unwrap();
        let payload_abs = crate::web::FullTextIngestionPayload {
            canonical_id: work_abs.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "abstract_only".to_string(),
            format: "abstract".to_string(),
            raw_original: String::new(),
            abstract_text: work_abs.abstract_excerpt.clone(),
            is_full_text: false,
            is_orientation_only: false,
            sections: vec![],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 0,
                equations_preserved: 0,
                tables_found: 0,
                low_fidelity: true,
                low_fidelity_reason: Some("Abstract only".to_string()),
            },
        };
        store
            .store_ingestion_payload(
                work_abs,
                &payload_abs,
                Some("Abstract-Only Speculative Decoding Topic"),
                &cfg.domain.name,
            )
            .unwrap();

        // 3. Seed Topic 3 as CONTRADICTED (`contradicted`)
        let raw_contra = crate::web::ResearchSourcePaper {
            title: "AWQ: Activation-aware Weight Quantization".to_string(),
            doi: None,
            arxiv_id: Some("2306.00978".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2023-06-01T00:00:00Z".to_string()),
            authors: vec!["Ji Lin".to_string()],
            source_connector: "arxiv".to_string(),
            url: "https://arxiv.org/abs/2306.00978".to_string(),
            code_url: None,
            abstract_or_excerpt: "Protecting salient weight channels by activation scale preserves quantization accuracy.".to_string(),
            experimental_details: Some("INT4 evaluation.".to_string()),
            limitations_excerpt: Some("Calibration set dependency.".to_string()),
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2025-05-01T00:00:00Z".to_string(),
        };
        let work_contra = &crate::web::deduplicate_and_resolve_provenance(vec![raw_contra])[0];
        store.upsert_research_work(work_contra).unwrap();
        let payload_contra = crate::web::FullTextIngestionPayload {
            canonical_id: work_contra.canonical_id.clone(),
            connector: "arxiv".to_string(),
            ingestion_completeness: "full_text".to_string(),
            format: "html".to_string(),
            raw_original: "1 Introduction\nActivation-aware scaling protects 1% salient weights.".to_string(),
            abstract_text: work_contra.abstract_excerpt.clone(),
            is_full_text: true,
            is_orientation_only: false,
            sections: vec![crate::web::StructuredDocumentSection {
                section_index: 0,
                section_path: "1 Introduction".to_string(),
                section_kind: "method".to_string(),
                content: "Activation-aware scaling protects 1% salient weights.".to_string(),
                source_location: "https://arxiv.org/abs/2306.00978#sec1".to_string(),
                token_estimate: 15,
                is_full_text: true,
            }],
            harvested_references: vec![],
            extraction_quality: crate::web::DocumentExtractionQuality {
                sections_found: 1,
                equations_preserved: 1,
                tables_found: 1,
                low_fidelity: false,
                low_fidelity_reason: None,
            },
        };
        store
            .store_ingestion_payload(
                work_contra,
                &payload_contra,
                Some("Contradicted Quantization Topic"),
                &cfg.domain.name,
            )
            .unwrap();
        store
            .flag_source_contradiction(
                &work_contra.canonical_id,
                "Conflicting perplexity measurements under extreme outlier activations",
            )
            .unwrap();

        // 4. Record 1 real_data experiment and 1 synthetic_data experiment in this domain
        let mut spec_real = sample_complete_experiment_spec("real_data_reanalysis_tool");
        spec_real.domain = cfg.domain.name.clone();
        spec_real.executable_check_kind = "data_reanalysis".to_string();
        spec_real.data_kind = "real_data".to_string();
        spec_real.cited_source_ids = vec![work_ft.canonical_id.clone()];
        let reg_real = store.register_and_validate_experiment_spec(&mut spec_real).unwrap();
        store
            .record_structured_research_experiment(
                1,
                Some(&reg_real.id),
                Some("real_data_reanalysis_tool"),
                None,
                &reg_real.falsifiable_hypothesis,
                "Process Reward Models & Test-Time Compute",
                &reg_real.mechanism_under_test,
                r#"{"delta":-0.09}"#,
                r#"{"held_out_mse":0.12}"#,
                r#"{"baseline_mse":0.20,"candidate_mse":0.11,"held_out_mse":0.12}"#,
                "full_text_verified",
                false,
                &[],
                "Real-data reanalysis succeeded",
                "completed",
                "Verified on real data",
                false,
                None,
            )
            .unwrap();

        let mut spec_synth = sample_complete_experiment_spec("synthetic_simulation_tool");
        spec_synth.domain = cfg.domain.name.clone();
        spec_synth.executable_check_kind = "simulation".to_string();
        spec_synth.data_kind = "synthetic_data".to_string();
        spec_synth.cited_source_ids = vec![work_ft.canonical_id.clone()];
        let reg_synth = store.register_and_validate_experiment_spec(&mut spec_synth).unwrap();
        store
            .record_structured_research_experiment(
                2,
                Some(&reg_synth.id),
                Some("synthetic_simulation_tool"),
                None,
                &reg_synth.falsifiable_hypothesis,
                "Process Reward Models & Test-Time Compute",
                &reg_synth.mechanism_under_test,
                r#"{"delta":-0.09}"#,
                r#"{"held_out_mse":0.14}"#,
                r#"{"baseline_mse":0.22,"candidate_mse":0.13,"held_out_mse":0.14}"#,
                "full_text_verified",
                false,
                &[],
                "Synthetic simulation succeeded",
                "completed",
                "Verified on synthetic data",
                false,
                None,
            )
            .unwrap();

        // 5. Create external evaluation set with 1 full-text real_data problem and 1 abstract-only synthetic_data problem
        let eval_path = dir.path().join("split_eval_set.json");
        std::fs::write(
            &eval_path,
            r#"[
                {
                    "id": "ft_real_item",
                    "source_origin": "arXiv:2305.20050 (Lightman et al., 2023)",
                    "author": "external_domain_config",
                    "case_kind": "problem",
                    "topic": "Process Reward Models & Test-Time Compute",
                    "category": "Process Reward Models",
                    "in_triad_strong_coverage": true,
                    "question": "Calculate the budget split for Process Reward Models (arXiv:2305.20050) with 12 and 8 rollouts and explain step-level credit assignment.",
                    "expected_keywords": ["step", "credit assignment"],
                    "expected_citation_markers": ["2305.20050"],
                    "requires_solver": true,
                    "expects_contradiction_detection": false,
                    "data_kind": "real_data",
                    "ground_truth_summary": "Process supervision provides step-level credit assignment across 12 and 8 rollouts."
                },
                {
                    "id": "abs_synth_item",
                    "source_origin": "arXiv:2211.17192 (Leviathan et al., 2022)",
                    "author": "external_domain_config",
                    "case_kind": "review_case",
                    "topic": "Abstract-Only Speculative Decoding Topic",
                    "category": "Speculative Decoding",
                    "in_triad_strong_coverage": false,
                    "question": "Review conflicting evidence on speculative decoding draft acceptance rate (arXiv:2211.17192) and calculate 10 and 5.",
                    "expected_keywords": ["draft", "acceptance"],
                    "expected_citation_markers": ["2211.17192"],
                    "requires_solver": true,
                    "expects_contradiction_detection": true,
                    "data_kind": "synthetic_data",
                    "ground_truth_summary": "Draft model acceptance rate determines speculative decoding speedup."
                }
            ]"#,
        )
        .unwrap();
        cfg.domain.evaluation_set_path = eval_path.display().to_string();

        let m_client = crate::model::ModelClient::from_models_config(
            &cfg.models,
            Some(store.clone()),
            Some(logger.clone()),
        );
        let web = crate::web::WebExplorer::new(&cfg.web.wikipedia_api, cfg.web.request_timeout_seconds);
        let adapter: Arc<dyn crate::execution::ExecutionAdapter> =
            Arc::new(crate::execution::BubblewrapAdapter::probe());
        let (constitution, constitution_raw, constitution_digest) =
            crate::constitution::load_and_verify_constitution().unwrap();
        let coordinator = Arc::new(crate::coordinator::Coordinator::new(
            cfg.clone(),
            constitution,
            constitution_raw,
            constitution_digest,
            store.clone(),
            m_client,
            web,
            adapter,
            logger,
            ws_dir,
        ));
        let engine = Arc::new(crate::consultation::ConsultationEngine::new(coordinator.clone()));
        let runner = crate::evaluation::EvaluationRunner::new(engine.clone());

        let record = runner.run_domain_evaluation(None).await.unwrap();

        // Verify split reporting: 1 item in full_text_scores, 1 item in abstract_or_low_fidelity_scores
        assert_eq!(record.item_count, 2);
        assert_eq!(record.full_text_scores.item_count, 1);
        assert_eq!(record.abstract_or_low_fidelity_scores.item_count, 1);
        assert!(
            record.evaluated_corpus_full_text_share > 0.0
                && record.evaluated_corpus_full_text_share < 1.0,
            "Corpus full-text share must reflect mix of full-text and abstract-only sources: {}",
            record.evaluated_corpus_full_text_share
        );
        assert_eq!(record.real_data_experiment_count, 1);
        assert_eq!(record.synthetic_data_experiment_count, 1);

        // Domain-level headline score must equal ONLY the full-text score split, never mixed with thin/abstract-only!
        assert!(
            (record.triad_correctness - record.full_text_scores.triad_correctness).abs() < 1e-6,
            "Domain headline score must be backed strictly by full_text_scores"
        );

        // Verify topic coverage states: covered, low_fidelity, contradicted, thin
        let cov = store.list_domain_topic_coverage(&cfg.domain.name).unwrap();
        let status_for = |name: &str| -> String {
            cov.iter()
                .find(|c| c.topic == name)
                .map(|c| c.coverage_status.clone())
                .unwrap_or_default()
        };
        assert_eq!(
            status_for("Process Reward Models & Test-Time Compute"),
            "covered"
        );
        assert_eq!(
            status_for("Abstract-Only Speculative Decoding Topic"),
            "low_fidelity"
        );
        assert_eq!(status_for("Contradicted Quantization Topic"), "contradicted");
        assert_eq!(status_for("Thin Uninvestigated Topic"), "thin");

        // Verify thin, contradicted, and low-fidelity topics were fed into research_backlog and re-ingestion requests
        let backlog = store.list_research_backlog(20).unwrap();
        assert!(
            backlog
                .iter()
                .any(|b| b.topic == "Abstract-Only Speculative Decoding Topic" && b.priority >= 90),
            "Low-fidelity topic must be prioritized in research_backlog"
        );
        assert!(
            backlog
                .iter()
                .any(|b| b.topic == "Contradicted Quantization Topic" && b.priority >= 95),
            "Contradicted topic must be prioritized in research_backlog"
        );
        assert!(
            backlog
                .iter()
                .any(|b| b.topic == "Thin Uninvestigated Topic" && b.priority >= 85),
            "Thin topic must be prioritized in research_backlog"
        );

        let abs_src = store.get_tracked_source("arxiv:2211.17192").unwrap().unwrap();
        assert!(
            abs_src.retry_eligible,
            "Abstract-only source on low-fidelity topic must be marked retry_eligible for re-ingestion"
        );

        // Verify strict rule: when a domain has ONLY thin/abstract-only ingestion (0 full-text topics),
        // a score on thin ingestion is NEVER reported as a score on the domain (`domain_competence_verified == false` and domain score == 0.0)
        let thin_only_eval_path = dir.path().join("thin_only_eval.json");
        std::fs::write(
            &thin_only_eval_path,
            r#"[
                {
                    "id": "only_abs_item",
                    "source_origin": "arXiv:2211.17192 (Leviathan et al., 2022)",
                    "author": "external_domain_config",
                    "case_kind": "question",
                    "topic": "Abstract-Only Speculative Decoding Topic",
                    "category": "Speculative Decoding",
                    "in_triad_strong_coverage": false,
                    "question": "What does speculative decoding do (arXiv:2211.17192)?",
                    "expected_keywords": ["draft", "acceptance"],
                    "expected_citation_markers": ["2211.17192"],
                    "requires_solver": false,
                    "expects_contradiction_detection": false,
                    "data_kind": "synthetic_data",
                    "ground_truth_summary": "Draft model acceptance accelerates decoding."
                }
            ]"#,
        )
        .unwrap();
        let thin_rec = runner
            .run_domain_evaluation(Some(&thin_only_eval_path))
            .await
            .unwrap();
        assert_eq!(thin_rec.full_text_scores.item_count, 0);
        assert_eq!(thin_rec.abstract_or_low_fidelity_scores.item_count, 1);
        assert!(
            !thin_rec.domain_competence_verified,
            "Score on thin/abstract-only ingestion must never mark domain_competence_verified = true"
        );
        assert_eq!(
            thin_rec.triad_correctness, 0.0,
            "Score on thin ingestion must never be reported as a positive domain-level correctness score"
        );

        // Verify MCP `expert_profile` publishes all required fields and epistemic disclaimer
        let profile = engine.expert_profile().await.unwrap();
        assert!(profile["epistemic_policy"]
            .as_str()
            .unwrap_or_default()
            .contains("Coverage is never a competence claim"));
        assert!(profile["coverage"]["topics"].is_array());
        assert!(profile["full_text_share"]["corpus_full_text_share"].is_f64());
        assert_eq!(
            profile["experiments_breakdown"]["real_data_experiment_count"]
                .as_u64()
                .unwrap(),
            1
        );
        assert_eq!(
            profile["experiments_breakdown"]["synthetic_data_experiment_count"]
                .as_u64()
                .unwrap(),
            1
        );
        assert!(profile["connector_health"].is_array());
        assert!(profile["frontier_availability"].is_object());
        assert!(profile["last_evaluation_date"].is_string());
    }
}
