use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcRequest {
    Status,
    Pause,
    Resume,
    Stop,
    Ask { question: String },
    Challenge { task: String },
    Tell { info: String, source: Option<String> },
    Answer { question_id: i64, answer: String },
    ConfigReload,
    Audit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcResponse {
    Ok { message: String },
    Error { message: String },
    TaskSubmitted { task_id: String },
    Status(DaemonStatusReport),
    Audit(AuditReport),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatusReport {
    pub pid: u32,
    pub running: bool,
    pub paused: bool,
    pub current_cycle: i64,
    pub active_task: Option<String>,
    pub model_chat_status: String,
    pub model_emb_status: String,
    pub antigravity_status: String,
    pub isolation_status: String,
    pub trusted_learnings_count: usize,
    pub tools_count: usize,
    pub uptime_seconds: u64,
    pub constitution_version: String,
    pub constitution_digest: String,
    pub learning_progress: String,
    pub latest_audit: String,
    pub repair_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub timestamp: String,
    pub constitution_digest: String,
    pub constitution_version: String,
    pub semantic_nodes_count: usize,
    pub semantic_edges_count: usize,
    pub graph_density_description: String,
    pub source_evidence: Vec<SourceEvidenceItem>,
    pub tool_execution_records: Vec<crate::store::ToolExecutionRecord>,
    pub memory_reuse: Vec<MemoryReuseItem>,
    pub curriculum_evaluations: Vec<CurriculumEvaluationItem>,
    pub research_backlog: Vec<crate::store::ResearchBacklogItem>,
    pub research_experiments: Vec<crate::store::ResearchExperimentRecord>,
    pub capability_map: Vec<crate::store::CapabilityMapItem>,
    pub extensions: Vec<crate::extensions::ExtensionRecord>,
    pub coordinator_change_tasks: Vec<CoordinatorChangeTaskReportItem>,
    #[serde(default)]
    pub research_works: Vec<crate::web::ResolvedResearchWork>,
    #[serde(default)]
    pub research_syntheses: Vec<crate::store::ResearchSynthesisRecord>,
    pub failures_recorded: Vec<FailureRecordItem>,
    pub progress_summary: String,
    pub repair_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEvidenceItem {
    pub url: String,
    pub title: String,
    pub retrieved_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReuseItem {
    pub memory_id: String,
    pub task_id: String,
    pub context: String,
    pub reused_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumEvaluationItem {
    pub cycle_id: i64,
    pub step: String,
    pub status: String,
    pub details: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorChangeTaskReportItem {
    pub id: i64,
    pub title: String,
    pub rationale: String,
    pub proposed_interface: String,
    pub experiment_evidence: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecordItem {
    pub event: String,
    pub message: String,
    pub timestamp: String,
}
