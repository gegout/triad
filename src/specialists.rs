use anyhow::{bail, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::constitution::Constitution;
use crate::logging::Logger;
use crate::model::ModelClient;
use crate::store::{SpecialistPhase2Proposal, Store};
use crate::tools::ToolManager;

fn default_specialist_phase() -> String {
    "phase1_investigate_and_execute".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistRecord {
    pub id: String,
    pub name: String,
    pub role: String,
    pub task: String,
    pub completion_criteria: String,
    pub max_tokens: usize,
    pub deadline: String,
    pub endorser_1: String,
    pub endorser_2: String,
    pub status: String, // "queued", "running", "completed", "failed", "cancelled", "checkpointed"
    pub result: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default = "default_specialist_phase")]
    pub phase: String, // "phase1_investigate_and_execute", "phase2_validate_and_commit", "completed", "failed", "cancelled", "timed_out", "checkpointed"
    #[serde(default)]
    pub plan_json: Option<String>,
    #[serde(default)]
    pub execution_ids: Vec<i64>,
    #[serde(default)]
    pub pending_commit_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitIncreaseProposal {
    pub id: i64,
    pub current_limit: usize,
    pub requested_limit: usize,
    pub expected_benefit: String,
    pub available_resources: String,
    pub config_revision: String,
    pub votes_explorer: bool,
    pub votes_sceptic: bool,
    pub votes_builder: bool,
    pub status: String, // "pending", "approved", "rejected"
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialistStatusReport {
    pub current_limit: usize,
    pub running_specialists: usize,
    pub queued_tasks: usize,
    pub pending_increase_proposals: usize,
    pub next_eligible_increase_time: Option<String>,
    pub cooldown_active: bool,
}

pub struct SpecialistManager {
    store: Store,
    logger: Logger,
    model_client: Arc<Mutex<ModelClient>>,
    constitution_raw: Arc<String>,
    constitution_digest: Arc<String>,
    tool_manager: Option<Arc<ToolManager>>,
    active_running: Arc<Mutex<Vec<String>>>,
}

impl SpecialistManager {
    pub fn new(
        store: Store,
        logger: Logger,
        model_client: Arc<Mutex<ModelClient>>,
        constitution_raw: Arc<String>,
        constitution_digest: Arc<String>,
    ) -> Self {
        Self {
            store,
            logger,
            model_client,
            constitution_raw,
            constitution_digest,
            tool_manager: None,
            active_running: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_tool_manager(mut self, tool_manager: Arc<ToolManager>) -> Self {
        self.tool_manager = Some(tool_manager);
        self
    }

    /// Called once on daemon startup to handle any interrupted 'running' specialists
    /// from a previous daemon PID without duplicate commits or fabricated completion.
    pub async fn recover_stale_running_on_daemon_start(&self) {
        let pending_commits = self
            .store
            .list_specialist_pending_commits(200)
            .unwrap_or_default();
        if let Ok(stale_running) = self.store.list_running_specialists_newest_first() {
            for spec in stale_running {
                // If this specialist already completed its Phase 2 commit before the crash/restart,
                // mark it completed idempotently without re-running or duplicating commits.
                if let Some(committed) = pending_commits
                    .iter()
                    .find(|c| c.specialist_id == spec.id && c.validation_status == "committed")
                {
                    let msg = format!(
                        "Recovered committed Phase 2 finding (claim={}) across restart without duplicate commit",
                        committed.committed_claim_id.as_deref().unwrap_or("none")
                    );
                    let _ = self.store.update_specialist_phase(
                        &spec.id,
                        "completed",
                        Some("completed"),
                        Some(&msg),
                    );
                    continue;
                }

                // If deadline has already passed during downtime, mark failed/timed_out accurately
                if let Ok(deadline_dt) = DateTime::parse_from_rfc3339(&spec.deadline) {
                    if Utc::now() > deadline_dt.with_timezone(&Utc) {
                        let _ = self
                            .store
                            .reject_pending_commits_for_specialist(&spec.id, "Deadline expired across daemon restart");
                        let _ = self.store.update_specialist_phase(
                            &spec.id,
                            "timed_out",
                            Some("failed"),
                            Some("Specialist deadline expired across restart; no completion fabricated"),
                        );
                        continue;
                    }
                }

                // Otherwise reset to Phase 1 queued so it can run cleanly
                let _ = self.store.update_specialist_phase(
                    &spec.id,
                    "phase1_investigate_and_execute",
                    Some("queued"),
                    None,
                );
            }
        }
        let _ = self.dispatch_queued_specialists().await;
    }

    /// Explicitly cancels a specialist, terminating any active sandbox process groups,
    /// rejecting any uncommitted pending commits, and releasing its slot after exit confirmation.
    pub async fn cancel_specialist(&self, spec_id: &str, reason: &str) -> Result<()> {
        let mut running_ids = self.active_running.lock().await;
        let _ = crate::execution::terminate_all_active_sandbox_process_groups(None);
        let _ = self
            .store
            .reject_pending_commits_for_specialist(spec_id, reason);
        self.store.update_specialist_phase(
            spec_id,
            "cancelled",
            Some("cancelled"),
            Some(reason),
        )?;
        running_ids.retain(|id| id != spec_id);
        let _ = self.store.record_research_experiment(
            1,
            &format!("Cancelled specialist {}", spec_id),
            "Cancelled before Phase 2 commit",
            "Specialist cancellation",
            reason,
            "Cancelled specialists terminate sandbox process groups, release slots, and never fabricate completion",
            true,
            Some("Re-queue if investigation is still needed"),
        );
        Ok(())
    }

    /// Retrieve specialist status summary
    pub async fn get_status_report(&self) -> Result<SpecialistStatusReport> {
        let (limit, last_increase, last_reduction) = self.store.get_specialist_limit_info()?;
        let running_count = self.store.count_running_specialists()?;
        let queued_count = self.store.count_queued_specialists()?;
        let pending_proposals = self.store.count_pending_limit_proposals()?;

        let now = Utc::now();
        let mut cooldown_active = false;
        let mut next_eligible = None;

        // Check 24-hour cooldown from last increase or reduction
        let mut latest_cooldown_anchor: Option<DateTime<Utc>> = None;
        if let Some(ref inc_str) = last_increase {
            if let Ok(ts) = DateTime::parse_from_rfc3339(inc_str) {
                let utc_ts = ts.with_timezone(&Utc);
                latest_cooldown_anchor = Some(utc_ts);
            }
        }
        if let Some(ref red_str) = last_reduction {
            if let Ok(ts) = DateTime::parse_from_rfc3339(red_str) {
                let utc_ts = ts.with_timezone(&Utc);
                if let Some(prev) = latest_cooldown_anchor {
                    if utc_ts > prev {
                        latest_cooldown_anchor = Some(utc_ts);
                    }
                } else {
                    latest_cooldown_anchor = Some(utc_ts);
                }
            }
        }

        if let Some(anchor) = latest_cooldown_anchor {
            let next_time = anchor + Duration::hours(24);
            if next_time > now {
                cooldown_active = true;
                next_eligible = Some(next_time.to_rfc3339());
            }
        }

        Ok(SpecialistStatusReport {
            current_limit: limit,
            running_specialists: running_count,
            queued_tasks: queued_count,
            pending_increase_proposals: pending_proposals,
            next_eligible_increase_time: next_eligible,
            cooldown_active,
        })
    }

    /// User immediately reduces the specialist limit to N.
    /// Checkpoints and terminates excess running specialists from newest to oldest.
    pub async fn set_user_limit(&self, new_limit: usize) -> Result<String> {
        let mut running_ids = self.active_running.lock().await;
        let old_limit = self.store.get_specialist_concurrency_limit()?;

        self.logger.log(
            "warn",
            "specialists.user_reduction",
            None,
            None,
            &format!("User setting specialist limit to {} (was {})", new_limit, old_limit),
            None,
        );

        // Record the reduction and activate 24-hour autonomous increase block
        self.store.record_user_specialist_reduction(new_limit)?;

        // If currently running specialists exceed new_limit, checkpoint and terminate excess
        let current_running = self.store.list_running_specialists_newest_first()?;
        let mut terminated = 0;
        if current_running.len() > new_limit {
            let excess = current_running.len() - new_limit;
            for spec in current_running.into_iter().take(excess) {
                self.store.update_specialist_status(
                    &spec.id,
                    "checkpointed",
                    Some("Checkpointed and terminated due to user concurrency limit reduction"),
                )?;
                running_ids.retain(|id| id != &spec.id);
                terminated += 1;
                self.logger.log(
                    "info",
                    "specialists.checkpointed",
                    Some(&spec.name),
                    Some(&spec.id),
                    "Specialist checkpointed due to concurrency reduction",
                    None,
                );
            }
        }

        // Cancel any pending limit increase proposals
        self.store.reject_all_pending_limit_proposals("User reduced specialist limit")?;

        Ok(format!(
            "Specialist concurrency limit successfully set to {}. {} excess specialists checkpointed/terminated. Autonomous increase blocked for 24h.",
            new_limit, terminated
        ))
    }

    /// Create specialist proposal (requires 2 permanent governing agents)
    pub async fn propose_and_create_specialist(
        &self,
        constitution: &Constitution,
        name: &str,
        role: &str,
        task: &str,
        completion_criteria: &str,
        max_tokens: usize,
        deadline_rfc3339: &str,
        endorser_1: &str,
        endorser_2: &str,
    ) -> Result<String> {
        // Enforce governance: Endorsers must be two distinct permanent agents
        if !constitution.is_valid_agent(endorser_1) || !constitution.is_valid_agent(endorser_2) {
            bail!("Specialist creation must be endorsed by permanent governing agents (Explorer, Sceptic, Builder)");
        }
        if endorser_1 == endorser_2 {
            bail!("Specialist creation requires two distinct permanent agent endorsements");
        }

        let spec_id = format!("spec-{}-{}", name.to_lowercase().replace(' ', "-"), Utc::now().timestamp_millis());

        // Acknowledge constitution for specialist
        let _ = self.store.record_constitution_acknowledgment(name, &self.constitution_digest, "specialist_spawn");

        let record = SpecialistRecord {
            id: spec_id.clone(),
            name: name.to_string(),
            role: role.to_string(),
            task: task.to_string(),
            completion_criteria: completion_criteria.to_string(),
            max_tokens,
            deadline: deadline_rfc3339.to_string(),
            endorser_1: endorser_1.to_string(),
            endorser_2: endorser_2.to_string(),
            status: "queued".to_string(),
            result: None,
            created_at: Utc::now().to_rfc3339(),
            updated_at: Utc::now().to_rfc3339(),
            phase: "phase1_investigate_and_execute".to_string(),
            plan_json: None,
            execution_ids: Vec::new(),
            pending_commit_id: None,
        };

        self.store.register_specialist(&record)?;
        self.logger.log(
            "info",
            "specialist.queued",
            Some(name),
            Some(&spec_id),
            &format!("Specialist {} queued for task: {}", name, task),
            None,
        );

        // Attempt dispatch if slot available
        self.dispatch_queued_specialists().await?;

        Ok(spec_id)
    }

    /// Dispatches queued specialists up to concurrency limit and runs the enforced 2-phase state machine:
    /// - Phase 1 (`phase1_investigate_and_execute`): retrieve sources, propose hypotheses, run tools,
    ///   and inspect failures. Plans and predictions stay task records; they never become findings.
    /// - Phase 2 (`phase2_validate_and_commit`): host validates execution records or source passages
    ///   with extraction fidelity before committing findings and applying the two-agent trust rule.
    pub async fn dispatch_queued_specialists(&self) -> Result<()> {
        let mut running_ids = self.active_running.lock().await;
        let limit = self.store.get_specialist_concurrency_limit()?;

        // Clean up any stale running records in memory
        let running_from_db = self.store.list_running_specialists_newest_first()?;
        *running_ids = running_from_db.into_iter().map(|s| s.id).collect();

        if running_ids.len() >= limit {
            return Ok(()); // At limit, remain queued
        }

        let available_slots = limit - running_ids.len();
        let queued = self.store.list_queued_specialists(available_slots)?;

        for spec in queued {
            self.store.update_specialist_phase(
                &spec.id,
                "phase1_investigate_and_execute",
                Some("running"),
                None,
            )?;
            running_ids.push(spec.id.clone());

            let store_clone = self.store.clone();
            let logger_clone = self.logger.clone();
            let m_client_arc = self.model_client.clone();
            let const_raw = self.constitution_raw.clone();
            let const_digest = self.constitution_digest.clone();
            let tool_mgr_opt = self.tool_manager.clone();
            let active_ids_arc = self.active_running.clone();

            tokio::spawn(async move {
                let inferred_cycle = spec
                    .name
                    .split('-')
                    .find_map(|part| part.parse::<i64>().ok())
                    .unwrap_or(1);

                // Compute timeout from deadline (bounded between 5s and 120s for responsive slot release)
                let timeout_secs = DateTime::parse_from_rfc3339(&spec.deadline)
                    .ok()
                    .map(|dt| (dt.with_timezone(&Utc) - Utc::now()).num_seconds())
                    .unwrap_or(90);

                if timeout_secs <= 0 {
                    let msg = "Specialist execution deadline exceeded before Phase 1 start";
                    let _ = store_clone.reject_pending_commits_for_specialist(&spec.id, msg);
                    let _ = store_clone.update_specialist_phase(
                        &spec.id,
                        "timed_out",
                        Some("failed"),
                        Some(msg),
                    );
                    let _ = store_clone.record_research_experiment(
                        inferred_cycle,
                        &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                        &format!("Endorsers: {} & {} | Budget: {} tokens", spec.endorser_1, spec.endorser_2, spec.max_tokens),
                        &spec.completion_criteria,
                        msg,
                        "Deadline expired before execution; no completion or finding fabricated",
                        true,
                        Some("Increase deadline budget"),
                    );
                    let mut lock = active_ids_arc.lock().await;
                    lock.retain(|id| id != &spec.id);
                    return;
                }

                let effective_timeout = std::time::Duration::from_secs((timeout_secs as u64).clamp(5, 120));
                let exec_res = tokio::time::timeout(
                    effective_timeout,
                    Self::run_enforced_two_phase_execution(
                        spec.clone(),
                        inferred_cycle,
                        store_clone.clone(),
                        m_client_arc,
                        const_raw,
                        const_digest,
                        tool_mgr_opt,
                        logger_clone.clone(),
                    ),
                )
                .await;

                // Check if specialist was cancelled or checkpointed externally while running
                let current_status = store_clone
                    .get_specialist(&spec.id)
                    .ok()
                    .flatten()
                    .map(|s| s.status)
                    .unwrap_or_default();
                if current_status == "cancelled" || current_status == "checkpointed" {
                    let _ = crate::execution::terminate_all_active_sandbox_process_groups(None);
                    let _ = store_clone.reject_pending_commits_for_specialist(
                        &spec.id,
                        "Specialist cancelled or checkpointed before completion",
                    );
                    let mut lock = active_ids_arc.lock().await;
                    lock.retain(|id| id != &spec.id);
                    return;
                }

                match exec_res {
                    Ok(Ok(summary)) => {
                        let _ = store_clone.update_specialist_phase(
                            &spec.id,
                            "completed",
                            Some("completed"),
                            Some(&summary),
                        );
                        logger_clone.log(
                            "info",
                            "specialist.completed",
                            Some(&spec.name),
                            Some(&spec.id),
                            &summary,
                            None,
                        );
                    }
                    Ok(Err(e)) => {
                        let err_str = format!("Error: {}", e);
                        let _ = store_clone
                            .reject_pending_commits_for_specialist(&spec.id, &err_str);
                        let _ = store_clone.update_specialist_phase(
                            &spec.id,
                            "failed",
                            Some("failed"),
                            Some(&err_str),
                        );
                        let _ = store_clone.record_research_experiment(
                            inferred_cycle,
                            &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                            &format!("Endorsers: {} & {} | Budget: {} tokens", spec.endorser_1, spec.endorser_2, spec.max_tokens),
                            &spec.completion_criteria,
                            &crate::model::clip_text(&err_str, 280),
                            "Failed specialist execution recorded accurately without fabricating completion",
                            true,
                            Some("Inspect error and retry with adjusted tool/source parameters"),
                        );
                        logger_clone.log(
                            "error",
                            "specialist.failed",
                            Some(&spec.name),
                            Some(&spec.id),
                            &err_str,
                            None,
                        );
                    }
                    Err(_elapsed) => {
                        let killed_groups =
                            crate::execution::terminate_all_active_sandbox_process_groups(None);
                        let timeout_msg = format!(
                            "Timed out after {}s ({} sandbox process groups terminated and confirmed exited); uncommitted work aborted without fabricated completion",
                            effective_timeout.as_secs(),
                            killed_groups
                        );
                        let _ = store_clone
                            .reject_pending_commits_for_specialist(&spec.id, &timeout_msg);
                        let _ = store_clone.update_specialist_phase(
                            &spec.id,
                            "timed_out",
                            Some("failed"),
                            Some(&timeout_msg),
                        );
                        let _ = store_clone.record_research_experiment(
                            inferred_cycle,
                            &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                            &format!("Endorsers: {} & {} | Budget: {} tokens", spec.endorser_1, spec.endorser_2, spec.max_tokens),
                            &spec.completion_criteria,
                            &timeout_msg,
                            "Specialist timeout killed sandbox process groups, confirmed exit, recorded timeout outcome, and released concurrency slot",
                            true,
                            Some("Increase timeout or decompose subtask"),
                        );
                        logger_clone.log(
                            "warn",
                            "specialist.timed_out",
                            Some(&spec.name),
                            Some(&spec.id),
                            &timeout_msg,
                            None,
                        );
                    }
                }

                // Always release concurrency slot when done
                let mut lock = active_ids_arc.lock().await;
                lock.retain(|id| id != &spec.id);
            });
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_enforced_two_phase_execution(
        spec: SpecialistRecord,
        inferred_cycle: i64,
        store: Store,
        m_client_arc: Arc<Mutex<ModelClient>>,
        const_raw: Arc<String>,
        const_digest: Arc<String>,
        tool_mgr_opt: Option<Arc<ToolManager>>,
        logger: Logger,
    ) -> Result<String> {
        // =====================================================================
        // PHASE 1: INVESTIGATE AND EXECUTE (`phase1_investigate_and_execute`)
        // Retrieve sources, propose hypotheses, run tools, and inspect failures.
        // Plans and predictions stay task records; they never become findings.
        // =====================================================================
        let _ = store.update_specialist_phase(
            &spec.id,
            "phase1_investigate_and_execute",
            Some("running"),
            None,
        );
        logger.log(
            "info",
            "specialist.phase1_start",
            Some(&spec.name),
            Some(&spec.id),
            &format!("Phase 1 (investigate & execute): {}", spec.task),
            None,
        );

        let retrieved_sections = store
            .retrieve_relevant_source_sections(&spec.task, &[], 450)
            .unwrap_or_default();
        let section_context = if retrieved_sections.is_empty() {
            "No full-text passages retrieved yet.".to_string()
        } else {
            retrieved_sections
                .iter()
                .take(4)
                .map(|s| {
                    format!(
                        "[passage_id={} | source={} | {} ({})]: {}",
                        s.id,
                        s.canonical_source_id,
                        s.section_path,
                        s.section_kind,
                        crate::model::clip_text(&s.content, 180)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Execute tool in sandbox if available to gather immutable host execution evidence
        let mut execution_record: Option<crate::store::ToolExecutionRecord> = None;
        if let Some(ref tm) = tool_mgr_opt {
            if let Ok(tools) = store.list_tools() {
                let chosen_tool = tools
                    .iter()
                    .find(|t| t.status == "active" && t.experiment_spec_id.is_some())
                    .or_else(|| tools.iter().find(|t| t.status == "active"));
                if let Some(active_tool) = chosen_tool {
                    let payload = serde_json::json!({
                        "text": format!("{} | {}", spec.task, spec.completion_criteria),
                        "seed": inferred_cycle,
                    })
                    .to_string();
                    if active_tool.experiment_spec_id.is_some() {
                        if let Ok(exp_outcome) = tm
                            .execute_research_experiment(
                                inferred_cycle,
                                &active_tool.name,
                                &payload,
                                Some(inferred_cycle),
                                Some(&spec.id),
                                None,
                            )
                            .await
                        {
                            execution_record = Some(exp_outcome.execution_record);
                        }
                    } else if let Ok(rec) = tm
                        .execute_tool_with_evidence(
                            &active_tool.name,
                            &payload,
                            Some(inferred_cycle),
                            Some(&spec.id),
                        )
                        .await
                    {
                        execution_record = Some(rec);
                    }
                }
            }
        }

        let m_client = m_client_arc.lock().await.clone();
        let exec_summary_for_prompt = match execution_record {
            Some(ref r) => format!(
                "Host Execution #{} (tool={}@{}, exit_code={}, output={})",
                r.id,
                r.tool_name,
                r.tool_version,
                r.exit_code,
                r.output_json.as_deref().unwrap_or("none")
            ),
            None => "No sandboxed tool execution recorded for this subtask.".to_string(),
        };

        let prompt = format!(
            "You are temporary specialist agent '{}' with role '{}'.\n\
             Assigned task: {}\n\
             Completion criteria: {}\n\
             Retrieved source passages:\n{}\n\
             Host execution evidence:\n{}\n\
             Budget limit: {} tokens.\n\
             Provide a bounded conclusion grounded strictly in the retrieved source passages or host execution evidence, plus explicit assumptions and any negative results.",
            spec.name,
            spec.role,
            spec.task,
            spec.completion_criteria,
            section_context,
            exec_summary_for_prompt,
            spec.max_tokens
        );

        let role_low = spec.role.to_lowercase();
        let workload_kind = if role_low.contains("solver")
            || role_low.contains("synthesis")
            || role_low.contains("architecture")
        {
            crate::model::WorkloadKind::ComplexRustGeneration
        } else if role_low.contains("auditor") || role_low.contains("critic") {
            crate::model::WorkloadKind::CriticalReview
        } else if role_low.contains("reproduction") || role_low.contains("ablation") {
            crate::model::WorkloadKind::ExperimentDesign
        } else {
            crate::model::WorkloadKind::HardSynthesis
        };
        let spec_ctx = crate::model::WorkloadSelectionContext {
            task_id: spec.id.clone(),
            caller_kind: "specialist".to_string(),
            caller_id: spec.name.clone(),
            workload: workload_kind,
            required_context_tokens: (prompt.len() / 4).max(1024),
            require_structured_json: true,
            explicit_override: None,
        };
        let turn = m_client
            .agent_turn_for_workload(
                &spec_ctx,
                &spec.name,
                &format!("Specialist in {}: execute strictly within scope", spec.role),
                &const_raw,
                &const_digest,
                &prompt,
                &[],
            )
            .await?;

        let raw_message = turn.public_message.trim().to_string();
        let hypothesis = turn
            .research_hypothesis
            .clone()
            .unwrap_or_else(|| spec.task.clone());

        // Always record Phase 1 hypothesis/plan/predictions in task records (never as a claim)
        let _ = store.record_specialist_phase1_plan(
            &spec.id,
            &hypothesis,
            &[spec.completion_criteria.clone()],
            &[turn
                .research_outcome
                .clone()
                .unwrap_or_else(|| "Bounded specialist investigation".to_string())],
        );

        // If the specialist output is only a plan or prediction, keep it strictly as a Phase 1 task
        // record and do NOT promote it to a finding.
        if Store::is_plan_or_prediction_only(&raw_message) {
            let _ = store.record_research_experiment(
                inferred_cycle,
                &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                &format!("Endorsers: {} & {} | Phase 1 plan only", spec.endorser_1, spec.endorser_2),
                &spec.completion_criteria,
                &crate::model::clip_text(&raw_message, 240),
                "Plan/prediction kept strictly as task record per enforced specialist state machine; never promoted to finding",
                false,
                None,
            );
            return Ok(format!(
                "[Phase 1 Task Record Only (plan/prediction not promoted to finding)]: {}",
                crate::model::clip_text(&raw_message, 220)
            ));
        }

        // =====================================================================
        // PHASE 2: VALIDATE AND COMMIT (`phase2_validate_and_commit`)
        // Host code validates execution references or source passages before
        // committing findings and applying the two-agent trust rule.
        // =====================================================================
        let _ = store.update_specialist_phase(
            &spec.id,
            "phase2_validate_and_commit",
            Some("running"),
            None,
        );

        let mut committed_summaries = Vec::new();
        let assumptions_str = turn
            .remaining_uncertainty
            .clone()
            .unwrap_or_else(|| "Bounded to evaluated inputs and retrieved passages".to_string());

        // 1. If a host tool execution occurred in Phase 1, validate and commit the locally_reproduced finding
        if let Some(ref exec_rec) = execution_record {
            let run_failed = exec_rec.exit_code != 0 || exec_rec.error.is_some();
            let _ = store.record_research_experiment(
                inferred_cycle,
                &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                &format!(
                    "Execution #{} (tool={}@{}, hash={}, seed={:?})",
                    exec_rec.id,
                    exec_rec.tool_name,
                    exec_rec.tool_version,
                    &exec_rec.tool_hash[..exec_rec.tool_hash.len().min(10)],
                    exec_rec.seed
                ),
                &spec.completion_criteria,
                &format!(
                    "exit_code={} | output={} | error={:?}",
                    exec_rec.exit_code,
                    exec_rec.output_json.as_deref().unwrap_or(""),
                    exec_rec.error
                ),
                if run_failed {
                    "Failed specialist experiment recorded with accurate exit code and error outcome"
                } else {
                    "Completed specialist experiment verified by host execution record"
                },
                run_failed,
                if run_failed {
                    Some("Inspect stderr and input schema before retrying")
                } else {
                    None
                },
            );

            let measured_val = serde_json::from_str::<serde_json::Value>(
                &exec_rec.measured_results_json,
            )
            .unwrap_or_else(|_| serde_json::json!({}));

            let conclusion_text = if run_failed {
                format!(
                    "Negative result on '{}': tool {}@{} exited with code {} ({})",
                    spec.task,
                    exec_rec.tool_name,
                    exec_rec.tool_version,
                    exec_rec.exit_code,
                    exec_rec.error.as_deref().unwrap_or("non-zero exit")
                )
            } else {
                format!(
                    "{} — verified by host execution #{} ({}@{}): {}",
                    crate::model::clip_text(&raw_message, 180),
                    exec_rec.id,
                    exec_rec.tool_name,
                    exec_rec.tool_version,
                    exec_rec.output_json.as_deref().unwrap_or("{}").trim()
                )
            };

            let emp_proposal = SpecialistPhase2Proposal {
                specialist_id: spec.id.clone(),
                finding_kind: "locally_reproduced".to_string(),
                conclusion: conclusion_text,
                assumptions: assumptions_str.clone(),
                negative_results: if run_failed {
                    exec_rec
                        .error
                        .clone()
                        .unwrap_or_else(|| format!("exit_code={}", exec_rec.exit_code))
                } else {
                    "No execution failure observed under tested inputs".to_string()
                },
                is_negative_result: run_failed,
                execution_id: Some(exec_rec.id),
                claimed_measurements: if run_failed {
                    serde_json::json!({})
                } else {
                    measured_val
                },
                source_passage_ids: Vec::new(),
                canonical_source_ids: Vec::new(),
                cycle_id: inferred_cycle,
                domain: "frontier_ai_research_and_verification".to_string(),
            };

            if let Ok(outcome) = store.validate_and_commit_specialist_finding(&emp_proposal) {
                if outcome.validation_status == "committed" {
                    committed_summaries.push(format!(
                        "committed locally_reproduced claim={} (exec_id={}, trusted={})",
                        outcome.committed_claim_id.as_deref().unwrap_or("none"),
                        exec_rec.id,
                        outcome.trusted
                    ));
                }
            }
        }

        // 2. If retrieved source passages exist, validate and commit a distinct source_supported literature finding
        if !retrieved_sections.is_empty() {
            let passage_ids: Vec<i64> = retrieved_sections.iter().take(3).map(|s| s.id).collect();
            let lit_proposal = SpecialistPhase2Proposal {
                specialist_id: spec.id.clone(),
                finding_kind: "source_supported".to_string(),
                conclusion: crate::model::clip_text(&raw_message, 220),
                assumptions: assumptions_str,
                negative_results: "Literature finding only; kept distinct from locally reproduced execution".to_string(),
                is_negative_result: false,
                execution_id: None,
                claimed_measurements: serde_json::json!({}),
                source_passage_ids: passage_ids,
                canonical_source_ids: Vec::new(),
                cycle_id: inferred_cycle,
                domain: "frontier_ai_research_and_verification".to_string(),
            };
            if let Ok(outcome) = store.validate_and_commit_specialist_finding(&lit_proposal) {
                if outcome.validation_status == "committed" {
                    committed_summaries.push(format!(
                        "committed source_supported claim={} (sources={}, fidelity={})",
                        outcome.committed_claim_id.as_deref().unwrap_or("none"),
                        outcome.independent_source_count,
                        outcome.source_fidelity_summary
                    ));
                }
            }
        }

        if committed_summaries.is_empty() {
            Ok(format!(
                "[Phase 1 Task Recorded; no Phase 2 finding committed without validated execution/source references]: {}",
                crate::model::clip_text(&raw_message, 200)
            ))
        } else {
            Ok(format!(
                "[Phase 2 Validated & Committed ({})]: {}",
                committed_summaries.join("; "),
                crate::model::clip_text(&raw_message, 180)
            ))
        }
    }

    /// Propose and vote on increasing specialist concurrency limit.
    /// Requires unanimous vote from all 3 permanent governing agents: Explorer, Sceptic, and Builder.
    /// Exactly 1 slot increase. Must satisfy 24-hour rolling cooldown.
    pub async fn propose_and_vote_limit_increase(
        &self,
        expected_benefit: &str,
        available_resources: &str,
        config_revision: &str,
        votes: (&str, &str, &str), // Must be ("Explorer", "Sceptic", "Builder")
    ) -> Result<usize> {
        let status = self.get_status_report().await?;
        if status.cooldown_active {
            bail!(
                "Concurrency limit increase rejected: 24-hour rolling cooldown active. Next eligible time: {}",
                status.next_eligible_increase_time.unwrap_or_default()
            );
        }

        // Unanimity verification: All 3 distinct permanent identities must vote in favor
        let mut voter_set = std::collections::HashSet::new();
        voter_set.insert(votes.0);
        voter_set.insert(votes.1);
        voter_set.insert(votes.2);

        if !voter_set.contains("Explorer") || !voter_set.contains("Sceptic") || !voter_set.contains("Builder") {
            bail!("Increasing specialist concurrency limit strictly requires unanimous consent of all 3 permanent agents (Explorer, Sceptic, Builder)");
        }

        let current_limit = status.current_limit;
        let requested_limit = current_limit + 1; // Exactly +1 slot

        let proposal_id = self.store.record_limit_increase_proposal(
            current_limit,
            requested_limit,
            expected_benefit,
            available_resources,
            config_revision,
            true,
            true,
            true,
            "approved",
        )?;

        self.store.apply_specialist_limit_increase(requested_limit)?;

        self.logger.log(
            "info",
            "specialists.limit_increased",
            None,
            Some(&proposal_id.to_string()),
            &format!("Unanimous consent achieved: specialist concurrency limit increased from {} to {}", current_limit, requested_limit),
            None,
        );

        // Attempt dispatch of any queued tasks with newly unlocked slot
        self.dispatch_queued_specialists().await?;

        Ok(requested_limit)
    }
}
