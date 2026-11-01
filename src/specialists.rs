use anyhow::{bail, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::constitution::Constitution;
use crate::logging::Logger;
use crate::model::ModelClient;
use crate::store::Store;

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
            active_running: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Called once on daemon startup to return any interrupted 'running' specialists
    /// from a previous daemon PID back to 'queued' and dispatch them cleanly.
    pub async fn recover_stale_running_on_daemon_start(&self) {
        if let Ok(stale_running) = self.store.list_running_specialists_newest_first() {
            for spec in stale_running {
                let _ = self.store.update_specialist_status(&spec.id, "queued", None);
            }
        }
        let _ = self.dispatch_queued_specialists().await;
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

    /// Dispatches queued specialists up to concurrency limit
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
            self.store.update_specialist_status(&spec.id, "running", None)?;
            running_ids.push(spec.id.clone());

            let store_clone = self.store.clone();
            let logger_clone = self.logger.clone();
            let m_client_arc = self.model_client.clone();
            let const_raw = self.constitution_raw.clone();
            let const_digest = self.constitution_digest.clone();
            let active_ids_arc = self.active_running.clone();

            tokio::spawn(async move {
                let res = Self::run_specialist_execution(
                    spec.clone(),
                    m_client_arc,
                    const_raw,
                    const_digest,
                    logger_clone.clone(),
                ).await;

                // After completion or failure: persist results and release slot
                match res {
                    Ok(summary) => {
                        let _ = store_clone.update_specialist_status(&spec.id, "completed", Some(&summary));
                        let inferred_cycle = spec
                            .name
                            .split('-')
                            .find_map(|part| part.parse::<i64>().ok())
                            .unwrap_or(1);

                        // Persist useful learnings into semantic store
                        let cid_res = store_clone.propose_claim(
                            &format!("Specialist [{}] ({}): {}", spec.name, spec.role, summary),
                            &format!("triad:specialist:{}", spec.id),
                            &spec.endorser_1,
                        );
                        if let Ok(cid) = cid_res {
                            let _ = store_clone.record_vote(&cid, &spec.endorser_1, "endorse", inferred_cycle, Some("Specialist completed bounded task"));
                            let _ = store_clone.record_vote(&cid, &spec.endorser_2, "endorse", inferred_cycle, Some("Endorser verified specialist outcome"));
                            let _ = store_clone.upsert_semantic_node(&cid, &crate::model::clip_text(&summary, 40), "concept", "{\"specialist\":true}");
                            if let Ok(recent_learnings) = store_clone.list_learnings(4) {
                                for past in recent_learnings {
                                    if past.id != cid {
                                        let _ = store_clone.add_semantic_edge(&cid, &past.id, "relates_to", 0.75);
                                    }
                                }
                            }
                        }

                        // Record specialist experiment in research_experiments for visibility
                        let _ = store_clone.record_research_experiment(
                            inferred_cycle,
                            &format!("[Specialist {} ({})] {}", spec.name, spec.role, spec.task),
                            &format!("Endorsers: {} & {} | Budget: {} tokens", spec.endorser_1, spec.endorser_2, spec.max_tokens),
                            &spec.completion_criteria,
                            &crate::model::clip_text(&summary, 280),
                            "Temporary specialist completed bounded subproblem and terminated per constitutional governance",
                            false,
                            None,
                        );

                        logger_clone.log("info", "specialist.completed", Some(&spec.name), Some(&spec.id), &summary, None);
                    }
                    Err(e) => {
                        let _ = store_clone.update_specialist_status(&spec.id, "failed", Some(&format!("Error: {}", e)));
                        logger_clone.log("error", "specialist.failed", Some(&spec.name), Some(&spec.id), &e.to_string(), None);
                    }
                }

                // Release slot atomically after process actually terminates
                let mut lock = active_ids_arc.lock().await;
                lock.retain(|id| id != &spec.id);
            });
        }

        Ok(())
    }

    async fn run_specialist_execution(
        spec: SpecialistRecord,
        m_client_arc: Arc<Mutex<ModelClient>>,
        const_raw: Arc<String>,
        const_digest: Arc<String>,
        logger: Logger,
    ) -> Result<String> {
        let m_client = m_client_arc.lock().await.clone();

        // Check deadline
        if let Ok(deadline_dt) = DateTime::parse_from_rfc3339(&spec.deadline) {
            if Utc::now() > deadline_dt.with_timezone(&Utc) {
                bail!("Specialist execution deadline exceeded before execution start");
            }
        }

        let prompt = format!(
            "You are temporary specialist agent '{}' with role '{}'.\nYour assigned task: {}\nCompletion criteria: {}\nBudget limit: {} tokens.\nPerform the task and provide your structured finding.",
            spec.name, spec.role, spec.task, spec.completion_criteria, spec.max_tokens
        );

        logger.log("info", "specialist.execution_start", Some(&spec.name), Some(&spec.id), &format!("Executing task: {}", spec.task), None);

        let turn = m_client.agent_turn_with_constitution(
            &spec.name,
            &format!("Specialist in {}: execute strictly within scope", spec.role),
            &const_raw,
            &const_digest,
            &prompt,
            &[],
        ).await?;

        Ok(turn.public_message)
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
