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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub id: String, // stable SHA256 of statement
    pub statement: String,
    pub source_url: String,
    pub proposer: String,
    pub status: String, // "unresolved", "trusted", "rejected", "superseded"
    pub version: i64,
    pub created_at: String,
    pub updated_at: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionRecord {
    pub id: i64,
    pub tool_name: String,
    pub tool_version: i64,
    pub input_json: String,
    pub output_json: Option<String>,
    pub error: Option<String>,
    pub exit_code: i32,
    pub duration_ms: i64,
    pub executed_at: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchExperimentRecord {
    pub id: i64,
    pub cycle_id: i64,
    pub hypothesis: String,
    pub conditions: String,
    pub attempted_method: String,
    pub observed_outcome: String,
    pub lessons_learned: String,
    pub is_failure: bool,
    pub justification_to_retry: Option<String>,
    pub created_at: String,
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
        ")?;

        // Migration check from older v0.2.0 memories table if it exists
        Self::migrate_legacy_memories_if_present(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
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

    pub fn get_claim(&self, claim_id: &str) -> Result<Option<Claim>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, statement, source_url, proposer, status, version, created_at, updated_at FROM claims WHERE id=?1")?;
        let mut rows = stmt.query([claim_id])?;
        if let Some(r) = rows.next()? {
            Ok(Some(Claim {
                id: r.get(0)?,
                statement: r.get(1)?,
                source_url: r.get(2)?,
                proposer: r.get(3)?,
                status: r.get(4)?,
                version: r.get(5)?,
                created_at: r.get(6)?,
                updated_at: r.get(7)?,
            }))
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

    pub fn list_learnings(&self, limit: usize) -> Result<Vec<Claim>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, statement, source_url, proposer, status, version, created_at, updated_at \
             FROM claims WHERE status='trusted' ORDER BY updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(Claim {
                id: r.get(0)?,
                statement: r.get(1)?,
                source_url: r.get(2)?,
                proposer: r.get(3)?,
                status: r.get(4)?,
                version: r.get(5)?,
                created_at: r.get(6)?,
                updated_at: r.get(7)?,
            })
        })?;
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
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        let status = if test_passed && endorser_1.is_some() && endorser_2.is_some() {
            "active"
        } else {
            "draft"
        };
        conn.execute(
            "INSERT INTO tools (name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, if test_passed { 1 } else { 0 }, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_tools(&self) -> Result<Vec<ToolRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at \
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
            "SELECT id, name, version, description, schema_json, source_code, binary_path, source_hash, status, endorser_1, endorser_2, test_passed, created_at \
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
            }))
        } else {
            Ok(None)
        }
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
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO tool_executions (tool_name, tool_version, input_json, output_json, error, exit_code, duration_ms, executed_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![name, version, input_json, output_json, error, exit_code, duration_ms as i64, now],
        )?;
        Ok(())
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
            "SELECT id, tool_name, tool_version, input_json, output_json, error, exit_code, duration_ms, executed_at
             FROM tool_executions ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            Ok(ToolExecutionRecord {
                id: r.get(0)?,
                tool_name: r.get(1)?,
                tool_version: r.get(2)?,
                input_json: r.get(3)?,
                output_json: r.get(4)?,
                error: r.get(5)?,
                exit_code: r.get(6)?,
                duration_ms: r.get(7)?,
                executed_at: r.get(8)?,
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
        conn.execute(
            "INSERT INTO research_experiments (cycle_id, hypothesis, conditions, attempted_method, observed_outcome, lessons_learned, is_failure, justification_to_retry, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![cycle_id, hypothesis, conditions, attempted_method, observed_outcome, lessons_learned, is_fail_int, justification_to_retry, now],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_research_experiments(&self, limit: usize) -> Result<Vec<ResearchExperimentRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, cycle_id, hypothesis, conditions, attempted_method, observed_outcome, lessons_learned, is_failure, justification_to_retry, created_at
             FROM research_experiments ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit], |r| {
            let is_fail_int: i64 = r.get(7)?;
            Ok(ResearchExperimentRecord {
                id: r.get(0)?,
                cycle_id: r.get(1)?,
                hypothesis: r.get(2)?,
                conditions: r.get(3)?,
                attempted_method: r.get(4)?,
                observed_outcome: r.get(5)?,
                lessons_learned: r.get(6)?,
                is_failure: is_fail_int != 0,
                justification_to_retry: r.get(8)?,
                created_at: r.get(9)?,
            })
        })?;
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

    pub fn activate_extension(&self, id: i64, hook: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE extensions SET state = 'evaluated', updated_at=?1 WHERE hook=?2 AND state = 'active'",
            params![now, hook],
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

    pub fn register_specialist(&self, spec: &crate::specialists::SpecialistRecord) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO specialists (id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                spec.id, spec.name, spec.role, spec.task, spec.completion_criteria,
                spec.max_tokens as i64, spec.deadline, spec.endorser_1, spec.endorser_2,
                spec.status, spec.result, spec.created_at, spec.updated_at
            ],
        )?;
        Ok(())
    }

    pub fn update_specialist_status(&self, id: &str, status: &str, result: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE specialists SET status = ?1, result = ?2, updated_at = ?3 WHERE id = ?4",
            params![status, result, now, id],
        )?;
        Ok(())
    }

    pub fn list_running_specialists_newest_first(&self) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at
             FROM specialists WHERE status = 'running' ORDER BY created_at DESC"
        )?;
        let rows = stmt.query_map([], |r| {
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
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_queued_specialists(&self, limit: usize) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at
             FROM specialists WHERE status = 'queued' ORDER BY created_at ASC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
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
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn list_specialists(&self, limit: usize) -> Result<Vec<crate::specialists::SpecialistRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, role, task, completion_criteria, max_tokens, deadline, endorser_1, endorser_2, status, result, created_at, updated_at
             FROM specialists ORDER BY created_at DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
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
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
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
            Ok(crate::web::ResolvedResearchWork {
                canonical_id: r.get(0)?,
                title: r.get(1)?,
                normalized_title: r.get(2)?,
                doi: r.get(3)?,
                arxiv_id: r.get(4)?,
                versions: serde_json::from_str(&v_str).unwrap_or_default(),
                published_date: r.get(6)?,
                authors: serde_json::from_str(&a_str).unwrap_or_default(),
                primary_connector: r.get(8)?,
                corroborating_connectors: serde_json::from_str(&c_str).unwrap_or_default(),
                urls: serde_json::from_str(&u_str).unwrap_or_default(),
                code_urls: serde_json::from_str(&cu_str).unwrap_or_default(),
                abstract_excerpt: r.get(12)?,
                experimental_details: r.get(13)?,
                limitations_excerpt: r.get(14)?,
                citations: serde_json::from_str(&cit_str).unwrap_or_default(),
                availability_status: r.get(16)?,
                duplicate_copies_merged: r.get::<_, i64>(17)? as usize,
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
}
