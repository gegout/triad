use std::fs;
use tempfile::tempdir;

use triad::config::{AppConfig, DEFAULT_CONFIG_TOML};
use triad::execution::{BubblewrapAdapter, ExecutionAdapter};
use triad::store::Store;
use triad::tools::{ToolManager, ToolProposal};

#[test]
fn test_default_config_parsing_and_validation() {
    let cfg = AppConfig::parse_and_validate(DEFAULT_CONFIG_TOML).expect("Default config must parse and validate");
    assert_eq!(cfg.application.name, "triad");
    assert_eq!(cfg.learning.agreement_threshold, 2);
    assert_eq!(cfg.orchestration.discussion_rounds, 2);
    assert_eq!(cfg.orchestration.max_model_requests_per_cycle, 24);
    assert_eq!(cfg.models.chat.model, "smollm3-3b");
    assert_eq!(cfg.models.embedding.model, "nomic-embed-text-v1-5");
}

#[test]
fn test_two_distinct_votes_promote_exact_claim_version_dissent_retained() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let store = Store::open(&db_path).unwrap();

    let statement = "Gravitational waves travel at the speed of light.";
    let claim_id = store.propose_claim(statement, "https://en.wikipedia.org/wiki/Gravitational_wave", "Explorer").unwrap();

    // 1st vote by Explorer (endorse)
    let p1 = store.record_vote(&claim_id, "Explorer", "endorse", 1, Some("Theoretical prediction")).unwrap();
    assert!(!p1, "Single endorsement must not promote");
    let c1 = store.get_claim(&claim_id).unwrap().unwrap();
    assert_eq!(c1.status, "unresolved");

    // Sceptic dissents (reject)
    let p_scep = store.record_vote(&claim_id, "Sceptic", "reject", 1, Some("Awaiting empirical detection")).unwrap();
    assert!(!p_scep);
    let c2 = store.get_claim(&claim_id).unwrap().unwrap();
    assert_eq!(c2.status, "unresolved");

    // Builder endorses -> exactly 2 distinct endorsements in cycle 1 -> promoted to trusted!
    let p2 = store.record_vote(&claim_id, "Builder", "endorse", 1, Some("LIGO observation verified")).unwrap();
    assert!(p2, "Two distinct endorsements must promote to trusted");
    let c3 = store.get_claim(&claim_id).unwrap().unwrap();
    assert_eq!(c3.status, "trusted");

    // Check vote history: all 3 votes (2 endorse, 1 dissent) are preserved
    let votes = store.get_claim_votes(&claim_id).unwrap();
    assert_eq!(votes.len(), 3);
    assert_eq!(votes.iter().filter(|v| v.decision == "endorse").count(), 2);
    assert_eq!(votes.iter().filter(|v| v.decision == "reject").count(), 1);
}

#[test]
fn test_duplicate_votes_cannot_manufacture_quorum() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let store = Store::open(&db_path).unwrap();

    let claim_id = store.propose_claim("Quantum entanglement allows faster-than-light signaling", "https://wiki/test", "Explorer").unwrap();

    // Same agent votes endorse 5 times in the same cycle
    for i in 1..=5 {
        let p = store.record_vote(&claim_id, "Explorer", "endorse", 1, Some(&format!("Attempt {}", i))).unwrap();
        assert!(!p, "Duplicate votes from same agent must not promote");
    }

    let c = store.get_claim(&claim_id).unwrap().unwrap();
    assert_eq!(c.status, "unresolved");
}

#[tokio::test]
async fn test_tool_generation_test_pass_and_execution() {
    let adapter = BubblewrapAdapter::probe();
    if !adapter.is_available() {
        eprintln!("bwrap not installed, skipping test");
        return;
    }

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let ws_dir = dir.path().join("workspace");
    fs::create_dir_all(&ws_dir).unwrap();

    let store = Store::open(&db_path).unwrap();
    let mgr = ToolManager::new(store.clone(), ws_dir.clone(), std::sync::Arc::new(adapter));

    let proposal = ToolProposal {
        name: "math_adder".to_string(),
        description: "Adds two integers".to_string(),
        schema_json: r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#.to_string(),
        rust_source: r#"
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Deserialize)]
struct Input { a: i64, b: i64 }
#[derive(Serialize)]
struct Output { sum: i64 }

fn main() {
    let mut s = String::new();
    let _ = std::io::stdin().read_to_string(&mut s);
    let input: Input = serde_json::from_str(&s).unwrap();
    let out = Output { sum: input.a + input.b };
    println!("{}", serde_json::to_string(&out).unwrap());
}
"#.to_string(),
        test_code: r#"
    #[test]
    fn test_add() {
        assert_eq!(2 + 2, 4);
    }
"#.to_string(),
        experiment_spec: None,
    };

    let tool_id = mgr.build_and_register_tool(&proposal, "Builder", "Sceptic").await.unwrap();
    assert!(tool_id > 0);

    let exec_out = mgr.execute_tool("math_adder", r#"{"a":17,"b":25}"#).await.unwrap();
    assert!(exec_out.contains(r#""sum":42"#));
}

#[tokio::test]
async fn test_failed_tool_test_prevents_registration() {
    let adapter = BubblewrapAdapter::probe();
    if !adapter.is_available() {
        return;
    }

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let ws_dir = dir.path().join("workspace");
    fs::create_dir_all(&ws_dir).unwrap();

    let store = Store::open(&db_path).unwrap();
    let mgr = ToolManager::new(store.clone(), ws_dir.clone(), std::sync::Arc::new(adapter));

    let bad_proposal = ToolProposal {
        name: "broken_tool".to_string(),
        description: "A tool that fails tests".to_string(),
        schema_json: "{}".to_string(),
        rust_source: "fn main() {}".to_string(),
        test_code: r#"
    #[test]
    fn test_fail() {
        panic!("Intended test failure");
    }
"#.to_string(),
        experiment_spec: None,
    };

    let res = mgr.build_and_register_tool(&bad_proposal, "Builder", "Sceptic").await;
    assert!(res.is_err(), "Tool registration must fail when test suite fails");

    let registered = store.get_tool_by_name("broken_tool").unwrap();
    assert!(registered.is_none());
}

#[test]
fn test_reindex_preserves_trust_and_updates_vectors_atomically() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let store = Store::open(&db_path).unwrap();

    let cid = store.propose_claim("Rust provides memory safety without GC", "https://rust-lang.org", "Explorer").unwrap();
    store.record_vote(&cid, "Explorer", "endorse", 1, None).unwrap();
    store.record_vote(&cid, "Builder", "endorse", 1, None).unwrap();

    store.save_embedding("claim", &cid, "Rust provides memory safety without GC", &[1.0, 0.0]).unwrap();

    // Verify trusted
    let claim_before = store.get_claim(&cid).unwrap().unwrap();
    assert_eq!(claim_before.status, "trusted");

    // Atomically replace embeddings with new 3D vector space
    let new_encoder = "nomic-embed-text-v1-5|dim=3|q_pfx=query: |doc_pfx=doc: ";
    let items = vec![("claim".to_string(), cid.clone(), "Rust provides memory safety without GC".to_string(), vec![0.5, 0.5, 0.5])];

    store.replace_embeddings_atomic(new_encoder, &items).unwrap();

    // Verify claim status and trust preserved
    let claim_after = store.get_claim(&cid).unwrap().unwrap();
    assert_eq!(claim_after.status, "trusted");

    // Verify new embedding vector is searchable
    let search_res = store.search_semantic(&[0.5, 0.5, 0.5], 5).unwrap();
    assert_eq!(search_res.len(), 1);
    assert_eq!(search_res[0].0.target_id, cid);
    assert_eq!(search_res[0].0.embedding.len(), 3);
}

#[test]
fn test_constitutional_invariants_and_tamper_detection() {
    let (c, _raw, digest) = triad::constitution::load_and_verify_constitution().unwrap();
    assert_eq!(c.constitution.version, "2.1.0");
    assert!(c.objective.frontier_ai_mandate.is_some());
    assert_eq!(c.identities.count, 3);
    assert_eq!(c.consensus.endorsement_threshold, 2);
    assert!(c.is_valid_agent("Explorer"));
    assert!(c.is_valid_agent("Sceptic"));
    assert!(c.is_valid_agent("Builder"));
    assert!(!c.is_valid_agent("UnauthorizedAgent"));

    // Digest must be 64-char sha256 hex string
    assert_eq!(digest.len(), 64);
    assert_eq!(digest, triad::constitution::Constitution::compute_digest(&triad::constitution::DEFAULT_CONSTITUTION_TOML));

    // Pinned path must exist
    let pin_file = triad::constitution::pinned_digest_path();
    assert!(pin_file.exists());
    let pinned_content = fs::read_to_string(pin_file).unwrap();
    assert_eq!(pinned_content.trim(), digest);
}

#[test]
fn test_semantic_object_graph_and_density() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let store = Store::open(&db_path).unwrap();

    // Insert semantic entities
    store.upsert_semantic_node("concept:rust", "Rust Programming Language", "concept", "{}").unwrap();
    store.upsert_semantic_node("concept:safety", "Memory Safety", "concept", "{}").unwrap();
    store.upsert_semantic_node("source:rust_book", "The Rust Book", "source", "{\"url\":\"https://doc.rust-lang.org\"}").unwrap();

    // Create relationships
    store.add_semantic_edge("concept:rust", "concept:safety", "guarantees", 1.0).unwrap();
    store.add_semantic_edge("concept:rust", "source:rust_book", "documented_in", 0.9).unwrap();

    let (nodes, edges) = store.get_semantic_graph_stats().unwrap();
    assert_eq!(nodes, 3);
    assert_eq!(edges, 2);

    // Record memory reuse
    store.record_memory_reuse("concept:rust", "task_01", "Referenced for concurrency").unwrap();
    let reuses = store.list_memory_reuses(10).unwrap();
    assert_eq!(reuses.len(), 1);
    assert_eq!(reuses[0].0, "concept:rust");
    assert_eq!(reuses[0].1, "task_01");
}

#[test]
fn test_tool_endorser_enforcement_and_unauthorized_agent_rejection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let store = Store::open(&db_path).unwrap();

    let cid = store.propose_claim("Test claim", "src", "Explorer").unwrap();

    // Reject unknown agent voting
    let res_bad_agent = store.record_vote(&cid, "HackerAgent", "endorse", 1, None);
    assert!(res_bad_agent.is_err(), "Voting by non-constitutional agent must be rejected");

    // Same agent voting twice in same cycle cannot double count
    let res1 = store.record_vote(&cid, "Explorer", "endorse", 1, None).unwrap();
    assert!(!res1);
    let res2 = store.record_vote(&cid, "Explorer", "endorse", 1, None).unwrap();
    assert!(!res2);
}

#[tokio::test]
async fn test_stagnation_recovery_simulation() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let ws_dir = dir.path().join("workspace");
    fs::create_dir_all(&ws_dir).unwrap();

    let store = Store::open(&db_path).unwrap();
    let config = AppConfig::parse_and_validate(DEFAULT_CONFIG_TOML).unwrap();
    let (constitution, raw, digest) = triad::constitution::load_and_verify_constitution().unwrap();
    let m_client = triad::model::ModelClient::new(
        "http://localhost:8366/v1", "smollm3-3b", 100, true, 1,
        "http://127.0.0.1:8348/v1", "nomic-embed-text-v1-5", "q:", "d:",
    );
    let web = triad::web::WebExplorer::new("https://en.wikipedia.org/w/api.php", 5);
    let adapter = std::sync::Arc::new(triad::execution::BubblewrapAdapter::probe());
    let logger = triad::logging::Logger::new(dir.path().join("logs"), 10, 3);

    let coord = triad::coordinator::Coordinator::new(
        config, constitution, raw, digest.clone(), store.clone(), m_client, web, adapter, logger, ws_dir,
    );

    // Trigger deliberate stagnation recovery
    coord.execute_stagnation_recovery().await;

    // Verify recovery claim was added without losing consistent memory
    let learnings = store.list_learnings(10).unwrap();
    assert!(!learnings.is_empty());
    assert!(learnings[0].statement.contains("Autonomous stagnation recovery successfully completed"));
    assert_eq!(learnings[0].status, "trusted");
}

#[tokio::test]
async fn test_extension_lifecycle_and_reranker_evaluation_and_activation() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let ws_dir = dir.path().join("workspace");
    fs::create_dir_all(&ws_dir).unwrap();

    let store = Store::open(&db_path).unwrap();
    let config = AppConfig::parse_and_validate(DEFAULT_CONFIG_TOML).unwrap();
    let (constitution, raw, digest) = triad::constitution::load_and_verify_constitution().unwrap();
    let m_client = triad::model::ModelClient::new(
        "http://localhost:8366/v1", "smollm3-3b", 100, true, 1,
        "http://127.0.0.1:8348/v1", "nomic-embed-text-v1-5", "q:", "d:",
    );
    let web = triad::web::WebExplorer::new("https://en.wikipedia.org/w/api.php", 5);
    let adapter = std::sync::Arc::new(triad::execution::BubblewrapAdapter::probe());
    let logger = triad::logging::Logger::new(dir.path().join("logs"), 10, 3);

    let coord = triad::coordinator::Coordinator::new(
        config, constitution, raw, digest.clone(), store.clone(), m_client, web, adapter, logger, ws_dir,
    );

    // 1. Propose and build the keyword boost reranker extension (proposer Explorer, endorser Builder)
    let ext_id = coord.build_and_activate_reranker_extension("Explorer", "Builder").await.expect("Build and activate reranker must succeed");
    
    // 2. Verify state is 'active' in store and in manager
    let ext = store.get_extension_by_id(ext_id).unwrap().expect("Extension must exist in store");
    assert_eq!(ext.state, "active");
    assert_eq!(ext.hook, "retrieval.rerank");
    assert_eq!(ext.endorser_1.as_deref(), Some("Explorer"));
    assert_eq!(ext.endorser_2.as_deref(), Some("Builder"));
    assert!(ext.evaluation_results_json.is_some(), "Must have evaluation score");

    let active_ext = coord.extension_manager.get_active_extension(triad::extensions::HookType::RetrievalRerank).await;
    assert!(active_ext.is_some(), "Must be registered as active hook in manager");
    assert_eq!(active_ext.unwrap().id, ext_id);

    // 3. Test execution of retrieval.rerank hook via ExtensionManager
    let candidates = vec![
        triad::extensions::CandidateClaim {
            id: "claim-1".to_string(),
            statement: "The speed of light in vacuum is approximately 299,792,458 m/s.".to_string(),
            score: 0.70,
            source_url: "https://en.wikipedia.org/wiki/Speed_of_light".to_string(),
            canonical_work_id: Some("wiki:speed_of_light".to_string()),
            graph_degree: 3,
        },
        triad::extensions::CandidateClaim {
            id: "claim-2".to_string(),
            statement: "Sound travels much slower than light.".to_string(),
            score: 0.85,
            source_url: "https://en.wikipedia.org/wiki/Speed_of_sound".to_string(),
            canonical_work_id: Some("wiki:speed_of_sound".to_string()),
            graph_degree: 1,
        },
    ];

    let task_ctx = triad::extensions::TaskContext {
        task_id: "test-eval-1".to_string(),
        cycle_id: 1,
        step: "test".to_string(),
        active_agent: Some("Explorer".to_string()),
    };

    let rerank_input = triad::extensions::RetrievalRerankInput {
        query: "speed of light".to_string(),
        candidates: candidates.clone(),
        top_k: 2,
    };

    let rerank_output = coord.extension_manager.execute_retrieval_rerank(
        &task_ctx,
        &digest,
        &rerank_input,
    ).await;

    assert_eq!(rerank_output.ranked_ids.len(), 2);
    // claim-1 has exact keywords "speed", "of", "light" so keyword boost should place it first!
    assert_eq!(rerank_output.ranked_ids[0], "claim-1");

    // 4. Test automatic rollback after 3 consecutive failures
    for _ in 0..3 {
        coord.extension_manager.handle_extension_failure(ext_id, "Simulated failure for rollback test").await;
    }
    let ext_after_fails = store.get_extension_by_id(ext_id).unwrap().expect("Extension must exist");
    assert_eq!(ext_after_fails.state, "rolled_back", "Extension must be rolled back after 3 consecutive failures");
    
    let active_after_rollback = coord.extension_manager.get_active_extension(triad::extensions::HookType::RetrievalRerank).await;
    assert!(active_after_rollback.is_none(), "Rolled back hook must not be in active map");

    // 5. Test fallback behavior: when no active extension exists, original candidate ordering is preserved
    let fallback_output = coord.extension_manager.execute_retrieval_rerank(
        &task_ctx,
        &digest,
        &rerank_input,
    ).await;
    // Without active extension, candidate-2 has higher original score (0.85 > 0.70), so it comes first
    assert_eq!(fallback_output.ranked_ids[0], "claim-2");
}

#[tokio::test]
async fn test_specialist_lifecycle_concurrency_and_governance() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("triad.db");
    let store = Store::open(&db_path).unwrap();

    let (constitution, raw, digest) = triad::constitution::load_and_verify_constitution().unwrap();
    let model = triad::model::ModelClient::new(
        "http://localhost:8366/v1", "smollm3-3b", 100, true, 1,
        "http://127.0.0.1:8348/v1", "nomic-embed-text-v1-5", "q:", "d:",
    );
    let logger = triad::logging::Logger::new(temp_dir.path().join("logs"), 10, 3);

    let spec_mgr = triad::specialists::SpecialistManager::new(
        store.clone(),
        logger,
        std::sync::Arc::new(tokio::sync::Mutex::new(model)),
        std::sync::Arc::new(raw),
        std::sync::Arc::new(digest),
    );

    // 1. Initial concurrency limit must be 6
    let status = spec_mgr.get_status_report().await.unwrap();
    assert_eq!(status.current_limit, 6);

    // 2. Reject specialist creation without 2 distinct permanent agent endorsers
    let deadline = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
    let unauth_creation = spec_mgr.propose_and_create_specialist(
        &constitution,
        "spec-unauth",
        "Literature Reviewer",
        "Deep literature extraction",
        "Find paper",
        500,
        &deadline,
        "Explorer",
        "UnauthorizedAgent",
    ).await;
    assert!(unauth_creation.is_err(), "Must reject non-permanent agent endorser");

    let dup_creation = spec_mgr.propose_and_create_specialist(
        &constitution,
        "spec-dup",
        "Literature Reviewer",
        "Deep literature extraction",
        "Find paper",
        500,
        &deadline,
        "Explorer",
        "Explorer",
    ).await;
    assert!(dup_creation.is_err(), "Must reject duplicate endorser");

    // 3. Valid creation with Explorer and Sceptic
    let valid_id = spec_mgr.propose_and_create_specialist(
        &constitution,
        "spec-lit",
        "Literature Reviewer",
        "Deep literature extraction",
        "Find paper on hierarchical planning",
        500,
        &deadline,
        "Explorer",
        "Sceptic",
    ).await.unwrap();
    assert!(valid_id.starts_with("spec-spec-lit-"));

    let st = spec_mgr.get_status_report().await.unwrap();
    assert_eq!(st.running_specialists, 1);

    // 4. Fill up remaining 5 slots to reach full capacity of 6
    for i in 2..=6 {
        let name = format!("spec-worker-{}", i);
        let _ = spec_mgr.propose_and_create_specialist(
            &constitution,
            &name,
            "Worker",
            "Subtask work",
            "Complete subtask",
            500,
            &deadline,
            "Explorer",
            "Builder",
        ).await.unwrap();
    }
    let st6 = spec_mgr.get_status_report().await.unwrap();
    assert_eq!(st6.running_specialists, 6);

    // 7th specialist must be queued
    let _ = spec_mgr.propose_and_create_specialist(
        &constitution,
        "spec-queued-7",
        "Queued Worker",
        "Queued task",
        "Wait for slot",
        500,
        &deadline,
        "Sceptic",
        "Builder",
    ).await.unwrap();

    let st7 = spec_mgr.get_status_report().await.unwrap();
    assert_eq!(st7.running_specialists, 6);
    assert_eq!(st7.queued_tasks, 1);

    // 5. User reduction command: triad specialists --limit 3 terminates newest running specialists down to limit 3
    let res_msg = spec_mgr.set_user_limit(3).await.unwrap();
    assert!(res_msg.contains("3 excess specialists checkpointed/terminated"));

    let st_after_red = spec_mgr.get_status_report().await.unwrap();
    assert_eq!(st_after_red.current_limit, 3);
    assert_eq!(st_after_red.running_specialists, 3);
    assert!(st_after_red.cooldown_active, "24h cooldown must be active after user reduction");

    // 6. Test unanimous limit increase voting (+1 slot only)
    // Non-unanimous proposal (e.g. Sceptic votes no / duplicate votes) must be rejected
    let non_unanimous = spec_mgr.propose_and_vote_limit_increase(
        "Need more slots for benchmark exploration",
        "CPU & RAM capacity idle",
        "rev-1",
        ("Explorer", "Builder", "Explorer"),
    ).await;
    assert!(non_unanimous.is_err(), "Non-unanimous increase must be rejected");

    // Unanimous proposal right after reduction must be rejected by 24h cooldown
    let cooldown_rejection = spec_mgr.propose_and_vote_limit_increase(
        "Need more slots for benchmark exploration",
        "CPU & RAM capacity idle",
        "rev-1",
        ("Explorer", "Sceptic", "Builder"),
    ).await;
    assert!(cooldown_rejection.is_err(), "Must enforce 24-hour cooldown after user reduction");
}

#[tokio::test]
async fn test_reasoning_and_deep_research_extension_suite_end_to_end() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test_suite.db");
    let ws_dir = dir.path().join("workspace");
    fs::create_dir_all(&ws_dir).unwrap();

    let store = Store::open(&db_path).unwrap();
    let config = AppConfig::parse_and_validate(DEFAULT_CONFIG_TOML).unwrap();
    let (constitution, raw, digest) = triad::constitution::load_and_verify_constitution().unwrap();
    let m_client = triad::model::ModelClient::new(
        "http://localhost:8366/v1", "smollm3-3b", 100, true, 1,
        "http://127.0.0.1:8348/v1", "nomic-embed-text-v1-5", "q:", "d:",
    );
    let web = triad::web::WebExplorer::new("https://en.wikipedia.org/w/api.php", 5);
    let adapter = std::sync::Arc::new(triad::execution::BubblewrapAdapter::probe());
    let logger = triad::logging::Logger::new(dir.path().join("logs"), 10, 3);

    let coord = triad::coordinator::Coordinator::new(
        config, constitution, raw, digest.clone(), store.clone(), m_client, web, adapter, logger, ws_dir,
    );

    // 1. Autonomously synthesize, compile, test, evaluate, endorse, and activate all 5 hooks
    coord.ensure_reasoning_and_research_extensions_active(1).await;

    let active_exts = store.list_extensions().unwrap();
    assert_eq!(active_exts.len(), 5, "All 5 coordinator hook extensions must be registered and active");
    for ext in &active_exts {
        assert_eq!(ext.state, "active", "Extension {} must be active", ext.name);
    }

    let task_ctx = triad::extensions::TaskContext {
        task_id: "test-reasoning-pipeline".to_string(),
        cycle_id: 1,
        step: "verify".to_string(),
        active_agent: Some("Explorer".to_string()),
    };

    // 2. Verify query.transform hook (multi_source_research_decomposer)
    let q_out = coord.extension_manager.execute_query_transform(
        &task_ctx,
        &digest,
        &triad::extensions::QueryTransformInput {
            original_query: "Tree of Thoughts reasoning and process reward verification".to_string(),
            target_domain: Some("AI_Reasoning".to_string()),
        },
    ).await;
    assert!(q_out.sub_queries.len() >= 3, "Must decompose into at least 3 complementary sub-queries");
    assert!(q_out.target_connectors.contains(&"arxiv".to_string()));
    assert!(q_out.target_connectors.contains(&"openreview".to_string()));
    assert!(!q_out.follow_up_targets.is_empty());

    // 3. Verify provenance deduplication across multiple repositories & retrieval.rerank
    let raw_papers = vec![
        triad::web::ResearchSourcePaper {
            source_connector: "arxiv".to_string(),
            title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models".to_string(),
            authors: vec!["Shunyu Yao".to_string()],
            published_date: Some("2023-05-17".to_string()),
            url: "https://arxiv.org/abs/2305.10601v2".to_string(),
            code_url: Some("https://github.com/princeton-nlp/tree-of-thought-llm".to_string()),
            doi: Some("10.48550/arXiv.2305.10601".to_string()),
            arxiv_id: Some("2305.10601".to_string()),
            version: Some("v2".to_string()),
            abstract_or_excerpt: "We propose Tree of Thoughts (ToT) for deliberate reasoning search. Experiments on Game of 24 show 74% success. Limitation: higher token cost.".to_string(),
            experimental_details: Some("Tree search over coherent language thought units".to_string()),
            limitations_excerpt: Some("Requires more inference compute than IO prompting".to_string()),
            citations: vec!["2201.11903".to_string()],
            retrieved_at: "2026-10-31T06:00:00Z".to_string(),
            availability_status: "accessible".to_string(),
        },
        triad::web::ResearchSourcePaper {
            source_connector: "semantic_scholar".to_string(),
            title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models".to_string(),
            authors: vec!["Shunyu Yao".to_string()],
            published_date: Some("2023-05-17".to_string()),
            url: "https://www.semanticscholar.org/paper/2305.10601".to_string(),
            code_url: None,
            doi: Some("10.48550/arxiv.2305.10601".to_string()),
            arxiv_id: Some("2305.10601".to_string()),
            version: None,
            abstract_or_excerpt: "Tree of Thoughts enables exploration and backtracking over reasoning branches.".to_string(),
            experimental_details: Some("BFS/DFS over thoughts".to_string()),
            limitations_excerpt: Some("Compute overhead".to_string()),
            citations: vec![],
            retrieved_at: "2026-10-31T06:00:01Z".to_string(),
            availability_status: "accessible".to_string(),
        },
    ];
    let deduplicated = triad::web::deduplicate_and_resolve_provenance(raw_papers);
    assert_eq!(deduplicated.len(), 1, "Multiple copies of the same paper across repositories must resolve to 1 canonical work");
    assert_eq!(deduplicated[0].canonical_id, "arxiv:2305.10601");
    assert_eq!(deduplicated[0].corroborating_connectors.len(), 2);

    // 4. Verify context.compose hook (evidence_chain_composer)
    let c_out = coord.extension_manager.execute_context_compose(
        &task_ctx,
        &digest,
        &triad::extensions::ContextComposeInput {
            prompt_preamble: "Question: How does Tree of Thoughts compare to standard CoT?".to_string(),
            retrieved_memories: vec![triad::extensions::MemoryItem {
                id: deduplicated[0].canonical_id.clone(),
                statement: deduplicated[0].abstract_excerpt.clone(),
                source_url: deduplicated[0].urls[0].clone(),
                proposer: "Explorer".to_string(),
                epistemic_type: "observation".to_string(),
            }],
            constitutional_digest: digest.clone(),
        },
    ).await;
    assert!(!c_out.premise_conclusion_chains.is_empty());
    assert!(c_out.preserved_sources.contains(&"https://arxiv.org/abs/2305.10601v2".to_string()));

    // 5. Verify turn.plan hook (hierarchical_branch_planner)
    let tp_out = coord.extension_manager.execute_turn_plan(
        &task_ctx,
        &digest,
        &triad::extensions::TurnPlanInput {
            topic: "Compare and calculate branch utility for 14 and 28 reasoning states".to_string(),
            available_agents: vec!["Explorer".into(), "Sceptic".into(), "Builder".into()],
            budget_tokens: 1536,
            past_strategies: vec!["Evidence-grounded decomposition".into()],
            past_failures: vec!["Unverified single-pass heuristic".into()],
        },
    ).await;
    assert!(tp_out.requires_program_solver);
    assert!(tp_out.reasoning_branches.len() >= 2);
    assert!(tp_out.subproblems.len() >= 2);
    assert!(!tp_out.specialist_assignments.is_empty());

    // 6. Verify program- and solver-assisted reasoning execution in sandbox
    let solver_out = coord
        .execute_program_solver_verification(
            "Compare and calculate branch utility for 14 and 28 reasoning states",
            &deduplicated,
            "test_solver_task",
        )
        .await
        .expect("Sandboxed program solver must compile and execute");
    assert!(solver_out.contains("SOLVER_VERIFIED: formalization_match=true"));
    assert!(solver_out.contains("inputs=(14,28)"));
    assert!(solver_out.contains("sum=42"));

    // 7. Verify learning.consolidate hook (executable_verifier_consolidator)
    let lc_out = coord.extension_manager.execute_learning_consolidate(
        &task_ctx,
        &digest,
        &triad::extensions::LearningConsolidateInput {
            claims: vec![triad::extensions::UnconsolidatedClaim {
                id: "claim-tot-1".to_string(),
                statement: "Tree of Thoughts with backtracking improves Game of 24 accuracy to 74% under bounded search.".to_string(),
                endorsements: vec!["Explorer".into(), "Builder".into()],
                canonical_work_id: Some("arxiv:2305.10601".to_string()),
                source_connectors: vec!["arxiv".into(), "semantic_scholar".into()],
            }],
        },
    ).await;
    assert_eq!(lc_out.decisions.len(), 1);
    assert_eq!(lc_out.decisions[0].action, "promote");
    assert_eq!(lc_out.decisions[0].independent_sources_verified, 2);
    assert!(!lc_out.decisions[0].verification_checks.is_empty());
}




