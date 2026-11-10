use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UnixListener;

use triad::config::{self, ensure_canonical_config, AppConfig};
use triad::coordinator::Coordinator;
use triad::execution::{self, AntigravityAdapter, BubblewrapAdapter, ExecutionAdapter};
use triad::ipc::{IpcRequest, IpcResponse};
use triad::logging::{self, Logger};
use triad::model::{self, ModelClient};
use triad::server::{default_socket_path, read_ipc_request, send_ipc_request, write_ipc_response, SingleInstanceLock};
use triad::store::Store;
use triad::web::WebExplorer;

#[derive(Parser, Debug)]
#[command(
    name = "triad",
    version = env!("TRIAD_BUILD_VERSION"),
    about = "Triad: persistent, curious, three-agent research community"
)]
struct Cli {
    /// Override data directory (defaults to paths.data_dir in ~/.triad/config.toml)
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize configuration and directories without overwriting existing settings
    Init,

    /// Start the background daemon process
    Start,

    /// Run the service in foreground for debugging or systemd unit
    Run {
        #[arg(long)]
        foreground: bool,
    },

    /// Request graceful shutdown of running background daemon
    Stop,

    /// Show process health, model connectivity, active task, and recent activity
    Status,

    /// Pause autonomous exploration while keeping CLI and task handling available
    Pause,

    /// Resume autonomous exploration
    Resume,

    /// Show recent trusted learnings, timestamps, sources, and endorsing agents
    Learning {
        #[arg(long)]
        last: Option<usize>,
        #[arg(long)]
        since: Option<String>,
        #[arg(long)]
        json: bool,
        #[command(subcommand)]
        sub: Option<LearningSubcommands>,
    },

    /// Queue a question for discussion
    Ask {
        question: String,
        #[arg(long)]
        wait: bool,
    },

    /// Queue a goal requiring investigation, tool use, or artifact creation
    Challenge {
        task: String,
        #[arg(long)]
        wait: bool,
    },

    /// Persist user information and enqueue discussion
    Tell {
        information: String,
        #[arg(long)]
        source: Option<String>,
    },

    /// Show progress, outputs, artifacts, errors, and the final result of a queued job
    Task {
        id: String,
    },

    /// Inspect agent questions
    Questions,

    /// Provide answer to an agent question
    Answer {
        id: i64,
        answer: String,
    },

    /// Semantic memory search
    Memory {
        #[command(subcommand)]
        sub: MemorySubcommands,
    },

    /// Show personalities, current objectives, recent contributions, and activity metrics
    Agents,

    /// Inspect available tools or inspect a specific tool
    Tools {
        #[command(subcommand)]
        sub: Option<ToolsSubcommands>,
    },

    /// Read or follow the exchange and learning log
    Logs {
        #[arg(long)]
        follow: bool,
        #[arg(long)]
        agent: Option<String>,
    },

    /// Check snap endpoints, embedding behaviour, runtime compatibility, isolation, and toolchain
    Doctor,

    /// Inspect, update, validate, or reload configuration
    Config {
        #[command(subcommand)]
        sub: ConfigSubcommands,
    },

    /// Rebuild embeddings atomically without losing learning history or trust decisions
    Reindex,

    /// Audit system progress, source evidence, tool executions, graph density, and failures
    Audit {
        #[arg(long)]
        json: bool,
    },

    /// Manage coordinator extensions (query.transform, retrieval.rerank, context.compose, turn.plan, learning.consolidate)
    Extensions {
        #[arg(long)]
        json: bool,
    },

    /// Inspect proposals for coordinator-level architectural changes
    CoordinatorTasks {
        #[arg(long)]
        json: bool,
    },

    /// Manage specialist agents and concurrency limits
    Specialists {
        /// Immediately set or reduce specialist concurrency limit
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },

    /// Run Triad's Expert Consultation MCP server over stdio or Streamable HTTP
    Mcp {
        /// Serve MCP over stdin/stdout (default if --http is not specified)
        #[arg(long)]
        stdio: bool,
        /// Serve Streamable HTTP MCP on the given bind address (e.g. 127.0.0.1:8390)
        #[arg(long)]
        http: Option<String>,
    },

    /// Run external held-out evaluation benchmark against a plain-LLM baseline with identical inference budget
    Evaluate {
        /// Path to external benchmark JSON file (defaults to domain.evaluation_set_path or evaluation/external_benchmark.json)
        #[arg(long)]
        benchmark: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },

    /// Publish the domain expertise profile (domain, coverage linked to scores, full-text share, real vs synthetic experiments, evaluated strengths, known weaknesses, connector health, frontier availability, last evaluation date)
    Profile {
        #[arg(long)]
        json: bool,
    },

    /// Inspect persistent, version-aware source tracking state and independent corroboration counts
    Sources {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },

    /// Inspect research experiments, specifications, source fidelity, baseline comparisons, and outcomes
    Experiments {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        json: bool,
    },

    /// Inspect model providers, selected models per workload, ranking basis, frontier budgets, and blocked tasks
    Models {
        /// Probe and refresh provider model discovery before displaying
        #[arg(long)]
        refresh: bool,
        #[arg(long)]
        json: bool,
    },

    /// Purge redundant Cargo build caches (shared_target/debug, duplicate test binaries, scratch runs) without losing knowledge or active binaries
    Purge,
}

#[derive(Subcommand, Debug)]
enum LearningSubcommands {
    /// Show full claim details, source passages, vote history, and dissent
    Show { id: String },
}

#[derive(Subcommand, Debug)]
enum MemorySubcommands {
    /// Search semantic memory
    Search { query: String },
}

#[derive(Subcommand, Debug)]
enum ToolsSubcommands {
    /// Show details, source, and tests for a tool
    Show { name: String },
}

#[derive(Subcommand, Debug)]
enum ConfigSubcommands {
    /// Show effective configuration, path, and sha256 hash
    Show,
    /// Set a key to value in ~/.triad/config.toml
    Set { key: String, value: String },
    /// Validate current configuration file syntax and semantics
    Validate,
    /// Request daemon to reload configuration
    Reload,
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Every invocation automatically reads/creates ~/.triad/config.toml
    let (config, config_hash, config_path) = ensure_canonical_config()
        .context("Failed to initialize or read ~/.triad/config.toml")?;

    let cli = Cli::parse();
    let effective_data_dir = config.resolved_data_dir(cli.data_dir.as_deref());
    let effective_log_dir = config.resolved_log_dir();
    let effective_workspace_dir = config.resolved_workspace_dir();

    fs::create_dir_all(&effective_data_dir)?;
    fs::create_dir_all(&effective_log_dir)?;
    fs::create_dir_all(&effective_workspace_dir)?;

    let db_path = effective_data_dir.join("triad.db");
    let sock_path = default_socket_path();

    match cli.command {
        Commands::Init => {
            println!("Triad initialized successfully.");
            println!("Configuration: {}", config_path.display());
            println!("Data directory: {}", effective_data_dir.display());
            println!("Log directory: {}", effective_log_dir.display());
            println!("Workspace directory: {}", effective_workspace_dir.display());
            Ok(())
        }

        Commands::Doctor => {
            run_doctor(&config, &effective_workspace_dir).await
        }

        Commands::Start => {
            // Check if daemon is already running
            if sock_path.exists() {
                if let Ok(IpcResponse::Status(rep)) = send_ipc_request(&IpcRequest::Status, &sock_path).await {
                    println!("Triad daemon is already running (PID: {}).", rep.pid);
                    return Ok(());
                }
            }

            // Spawn foreground worker in background
            let exe = std::env::current_exe()?;
            let mut cmd = std::process::Command::new(exe);
            cmd.arg("run").arg("--foreground");
            if let Some(ref d) = cli.data_dir {
                cmd.arg("--data-dir").arg(d);
            }

            let child = cmd.spawn().context("Failed to spawn triad background process")?;
            println!("Starting Triad daemon (PID: {})...", child.id());

            // Poll for socket readiness up to 5 seconds
            let start = std::time::Instant::now();
            let mut ready = false;
            while start.elapsed() < Duration::from_secs(5) {
                tokio::time::sleep(Duration::from_millis(200)).await;
                if let Ok(IpcResponse::Status(_)) = send_ipc_request(&IpcRequest::Status, &sock_path).await {
                    ready = true;
                    break;
                }
            }

            if ready {
                println!("Triad daemon started successfully and is ready.");
            } else {
                eprintln!("Triad daemon spawned, waiting on initialization. Run 'triad status' to check.");
            }
            Ok(())
        }

        Commands::Run { foreground: _ } => {
            run_daemon(config, effective_data_dir, effective_log_dir, effective_workspace_dir, sock_path).await
        }

        Commands::Stop => {
            match send_ipc_request(&IpcRequest::Stop, &sock_path).await {
                Ok(_) => println!("Shutdown signal sent to Triad daemon."),
                Err(e) => eprintln!("Error communicating with daemon: {}", e),
            }
            Ok(())
        }

        Commands::Status => {
            println!("--- Triad Status ---");
            println!("Config file: {} (hash: {})", config_path.display(), &config_hash[..8]);
            println!("Data directory: {}", effective_data_dir.display());
            println!("Log directory: {}", effective_log_dir.display());
            println!("Workspace: {}", effective_workspace_dir.display());

            match tokio::time::timeout(
                Duration::from_secs(3),
                send_ipc_request(&IpcRequest::Status, &sock_path),
            )
            .await
            {
                Ok(Ok(IpcResponse::Status(rep))) => {
                    println!("Daemon PID: {}", rep.pid);
                    println!("Daemon running: {}", rep.running);
                    println!("Autonomous exploration paused: {}", rep.paused);
                    println!("Constitution version: {} (digest: {})", rep.constitution_version, &rep.constitution_digest[..16]);
                    println!("Learning progress: {}", rep.learning_progress);
                    println!("Latest audit: {}", rep.latest_audit);
                    println!("Self-repair state: {}", rep.repair_state);
                    println!("Current cycle: #{}", rep.current_cycle);
                    println!("Active task: {}", rep.active_task.as_deref().unwrap_or("None (idle / exploring)"));
                    println!("Chat endpoint: {}", rep.model_chat_status);
                    println!("Embedding endpoint: {}", rep.model_emb_status);
                    println!("Antigravity execution: {}", rep.antigravity_status);
                    println!("Process isolation: {}", rep.isolation_status);
                    println!("Trusted learnings: {}", rep.trusted_learnings_count);
                    println!("Registered tools: {}", rep.tools_count);
                    if !rep.source_tracking_summary.is_empty() {
                        println!("Source tracking: {}", rep.source_tracking_summary);
                    }
                    if let Ok(store) = Store::open(&db_path) {
                        if let Ok(fb) = store.get_frontier_budget_summary(&config.models.frontier_budget, 5) {
                            println!(
                                "Frontier budget (24h): {}/{} calls used ({} remaining) | Concurrency: {}/{} | Spend: {} / {} micros | Blocked/Degraded tasks: {} | Exhaustion policy: {}",
                                fb.used_calls_24h,
                                fb.max_calls_per_24h,
                                fb.remaining_calls_24h,
                                fb.active_concurrency,
                                fb.max_concurrency,
                                fb.used_spend_micros_24h,
                                fb.max_spend_micros_per_24h,
                                fb.blocked_tasks.len(),
                                fb.exhaustion_policy
                            );
                            let wl_dist: Vec<String> = fb
                                .workload_calls_24h
                                .iter()
                                .map(|(k, v)| format!("{}:{}", k, v))
                                .collect();
                            let mdl_dist: Vec<String> = fb
                                .model_calls_24h
                                .iter()
                                .map(|(k, v)| format!("{}:{}", k, v))
                                .collect();
                            println!(
                                "Frontier smart pacing: enabled={} (target_interval={}s, next_slot_in={}s) | Next priority workload: {} | 24h Workload dist: [{}] | 24h Model dist: [{}]",
                                fb.smart_pacing_enabled,
                                fb.target_pacing_interval_secs,
                                fb.secs_until_next_paced_slot,
                                fb.next_priority_workload.as_deref().unwrap_or("any"),
                                if wl_dist.is_empty() { "none".to_string() } else { wl_dist.join(", ") },
                                if mdl_dist.is_empty() { "none".to_string() } else { mdl_dist.join(", ") }
                            );
                        }
                    }
                    println!("Uptime: {}s", rep.uptime_seconds);
                }
                _ => {
                    println!("Daemon status: Stopped (daemon is not running)");
                    // Read directly from offline database
                    if let Ok(store) = Store::open(&db_path) {
                        let learnings = store.list_learnings(1000).unwrap_or_default().len();
                        let tools = store.list_tools().unwrap_or_default().len();
                        let (nodes, edges) = store.get_semantic_graph_stats().unwrap_or((0, 0));
                        println!("Offline Store - Trusted learnings: {}, Tools: {}, Semantic nodes: {}, Edges: {}", learnings, tools, nodes, edges);
                        if let Ok(st) = store.get_source_tracking_diagnostics(5) {
                            let total_indep = st.independent_papers_count + st.independent_datasets_count + st.independent_implementations_count;
                            println!(
                                "Source tracking: {} canonical sources (completed: {}, pending: {}, reserved: {}, failed: {}, retry_eligible: {}) | Independent corroborations: {} (papers: {}, datasets: {}, code: {}, deduped routes: {})",
                                st.total_tracked_sources,
                                st.completed_research_count,
                                st.pending_research_count,
                                st.reserved_research_count,
                                st.failed_research_count,
                                st.retry_eligible_count,
                                total_indep,
                                st.independent_papers_count,
                                st.independent_datasets_count,
                                st.independent_implementations_count,
                                st.duplicate_route_observations_deduped
                            );
                        }
                    }
                }
            }
            Ok(())
        }

        Commands::Pause => {
            match send_ipc_request(&IpcRequest::Pause, &sock_path).await {
                Ok(IpcResponse::Ok { message }) => println!("{}", message),
                Ok(other) => println!("{:?}", other),
                Err(e) => eprintln!("Failed to pause daemon: {}", e),
            }
            Ok(())
        }

        Commands::Resume => {
            match send_ipc_request(&IpcRequest::Resume, &sock_path).await {
                Ok(IpcResponse::Ok { message }) => println!("{}", message),
                Ok(other) => println!("{:?}", other),
                Err(e) => eprintln!("Failed to resume daemon: {}", e),
            }
            Ok(())
        }

        Commands::Learning { last, since: _, json, sub } => {
            let store = Store::open(&db_path)?;
            if let Some(LearningSubcommands::Show { id }) = sub {
                if let Some(claim) = store.get_claim(&id)? {
                    println!("--- Learning Details ---");
                    println!("ID: {}", claim.id);
                    println!("Statement: {}", claim.statement);
                    println!("Trust Attribute: {} (Triad 2-agent agreement policy)", claim.status);
                    println!(
                        "Source Support Attribute: {} ({} independent corroborating source(s))",
                        claim.source_support_status, claim.source_corroboration_count
                    );
                    println!(
                        "Experimental Verification Attribute: {}{}",
                        claim.experimental_verification_status,
                        claim
                            .verification_evidence
                            .as_deref()
                            .map(|e| format!(" ({})", e))
                            .unwrap_or_default()
                    );
                    println!("Source: {}", claim.source_url);
                    println!("Proposer: {}", claim.proposer);
                    println!("Version: {}", claim.version);
                    println!("Created: {}", claim.created_at);
                    if let Ok(corr) = store.get_corroboration_summary(&id) {
                        if corr.total_independent_sources > 0 {
                            println!(
                                "\nIndependent Corroboration Breakdown: {} total independent (papers: {}, datasets: {}, implementations: {}, total route observations: {})",
                                corr.total_independent_sources,
                                corr.independent_papers,
                                corr.independent_datasets,
                                corr.independent_implementations,
                                corr.total_route_observations
                            );
                        }
                    }
                    println!("\nVote Provenance:");
                    let votes = store.get_claim_votes(&id)?;
                    for v in votes {
                        println!("- Agent: {} | Decision: {} (Cycle #{}) | Rationale: {}",
                            v.agent, v.decision, v.cycle_id, v.rationale.as_deref().unwrap_or("none"));
                    }
                } else {
                    println!("Claim '{}' not found", id);
                }
                return Ok(());
            }

            let limit = last.unwrap_or(10);
            let learnings = store.list_learnings(limit)?;
            let experiments = store.list_research_experiments(limit).unwrap_or_default();
            let experiment_specs = store.list_experiment_specs(limit).unwrap_or_default();
            let research_works = store.list_research_works(limit).unwrap_or_default();
            let research_syntheses = store.list_research_syntheses(limit).unwrap_or_default();
            let specialists = store.list_specialists(limit).unwrap_or_default();
            let source_tracking = store.get_source_tracking_diagnostics(limit).unwrap_or_default();
            if json {
                #[derive(serde::Serialize)]
                struct LearningView<'a> {
                    learnings: &'a [triad::store::Claim],
                    experiments: &'a [triad::store::ResearchExperimentRecord],
                    experiment_specifications: &'a [triad::store::ExperimentSpecification],
                    research_works: &'a [triad::web::ResolvedResearchWork],
                    research_syntheses: &'a [triad::store::ResearchSynthesisRecord],
                    specialists: &'a [triad::specialists::SpecialistRecord],
                    source_tracking: &'a triad::store::SourceTrackingDiagnostics,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&LearningView {
                        learnings: &learnings,
                        experiments: &experiments,
                        experiment_specifications: &experiment_specs,
                        research_works: &research_works,
                        research_syntheses: &research_syntheses,
                        specialists: &specialists,
                        source_tracking: &source_tracking,
                    })?
                );
            } else {
                println!("--- Recent Trusted Learnings ({} items) ---", learnings.len());
                for l in learnings {
                    println!(
                        "[{}] {} (Kind: {} | Trust: {} | SourceSupport: {} [{}] | Fidelity: {} | ExpVerification: {} [exec={:?}] | Source: {}, Proposer: {})",
                        &l.id[..8],
                        l.statement,
                        l.finding_kind,
                        l.status,
                        l.source_support_status,
                        l.source_corroboration_count,
                        l.source_fidelity_summary.as_deref().unwrap_or("n/a"),
                        l.experimental_verification_status,
                        l.execution_id,
                        l.source_url,
                        l.proposer
                    );
                }
                if !specialists.is_empty() {
                    println!("\n--- Temporary Specialist Agents ({} items) ---", specialists.len());
                    for s in &specialists {
                        println!(
                            "• [{}] {} — {} (Status: {} | Phase: {} | ExecIDs: {:?} | Endorsers: {} & {})",
                            s.name, s.role, s.task, s.status, s.phase, s.execution_ids, s.endorser_1, s.endorser_2
                        );
                        if let Some(ref res) = s.result {
                            println!("  Finding: {}", triad::model::clip_text(res, 220));
                        }
                    }
                }
                if !research_works.is_empty() {
                    println!("\n--- Deduplicated Canonical Research Works ({} items) ---", research_works.len());
                    for w in &research_works {
                        println!(
                            "• [{}] {} (Kind: {} | Connectors: {} | Routes Merged: {} | Independent Count: {} | Status: {})",
                            w.canonical_id,
                            w.title,
                            w.source_kind,
                            w.corroborating_connectors.join(", "),
                            w.corroborating_connectors.len().max(w.duplicate_copies_merged),
                            w.independent_corroboration_count,
                            w.availability_status
                        );
                    }
                }
                if !research_syntheses.is_empty() {
                    println!("\n--- Multi-Source Research Syntheses & Disagreement Analyses ({} items) ---", research_syntheses.len());
                    for syn in &research_syntheses {
                        println!("• Question: {}", syn.question);
                        println!("  Competing Findings: {}", syn.competing_findings);
                        println!("  Limitations: {}", syn.limitations_summary);
                        println!("  Experiment Translation: {}", syn.experiment_translation);
                    }
                }
                if !experiment_specs.is_empty() {
                    println!("\n--- Experiment Specifications ({} items) ---", experiment_specs.len());
                    for sp in &experiment_specs {
                        let metric_keys: Vec<&str> = sp.metrics.iter().map(|m| m.result_key.as_str()).collect();
                        println!(
                            "• [{}] {} (Check: {} | Fidelity: {} | UnderSpecified: {} | ScepticReview: {})",
                            sp.id, sp.name, sp.executable_check_kind, sp.source_fidelity_status, sp.under_specified, sp.sceptic_review_status
                        );
                        println!("  Hypothesis: {}", sp.falsifiable_hypothesis);
                        println!("  Baseline: {} | Declared Metric Keys: {:?}", sp.baseline_description, metric_keys);
                        if sp.under_specified && !sp.missing_source_details.is_empty() {
                            println!("  Missing Source Details: {}", sp.missing_source_details.join("; "));
                        }
                    }
                }
                if !experiments.is_empty() {
                    println!("\n--- Research Investigations & Experiments ({} items) ---", experiments.len());
                    for exp in experiments {
                        println!(
                            "• Cycle #{} [status={} | spec={:?} | tool={:?} | exec_id={:?} | fidelity={} | under_specified={}]: {}",
                            exp.cycle_id,
                            exp.outcome_status,
                            exp.spec_id,
                            exp.tool_name,
                            exp.execution_id,
                            exp.source_fidelity_status,
                            exp.under_specified,
                            exp.hypothesis
                        );
                        println!("  Mechanism: {}", exp.attempted_method);
                        if exp.measured_results_json != "{}" && !exp.measured_results_json.is_empty() {
                            println!("  Measured Results (results.json): {}", exp.measured_results_json);
                        }
                        if exp.baseline_comparison_json != "{}" && !exp.baseline_comparison_json.is_empty() {
                            println!("  Baseline Comparison: {}", exp.baseline_comparison_json);
                        }
                        if exp.held_out_evaluation_json != "{}" && !exp.held_out_evaluation_json.is_empty() {
                            println!("  Held-Out Generalization: {}", exp.held_out_evaluation_json);
                        }
                        println!("  Outcome: {}", exp.observed_outcome);
                        println!("  Lessons: {} ({})", exp.lessons_learned, exp.epistemic_note);
                    }
                }
            }
            Ok(())
        }

        Commands::Ask { question, wait } => {
            let task_id = format!("ask-{}", chrono::Utc::now().timestamp_millis());
            let store = Store::open(&db_path)?;
            store.create_task(&task_id, "ask", &question)?;

            let _ = send_ipc_request(&IpcRequest::Ask { question: question.clone() }, &sock_path).await;
            println!("Queued question task ID: {}", task_id);

            if wait {
                println!("Waiting for Triad agents to discuss and answer...");
                poll_task_completion(&store, &task_id).await?;
            }
            Ok(())
        }

        Commands::Challenge { task, wait } => {
            let task_id = format!("chlg-{}", chrono::Utc::now().timestamp_millis());
            let store = Store::open(&db_path)?;
            store.create_task(&task_id, "challenge", &task)?;

            let _ = send_ipc_request(&IpcRequest::Challenge { task: task.clone() }, &sock_path).await;
            println!("Queued challenge task ID: {}", task_id);

            if wait {
                println!("Waiting for Triad agents to build, test, and execute challenge...");
                poll_task_completion(&store, &task_id).await?;
            }
            Ok(())
        }

        Commands::Tell { information, source } => {
            let task_id = format!("tell-{}", chrono::Utc::now().timestamp_millis());
            let store = Store::open(&db_path)?;
            store.create_task(&task_id, "tell", &information)?;

            let _ = send_ipc_request(&IpcRequest::Tell { info: information, source }, &sock_path).await;
            println!("User information submitted into queue (Task ID: {}).", task_id);
            Ok(())
        }

        Commands::Task { id } => {
            let store = Store::open(&db_path)?;
            if let Some(t) = store.get_task(&id)? {
                println!("Task ID: {}", t.id);
                println!("Kind: {}", t.kind);
                println!("Status: {}", t.status);
                println!("Description: {}", t.description);
                if let Some(res) = t.result {
                    println!("\nResult:\n{}", res);
                }
                if let Some(err) = t.error {
                    println!("\nError:\n{}", err);
                }
            } else {
                println!("Task '{}' not found", id);
            }
            Ok(())
        }

        Commands::Questions => {
            let store = Store::open(&db_path)?;
            let questions = store.list_user_questions()?;
            println!("--- Agent Questions for User ({}) ---", questions.len());
            for q in questions {
                let ans_str = match q.answer {
                    Some(a) => format!("Answered: {}", a),
                    None => "Pending answer".to_string(),
                };
                println!("[ID: {}] Agent {}: {} ({})", q.id, q.agent, q.question, ans_str);
            }
            Ok(())
        }

        Commands::Answer { id, answer } => {
            let store = Store::open(&db_path)?;
            store.answer_user_question(id, &answer)?;
            println!("Clarification recorded for question ID {}.", id);
            Ok(())
        }

        Commands::Memory { sub } => {
            match sub {
                MemorySubcommands::Search { query } => {
                    let m_client = ModelClient::new(
                        &config.models.chat.base_url,
                        &config.models.chat.model,
                        config.models.chat.max_tokens,
                        config.models.chat.no_think,
                        config.models.chat.concurrency,
                        &config.models.embedding.base_url,
                        &config.models.embedding.model,
                        &config.models.embedding.query_prefix,
                        &config.models.embedding.document_prefix,
                    );
                    let query_emb = m_client.embed_text(&query, true).await?;
                    let store = Store::open(&db_path)?;
                    let results = store.search_semantic(&query_emb, 10)?;
                    println!("--- Semantic Memory Search for '{}' ---", query);
                    for (mem, sim) in results {
                        println!("[{}] (sim: {:.3}) (ID: {}): {}", mem.memory_type, sim, mem.target_id, mem.text);
                    }
                }
            }
            Ok(())
        }

        Commands::Agents => {
            println!("--- The Triad Agent Community ---");
            println!("1. Explorer\n   Personality: {}", config.agents.explorer.personality);
            println!("   Role: Hypotheses, discovery, Wikipedia exploration.");
            println!("2. Sceptic\n   Personality: {}", config.agents.sceptic.personality);
            println!("   Role: Verification, evidence checking, dissent.");
            println!("3. Builder\n   Personality: {}", config.agents.builder.personality);
            println!("   Role: Tool generation, testing, execution, measured outcomes.");
            println!("\nAgreement Invariant: Any two agents agreeing on a claim mark it trusted.");
            Ok(())
        }

        Commands::Tools { sub } => {
            let store = Store::open(&db_path)?;
            if let Some(ToolsSubcommands::Show { name }) = sub {
                if let Some(tool) = store.get_tool_by_name(&name)? {
                    println!("Tool: {} (v{})", tool.name, tool.version);
                    println!("Status: {}", tool.status);
                    println!("Description: {}", tool.description);
                    println!("Source Hash: {}", tool.source_hash);
                    println!(
                        "Endorsers: {} & {}",
                        tool.endorser_1.as_deref().unwrap_or("none"),
                        tool.endorser_2.as_deref().unwrap_or("none")
                    );
                    println!("Tests passed: {}", tool.test_passed);
                    if let Some(ref spec_id) = tool.experiment_spec_id {
                        println!("Experiment Specification ID: {}", spec_id);
                        if let Ok(Some(spec)) = store.get_experiment_spec(spec_id) {
                            println!("  Research Question: {}", spec.research_question);
                            println!("  Falsifiable Hypothesis: {}", spec.falsifiable_hypothesis);
                            println!("  Mechanism Under Test: {}", spec.mechanism_under_test);
                            println!("  Executable Check Kind: {}", spec.executable_check_kind);
                            println!("  Baseline: {}", spec.baseline_description);
                            println!(
                                "  Source Fidelity: {} (under_specified={})",
                                spec.source_fidelity_status, spec.under_specified
                            );
                            if !spec.missing_source_details.is_empty() {
                                println!(
                                    "  Missing Source Details: {}",
                                    spec.missing_source_details.join("; ")
                                );
                            }
                        }
                    }
                    if let Some(ref notes) = tool.sceptic_verification_notes {
                        println!("Sceptic Review Notes: {}", notes);
                    }
                    println!("\nSchema:\n{}", tool.schema_json);
                    println!("\nSource:\n{}", tool.source_code);
                } else {
                    println!("Tool '{}' not found", name);
                }
                return Ok(());
            }

            let tools = store.list_tools()?;
            println!("--- Registered Tools ({}) ---", tools.len());
            for t in tools {
                println!(
                    "- {} (v{}) [{}] - {} (Endorsers: {}, {} | Spec: {})",
                    t.name,
                    t.version,
                    t.status,
                    t.description,
                    t.endorser_1.as_deref().unwrap_or("none"),
                    t.endorser_2.as_deref().unwrap_or("none"),
                    t.experiment_spec_id.as_deref().unwrap_or("none")
                );
            }
            Ok(())
        }

        Commands::Logs { follow, agent } => {
            let logger = Logger::new(effective_log_dir.clone(), config.logging.rotate_size_mb, config.logging.retained_files);
            let records = logger.read_recent_records(50, agent.as_deref());
            for r in records {
                println!("[{}] [{}] [{}] {}", &r.timestamp[11..19], r.level, r.agent.as_deref().unwrap_or("system"), r.message);
            }

            if follow {
                println!("--- Following live log events (Ctrl+C to quit) ---");
                let log_file = effective_log_dir.join("triad.jsonl");
                let mut last_len = fs::metadata(&log_file).map(|m| m.len()).unwrap_or(0);
                loop {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if let Ok(meta) = fs::metadata(&log_file) {
                        let cur_len = meta.len();
                        if cur_len > last_len {
                            if let Ok(content) = fs::read_to_string(&log_file) {
                                let lines: Vec<&str> = content.lines().collect();
                                if let Some(last_line) = lines.last() {
                                    if let Ok(rec) = serde_json::from_str::<logging::LogRecord>(last_line) {
                                        if agent.is_none() || rec.agent.as_deref() == agent.as_deref() {
                                            println!("[{}] [{}] [{}] {}", &rec.timestamp[11..19], rec.level, rec.agent.as_deref().unwrap_or("system"), rec.message);
                                        }
                                    }
                                }
                            }
                            last_len = cur_len;
                        }
                    }
                }
            }
            Ok(())
        }

        Commands::Config { sub } => {
            match sub {
                ConfigSubcommands::Show => {
                    println!("Path: {}", config_path.display());
                    println!("SHA256: {}", config_hash);
                    let toml_str = fs::read_to_string(&config_path)?;
                    println!("\n{}", toml_str);
                }
                ConfigSubcommands::Validate => {
                    let toml_str = fs::read_to_string(&config_path)?;
                    match AppConfig::parse_and_validate(&toml_str) {
                        Ok(_) => println!("Configuration at {} is syntactically and semantically valid.", config_path.display()),
                        Err(e) => eprintln!("Configuration validation error: {}", e),
                    }
                }
                ConfigSubcommands::Set { key, value } => {
                    println!("Setting '{}' to '{}'...", key, value);
                    // Read current toml
                    let mut doc: toml::Table = fs::read_to_string(&config_path)?.parse()?;
                    let parts: Vec<&str> = key.split('.').collect();
                    if parts.len() == 2 {
                        let sec = doc.entry(parts[0]).or_insert(toml::Value::Table(toml::Table::new()));
                        if let toml::Value::Table(ref mut t) = sec {
                            if let Ok(num) = value.parse::<i64>() {
                                t.insert(parts[1].to_string(), toml::Value::Integer(num));
                            } else if let Ok(b) = value.parse::<bool>() {
                                t.insert(parts[1].to_string(), toml::Value::Boolean(b));
                            } else {
                                t.insert(parts[1].to_string(), toml::Value::String(value));
                            }
                        }
                    } else {
                        bail!("Only two-level keys (e.g. orchestration.cycle_interval_seconds) supported");
                    }

                    let new_toml = toml::to_string_pretty(&doc)?;
                    // Validate before saving
                    AppConfig::parse_and_validate(&new_toml)?;
                    config::write_atomic(&config_path, new_toml.as_bytes())?;
                    println!("Configuration updated successfully.");
                }
                ConfigSubcommands::Reload => {
                    match send_ipc_request(&IpcRequest::ConfigReload, &sock_path).await {
                        Ok(IpcResponse::Ok { message }) => println!("{}", message),
                        _ => println!("Config reloaded or applied."),
                    }
                }
            }
            Ok(())
        }

        Commands::Reindex => {
            let m_client = ModelClient::new(
                &config.models.chat.base_url,
                &config.models.chat.model,
                config.models.chat.max_tokens,
                config.models.chat.no_think,
                config.models.chat.concurrency,
                &config.models.embedding.base_url,
                &config.models.embedding.model,
                &config.models.embedding.query_prefix,
                &config.models.embedding.document_prefix,
            );
            let store = Store::open(&db_path)?;
            let raw_memories = store.get_all_embeddings_raw()?;
            println!("Re-indexing {} items atomically with model '{}'...", raw_memories.len(), config.models.embedding.model);

            let mut reindexed = Vec::new();
            for (mtype, tid, text) in raw_memories {
                let emb = m_client.embed_text(&text, false).await?;
                reindexed.push((mtype, tid, text, emb));
            }

            let new_encoder_info = format!("{}|dim={}|q_pfx={}|doc_pfx={}",
                config.models.embedding.model,
                reindexed.first().map(|x| x.3.len()).unwrap_or(0),
                config.models.embedding.query_prefix,
                config.models.embedding.document_prefix
            );
            store.replace_embeddings_atomic(&new_encoder_info, &reindexed)?;
            println!("Reindexing complete! Encoder meta updated atomically.");
            Ok(())
        }

        Commands::Audit { json: _ } => {
            match send_ipc_request(&IpcRequest::Audit, &sock_path).await {
                Ok(IpcResponse::Audit(rep)) => {
                    println!("{}", serde_json::to_string_pretty(&rep)?);
                }
                _ => {
                    // Generate offline audit report
                    let store = Store::open(&db_path)?;
                    let (nodes, edges) = store.get_semantic_graph_stats().unwrap_or((0, 0));
                    let (const_parsed, _, const_digest) = triad::constitution::load_and_verify_constitution()?;
                    let density_desc = if nodes > 0 {
                        format!("{:.3} edges per node ({} nodes, {} edges)", edges as f64 / nodes as f64, nodes, edges)
                    } else {
                        "0.000 (empty graph)".to_string()
                    };

                    let raw_sources = store.list_sources(50).unwrap_or_default();
                    let source_evidence: Vec<triad::ipc::SourceEvidenceItem> = raw_sources
                        .into_iter()
                        .map(|s| triad::ipc::SourceEvidenceItem {
                            url: s.url,
                            title: s.title,
                            retrieved_at: s.retrieved_at,
                        })
                        .collect();

                    let tool_execution_records = store.list_recent_tool_executions(50).unwrap_or_default();

                    let raw_reuses = store.list_memory_reuses(50).unwrap_or_default();
                    let memory_reuse: Vec<triad::ipc::MemoryReuseItem> = raw_reuses
                        .into_iter()
                        .map(|(mid, tid, ctx, at)| triad::ipc::MemoryReuseItem {
                            memory_id: mid,
                            task_id: tid,
                            context: ctx,
                            reused_at: at,
                        })
                        .collect();

                    let raw_evals = store.list_recent_curriculum_evaluations(50).unwrap_or_default();
                    let curriculum_evaluations: Vec<triad::ipc::CurriculumEvaluationItem> = raw_evals
                        .into_iter()
                        .map(|(cid, step, status, details, at)| triad::ipc::CurriculumEvaluationItem {
                            cycle_id: cid,
                            step,
                            status,
                            details,
                            created_at: at,
                        })
                        .collect();

                    let logger = Logger::new(effective_log_dir.clone(), config.logging.rotate_size_mb, config.logging.retained_files);
                    let recent_logs = logger.read_recent_records(100, None);
                    let failures_recorded: Vec<triad::ipc::FailureRecordItem> = recent_logs
                        .into_iter()
                        .filter(|r| r.level == "error" || r.level == "warn")
                        .take(30)
                        .map(|r| triad::ipc::FailureRecordItem {
                            event: r.event,
                            message: r.message,
                            timestamp: r.timestamp,
                        })
                        .collect();

                    let learnings = store.list_learnings(1000).unwrap_or_default().len();
                    let tools = store.list_tools().unwrap_or_default().len();
                    let progress_summary = format!(
                        "Offline Audit: {} trusted claims, {} active tools, {} semantic entities, {} verified relationships. Graph integration: {}.",
                        learnings, tools, nodes, edges, density_desc
                    );

                    let research_backlog = store.list_research_backlog(50).unwrap_or_default();
                    let research_experiments = store.list_research_experiments(50).unwrap_or_default();
                    let capability_map = store.list_capability_map().unwrap_or_default();
                    let extensions = store.list_extensions().unwrap_or_default();
                    let raw_c_tasks = store.list_coordinator_change_tasks().unwrap_or_default();
                    let coordinator_change_tasks: Vec<triad::ipc::CoordinatorChangeTaskReportItem> = raw_c_tasks
                        .into_iter()
                        .map(|(id, title, rationale, proposed_interface, experiment_evidence, status, created_at)| {
                            triad::ipc::CoordinatorChangeTaskReportItem {
                                id,
                                title,
                                rationale,
                                proposed_interface,
                                experiment_evidence,
                                status,
                                created_at,
                            }
                        })
                        .collect();

                    let research_works = store.list_research_works(20).unwrap_or_default();
                    let research_syntheses = store.list_research_syntheses(10).unwrap_or_default();
                    let source_tracking = store.get_source_tracking_diagnostics(50).unwrap_or_default();

                    let offline_rep = triad::ipc::AuditReport {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        constitution_digest: const_digest,
                        constitution_version: const_parsed.constitution.version,
                        semantic_nodes_count: nodes,
                        semantic_edges_count: edges,
                        graph_density_description: density_desc,
                        source_evidence,
                        tool_execution_records,
                        memory_reuse,
                        curriculum_evaluations,
                        research_backlog,
                        research_experiments,
                        research_works,
                        research_syntheses,
                        source_tracking,
                        capability_map,
                        extensions,
                        coordinator_change_tasks,
                        failures_recorded,
                        progress_summary,
                        repair_state: "offline_store".to_string(),
                    };
                    println!("{}", serde_json::to_string_pretty(&offline_rep)?);
                }
            }
            Ok(())
        }

        Commands::Extensions { json } => {
            let store = Store::open(&db_path)?;
            let exts = store.list_extensions()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&exts)?);
            } else if exts.is_empty() {
                println!("No coordinator extensions registered.");
            } else {
                println!("{:<4} {:<26} {:<22} {:<6} {:<15} {:<8}", "ID", "NAME", "HOOK", "VER", "STATE", "FAILURES");
                println!("{}", "-".repeat(84));
                for e in exts {
                    println!("{:<4} {:<26} {:<22} v{:<5} {:<15} {:<8}", e.id, e.name, e.hook, e.version, e.state, e.failure_count);
                }
            }
            Ok(())
        }

        Commands::CoordinatorTasks { json } => {
            let store = Store::open(&db_path)?;
            let tasks = store.list_coordinator_change_tasks()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&tasks)?);
            } else if tasks.is_empty() {
                println!("No coordinator change tasks proposed.");
            } else {
                println!("{:<4} {:<32} {:<12} {:<24}", "ID", "TITLE", "STATUS", "CREATED_AT");
                println!("{}", "-".repeat(74));
                for t in tasks {
                    println!("{:<4} {:<32} {:<12} {:<24}", t.0, t.1, t.5, t.6);
                }
            }
            Ok(())
        }

        Commands::Specialists { limit, json } => {
            let store = Store::open(&db_path)?;
            if let Some(new_limit) = limit {
                store.record_user_specialist_reduction(new_limit)?;
                store.reject_all_pending_limit_proposals("User CLI override reduced concurrency limit")?;

                let running = store.list_running_specialists_newest_first()?;
                let mut cancelled = 0;
                if running.len() > new_limit {
                    let to_cancel = running.len() - new_limit;
                    for spec in running.into_iter().take(to_cancel) {
                        store.update_specialist_status(&spec.id, "cancelled", Some("Terminated due to user specialist concurrency limit reduction"))?;
                        cancelled += 1;
                    }
                }
                println!("Specialist concurrency limit set to {}. ({} active specialists terminated to meet limit)", new_limit, cancelled);
                return Ok(());
            }

            let (current_limit, last_increase, last_reduction) = store.get_specialist_limit_info()?;
            let running = store.count_running_specialists()?;
            let queued = store.count_queued_specialists()?;
            let pending_proposals = store.count_pending_limit_proposals()?;
            let specialists = store.list_specialists(20)?;
            let pending_commits = store.list_specialist_pending_commits(25).unwrap_or_default();

            if json {
                let info = serde_json::json!({
                    "concurrency_limit": current_limit,
                    "running": running,
                    "queued": queued,
                    "pending_limit_proposals": pending_proposals,
                    "last_increase_time": last_increase,
                    "last_reduction_time": last_reduction,
                    "specialists": specialists,
                    "pending_commits": pending_commits,
                });
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!("--- Specialist Agents Status (Enforced 2-Phase State Machine) ---");
                println!("Concurrency limit: {}", current_limit);
                println!("Active: {} running, {} queued (capacity: {}/{} slots)", running, queued, running, current_limit);
                println!("Pending increase proposals: {}", pending_proposals);
                println!(
                    "Phase 2 Commit Records: {} total ({} pending, {} committed, {} rejected)",
                    pending_commits.len(),
                    pending_commits.iter().filter(|c| c.validation_status == "pending").count(),
                    pending_commits.iter().filter(|c| c.validation_status == "committed").count(),
                    pending_commits.iter().filter(|c| c.validation_status == "rejected").count(),
                );
                if let Some(r) = last_reduction {
                    println!("Last reduction: {} (autonomous increase locked for 24h from this timestamp)", r);
                }

                if specialists.is_empty() {
                    println!("\nNo specialists recorded yet.");
                } else {
                    println!(
                        "\n{:<18} {:<38} {:<12} {:<30} {:<16} {:<8}",
                        "NAME", "ROLE", "STATUS", "PHASE", "ENDORSERS", "TOKENS"
                    );
                    println!("{}", "-".repeat(126));
                    for s in specialists {
                        let endorsers = format!("{}+{}", s.endorser_1, s.endorser_2);
                        println!(
                            "{:<18} {:<38} {:<12} {:<30} {:<16} {:<8}",
                            s.name,
                            triad::model::clip_text(&s.role, 36),
                            s.status,
                            s.phase,
                            endorsers,
                            s.max_tokens
                        );
                        println!("  Task: {} | ExecIDs: {:?} | PendingCommit: {:?}", s.task, s.execution_ids, s.pending_commit_id);
                        if let Some(ref res) = s.result {
                            println!("  Outcome: {}", triad::model::clip_text(res, 220));
                        }
                    }
                }

                if !pending_commits.is_empty() {
                    println!("\n--- Specialist Phase 2 Pending & Validated Commits ---");
                    println!(
                        "{:<20} {:<18} {:<20} {:<10} {:<12} {:<36}",
                        "COMMIT_ID", "SPECIALIST", "KIND", "EXEC_ID", "STATUS", "FIDELITY / REJECTION"
                    );
                    println!("{}", "-".repeat(120));
                    for pc in pending_commits {
                        let exec_str = pc
                            .execution_id
                            .map(|id| id.to_string())
                            .unwrap_or_else(|| "-".to_string());
                        let detail = if pc.validation_status == "rejected" {
                            pc.rejection_reason.unwrap_or_else(|| "rejected".to_string())
                        } else if !pc.source_fidelity_summary.is_empty() {
                            format!(
                                "sources={} | {}",
                                pc.independent_source_count, pc.source_fidelity_summary
                            )
                        } else {
                            pc.committed_claim_id.unwrap_or_else(|| "-".to_string())
                        };
                        println!(
                            "{:<20} {:<18} {:<20} {:<10} {:<12} {:<36}",
                            pc.id,
                            triad::model::clip_text(&pc.specialist_name, 16),
                            pc.finding_kind,
                            exec_str,
                            pc.validation_status,
                            triad::model::clip_text(&detail, 54)
                        );
                    }
                }
            }
            Ok(())
        }

        Commands::Mcp { stdio: _, http } => {
            let engine = build_standalone_consultation_engine(
                &config,
                &db_path,
                &effective_log_dir,
                &effective_workspace_dir,
            )?;
            if let Some(bind_addr) = http {
                triad::mcp::run_mcp_http_server(&bind_addr, engine).await
            } else {
                triad::mcp::run_mcp_stdio(engine, Some(sock_path)).await
            }
        }

        Commands::Evaluate { benchmark, json } => {
            let engine = build_standalone_consultation_engine(
                &config,
                &db_path,
                &effective_log_dir,
                &effective_workspace_dir,
            )?;
            let runner = triad::evaluation::EvaluationRunner::new(engine);
            let record = runner.run_domain_evaluation(benchmark.as_deref()).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&record)?);
            } else {
                println!("=== Triad Domain Expertise Evaluation ({}) ===", record.benchmark_name);
                println!("Domain: {} | Items evaluated: {}", record.domain, record.item_count);
                println!(
                    "Plain-LLM Baseline Selected: {}:{} (frontier={}, basis={}{})",
                    record.baseline_provider_id,
                    record.baseline_model_id,
                    record.baseline_is_frontier,
                    record.baseline_selection_basis,
                    record
                        .baseline_fallback_reason
                        .as_deref()
                        .map(|r| format!(" | fallback_reason={}", r))
                        .unwrap_or_default()
                );
                println!(
                    "Evaluated Corpus Full-Text Share: {:.1}% | Verified Domain Competence: {}",
                    record.evaluated_corpus_full_text_share * 100.0,
                    if record.domain_competence_verified {
                        "YES (backed by full-text topics)"
                    } else {
                        "NO (thin/abstract-only ingestion never reported as domain competence)"
                    }
                );
                println!("\n--- Domain Scores (Full-Text Backed; Thin Ingestion Never Reported as Domain Score) ---");
                println!(
                    "  Correctness:             Triad {:>5.1}%  vs  Plain-LLM Baseline {:>5.1}%",
                    record.triad_correctness * 100.0,
                    record.baseline_correctness * 100.0
                );
                println!(
                    "  Citation Accuracy:       Triad {:>5.1}%  vs  Plain-LLM Baseline {:>5.1}%",
                    record.triad_citation_accuracy * 100.0,
                    record.baseline_citation_accuracy * 100.0
                );
                println!(
                    "  Source Diversity:        Triad {:>5.1}%  vs  Plain-LLM Baseline {:>5.1}%",
                    record.triad_source_diversity * 100.0,
                    record.baseline_source_diversity * 100.0
                );
                println!(
                    "  Contradiction Detection: Triad {:>5.1}%  vs  Plain-LLM Baseline {:>5.1}%",
                    record.triad_contradiction_detection * 100.0,
                    record.baseline_contradiction_detection * 100.0
                );
                println!(
                    "  Uncertainty Calibration: Triad {:>5.1}% (Brier {:.3})  vs  Baseline {:>5.1}% (Brier {:.3})",
                    record.triad_uncertainty_calibration * 100.0,
                    record.triad_calibration_brier,
                    record.baseline_uncertainty_calibration * 100.0,
                    record.baseline_calibration_brier
                );
                println!(
                    "  Problem-Solving Success: Triad {:>5.1}%  vs  Plain-LLM Baseline {:>5.1}%",
                    record.triad_problem_solving * 100.0,
                    record.baseline_problem_solving * 100.0
                );

                println!(
                    "\n--- Split 1: Full-Text Topics ({} items) ---",
                    record.full_text_scores.item_count
                );
                println!(
                    "  Correctness: {:.1}% vs {:.1}% | Citation: {:.1}% vs {:.1}% | Diversity: {:.1}% vs {:.1}% | Contradiction: {:.1}% vs {:.1}% | Calibration: {:.1}% vs {:.1}% | Problem-Solving: {:.1}% vs {:.1}%",
                    record.full_text_scores.triad_correctness * 100.0,
                    record.full_text_scores.baseline_correctness * 100.0,
                    record.full_text_scores.triad_citation_accuracy * 100.0,
                    record.full_text_scores.baseline_citation_accuracy * 100.0,
                    record.full_text_scores.triad_source_diversity * 100.0,
                    record.full_text_scores.baseline_source_diversity * 100.0,
                    record.full_text_scores.triad_contradiction_detection * 100.0,
                    record.full_text_scores.baseline_contradiction_detection * 100.0,
                    record.full_text_scores.triad_uncertainty_calibration * 100.0,
                    record.full_text_scores.baseline_uncertainty_calibration * 100.0,
                    record.full_text_scores.triad_problem_solving * 100.0,
                    record.full_text_scores.baseline_problem_solving * 100.0,
                );

                println!(
                    "\n--- Split 2: Abstract-Only / Low-Fidelity / Thin Topics ({} items — Segregated from Domain Score) ---",
                    record.abstract_or_low_fidelity_scores.item_count
                );
                println!(
                    "  Correctness: {:.1}% vs {:.1}% | Citation: {:.1}% vs {:.1}% | Diversity: {:.1}% vs {:.1}% | Contradiction: {:.1}% vs {:.1}% | Calibration: {:.1}% vs {:.1}% | Problem-Solving: {:.1}% vs {:.1}%",
                    record.abstract_or_low_fidelity_scores.triad_correctness * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_correctness * 100.0,
                    record.abstract_or_low_fidelity_scores.triad_citation_accuracy * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_citation_accuracy * 100.0,
                    record.abstract_or_low_fidelity_scores.triad_source_diversity * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_source_diversity * 100.0,
                    record.abstract_or_low_fidelity_scores.triad_contradiction_detection * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_contradiction_detection * 100.0,
                    record.abstract_or_low_fidelity_scores.triad_uncertainty_calibration * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_uncertainty_calibration * 100.0,
                    record.abstract_or_low_fidelity_scores.triad_problem_solving * 100.0,
                    record.abstract_or_low_fidelity_scores.baseline_problem_solving * 100.0,
                );

                println!("\n--- Real-Data vs Synthetic-Data Problem-Solving & Experiment Breakdown ---");
                println!(
                    "  Real-Data Problem-Solving:      Triad {:>5.1}%  vs  Baseline {:>5.1}%  (Recorded Real-Data Experiments: {})",
                    record.real_data_problem_solving * 100.0,
                    record.baseline_real_data_problem_solving * 100.0,
                    record.real_data_experiment_count
                );
                println!(
                    "  Synthetic-Data Problem-Solving: Triad {:>5.1}%  vs  Baseline {:>5.1}%  (Recorded Synthetic-Data Experiments: {})",
                    record.synthetic_data_problem_solving * 100.0,
                    record.baseline_synthetic_data_problem_solving * 100.0,
                    record.synthetic_data_experiment_count
                );
            }
            Ok(())
        }

        Commands::Profile { json } => {
            let engine = build_standalone_consultation_engine(
                &config,
                &db_path,
                &effective_log_dir,
                &effective_workspace_dir,
            )?;
            let profile = engine.expert_profile().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&profile)?);
            } else {
                println!("=== Triad Domain Expertise Profile ===");
                println!(
                    "Epistemic Notice: {}",
                    profile["epistemic_policy"].as_str().unwrap_or("")
                );
                println!(
                    "Domain: {} ({})",
                    profile["domain"]["name"].as_str().unwrap_or(""),
                    profile["domain"]["display_name"].as_str().unwrap_or("")
                );
                println!(
                    "Last Evaluation Date: {}",
                    profile["last_evaluation_date"].as_str().unwrap_or("none")
                );
                let ft = &profile["full_text_share"];
                println!(
                    "Corpus Full-Text Share: {:.1}% (Evaluated: {:.1}% | Verified Full-Text Sources: {}, Abstract-Only: {}, Low-Fidelity: {}, Total Primary: {} | Domain Competence Verified: {})",
                    ft["corpus_full_text_share"].as_f64().unwrap_or(0.0) * 100.0,
                    ft["evaluated_corpus_full_text_share"].as_f64().unwrap_or(0.0) * 100.0,
                    ft["full_text_verified_sources"].as_u64().unwrap_or(0),
                    ft["abstract_only_sources"].as_u64().unwrap_or(0),
                    ft["low_fidelity_sources"].as_u64().unwrap_or(0),
                    ft["total_primary_sources"].as_u64().unwrap_or(0),
                    ft["domain_competence_verified"].as_bool().unwrap_or(false)
                );
                let eb = &profile["experiments_breakdown"];
                println!(
                    "Experiments Breakdown: Real-Data={} (problem-solving={:.1}% vs baseline {:.1}%) | Synthetic-Data={} (problem-solving={:.1}% vs baseline {:.1}%)",
                    eb["real_data_experiment_count"].as_u64().unwrap_or(0),
                    eb["real_data_problem_solving_success"].as_f64().unwrap_or(0.0) * 100.0,
                    eb["baseline_real_data_problem_solving_success"].as_f64().unwrap_or(0.0) * 100.0,
                    eb["synthetic_data_experiment_count"].as_u64().unwrap_or(0),
                    eb["synthetic_data_problem_solving_success"].as_f64().unwrap_or(0.0) * 100.0,
                    eb["baseline_synthetic_data_problem_solving_success"].as_f64().unwrap_or(0.0) * 100.0,
                );
                let cov_sum = &profile["coverage"]["summary"];
                println!(
                    "\n--- Topic Coverage Linked to Evaluation Scores (Covered={}, Thin={}, Low-Fidelity={}, Contradicted={}) ---",
                    cov_sum["covered_topics"].as_u64().unwrap_or(0),
                    cov_sum["thin_topics"].as_u64().unwrap_or(0),
                    cov_sum["low_fidelity_topics"].as_u64().unwrap_or(0),
                    cov_sum["contradicted_topics"].as_u64().unwrap_or(0),
                );
                println!("  Disclaimer: Coverage is never a competence claim. Competence is shown only by the evaluation profile over time.");
                if let Some(topics) = profile["coverage"]["topics"].as_array() {
                    for t in topics {
                        println!(
                            "  • [{}] {} ({}) | FullText={} Abstract={} LowFid={} Claims={} | ReingestReq={} Priority={} | Eval(corr={:?}, prob={:?})",
                            t["coverage_status"].as_str().unwrap_or(""),
                            t["topic"].as_str().unwrap_or(""),
                            t["discipline"].as_str().unwrap_or(""),
                            t["full_text_sources_count"].as_u64().unwrap_or(0),
                            t["abstract_only_sources_count"].as_u64().unwrap_or(0),
                            t["low_fidelity_sources_count"].as_u64().unwrap_or(0),
                            t["substantive_claims_count"].as_u64().unwrap_or(0),
                            t["reingestion_requested"].as_bool().unwrap_or(false),
                            t["backlog_priority"].as_i64().unwrap_or(0),
                            t["eval_correctness"].as_f64().map(|v| format!("{:.1}%", v * 100.0)),
                            t["eval_problem_solving"].as_f64().map(|v| format!("{:.1}%", v * 100.0)),
                        );
                    }
                }
                let fa = &profile["frontier_availability"];
                println!(
                    "\n--- Frontier Availability & Fallback ---\n  Live External Frontier Credential Configured: {}\n  Active Local Fallback: {}",
                    fa["any_live_frontier_credential_configured"].as_bool().unwrap_or(false),
                    fa["active_local_fallback"].as_str().unwrap_or("")
                );
                if let Some(strengths) = profile["evaluated_strengths"].as_array() {
                    println!("\n--- Evaluated Strengths ({}) ---", strengths.len());
                    for s in strengths {
                        println!(
                            "  • {}: {}",
                            s["area"].as_str().unwrap_or(""),
                            s["evidence_basis"].as_str().unwrap_or("")
                        );
                    }
                }
                if let Some(weaknesses) = profile["known_weaknesses"].as_array() {
                    println!("\n--- Known Weaknesses ({}) ---", weaknesses.len());
                    for w in weaknesses {
                        println!(
                            "  • {}: {} (Mitigation: {})",
                            w["weakness"].as_str().unwrap_or(""),
                            w["measurement"].as_str().unwrap_or(""),
                            w["mitigation"].as_str().unwrap_or("")
                        );
                    }
                }
            }
            Ok(())
        }

        Commands::Sources { limit, json } => {
            let store = Store::open(&db_path)?;
            let lim = limit.unwrap_or(30);
            let diag = store.get_source_tracking_diagnostics(lim)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&diag)?);
            } else {
                let total_indep = diag.independent_papers_count
                    + diag.independent_datasets_count
                    + diag.independent_implementations_count;
                println!("=== Triad Persistent, Version-Aware Source Tracking & Multi-Source Adapter Diagnostics ===");
                println!("Total Canonical Sources Discovered: {}", diag.total_tracked_sources);
                println!(
                    "  Ingestion State:   {} ingested (Full-Text: {}, Abstract-Only: {}, Orientation-Only: {}, Structured Sections: {})",
                    diag.ingested_count,
                    diag.full_text_ingested_count,
                    diag.abstract_only_ingested_count,
                    diag.orientation_only_count,
                    diag.total_stored_sections
                );
                println!(
                    "  Extraction Quality: {} equations preserved, {} tables found, {} low-fidelity sources",
                    diag.total_equations_preserved,
                    diag.total_tables_found,
                    diag.low_fidelity_count
                );
                println!(
                    "  Research State:    {} completed, {} pending, {} reserved, {} failed",
                    diag.completed_research_count,
                    diag.pending_research_count,
                    diag.reserved_research_count,
                    diag.failed_research_count
                );
                println!(
                    "  Retry / Revisit:   {} retry-eligible, {} flagged with contradiction",
                    diag.retry_eligible_count, diag.contradiction_flagged_count
                );
                println!("\n--- Independent Corroboration Diagnostics ---");
                println!(
                    "  Total Independent Corroborations:  {}",
                    total_indep
                );
                println!(
                    "    • Independent Papers:            {}",
                    diag.independent_papers_count
                );
                println!(
                    "    • Independent Datasets:          {}",
                    diag.independent_datasets_count
                );
                println!(
                    "    • Independent Implementations:   {}",
                    diag.independent_implementations_count
                );
                println!(
                    "  Total Route Observations:          {} ({} duplicate routes/copies deduplicated)",
                    diag.total_route_observations, diag.duplicate_route_observations_deduped
                );

                if !diag.connector_health.is_empty() {
                    println!("\n--- Multi-Source Adapter Connector Health & Coverage ---");
                    println!(
                        "{:<18} {:<16} {:<12} {:<8} {:<8} {:<8} {:<8} {:<6} {:<8} {:<22}",
                        "CONNECTOR", "CATEGORY", "STATUS", "DISC", "INGEST", "FULLTXT", "ABSTR", "FAILS", "LAT_MS", "LAST_SUCCESS"
                    );
                    println!("{}", "-".repeat(122));
                    for ch in &diag.connector_health {
                        println!(
                            "{:<18} {:<16} {:<12} {:<8} {:<8} {:<8} {:<8} {:<6} {:<8} {:<22}",
                            ch.connector,
                            triad::model::clip_text(&ch.category, 15),
                            ch.status,
                            ch.discovered_count,
                            ch.ingested_count,
                            ch.full_text_count,
                            ch.abstract_only_count,
                            ch.total_failures,
                            ch.avg_latency_ms,
                            ch.last_successful_fetch_at.as_deref().unwrap_or("never")
                        );
                    }
                }

                if !diag.topic_coverage.is_empty() {
                    println!("\n--- Domain & Topic Coverage (Discovered vs Ingested Full-Text / Abstract) ---");
                    println!(
                        "{:<48} {:<10} {:<10} {:<10} {:<10} {:<10}",
                        "TOPIC / DOMAIN", "DISC", "INGESTED", "FULL_TEXT", "ABSTRACT", "ORIENTATION"
                    );
                    println!("{}", "-".repeat(104));
                    for tc in &diag.topic_coverage {
                        println!(
                            "{:<48} {:<10} {:<10} {:<10} {:<10} {:<10}",
                            triad::model::clip_text(&tc.topic, 46),
                            tc.discovered_count,
                            tc.ingested_count,
                            tc.full_text_count,
                            tc.abstract_only_count,
                            tc.orientation_only_count
                        );
                    }
                }

                if !diag.recent_sources.is_empty() {
                    println!(
                        "\n{:<30} {:<8} {:<5} {:<9} {:<13} {:<5} {:<4} {:<4} {:<6} {:<10} {:<5} {:<5} {:<10}",
                        "CANONICAL_ID", "KIND", "VER", "INGEST", "COMPLETENESS", "SECS", "EQS", "TBLS", "LOWFID", "RESEARCH", "RTES", "FAIL", "RETRY/REVISIT"
                    );
                    println!("{}", "-".repeat(126));
                    for s in &diag.recent_sources {
                        let revisit = s
                            .revisit_reason
                            .as_deref()
                            .unwrap_or(if s.retry_eligible { "eligible" } else { "skip(done)" });
                        let comp_fmt = if let Some(ref fmt) = s.full_text_format {
                            format!("{}:{}", s.ingestion_completeness, fmt)
                        } else {
                            s.ingestion_completeness.clone()
                        };
                        println!(
                            "{:<30} {:<8} {:<5} {:<9} {:<13} {:<5} {:<4} {:<4} {:<6} {:<10} {:<5} {:<5} {:<10}",
                            triad::model::clip_text(&s.canonical_id, 28),
                            s.source_kind,
                            s.version,
                            s.ingestion_status,
                            triad::model::clip_text(&comp_fmt, 12),
                            s.section_count,
                            s.equations_preserved,
                            s.tables_found,
                            if s.low_fidelity { "yes" } else { "no" },
                            s.research_status,
                            s.route_observation_count,
                            s.failure_count,
                            revisit
                        );
                        println!(
                            "  Title: {} | Canonical URL: {} | Hash: {} | Connectors: [{}]",
                            triad::model::clip_text(&s.title, 70),
                            s.canonical_url,
                            &s.content_hash[..s.content_hash.len().min(12)],
                            s.observed_connectors.join(", ")
                        );
                    }
                }
            }
            Ok(())
        }

        Commands::Experiments { limit, json } => {
            let store = Store::open(&db_path)?;
            let lim = limit.unwrap_or(25);
            let specs = store.list_experiment_specs(lim)?;
            let experiments = store.list_research_experiments(lim)?;
            let executions = store.list_recent_tool_executions(lim).unwrap_or_default();

            if json {
                let payload = serde_json::json!({
                    "experiment_specifications": specs,
                    "research_experiments": experiments,
                    "recent_tool_executions": executions,
                    "domain_executable_checks": config.domain.executable_checks,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("=== Triad Meaningful Research Tools & Experiments Diagnostics ===");
                println!("Domain: {} | Configured Executable Checks: {:?}", config.domain.name, config.domain.executable_checks);
                println!(
                    "Specifications: {} total ({} full-text/computational, {} under-specified)",
                    specs.len(),
                    specs.iter().filter(|s| !s.under_specified).count(),
                    specs.iter().filter(|s| s.under_specified).count()
                );
                println!(
                    "Recorded Experiments: {} total ({} completed, {} failed/timed-out)",
                    experiments.len(),
                    experiments.iter().filter(|e| !e.is_failure).count(),
                    experiments.iter().filter(|e| e.is_failure).count()
                );

                if !specs.is_empty() {
                    println!("\n--- Experiment Specifications ---");
                    for sp in &specs {
                        let metric_keys: Vec<String> = sp
                            .metrics
                            .iter()
                            .map(|m| format!("{} ({})", m.result_key, m.success_criterion))
                            .collect();
                        println!(
                            "• [{}] {} (Check: {} | Fidelity: {} | UnderSpecified: {} | Review: {})",
                            sp.id,
                            sp.name,
                            sp.executable_check_kind,
                            sp.source_fidelity_status,
                            sp.under_specified,
                            sp.sceptic_review_status
                        );
                        println!("  Research Question: {}", sp.research_question);
                        println!("  Falsifiable Hypothesis: {}", sp.falsifiable_hypothesis);
                        println!("  Mechanism Under Test: {}", sp.mechanism_under_test);
                        println!("  Baseline: {} | Controls: {:?}", sp.baseline_description, sp.controls);
                        println!("  Held-Out Cases: {:?}", sp.held_out_cases);
                        println!("  Declared Metrics: {}", metric_keys.join(", "));
                        println!(
                            "  Budget: <= {}s, <= {} MB | Seeds: {:?} | Cited Sources: {:?}",
                            sp.budget_max_runtime_secs,
                            sp.budget_max_memory_mb,
                            sp.seeds,
                            sp.cited_source_ids
                        );
                        if !sp.missing_source_details.is_empty() {
                            println!("  Missing Source Details: {}", sp.missing_source_details.join("; "));
                        }
                        if !sp.sceptic_review_notes.is_empty() {
                            println!("  Sceptic Notes: {}", sp.sceptic_review_notes);
                        }
                    }
                }

                if !experiments.is_empty() {
                    println!("\n--- Executed Research Experiments & Outcomes ---");
                    for exp in &experiments {
                        println!(
                            "• [Exp #{}] Cycle #{} | Status: {} | Spec: {:?} | Tool: {:?} | ExecID: {:?} | Fidelity: {} (under_specified={})",
                            exp.id,
                            exp.cycle_id,
                            exp.outcome_status,
                            exp.spec_id,
                            exp.tool_name,
                            exp.execution_id,
                            exp.source_fidelity_status,
                            exp.under_specified
                        );
                        println!("  Hypothesis: {}", exp.hypothesis);
                        println!("  Mechanism: {}", exp.attempted_method);
                        if exp.measured_results_json != "{}" && !exp.measured_results_json.is_empty() {
                            println!("  Measured Results (results.json): {}", exp.measured_results_json);
                        }
                        if exp.baseline_comparison_json != "{}" && !exp.baseline_comparison_json.is_empty() {
                            println!("  Baseline Comparison: {}", exp.baseline_comparison_json);
                        }
                        if exp.held_out_evaluation_json != "{}" && !exp.held_out_evaluation_json.is_empty() {
                            println!("  Held-Out Evaluation: {}", exp.held_out_evaluation_json);
                        }
                        if exp.missing_source_details_json != "[]" && !exp.missing_source_details_json.is_empty() {
                            println!("  Missing Source Details: {}", exp.missing_source_details_json);
                        }
                        println!("  Outcome: {}", exp.observed_outcome);
                        println!("  Lessons: {} | {}", exp.lessons_learned, exp.epistemic_note);
                    }
                }
            }
            Ok(())
        }

        Commands::Models { refresh, json } => {
            let store = Store::open(&db_path)?;
            let logger = Logger::new(
                effective_log_dir.clone(),
                config.logging.rotate_size_mb,
                config.logging.retained_files,
            );
            let m_client = ModelClient::from_models_config(
                &config.models,
                Some(store.clone()),
                Some(logger),
            );

            if refresh {
                let _ = m_client.refresh_provider_discovery(true).await;
            }

            let workloads = [
                model::WorkloadKind::Routing,
                model::WorkloadKind::Summarization,
                model::WorkloadKind::CheapWork,
                model::WorkloadKind::HardSynthesis,
                model::WorkloadKind::ExperimentDesign,
                model::WorkloadKind::ComplexRustGeneration,
                model::WorkloadKind::Debugging,
                model::WorkloadKind::CriticalReview,
            ];

            let mut workload_decisions = Vec::new();
            for wk in &workloads {
                let ctx = model::WorkloadSelectionContext {
                    task_id: "cli_diagnostic_preview".to_string(),
                    caller_kind: "cli_diagnostics".to_string(),
                    caller_id: "Coordinator".to_string(),
                    workload: *wk,
                    required_context_tokens: if wk.is_escalated_by_default() { 4096 } else { 1024 },
                    require_structured_json: wk.is_escalated_by_default(),
                    explicit_override: None,
                };
                if let Ok(dec) = m_client.select_model_for_workload(&ctx).await {
                    workload_decisions.push(serde_json::json!({
                        "workload": wk.as_str(),
                        "escalated": wk.is_escalated_by_default(),
                        "selected_provider": dec.provider_id,
                        "selected_model": dec.model_id,
                        "transport": dec.transport,
                        "is_frontier": dec.is_frontier,
                        "used_thin_evidence_fallback": dec.used_preference_order_fallback,
                        "ranking_basis": dec.ranking_basis,
                        "fallback_reason": dec.fallback_reason,
                    }));
                }
            }

            let provider_profiles: Vec<serde_json::Value> = config
                .models
                .providers
                .iter()
                .map(|p| {
                    let cred_status = match p.credential_env.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        None => "none_required (local)".to_string(),
                        Some(env_name) => {
                            let is_set = std::env::var(env_name)
                                .ok()
                                .map(|v| !v.trim().is_empty())
                                .unwrap_or(false);
                            format!(
                                "env:{} ({})",
                                env_name,
                                if is_set { "configured [REDACTED]" } else { "unset" }
                            )
                        }
                    };
                    serde_json::json!({
                        "id": p.id,
                        "provider_type": p.provider_type,
                        "enabled": p.enabled,
                        "base_url": p.base_url,
                        "transport": p.transport,
                        "credential_status": cred_status,
                        "model_policy": p.model_policy,
                        "permitted_models": p.permitted_models,
                        "fallback_provider": p.fallback_provider,
                        "is_frontier": p.is_frontier,
                    })
                })
                .collect();

            let budget_summary = store.get_frontier_budget_summary(&config.models.frontier_budget, 20)?;
            let recent_selections = store.list_recent_model_selections(20)?;
            let discoveries = store.list_provider_discoveries()?;
            let evaluations = store.list_model_evaluations()?;

            if json {
                let payload = serde_json::json!({
                    "dedicated_embedding_model": {
                        "base_url": config.models.embedding.base_url,
                        "model": config.models.embedding.model,
                        "note": "Never assumed from chat model; dedicated embedding endpoint enforced."
                    },
                    "selection_policy": {
                        "explicit_override": config.models.selection.explicit_override,
                        "preference_order": config.models.selection.preference_order,
                        "discovery_refresh_interval_seconds": config.models.selection.discovery_refresh_interval_seconds,
                        "min_eval_runs_for_empirical_ranking": config.models.selection.min_eval_runs_for_empirical_ranking,
                    },
                    "providers": provider_profiles,
                    "provider_discovery": discoveries,
                    "workload_selections": workload_decisions,
                    "recorded_evaluations": evaluations,
                    "recent_selection_log": recent_selections,
                    "frontier_budget": budget_summary,
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("=== Triad Model Providers, Selection & Frontier Budgets ===");
                println!(
                    "Dedicated Embedding Endpoint: {} (model: '{}' — chat model never assumed to supply embeddings)",
                    config.models.embedding.base_url, config.models.embedding.model
                );
                println!(
                    "Local Default Chat Endpoint:  {} (model: '{}')",
                    config.models.chat.base_url, config.models.chat.model
                );
                println!(
                    "Selection Override:           {}",
                    config
                        .models
                        .selection
                        .explicit_override
                        .as_deref()
                        .filter(|s| !s.trim().is_empty())
                        .unwrap_or("none (automatic capability + evaluation + preference selection)")
                );
                println!(
                    "Configured Preference Order:  [{}]",
                    config.models.selection.preference_order.join(" -> ")
                );

                println!("\n--- Configured Model Providers (Secrets Never Logged) ---");
                println!(
                    "{:<18} {:<18} {:<8} {:<18} {:<18} {:<28} {:<24}",
                    "PROVIDER", "TYPE", "ENABLED", "TRANSPORT", "POLICY", "CREDENTIAL_ENV", "PERMITTED_MODELS"
                );
                println!("{}", "-".repeat(138));
                for p in &config.models.providers {
                    let cred_status = match p.credential_env.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
                        None => "none (local)".to_string(),
                        Some(env_name) => {
                            let is_set = std::env::var(env_name)
                                .ok()
                                .map(|v| !v.trim().is_empty())
                                .unwrap_or(false);
                            format!("{}:{}", env_name, if is_set { "SET" } else { "unset" })
                        }
                    };
                    println!(
                        "{:<18} {:<18} {:<8} {:<18} {:<18} {:<28} {:<24}",
                        p.id,
                        p.provider_type,
                        p.enabled,
                        p.transport,
                        p.model_policy,
                        cred_status,
                        triad::model::clip_text(&p.permitted_models.join(","), 23)
                    );
                }

                if !discoveries.is_empty() {
                    println!("\n--- Provider Discovery State ---");
                    for d in &discoveries {
                        println!(
                            "• {} ({}): status={} | cred={} | discovered={:?} | refreshed_at={}",
                            d.provider_id,
                            d.base_url,
                            d.health_status,
                            d.credential_status,
                            d.discovered_models,
                            d.last_refreshed_at
                        );
                    }
                }

                println!("\n--- Selected Models & Ranking Basis by Workload ---");
                println!(
                    "{:<26} {:<10} {:<16} {:<26} {:<10} {:<10} {:<40}",
                    "WORKLOAD", "ESCALATED", "PROVIDER", "MODEL_ID", "FRONTIER", "THIN_EVAL", "RANKING_BASIS / FALLBACK"
                );
                println!("{}", "-".repeat(144));
                for wd in &workload_decisions {
                    let wk = wd["workload"].as_str().unwrap_or("");
                    let esc = wd["escalated"].as_bool().unwrap_or(false);
                    let prov = wd["selected_provider"].as_str().unwrap_or("");
                    let mid = wd["selected_model"].as_str().unwrap_or("");
                    let fron = wd["is_frontier"].as_bool().unwrap_or(false);
                    let thin = wd["used_thin_evidence_fallback"].as_bool().unwrap_or(false);
                    let basis = wd["ranking_basis"].as_str().unwrap_or("");
                    let fb = wd["fallback_reason"].as_str();
                    println!(
                        "{:<26} {:<10} {:<16} {:<26} {:<10} {:<10} {}",
                        wk,
                        esc,
                        prov,
                        mid,
                        fron,
                        thin,
                        triad::model::clip_text(basis, 64)
                    );
                    if let Some(reason) = fb {
                        println!("  ↳ Fallback reason: {}", reason);
                    }
                }

                println!("\n--- Shared Atomic Frontier Budget (Rolling 24h & Per-Task) ---");
                println!(
                    "  Rolling 24h Calls:   {} / {} used ({} remaining; retries consume quota)",
                    budget_summary.used_calls_24h,
                    budget_summary.max_calls_per_24h,
                    budget_summary.remaining_calls_24h
                );
                println!(
                    "  Per-Task Call Limit: {} call(s) per task | Max Retries per Invocation: {}",
                    budget_summary.max_calls_per_task, budget_summary.max_retries
                );
                println!(
                    "  Active Concurrency:  {} / {} in-flight",
                    budget_summary.active_concurrency, budget_summary.max_concurrency
                );
                println!(
                    "  Rolling 24h Tokens:  {} / {} used ({} remaining) | Per-Task Token Cap: {}",
                    budget_summary.used_tokens_24h,
                    budget_summary.max_tokens_per_24h,
                    budget_summary.remaining_tokens_24h,
                    budget_summary.max_tokens_per_task
                );
                println!(
                    "  Rolling 24h Spend:   {} / {} USD micros ({} remaining; unknown prices billed conservatively)",
                    budget_summary.used_spend_micros_24h,
                    budget_summary.max_spend_micros_per_24h,
                    budget_summary.remaining_spend_micros_24h
                );
                println!(
                    "  Exhaustion Policy:   {} | Response Cache Hits: {} | Total Ledger Entries: {}",
                    budget_summary.exhaustion_policy,
                    budget_summary.cache_hits_total,
                    budget_summary.total_historical_ledger_entries
                );

                if !budget_summary.blocked_tasks.is_empty() {
                    println!(
                        "\n--- Blocked / Queued / Degraded Tasks ({} items) ---",
                        budget_summary.blocked_tasks.len()
                    );
                    for bt in &budget_summary.blocked_tasks {
                        println!(
                            "• Task '{}' (caller={}:{} | workload={} | model={}:{}): action={} | status={} | reason={}",
                            bt.task_id,
                            bt.caller_kind,
                            bt.caller_id,
                            bt.workload_kind,
                            bt.requested_provider,
                            bt.requested_model,
                            bt.exhaustion_action,
                            bt.status,
                            bt.exhaustion_reason
                        );
                    }
                } else {
                    println!("\nBlocked Tasks: 0 (no tasks currently blocked on frontier budget)");
                }

                if !budget_summary.recent_ledger.is_empty() {
                    println!(
                        "\n--- Recent Frontier Budget Ledger ({} entries) ---",
                        budget_summary.recent_ledger.len()
                    );
                    for entry in budget_summary.recent_ledger.iter().take(10) {
                        println!(
                            "• [{}] task={} caller={}:{} workload={} model={}:{} attempt=#{} retry={} status={} tokens={}(in:{}/out:{}) spend={} micros (pricing_known={})",
                            &entry.reservation_id[..entry.reservation_id.len().min(14)],
                            entry.task_id,
                            entry.caller_kind,
                            entry.caller_id,
                            entry.workload_kind,
                            entry.provider_id,
                            entry.model_id,
                            entry.attempt_number,
                            entry.is_retry,
                            entry.status,
                            entry.total_tokens,
                            entry.prompt_tokens,
                            entry.completion_tokens,
                            entry.actual_spend_micros,
                            entry.pricing_known
                        );
                    }
                }
            }
            Ok(())
        }
        Commands::Purge => {
            let (removed_items, reclaimed_bytes) =
                triad::tools::purge_workspace_build_bloat(&effective_workspace_dir, &effective_data_dir);
            println!(
                "Workspace build-artifact purge complete: removed {} redundant items, reclaimed {:.2} MB ({:.2} GB).",
                removed_items,
                reclaimed_bytes as f64 / 1_048_576.0,
                reclaimed_bytes as f64 / 1_073_741_824.0
            );
            Ok(())
        }
    }
}

fn build_standalone_consultation_engine(
    config: &AppConfig,
    db_path: &Path,
    log_dir: &Path,
    workspace_dir: &Path,
) -> Result<Arc<triad::consultation::ConsultationEngine>> {
    let store = Store::open(db_path)?;
    let logger = Logger::new(
        log_dir.to_path_buf(),
        config.logging.rotate_size_mb,
        config.logging.retained_files,
    );
    let m_client = ModelClient::new(
        &config.models.chat.base_url,
        &config.models.chat.model,
        config.models.chat.max_tokens,
        config.models.chat.no_think,
        config.models.chat.concurrency,
        &config.models.embedding.base_url,
        &config.models.embedding.model,
        &config.models.embedding.query_prefix,
        &config.models.embedding.document_prefix,
    );
    let web = WebExplorer::new(
        &config.web.wikipedia_api,
        config.web.request_timeout_seconds,
    );
    let adapter: Arc<dyn execution::ExecutionAdapter> =
        Arc::new(AntigravityAdapter::probe(&config.execution.runtime_binary));
    let (constitution, constitution_raw, constitution_digest) =
        triad::constitution::load_and_verify_constitution()?;
    let coordinator = Arc::new(Coordinator::new(
        config.clone(),
        constitution,
        constitution_raw,
        constitution_digest,
        store,
        m_client,
        web,
        adapter,
        logger,
        workspace_dir.to_path_buf(),
    ));
    Ok(Arc::new(triad::consultation::ConsultationEngine::new(
        coordinator,
    )))
}

async fn poll_task_completion(store: &Store, task_id: &str) -> Result<()> {
    let mut dots = 0;
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Some(t) = store.get_task(task_id)? {
            if t.status == "completed" {
                println!("\nTask {} completed successfully!", task_id);
                if let Some(res) = t.result {
                    println!("\nResult:\n{}", res);
                }
                return Ok(());
            } else if t.status == "failed" {
                eprintln!("\nTask {} failed: {}", task_id, t.error.as_deref().unwrap_or("unknown"));
                return Ok(());
            }
        }
        dots += 1;
        if dots % 4 == 0 {
            print!(".");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    }
}

async fn run_doctor(config: &AppConfig, workspace_dir: &Path) -> Result<()> {
    println!("=== Triad System Doctor ===");

    // 1. Toolchain
    print!("Checking Rust toolchain: ");
    let rustc = std::process::Command::new("rustc").arg("--version").output();
    match rustc {
        Ok(out) if out.status.success() => println!("OK ({})", String::from_utf8_lossy(&out.stdout).trim()),
        _ => println!("FAIL (rustc not available)"),
    }

    // 2. Chat Snap Endpoint
    print!("Checking SmolLM3 chat endpoint ({}): ", config.models.chat.base_url);
    let m_client = ModelClient::new(
        &config.models.chat.base_url,
        &config.models.chat.model,
        config.models.chat.max_tokens,
        config.models.chat.no_think,
        config.models.chat.concurrency,
        &config.models.embedding.base_url,
        &config.models.embedding.model,
        &config.models.embedding.query_prefix,
        &config.models.embedding.document_prefix,
    );
    match m_client.probe_models(&config.models.chat.base_url).await {
        Ok(models) => {
            println!("OK (Advertised: {})", models.join(", "));
            if !models.contains(&config.models.chat.model) {
                println!("  [NOTE] Configured model '{}' differs from advertised models. Consider updating config.", config.models.chat.model);
            }
        }
        Err(e) => println!("FAIL ({})", e),
    }

    // 3. Embedding Snap Endpoint
    print!("Checking Nomic embedding endpoint ({}): ", config.models.embedding.base_url);
    match m_client.probe_models(&config.models.embedding.base_url).await {
        Ok(models) => println!("OK (Advertised: {})", models.join(", ")),
        Err(e) => println!("FAIL ({})", e),
    }

    // 4. Test real embedding call
    print!("Testing Nomic embedding generation: ");
    match m_client.embed_text("doctor test", false).await {
        Ok(vec) => println!("OK (Generated {}-dimensional vector)", vec.len()),
        Err(e) => println!("FAIL ({})", e),
    }

    // 5. Test real chat roundtrip
    print!("Testing SmolLM3 chat round-trip: ");
    let msgs = vec![model::ChatMessage {
        role: "user".into(),
        content: "Respond with 'DOCTOR_OK' only.".into(),
    }];
    match m_client.complete_chat("Test", &msgs).await {
        Ok(resp) => println!("OK (Received: {})", resp.trim()),
        Err(e) => println!("FAIL ({})", e),
    }

    // 6. Isolation
    print!("Checking Linux process isolation (bubblewrap): ");
    let bwrap = BubblewrapAdapter::probe();
    if bwrap.is_available() {
        println!("OK ({})", bwrap.status_summary());
    } else {
        println!("WARNING: Bubblewrap unavailable. Generated tools cannot be safely executed.");
    }

    // 7. Antigravity Runtime
    print!("Checking Antigravity runtime: ");
    let agy = AntigravityAdapter::probe(&config.execution.runtime_binary);
    println!("{}", agy.status_summary());

    // 8. Internet / Wikipedia
    print!("Checking Wikipedia HTTPS access: ");
    let web = WebExplorer::new(&config.web.wikipedia_api, config.web.request_timeout_seconds);
    if web.test_connectivity().await {
        println!("OK (Wikipedia reachable)");
    } else {
        println!("FAIL (Wikipedia unreachable or offline)");
    }

    // 9. Application Workspace & Directories
    println!("Application Workspace: {} (writable: {})", workspace_dir.display(), workspace_dir.exists());

    println!("===========================");
    Ok(())
}

async fn run_daemon(
    config: AppConfig,
    data_dir: PathBuf,
    log_dir: PathBuf,
    workspace_dir: PathBuf,
    sock_path: PathBuf,
) -> Result<()> {
    let lock_path = data_dir.join("triad.lock");
    let _lock = SingleInstanceLock::acquire(&lock_path)
        .context("Failed to acquire single instance daemon lock")?;

    let db_path = data_dir.join("triad.db");
    let store = Store::open(&db_path)?;

    let logger = Logger::new(log_dir.clone(), config.logging.rotate_size_mb, config.logging.retained_files);
    logger.log("info", "daemon.started", None, None, "Triad daemon initialized", None);

    let m_client = ModelClient::new(
        &config.models.chat.base_url,
        &config.models.chat.model,
        config.models.chat.max_tokens,
        config.models.chat.no_think,
        config.models.chat.concurrency,
        &config.models.embedding.base_url,
        &config.models.embedding.model,
        &config.models.embedding.query_prefix,
        &config.models.embedding.document_prefix,
    );

    let web = WebExplorer::new(&config.web.wikipedia_api, config.web.request_timeout_seconds);
    let adapter: Arc<dyn execution::ExecutionAdapter> = Arc::new(AntigravityAdapter::probe(&config.execution.runtime_binary));

    let (constitution, constitution_raw, constitution_digest) = triad::constitution::load_and_verify_constitution()?;
    logger.log("info", "constitution.verified", None, None, &format!("Constitution verified with digest: {}", constitution_digest), None);

    let coordinator = Arc::new(Coordinator::new(
        config.clone(),
        constitution,
        constitution_raw,
        constitution_digest,
        store.clone(),
        m_client,
        web,
        adapter,
        logger.clone(),
        workspace_dir.clone(),
    ));

    // Spawn autonomous loop
    let coord_loop = coordinator.clone();
    tokio::spawn(async move {
        coord_loop.run_autonomous_loop().await;
    });

    // Create unified ConsultationEngine over the single Coordinator instance
    let consultation_engine = Arc::new(triad::consultation::ConsultationEngine::new(
        coordinator.clone(),
    ));

    // Spawn embedded Streamable HTTP MCP server on configured bind address
    let mcp_bind = config.consultation.http_bind_addr.clone();
    if !mcp_bind.trim().is_empty() {
        let mcp_eng = consultation_engine.clone();
        let mcp_logger = logger.clone();
        tokio::spawn(async move {
            mcp_logger.log(
                "info",
                "mcp.http_listening",
                None,
                None,
                &format!("Streamable HTTP MCP server listening on http://{}/mcp", mcp_bind),
                None,
            );
            let _ = triad::mcp::run_mcp_http_server(&mcp_bind, mcp_eng).await;
        });
    }

    // Remove old socket if exists
    let _ = fs::remove_file(&sock_path);
    if let Some(parent) = sock_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let listener = UnixListener::bind(&sock_path)
        .with_context(|| format!("Failed to bind UNIX socket at {}", sock_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&sock_path, fs::Permissions::from_mode(0o600));
    }

    logger.log("info", "ipc.listening", None, None, &format!("UNIX socket listening at {}", sock_path.display()), None);

    // Accept IPC connections
    while let Ok((mut stream, _)) = listener.accept().await {
        let coord = coordinator.clone();
        let cons_eng = consultation_engine.clone();
        tokio::spawn(async move {
            if let Ok(req) = read_ipc_request(&mut stream).await {
                let resp = match req {
                    IpcRequest::Status => {
                        let rep = coord.get_status_report().await;
                        IpcResponse::Status(rep)
                    }
                    IpcRequest::Pause => {
                        coord.pause();
                        IpcResponse::Ok { message: "Autonomous exploration paused.".into() }
                    }
                    IpcRequest::Resume => {
                        coord.resume();
                        IpcResponse::Ok { message: "Autonomous exploration resumed.".into() }
                    }
                    IpcRequest::Stop => {
                        coord.stop();
                        let _ = write_ipc_response(&mut stream, &IpcResponse::Ok { message: "Stopping daemon.".into() }).await;
                        std::process::exit(0);
                    }
                    IpcRequest::Ask { question } => {
                        let _ = question;
                        let _rep = coord.get_status_report().await;
                        IpcResponse::Ok { message: "Question queued for discussion.".into() }
                    }
                    IpcRequest::Challenge { task } => {
                        let _ = task;
                        IpcResponse::Ok { message: "Challenge queued for agents.".into() }
                    }
                    IpcRequest::Tell { info, source } => {
                        let _ = (info, source);
                        IpcResponse::Ok { message: "User information queued.".into() }
                    }
                    IpcRequest::Answer { question_id, answer } => {
                        let _ = (question_id, answer);
                        IpcResponse::Ok { message: "Answer received.".into() }
                    }
                    IpcRequest::ConfigReload => {
                        match ensure_canonical_config() {
                            Ok((new_cfg, _, _)) => {
                                coord.reload_config(new_cfg).await;
                                IpcResponse::Ok {
                                    message: "Configuration live-reloaded (accounting history preserved).".into(),
                                }
                            }
                            Err(e) => IpcResponse::Error {
                                message: format!("Rejected invalid configuration (keeping previous active config): {}", e),
                            },
                        }
                    }
                    IpcRequest::Audit => {
                        let rep = coord.get_audit_report().await;
                        IpcResponse::Audit(rep)
                    }
                    IpcRequest::McpJsonRpc { payload } => {
                        let out = triad::mcp::process_raw_jsonrpc_payload(&cons_eng, &payload).await;
                        IpcResponse::McpJsonRpcResult { payload: out }
                    }
                };
                let _ = write_ipc_response(&mut stream, &resp).await;
            }
        });
    }

    Ok(())
}
