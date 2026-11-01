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

            match send_ipc_request(&IpcRequest::Status, &sock_path).await {
                Ok(IpcResponse::Status(rep)) => {
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
                    println!("Status: {} (Triad agreement policy)", claim.status);
                    println!("Source: {}", claim.source_url);
                    println!("Proposer: {}", claim.proposer);
                    println!("Version: {}", claim.version);
                    println!("Created: {}", claim.created_at);
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
            let research_works = store.list_research_works(limit).unwrap_or_default();
            let research_syntheses = store.list_research_syntheses(limit).unwrap_or_default();
            let specialists = store.list_specialists(limit).unwrap_or_default();
            if json {
                #[derive(serde::Serialize)]
                struct LearningView<'a> {
                    learnings: &'a [triad::store::Claim],
                    experiments: &'a [triad::store::ResearchExperimentRecord],
                    research_works: &'a [triad::web::ResolvedResearchWork],
                    research_syntheses: &'a [triad::store::ResearchSynthesisRecord],
                    specialists: &'a [triad::specialists::SpecialistRecord],
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&LearningView {
                        learnings: &learnings,
                        experiments: &experiments,
                        research_works: &research_works,
                        research_syntheses: &research_syntheses,
                        specialists: &specialists,
                    })?
                );
            } else {
                println!("--- Recent Trusted Learnings ({} items) ---", learnings.len());
                for l in learnings {
                    println!("[{}] {} (Source: {}, Proposer: {})", &l.id[..8], l.statement, l.source_url, l.proposer);
                }
                if !specialists.is_empty() {
                    println!("\n--- Temporary Specialist Agents ({} items) ---", specialists.len());
                    for s in &specialists {
                        println!(
                            "• [{}] {} — {} (Status: {} | Endorsers: {} & {})",
                            s.name, s.role, s.task, s.status, s.endorser_1, s.endorser_2
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
                            "• [{}] {} (Connectors: {} | Corroboration: {} | Status: {})",
                            w.canonical_id,
                            w.title,
                            w.corroborating_connectors.join(", "),
                            w.corroborating_connectors.len(),
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
                if !experiments.is_empty() {
                    println!("\n--- Research Investigations & Experiments ({} items) ---", experiments.len());
                    for exp in experiments {
                        println!("• Cycle #{}: Hypothesis: {}", exp.cycle_id, exp.hypothesis);
                        println!("  Experiment: {}", exp.attempted_method);
                        println!("  Outcome: {}", exp.observed_outcome);
                        println!("  Lessons: {}", exp.lessons_learned);
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
                    println!("Endorsers: {} & {}", tool.endorser_1.as_deref().unwrap_or("none"), tool.endorser_2.as_deref().unwrap_or("none"));
                    println!("Tests passed: {}", tool.test_passed);
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
                println!("- {} (v{}) [{}] - {} (Endorsers: {}, {})",
                    t.name, t.version, t.status, t.description,
                    t.endorser_1.as_deref().unwrap_or("none"),
                    t.endorser_2.as_deref().unwrap_or("none"));
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

            if json {
                let info = serde_json::json!({
                    "concurrency_limit": current_limit,
                    "running": running,
                    "queued": queued,
                    "pending_limit_proposals": pending_proposals,
                    "last_increase_time": last_increase,
                    "last_reduction_time": last_reduction,
                    "specialists": specialists,
                });
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!("--- Specialist Agents Status ---");
                println!("Concurrency limit: {}", current_limit);
                println!("Active: {} running, {} queued (capacity: {}/{} slots)", running, queued, running, current_limit);
                println!("Pending increase proposals: {}", pending_proposals);
                if let Some(r) = last_reduction {
                    println!("Last reduction: {} (autonomous increase locked for 24h from this timestamp)", r);
                }

                if specialists.is_empty() {
                    println!("\nNo specialists recorded yet.");
                } else {
                    println!("\n{:<18} {:<44} {:<12} {:<18} {:<8}", "NAME", "ROLE", "STATUS", "ENDORSERS", "TOKENS");
                    println!("{}", "-".repeat(104));
                    for s in specialists {
                        let endorsers = format!("{}+{}", s.endorser_1, s.endorser_2);
                        println!("{:<18} {:<44} {:<12} {:<18} {:<8}", s.name, s.role, s.status, endorsers, s.max_tokens);
                        println!("  Task: {}", s.task);
                        if let Some(ref res) = s.result {
                            println!("  Finding: {}", triad::model::clip_text(res, 220));
                        }
                    }
                }
            }
            Ok(())
        }
    }
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
                        IpcResponse::Ok { message: "Configuration reloaded.".into() }
                    }
                    IpcRequest::Audit => {
                        let rep = coord.get_audit_report().await;
                        IpcResponse::Audit(rep)
                    }
                };
                let _ = write_ipc_response(&mut stream, &resp).await;
            }
        });
    }

    Ok(())
}
