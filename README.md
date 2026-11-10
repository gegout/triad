# Triad: Persistent, Self-Improving Three-Agent Autonomous Research Lab (`v0.3.0`)

[![Version: 0.3.0](https://img.shields.io/badge/Version-0.3.0-brightgreen.svg)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: Rust](https://img.shields.io/badge/Language-Rust_2021-orange.svg)](Cargo.toml)
[![Isolation: Bubblewrap](https://img.shields.io/badge/Sandbox-Bubblewrap_(bwrap)-success.svg)](#9-sandboxed-tool--program-solver-execution)

**Triad** is a local, persistent, self-improving autonomous research community composed of three governing agents—**Explorer**, **Sceptic**, and **Builder**—augmented by bounded temporary **Specialists**, implemented in native Rust.

> **Shared Mission:**  
> *“Discover and understand AI research, develop better reasoning methods, and build extensions that increase our ability to investigate, experiment, and learn. Apply what we discover to improve our own operational capabilities.”*

Triad operates out of the box on local OpenAI-compatible inference snaps (`smollm3-3b` and dedicated `nomic-embed-text-v1-5` 768-D embeddings) while supporting optional, strictly budget-governed and **smartly 24h-paced** escalation to frontier providers (`Anthropic`, `OpenAI`, `Gemini`, `OpenRouter`). It investigates across **19 scientific, metadata, venue, code/artifact, institutional, orientation, and web-fetch connectors** with a **multi-route Open-Access full-text extraction waterfall** (`arxiv.org/html`, `ar5iv`, gzipped/tarred LaTeX e-print archives, PDF streams, ACL Anthology, bioRxiv/medRxiv, Zenodo REST API, HAL, OpenAlex OA, and GitHub multi-branch READMEs), enforces a **two-phase specialist state machine** and **spec-driven real-data model training & evaluation harnesses** (`~/.triad/datasets/*.jsonl`), maintains a provenance-deduplicated **Semantic World Graph** in SQLite (WAL mode), and continuously evaluates its domain competence against external benchmarks.

---

## Table of Contents

1. [System Architecture & Module Map](#1-system-architecture--module-map)
2. [Constitutional Governance, Personalities & Enforced Specialist State Machine](#2-constitutional-governance-personalities--enforced-specialist-state-machine)
3. [Versioned Coordinator Extension Interface (5 Operational Hooks)](#3-versioned-coordinator-extension-interface-5-operational-hooks)
4. [Autonomous Reasoning Programme & Real-Data Model Training / Evaluation Harnesses](#4-autonomous-reasoning-programme--real-data-model-training--evaluation-harnesses)
5. [Deep-Research Pipeline, 19-Connector Adapter Interface & Multi-Route Full-Text Waterfall](#5-deep-research-pipeline-19-connector-adapter-interface--multi-route-full-text-waterfall)
6. [Model Providers, Capability Selection & Smart 24h Frontier Pacing](#6-model-providers-capability-selection--smart-24h-frontier-pacing)
7. [Domain Expertise Evaluation & Expertise Profile (`CLI` & `MCP`)](#7-domain-expertise-evaluation--expertise-profile-cli--mcp)
8. [Semantic World Graph & Knowledge Model](#8-semantic-world-graph--knowledge-model)
9. [Sandboxed Tool, Dataset & Program-Solver Execution](#9-sandboxed-tool-dataset--program-solver-execution)
10. [Prerequisites, Installation & Inference Services](#10-prerequisites-installation--inference-services)
11. [Quickstart, Storage Hygiene & Complete CLI Reference](#11-quickstart-storage-hygiene--complete-cli-reference)
12. [Systemd User Service Integration](#12-systemd-user-service-integration)
13. [Test Suite & Verification](#13-test-suite--verification)
14. [MIT License](#14-mit-license)

---

## 1. System Architecture & Module Map

```
                   +---------------------------------------------------+
                   |          triad CLI & MCP Server Interface         |
                   +-------------------------+-------------------------+
                                             | UNIX Domain Socket IPC / Streamable HTTP
                                             v
+---------------------------------------------------------------------------------------+
|                                     triad Daemon                                      |
|                                                                                       |
|  +-------------------------+  +-------------------------+  +-----------------------+  |
|  |     Explorer Agent      |  |      Sceptic Agent      |  |     Builder Agent     |  |
|  | (Hypotheses, Discovery) |  | (Verification, Dissent) |  | (Rust Tools & Exts)   |  |
|  +------------+------------+  +------------+------------+  +-----------+-----------+  |
|               |                            |                           |              |
|               +----------------------------+---------------------------+              |
|                                            | 2-of-3 Constitutional Quorum             |
|                                            v                                          |
|  +---------------------------------------------------------------------------------+  |
|  |                         Triad Coordinator & Hook Pipeline                       |  |
|  |  [query.transform] -> [retrieval.rerank] -> [context.compose]                   |  |
|  |                    -> [turn.plan]        -> [learning.consolidate]              |  |
|  +-----------------+-------------------------------+-------------------------------+  |
|                    |                               |                                  |
|     +--------------v---------------+ +-------------v---------------+ +-------------v--+
|     |   Shared Store (SQLite WAL)  | |  Sandboxed Execution (bwrap)| | 19-Connector   |
|     | - Claims, Votes & Dissent    | | - Two-Phase Specialist Exec | | OA Waterfall   |
|     | - Versioned Tracked Sources  | | - Real-Data Trainer Harness | | - HTML/ar5iv   |
|     | - Immutable Exec Evidence    | | - Read-Only /tmp/datasets   | | - Safe tar.gz  |
|     | - Smart 24h Frontier Ledger  | | - Extension Runner (v1 JSON)| | - Pure-Rust PDF|
|     | - Domain Topic Coverage      | | - Process-Group Cleanup     | | - Cross-ID OA  |
|     +------------------------------+ +-----------------------------+ +----------------+
+---------------------------------------------------------------------------------------+
```

### Source Code Structure (`src/`)

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **CLI & Daemon Entry** | [`src/main.rs`](src/main.rs) | CLI subcommand dispatcher (`status`, `learning`, `sources`, `experiments`, `models`, `profile`, `evaluate`, `purge`, etc.), daemon lifecycle, and `triad doctor` diagnostics. |
| **Coordinator Engine** | [`src/coordinator.rs`](src/coordinator.rs) | Domain-independent autonomous curiosity loop, topic coverage prioritization, full-text backlog draining (`drain_full_text_ingestion_backlog`), 5-hook extension pipeline, two-phase specialist orchestration, real-data model trainer synthesis, and self-repair engine. |
| **Extension Subsystem** | [`src/extensions.rs`](src/extensions.rs) | Versioned extension protocol (`v1`), sandboxed JSON runner, shared-target Cargo compilation, evaluation, quorum endorsement, atomic activation, automatic rollback, and stale workspace artifact cleanup. |
| **Multi-Source Web & Extraction** | [`src/web.rs`](src/web.rs) | 19 research connectors (`SourceAdapter`), multi-route cross-identifier Open-Access full-text waterfall (`ingest_source_via_adapter`), URL canonicalization, pure-Rust PDF (`lopdf`/`pdf-extract`) & in-memory `tar.gz` LaTeX extraction (`flate2`/`tar`), semantic HTML/Markdown section hierarchy extraction, and fidelity scoring. |
| **Persistent Semantic Store** | [`src/store.rs`](src/store.rs) | SQLite WAL store for `claims`, `tracked_sources`, `source_originals`, `source_sections`, `source_corroborations`, `tool_executions`, `experiment_specifications`, `research_experiments`, `frontier_budget_ledger`, `evaluation_runs`, and `domain_topic_coverage`. |
| **Specialist Governance** | [`src/specialists.rs`](src/specialists.rs) | Enforced two-phase specialist state machine (`phase1_investigate_and_execute` $\rightarrow$ `phase2_validate_and_commit`), concurrency slots, and timeout/cancel cleanup. |
| **Research Tools & Validation** | [`src/tools.rs`](src/tools.rs) | Static/AST anti-triviality validator for generated Rust tools, structured `results.json` verification against `ExperimentSpecification` keys, and baseline/held-out checks. |
| **Sandboxed Execution & Datasets** | [`src/execution.rs`](src/execution.rs) | Linux Bubblewrap (`bwrap`) isolation, persistent JSONL benchmark dataset provisioning (`~/.triad/datasets` mounted read-only at `TRIAD_DATASETS_DIR` and `/tmp/datasets`), process-group tracking and `SIGKILL` cleanup, and Antigravity (`agy`) CLI adapter. |
| **Constitutional Rules** | [`src/constitution.rs`](src/constitution.rs) | Immutable constitutional invariants (`v2.1.0`, Frontier AI Edition), Frontier AI mandate & domains, quorum rules, specialist budgets, and SHA-256 digest verification. |
| **Model Providers & Smart Pacing** | [`src/model.rs`](src/model.rs) | Multi-provider abstraction (`local_snap`, `openai`, `anthropic`, `gemini`, `openrouter`, `openai_compatible`), capability/evaluation model selector, 3-dimensional 24h smart frontier pacing & balancing (`check_frontier_budget`), atomic quota reservation, and dedicated `nomic-embed-text-v1-5` embedding client. |
| **Configuration & Reload** | [`src/config.rs`](src/config.rs) | `~/.triad/config.toml` management (`0600` permissions), external evaluation set validation, provider/budget/pacing policies, and live file-watcher hot-reload. |
| **Consultation Engine** | [`src/consultation.rs`](src/consultation.rs) | Unified domain consultation engine (`expert_profile`, `consult_expert`, `solve_problem`, `investigate_question`, `review_proposal`, `consultation_status`), routing/escalation, and epistemic labeling. |
| **MCP Server (`stdio` & HTTP)** | [`src/mcp.rs`](src/mcp.rs) | Model Context Protocol (`2025-03-26`) JSON-RPC 2.0 server supporting both `stdio` (Unix socket forwarding to live daemon) and Streamable HTTP (`POST /mcp`, `GET /mcp`, `DELETE /mcp`). |
| **Domain Expertise Evaluation** | [`src/evaluation.rs`](src/evaluation.rs) | External benchmark evaluation runner (`evaluation/external_benchmark.json`) scoring 6 dimensions against an equal-budget plain-LLM baseline, split by full-text vs. abstract-only/low-fidelity topics and real-data vs. synthetic-data experiments. |
| **IPC & Single-Instance Server** | [`src/ipc.rs`](src/ipc.rs), [`src/server.rs`](src/server.rs) | Typed UNIX domain socket request/response protocol and advisory PID lockfile enforcement. |

---

## 2. Constitutional Governance, Personalities & Enforced Specialist State Machine

Triad is governed by an immutable constitution (`~/.triad/constitution.toml`, version `2.1.0` — *Triad Community Invariant Constitution (Frontier AI Edition)*) whose SHA-256 digest is verified on every coordinator cycle and passed into every extension and specialist invocation.

### The Three Permanent Governing Agents
1. **Explorer (`Explorer`):** Driven by endogenous curiosity. Formulates falsifiable hypotheses, decomposes open questions across research sources, and identifies unexplored or thin nodes in the semantic graph.
2. **Sceptic (`Sceptic`):** Epistemic auditor and adversarial critic. Audits source provenance, reviews `ExperimentSpecification` definitions before implementation, checks benchmark comparability, detects contradictions, and records formal dissent.
3. **Builder (`Builder`):** Systems engineer and empiricist. Translates theoretical findings into spec-validated Rust model trainers, evaluators, coordinator extensions, and sandboxed verification experiments.

### Orthogonal Epistemic Attributes & Two-Agent Quorum
- **Two-Agent Quorum:** At least **2 distinct permanent agents** must endorse an exact claim version (or extension/tool build) to promote it to `trusted` / `active`. Agreement never fabricates evidence.
- **Orthogonal Claim Attributes:** Every claim in `claims` records three separate, orthogonal attributes:
  1. **Trust (`status`):** `proposed` vs. `trusted` (governed by the 2-of-3 permanent-agent vote).
  2. **Source Support (`source_support_status` + `source_corroboration_count` + `source_fidelity_summary`):** Tracks independent corroborating sources and per-source extraction fidelity (`full-text`, `low-fidelity`, `abstract-only`).
  3. **Experimental Verification (`experimental_verification_status` + `execution_id` + `reproducibility_artifacts_json`):** Links locally reproduced empirical claims to immutable host `tool_executions` records.

### Enforced Two-Phase Specialist State Machine (`src/specialists.rs`, `src/store.rs`)
Temporary bounded specialists (`initial_concurrency_limit = 6`) operate under a strictly host-enforced two-phase state machine:
- **Phase 1 — Investigate & Execute (`phase1_investigate_and_execute`):**
  - A specialist may retrieve sources, propose hypotheses, build tools, compile, run, and inspect failures.
  - **Plans and predictions stay task records:** [`Store::record_specialist_phase1_plan`](src/store.rs) stores plans/predictions in `specialist_phase1_plans`; [`Store::validate_and_commit_specialist_finding`](src/store.rs) rejects any attempt to promote a plan or prediction into a finding.
- **Phase 2 — Validate & Commit (`phase2_validate_and_commit`):**
  - **Locally Reproduced Empirical Findings (`locally_reproduced`):** Every empirical claim must reference a completed host execution record (`execution_id` in `tool_executions`) and its supporting outputs (`measured_results_json`). Host code verifies the SHA-256 `immutable_evidence_hash` (covering tool version/hash, inputs, seed, dependencies, stdout/stderr, exit code, duration, and artifacts) and checks that every number in the claim matches a declared metric key in `results.json`. Agent prose cannot invent runs or measurements.
  - **Literature Findings (`source_supported`):** Must reference real stored passages (`source_sections`) or canonical sources (`tracked_sources`), labeled `source-supported` with the count of independent sources and per-source extraction fidelity (`full-text`, `low-fidelity`, `abstract-only`). Orientation-only sources (`wikipedia`) are rejected as primary evidence.
  - **Post-Validation Quorum:** The two-agent trust rule is applied *after* host validation succeeds, committing bounded conclusions, explicit assumptions, negative results (`is_negative_result`), and reproducibility artifacts.
- **Process-Group Cancellation & Timeout Enforcement:** If a specialist times out or is cancelled, [`BubblewrapAdapter`](src/execution.rs) terminates the entire sandbox process group via `nix::sys::signal::killpg(..., Signal::SIGKILL)`, confirms exit, records the timeout/cancelled outcome, releases the concurrency slot, and blocks any uncommitted work from becoming a finding.

---

## 3. Versioned Coordinator Extension Interface (5 Operational Hooks)

Governing agents synthesize, compile, test, evaluate, endorse, and atomically activate native Rust extensions across **5 versioned coordinator hooks** (`protocol_version = "v1"`):

| Hook Name | Active Baseline Extension | Purpose & Validated Contract |
| :--- | :--- | :--- |
| **`query.transform`** | `multi_source_research_decomposer` (`v1`) | Expands user questions and curiosity topics into complementary search queries across 7 research aspects and selects target connectors. |
| **`retrieval.rerank`** | `keyword_boost_reranker` (`v2`) | Re-scores retrieved claims using lexical overlap, semantic graph node degree (`node_degree`), independent source corroboration (`provenance_count`), and recency. |
| **`context.compose`** | `evidence_chain_composer` (`v1`) | Constructs compact, source-attributed reasoning contexts stratified into *Trusted Claims*, *Observations*, *Unresolved Hypotheses*, *Contradictions*, and explicit `[MISSING_EVIDENCE_GAP]` markers. |
| **`turn.plan`** | `hierarchical_branch_planner` (`v1`) | Generates an explicit subproblem dependency DAG, evaluates 3 competing reasoning branches (*Direct Empirical*, *Counterfactual Ablation*, *Formal/Solver Verification*), prunes low-utility branches, and flags quantitative tasks for solver execution. |
| **`learning.consolidate`** | `executable_verifier_consolidator` (`v1`) | Audits candidate learnings against source passages, numerical/percentage bounds, and internal consistency before consolidation, feeding failures into a bounded correction loop. |

---

## 4. Autonomous Reasoning Programme & Real-Data Model Training / Evaluation Harnesses

### Spec-Driven Research Tools & Anti-Triviality Gate (`src/tools.rs`, `src/store.rs`)
Triad requires a formal [`ExperimentSpecification`](src/store.rs) before any research tool or experiment is implemented or executed:
- **Required Specification Fields:** `research_question`, `falsifiable_hypothesis`, `mechanism_under_test`, `executable_check_kind` (`simulation`, `numerical_optimization`, `algorithm_ablation`, `constraint_counterexample`, `data_reanalysis`, `real_implementation`, `statistical_replication`), `data_kind` (`real_data` or `synthetic_data`), `inputs_description`, `assumptions`, `baseline_method`, `controls`, `declared_metrics` (with explicit result keys and success criteria), `budget`, `seeds`, `expected_outputs`, and `known_limitations`.
- **Static & AST Anti-Triviality Rejection:** [`ToolManager::detect_superficial_verifier_patterns`](src/tools.rs) rejects generated tools that merely count words/characters, inspect strings, print predefined conclusions, or emit hardcoded/arbitrary scores without real computation.
- **Mandatory Structured `results.json`:** Every generated research tool must write a structured JSON results file whose keys match the `declared_metrics` in its `ExperimentSpecification`, including a `baseline_comparison` and `held_out_evaluation`. Numbers that appear only in prose or `stdout` cannot support an empirical claim.
- **Full-Text Source Grounding & `under_specified` Protection:** Experiments may only cite methods, algorithms, or equations from sources ingested at `full_text` fidelity (`low_fidelity = false`). If an experiment cites an `abstract_only` or `low_fidelity` source, [`Store::register_and_validate_experiment_spec`](src/store.rs) marks the specification `under_specified = true`, records the missing source details, and automatically triggers a source re-ingestion request (`request_source_reingestion`).

### Real-Data Model Training & Benchmark Evaluation Harnesses (`src/execution.rs`, `src/coordinator.rs`)
Rather than relying solely on synthetic micro-simulations, Triad provisions and continuously updates persistent JSONL benchmark datasets in `~/.triad/datasets` (seeded with live local `smollm3-3b` and `nomic-embed-text-v1-5` rollout traces) and mounts them read-only inside the Bubblewrap sandbox (`TRIAD_DATASETS_DIR` and `/tmp/datasets`):
1. **`gsm8k_math_step_traces.jsonl`:** Multi-branch mathematical reasoning step feature traces, step correctness labels, and outcome logits (`train` + `held_out` splits) for **Process Reward Model (PRM)** step-verifier training (`prm_step_verifier_trainer_harness`).
2. **`moe_token_routing_activations.jsonl`:** 8-D token activation hidden states and skewed burst routing targets (`train` + `held_out` splits) for **Sinkhorn-Knopp entropic MoE router** gradient training (`sinkhorn_moe_gate_trainer_harness`).
3. **`calibration_verifier_logits.jsonl`:** Raw verifier confidence logits and empirical correctness labels (`train` + `held_out` splits) for **Platt / temperature scaling NLL + Brier** gradient optimization (`ece_platt_verifier_calibration_harness`).
4. **`lora_adapter_regression_tensors.jsonl`:** 8-D transformer MLP activation pairs $(x, \text{base\_pred}, y_{\text{target}})$ (`train` + `held_out` splits) for **rank-2 Low-Rank Adapter ($\text{LoRA}: W_0 + BA$)** multi-epoch momentum SGD backpropagation (`lora_rank_adapter_trainer_harness` and `cycle_{id}_lora_prm_eval_harness`).

Every real-data trainer reports `initial_train_loss`, `final_train_loss`, `learned_temperature`, `lora_frobenius_norm`, `dataset_train_samples`, `dataset_held_out_samples`, `baseline_comparison`, and `held_out_evaluation` (`generalization_gap`, `held_out_gain`), tagged with `data_kind = "real_data"` and the constitutional guardrail: *"Numerical agreement supports an experiment; it is not a proof."*

---

## 5. Deep-Research Pipeline, 19-Connector Adapter Interface & Multi-Route Full-Text Waterfall

Triad’s [`WebExplorer`](src/web.rs) implements a unified [`SourceAdapter`](src/web.rs) trait across **19 connectors in 7 categories**:

1. **Preprints (`preprints`):** `arxiv`, `biorxiv`, `medrxiv`, `ssrn`, `hal`
2. **Metadata & Citation Graphs (`metadata`):** `openalex`, `crossref`, `semantic_scholar`
3. **Venues (`venues`):** `openreview`, `acl_anthology`, `dblp`, `pubmed`
4. **Code & Artifacts (`code_and_artifacts`):** `github`, `huggingface`, `zenodo`, `papers_with_code`
5. **Institutional (`institutional`):** `institutional` (lab reports, IETF RFCs, W3C, NIST, patents, regulatory documents)
6. **Orientation Only (`orientation_only`):** `wikipedia` (strictly marked `is_orientation_only = true`; never counted as primary evidence)
7. **Generic Web Fetch (`web_fetch`):** `web_fetch` (configured URL lists and RSS/Atom feeds)

### Multi-Route Open-Access Full-Text Extraction Waterfall (`src/web.rs`)
To maintain a high primary-corpus full-text share (**`77.8%` full-text** across tracked primary sources), [`WebExplorer::ingest_source_via_adapter`](src/web.rs) executes a 3-stage open-access promotion waterfall:
- **Route A — Cross-Identifier `ArxivAdapter` Promotion:** Any paper discovered via `openalex`, `semantic_scholar`, `crossref`, `dblp`, or `acl_anthology` that carries an `arxiv_id` or `10.48550/arXiv.*` DOI is automatically promoted to `ArxivAdapter`:
  1. `arxiv.org/html` and `ar5iv.labs.arxiv.org/html` structured HTML5/MathML extraction
  2. Gzipped/tarred LaTeX e-print source (`arxiv.org/e-print`, decompressed and parsed safely in memory via `flate2` + `tar`)
  3. Pure-Rust PDF stream extraction (`lopdf` + `pdf-extract`)
- **Route B — Primary Connector Adapter:** Direct ingestion via the discovering connector’s `SourceAdapter` (`openreview`, `github` multi-branch `main`/`master` `README.md`/`README.rst` + HTML fallback, `huggingface`, `hal`, etc.).
- **Route C — Connector-Aware Open-Access PDF/HTML Waterfall:** Resolves open-access full texts across:
  - **ACL Anthology (`10.18653/*`):** Direct `https://aclanthology.org/{id}.pdf` and HTML extraction
  - **bioRxiv / medRxiv (`10.1101/*`):** `.full` HTML and `.full.pdf` extraction
  - **Zenodo (`10.5281/zenodo.*`):** Zenodo REST API (`https://zenodo.org/api/records/{id}`) file/description extraction
  - **HAL (`hal-*`):** Direct `/document` PDF extraction
  - **OpenAlex Open-Access Lookup:** Resolves `best_oa_location.pdf_url`, `open_access.oa_url`, and `primary_location.pdf_url` via `https://api.openalex.org/works/...`
- **Active Full-Text Backlog Draining:** [`Coordinator::drain_full_text_ingestion_backlog`](src/coordinator.rs) and [`Store::reconcile_and_select_upgradeable_sources`](src/store.rs) continuously upgrade `abstract_only`, `metadata_only`, and `low_fidelity` primary sources into structured `full_text` documents with a `"0 Section Hierarchy"` index.

---

## 6. Model Providers, Capability Selection & Smart 24h Frontier Pacing

Triad separates local cheap/routing workloads from optional escalated frontier synthesis ([`src/model.rs`](src/model.rs), [`src/config.rs`](src/config.rs)):
- **Local Inference Snaps Preserved:** `routing`, `summarization`, and `cheap_work` stay on local `smollm3-3b`. Dedicated 768-D embeddings always use local `nomic-embed-text-v1-5`.
- **Provider Abstraction:** Supports `local_snap` / `openai` / `openrouter` / `openai_compatible` (`/v1/chat/completions`), native `anthropic` (`/v1/messages`), and native `gemini` (`/v1beta/models/{model}:generateContent`).
- **Capability & Evaluation-Driven Model Selection:** [`ModelClient::select_model_for_workload`](src/model.rs) selects the most capable eligible model using capability metadata (`reasoning_score_bps`, `coding_score_bps`, `synthesis_score_bps`, `critique_score_bps`), context window, required features, and empirical evaluations (`model_evaluations`), falling back to `preference_order` when empirical evidence is thin.
- **3-Dimensional Smart 24h Frontier Pacing & Balancing (`[models.frontier_budget]`):**
  - Rather than burning all daily frontier calls in the first few minutes and hitting a hard stop, Triad implements a 3-dimensional pacing and balancing strategy across the rolling 24-hour window (`86,400s`):
    1. **Time-Slot Pacing (`smart_pacing_enabled = true`, `pacing_utilization_target_pct = 90`, `pacing_burst_allowance = 1`):** Computes the dynamic target interval between calls from the target daily utilization (`target_calls = ceil(max_calls_per_24h * 0.90)`). Calls that arrive before the next time slot smoothly fall back to local inference (`frontier_paced -> continuing locally`) while preserving a burst allowance for urgent interactive tasks.
    2. **Workload Priority Balancing (`workload_balancing_enabled = true`):** Tracks 24h call distribution across escalated workloads (`hard_synthesis`, `experiment_design`, `complex_rust_generation`, `debugging`, `critical_review`) and grants an urgency discount to under-served high-value workloads while applying a backpressure multiplier to over-represented workloads.
    3. **Multi-Model Exploration & Rotation (`model_balancing_enabled = true`):** Balances frontier calls across the configured `preference_order` (e.g., `anthropic/claude-opus-5.5`, `openai/gpt-6-astra`, `qwen/qwen3.8-2.4t-a95b`, `openai/gpt-6.1-sol`, `anthropic/claude-sonnet-5.5`) so every permitted frontier model accumulates empirical evaluations (`min_eval_runs_for_empirical_ranking`) and no single model monopolizes the daily quota.

---

## 7. Domain Expertise Evaluation & Expertise Profile (`CLI` & `MCP`)

Triad measures domain competence periodically and on demand via [`EvaluationRunner::run_domain_evaluation`](src/evaluation.rs) (`./bin/triad evaluate` and `./bin/triad profile`):
- **External Benchmark Set Only:** Configured via `[domain].evaluation_set_path` ([`evaluation/external_benchmark.json`](evaluation/external_benchmark.json)) or `[[domain.evaluation_cases]]`. [`validate_evaluation_case_external`](src/config.rs) strictly rejects any evaluation case authored by Triad or its agents (`Explorer`, `Sceptic`, `Builder`, `specialist`, `triad_authored`).
- **Six Scored Dimensions vs. Equal-Budget Plain-LLM Baseline:**
  1. `correctness`
  2. `citation_accuracy`
  3. `source_diversity`
  4. `contradiction_detection`
  5. `uncertainty_calibration`
  6. `problem_solving_success`
- **Full-Text vs. Abstract-Only/Low-Fidelity Topic Split:** Reports `evaluated_corpus_full_text_share` and splits every score into `full_text_scores` and `abstract_or_low_fidelity_scores`. **A score on thin or abstract-only ingestion is never reported as a score on the domain.**
- **Real-Data vs. Synthetic-Data Experiment Split:** Tags every experiment in `research_experiments` as `real_data` or `synthetic_data` and reports problem-solving success separately for each (`real_data_problem_solving` vs. `synthetic_data_problem_solving`).
- **Topic Coverage Linked to Prioritization (`domain_topic_coverage`):** [`Store::sync_and_prioritize_domain_topic_coverage`](src/store.rs) marks each topic as `covered`, `thin`, `contradicted`, or `low_fidelity`, and automatically feeds `thin`, `contradicted`, and `low_fidelity` topics into `research_backlog` (priority `97..99`) with automatic re-ingestion requests (`reingestion_requested = true`).

---

## 8. Semantic World Graph & Knowledge Model

![Triad Semantic World Graph](triad_semantic_graph.png)

Triad’s live knowledge graph currently encompasses:
- **10,990+ Semantic Nodes** and **33,099+ Semantic Edges (~3.01 edges/node)** across `concept`, `research_work`, `source`, `dataset`, `implementation`, `tool`, `extension`, and `experiment` ontology types.
- **11,658+ Quorum-Verified Claims**, **820+ Registered Sandboxed Tools** (including multi-epoch real-data PRM, Sinkhorn MoE, Platt calibration, and rank-2 LoRA trainers), **545+ Active Coordinator Extensions**, and **260+ Canonical Tracked Sources** (`99+` duplicate routes merged, **`77.8%` primary corpus full-text share**).

---

## 9. Sandboxed Tool, Dataset & Program-Solver Execution

Every agent-synthesized tool, coordinator extension, and real-data model trainer runs inside an isolated Linux **Bubblewrap (`bwrap`)** container:
- Read-only root filesystem (`--ro-bind / /`) with isolated `/dev`, `/proc`, and `--tmpfs /tmp`.
- Read-only bind mounts for persistent benchmark datasets (`~/.triad/datasets` mounted at `TRIAD_DATASETS_DIR` and `/tmp/datasets`).
- `--unshare-all` (unsharing user, IPC, PID, net, and UTS namespaces) so compiled tool binaries execute with **zero network access** and cannot modify host files or the SQLite database.
- Separate process group (`process_group(0)`) with automatic `killpg(..., SIGKILL)` cleanup on timeout or cancellation.
- Shared Cargo build cache (`extensions/shared_target`, `tools/shared_target`) and automatic workspace artifact pruning (`purge_stale_build_artifacts`).

---

## 10. Prerequisites, Installation & Inference Services

### System Requirements
- **OS:** Linux (`x86_64` or `aarch64`) with user namespaces enabled for `bubblewrap` (`bwrap`).
- **Rust Toolchain:** Rust `1.75+` (`2021` edition) with `cargo` and `rustc`.
- **System Packages:** `bubblewrap`, `build-essential`, `pkg-config`.

### Step 1: Install Local Inference Snaps (or OpenAI-Compatible Local Endpoints)

| Service | Snap Name | Default Endpoint | Advertised Model ID | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Chat LLM** | `smollm3` | `http://localhost:8366/v1` | `smollm3-3b` | Local routing, summarization, cheap work, and fallback synthesis (`/no_think` mode) |
| **Embeddings** | `nomic-embed-text-v1-5` | `http://127.0.0.1:8348/v1` | `nomic-embed-text-v1-5` | Dedicated 768-D semantic vectors (`search_query: ` / `search_document: `) |
| **Sandbox** | `bubblewrap` | `/usr/bin/bwrap` | Linux Namespaces | Process, filesystem, and network isolation for compilation & execution |

### Step 2: Optional Frontier Model API Credentials
Triad reads frontier API credentials strictly from environment variables referenced by `credential_env` in `~/.triad/config.toml` (never stored in config files or logs):

```bash
export OPENROUTER_API_KEY="sk-or-..."   # For openrouter (anthropic/claude-opus-5.5, openai/gpt-6-astra, qwen/qwen3.8-2.4t-a95b, openai/gpt-6.1-sol, anthropic/claude-sonnet-5.5)
export ANTHROPIC_API_KEY="sk-ant-..."   # For anthropic
export OPENAI_API_KEY="sk-..."          # For openai
export GEMINI_API_KEY="AIza..."         # For gemini
```

### Step 3: Clone, Build & Initialize Triad

```bash
git clone https://github.com/gegout/triad.git
cd triad

cargo build --release
mkdir -p ./bin
cp target/release/triad ./bin/triad

./bin/triad init
./bin/triad doctor
```

---

## 11. Quickstart, Storage Hygiene & Complete CLI Reference

```bash
# Daemon lifecycle
./bin/triad start
./bin/triad run --foreground
./bin/triad status
./bin/triad pause
./bin/triad resume
./bin/triad stop

# Domain Expertise Evaluation & Published Expertise Profile
./bin/triad evaluate
./bin/triad evaluate --json
./bin/triad profile
./bin/triad profile --json

# Model Providers, Capability Selection & Smart 24h Frontier Pacing Status
./bin/triad models
./bin/triad models --refresh
./bin/triad models --json

# Persistent Source Tracking, Extraction Quality & Connector Health
./bin/triad sources
./bin/triad sources --json

# Spec-Backed Research Experiments & Real-Data Model Trainer Results
./bin/triad experiments
./bin/triad experiments --json

# Workspace Build-Artifact Purge (Reclaims Cargo Target Cache Without Losing Knowledge/Binaries)
./bin/triad purge

# Learnings, Provenance & Semantic Memory
./bin/triad learning --last 10
./bin/triad learning show <claim-id-prefix>
./bin/triad memory search "neuro-symbolic program synthesis"

# Interactive Research & Engineering Tasks
./bin/triad ask "Compare Tree of Thoughts search and Process Reward Model verification" --wait
./bin/triad challenge "Build and evaluate an extension that improves retrieval re-ranking" --wait
./bin/triad tell "Local Cargo shared target caching reduces extension compile times to <1.5s."

# Extensions, Tools, Specialists & Auditing
./bin/triad extensions
./bin/triad tools
./bin/triad specialists
./bin/triad audit

# MCP Expert Consultation Server (stdio & Streamable HTTP)
./bin/triad mcp --stdio
./bin/triad mcp --http 127.0.0.1:8390
```

---

## 12. Systemd User Service Integration

```bash
mkdir -p ~/.config/systemd/user
cp triad.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now triad
```

---

## 13. Test Suite & Verification

Triad’s automated unit and integration test suite verifies:
- **Persistent, Version-Aware Source Tracking:** Equivalent URL deduplication across agents and restarts, new document version (`v1 -> v2`) eligibility, and independent corroboration counting (1 paper via 3 routes counts once; 2 independent papers count twice; datasets and implementations count separately).
- **Multi-Source Adapter Interface & Open-Access Full-Text Waterfall:** Cross-identifier `ArxivAdapter` promotion, compressed-stream PDF (`lopdf`/`pdf-extract`) and gzipped LaTeX tarball (`flate2`/`tar`) extraction, semantic HTML/Markdown section hierarchies and preserved equations, structured section retrieval within token limits, and orientation-only (`wikipedia`) exclusion from primary evidence.
- **Enforced Specialist State Machine:** Phase 1 plans/predictions rejected as findings, empirical claims with fabricated execution IDs or unverified prose numbers rejected, literature claims verified with passage IDs and per-source extraction fidelity, and timed-out/cancelled runs cleaned up via process-group `SIGKILL` without committing findings.
- **Real-Data Model Training Harnesses & Anti-Triviality Verification:** Trivial word-counter/string-inspector/hardcoded-output tools rejected, experiment specifications citing abstract-only sources flagged `under_specified` with automatic re-ingestion requests, and multi-epoch gradient-descent trainers producing valid `results.json` on mounted `/tmp/datasets/*.jsonl` benchmark splits verified end-to-end.
- **Model Providers, Selection & 3-Dimensional Smart Frontier Pacing:** Capability + evaluation ranking without name/parameter/price heuristics, 24h time-slot pacing, workload balancing, and multi-model rotation across `preference_order`, missing credentials and exhausted budgets falling back cleanly under `continue_locally` / `queue` / `report`, retries consuming quota, and secret redaction across logs and errors.
- **Domain Expertise Evaluation:** Identical coordinator workflow running under two different domain configs (`quantum_error_correction` and `computational_biology_protein_design`) without code changes, external evaluation set execution with rejection of Triad-authored sets, and separate score reporting for `full_text` vs. `abstract_only`/`low_fidelity` topics and `real_data` vs. `synthetic_data` experiments.

```bash
cargo test -- --test-threads=1
```

---

## 14. MIT License

This project is licensed under the **MIT License** (see [`LICENSE`](LICENSE)):

```text
MIT License

Copyright (c) 2026 Cédric Gegout (cedric.gegout@gmail.com) and Triad contributors
```
