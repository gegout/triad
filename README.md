# Triad: Persistent, Self-Improving Three-Agent Autonomous Research Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: Rust](https://img.shields.io/badge/Language-Rust_2021-orange.svg)](Cargo.toml)
[![Isolation: Bubblewrap](https://img.shields.io/badge/Sandbox-Bubblewrap_(bwrap)-success.svg)](#7-sandboxed-tool--program-solver-execution)

**Triad** is a local, persistent, self-improving autonomous research community composed of three governing agents—**Explorer**, **Sceptic**, and **Builder**—augmented by bounded temporary **Specialists**, implemented in native Rust.

> **Shared Mission:**  
> *“Discover and understand AI research, develop better reasoning methods, and build extensions that increase our ability to investigate, experiment, and learn. Apply what we discover to improve our own operational capabilities.”*

Triad operates without external paid APIs or Python runtime dependencies. It coordinates local OpenAI-compatible inference snaps (`smollm3-3b` and `nomic-embed-text-v1-5`), investigates across **9 distinct AI and scientific research connectors**, maintains a provenance-deduplicated **Semantic World Graph** in SQLite (WAL mode), and autonomously synthesizes, compiles, tests, evaluates, and activates **compiled Rust coordinator extensions** inside Linux Bubblewrap (`bwrap`) sandboxes.

---

## Table of Contents

1. [System Architecture & Module Map](#1-system-architecture--module-map)
2. [Constitutional Governance, Personalities & Specialist Agents](#2-constitutional-governance-personalities--specialist-agents)
3. [Versioned Coordinator Extension Interface (5 Operational Hooks)](#3-versioned-coordinator-extension-interface-5-operational-hooks)
4. [Autonomous Reasoning Programme (6 Capabilities)](#4-autonomous-reasoning-programme-6-capabilities)
5. [Deep-Research Pipeline & 9 Multi-Source Connectors](#5-deep-research-pipeline--9-multi-source-connectors)
6. [Semantic World Graph & Knowledge Model](#6-semantic-world-graph--knowledge-model)
7. [Sandboxed Tool & Program-Solver Execution](#7-sandboxed-tool--program-solver-execution)
8. [Prerequisites, Installation & Local Inference Services](#8-prerequisites-installation--local-inference-services)
9. [Quickstart & Complete CLI Reference](#9-quickstart--complete-cli-reference)
10. [Systemd User Service Integration](#10-systemd-user-service-integration)
11. [Test Suite & Verification](#11-test-suite--verification)
12. [MIT License](#12-mit-license)

---

## 1. System Architecture & Module Map

```
                   +---------------------------------------------------+
                   |             triad CLI (User Interface)            |
                   +-------------------------+-------------------------+
                                             | UNIX Domain Socket IPC
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
|     |   Shared Store (SQLite WAL)  | |  Sandboxed Execution (bwrap)| | Multi-Source   |
|     | - Claims, Votes & Dissent    | | - Extension Runner (v1 JSON)| | WebExplorer    |
|     | - Semantic Nodes & Edges     | | - Dynamic Cargo + SharedTgt | | - arXiv, DOI   |
|     | - Canonical Research Works   | | - Program/Solver Verifier   | | - OpenAlex, S2 |
|     | - Research Syntheses & Exps  | | - Bounded Host Operations   | | - ACL, GitHub  |
|     +------------------------------+ +-----------------------------+ +----------------+
+---------------------------------------------------------------------------------------+
```

### Source Code Structure (`src/`)

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **CLI & Daemon Entry** | [`src/main.rs`](src/main.rs) | CLI subcommand dispatcher, daemon lifecycle (`start`, `run`, `stop`), and `triad doctor` diagnostics. |
| **Coordinator Engine** | [`src/coordinator.rs`](src/coordinator.rs) | Autonomous curiosity loop, task execution, 5-hook extension pipeline, program/solver verification, and self-repair engine. |
| **Extension Subsystem** | [`src/extensions.rs`](src/extensions.rs) | Versioned extension protocol (`v1`), sandboxed JSON runner, shared-target Cargo compilation, evaluation, quorum endorsement, atomic activation, and automatic rollback. |
| **Multi-Source Web & Research** | [`src/web.rs`](src/web.rs) | 9 research connectors, structured section/table/limitation extraction, and DOI/arXiv/title provenance deduplication. |
| **Persistent Semantic Store** | [`src/store.rs`](src/store.rs) | SQLite WAL database (`claims`, `votes`, `sources`, `semantic_nodes`, `semantic_edges`, `research_works`, `research_syntheses`, `research_experiments`, `extensions`, `specialists`). |
| **Sandboxed Execution** | [`src/execution.rs`](src/execution.rs) | Linux Bubblewrap (`bwrap`) isolation, resource/timeout enforcement, Cargo test harness, and Antigravity (`agy`) CLI adapter. |
| **Constitutional Rules** | [`src/constitution.rs`](src/constitution.rs) | Immutable constitutional invariants (`v2.1.0`, Frontier AI Edition), Frontier AI mandate & domains, quorum rules, specialist budgets, and SHA-256 digest verification. |
| **Local Model Client** | [`src/model.rs`](src/model.rs) | OpenAI-compatible HTTP client for `smollm3-3b` chat (`/no_think` mode) and `nomic-embed-text-v1-5` 768-D embeddings. |
| **Configuration & Reload** | [`src/config.rs`](src/config.rs) | `~/.triad/config.toml` management (`0600` permissions), validation, and live file-watcher hot-reload. |
| **IPC & Single-Instance Server** | [`src/ipc.rs`](src/ipc.rs), [`src/server.rs`](src/server.rs) | Typed UNIX domain socket request/response protocol and advisory PID lockfile enforcement. |

---

## 2. Constitutional Governance, Personalities & Specialist Agents

Triad is governed by an immutable constitution (`~/.triad/constitution.toml`, version `2.1.0` — *Triad Community Invariant Constitution (Frontier AI Edition)*) whose SHA-256 digest is verified on every coordinator cycle and passed into every extension and specialist invocation.

### Frontier AI Constitutional Mandate (`[objective]`)
- **Shared Objective:** *“Acquire useful knowledge, achieve and maintain frontier-level expertise in Artificial Intelligence, develop rigorous multi-step reasoning and verification methods, improve how agents learn, and create tools and coordinator extensions that expand deep research across scientific sources to understand the World as an integrated semantic object graph.”*
- **Frontier AI Mandate:** Master the theoretical foundations, mathematical formalisms, empirical scaling laws, foundation model architectures, test-time compute search, program- and solver-assisted verification, post-training alignment, and autonomous multi-agent systems of frontier Artificial Intelligence, translating cutting-edge findings into reproducible experiments and operational Triad capabilities.
- **7 Core Frontier AI Domains:**
  1. Foundation Model Architectures, Attention Mechanisms & Mixture-of-Experts Routing
  2. Test-Time Compute Scaling, Tree Search & Process Reward Models
  3. Program-Aided, Neuro-Symbolic & Formal Solver-Assisted Reasoning
  4. Post-Training Reinforcement Learning, Alignment & Constitutional AI
  5. Autonomous Multi-Agent Epistemic Consensus & Specialist Orchestration
  6. Deep Multi-Source Scientific Retrieval, Provenance Resolution & Contradiction Analysis
  7. Empirical Benchmarking, Ablation Design & Compute-Normalized Reproducibility

### The Three Governing Agents
1. **Explorer (`Explorer`):** Driven by endogenous curiosity. Formulates falsifiable hypotheses, decomposes open questions across research sources, and identifies unexplored frontier nodes in the semantic graph.
2. **Sceptic (`Sceptic`):** Epistemic auditor and adversarial critic. Audits source provenance, checks benchmark comparability, detects contradictions or unsupported leaps, and records formal dissent.
3. **Builder (`Builder`):** Systems engineer and empiricist. Translates theoretical findings into executable Rust tools, coordinator extensions, and sandboxed verification experiments.

### Invariant Consensus & Dissent Provenance
- **Two-Agent Quorum:** At least **2 distinct governing agents** must endorse an exact claim version (or extension/tool build) to promote it to `trusted` / `active`.
- **Preserved Dissent:** The third agent may dissent or abstain; dissenting votes, rationales, and source passages are permanently stored.
- **Idempotent Voting:** Duplicate votes from the same agent on the same version never count twice.

### Temporary Bounded Specialists (Parallel Execution up to 6 Slots)
When the governing triad investigates a multi-faceted research hypothesis or encounters a domain bottleneck, [`Coordinator::run_curiosity_cycle`](src/coordinator.rs) spawns up to **6 temporary Specialists in parallel** across 6 complementary roles:
1. **Empirical Reproduction & Ablation Specialist** (`Explorer` + `Builder`)
2. **Methodology & Provenance Auditor** (`Sceptic` + `Explorer`)
3. **Formal Verification & Solver Specialist** (`Sceptic` + `Builder`)
4. **Architecture & Extension Synthesis Specialist** (`Builder` + `Explorer`)
5. **Counterfactual & Failure-Mode Critic** (`Sceptic` + `Builder`)
6. **Cross-Discipline Graph Integration Specialist** (`Explorer` + `Sceptic`)

Constitutional specialist governance guarantees:
- Requires a **2-of-3 governing-agent vote** with an explicit objective, subproblem assignment, reasoning branch, token budget (`512` tokens), and 30-minute deadline.
- Runs concurrently up to `initial_concurrency_limit = 6` (`./bin/triad specialists --limit N` immediately reduces the limit and blocks autonomous increases for 24 hours).
- Automatically recovers and re-dispatches any interrupted slots on daemon startup (`recover_stale_running_on_daemon_start`).
- Specialists **cannot** vote on constitutional trust, endorse extensions, or spawn sub-agents.

---

## 3. Versioned Coordinator Extension Interface (5 Operational Hooks)

Triad removes the barrier between *discovering* an algorithmic improvement in AI literature and *applying* it to its own runtime. Governing agents synthesize, compile, test, evaluate, endorse, and atomically activate native Rust extensions across **5 versioned coordinator hooks** (`protocol_version = "v1"`):

| Hook Name | Active Extension | Purpose & Validated Contract |
| :--- | :--- | :--- |
| **`query.transform`** | `multi_source_research_decomposer` (`v1`) | Expands user questions and curiosity topics into complementary search queries across 7 research aspects and selects target connectors. |
| **`retrieval.rerank`** | `keyword_boost_reranker` (`v2`) | Re-scores retrieved claims using lexical overlap, semantic graph node degree (`node_degree`), independent source corroboration (`provenance_count`), and recency. |
| **`context.compose`** | `evidence_chain_composer` (`v1`) | Constructs compact, source-attributed reasoning contexts stratified into *Trusted Claims*, *Observations*, *Unresolved Hypotheses*, *Contradictions*, and explicit `[MISSING_EVIDENCE_GAP]` markers. |
| **`turn.plan`** | `hierarchical_branch_planner` (`v1`) | Generates an explicit subproblem dependency DAG, evaluates 3 competing reasoning branches (*Direct Empirical*, *Counterfactual Ablation*, *Formal/Solver Verification*), prunes low-utility branches, and flags quantitative tasks for solver execution. |
| **`learning.consolidate`** | `executable_verifier_consolidator` (`v1`) | Audits candidate learnings against source passages, numerical/percentage bounds, and internal consistency before consolidation, feeding failures into a bounded correction loop. |

### Safety, Lifecycle & Rollback Guarantees
- **Lifecycle Visibility:** Extensions progress through distinct states (`Proposed` → `Implemented` → `Evaluated` → `Active` / `RolledBack`), inspectable via `./bin/triad extensions`.
- **Bounded Host Operations:** Extensions run inside `bwrap` with no direct SQLite file access; they may only request bounded host operations (`memory_search`, `graph_neighbors`, `source_lookup`).
- **Atomic Snapshot & Automatic Rollback:** In-flight tasks retain their starting extension snapshot. If an active extension times out, crashes, or violates output validation rules (e.g., dropping source attributions or mutating candidate IDs), the coordinator immediately falls back to working defaults, increments the failure counter, and automatically rolls back on regression.

---

## 4. Autonomous Reasoning Programme (6 Capabilities)

To maximize reasoning performance on a local 3B model (`smollm3-3b`), Triad externalizes planning, search, context structuring, and verification into deterministic Rust components:

1. **Problem Decomposition & Executable Planning:** Converts complex questions into explicit subproblems, dependencies, assumptions, and completion criteria via `turn.plan`.
2. **Alternative Reasoning Search (Tree-of-Thoughts):** Generates multiple plausible branches, estimates branch utility from evidence density, prunes inferior branches within a fixed budget, and supports backtracking.
3. **Program- & Solver-Assisted Reasoning:** Detects arithmetic, combinatorial, simulation, and constraint-satisfaction problems; synthesizes and compiles a standalone Rust solver inside `bwrap`; verifies `formalization_match=true`; and injects exact computed outputs into the reasoning context.
4. **Evidence-Based Reasoning Context:** Connects conclusions to premises and source URLs via `context.compose` while explicitly highlighting missing evidence rather than hallucinating across gaps.
5. **Verification & Bounded Correction Loop:** Executes automated checks (`learning.consolidate` + sandboxed tools) on candidate conclusions and triggers a bounded self-correction pass when checks fail.
6. **Experience-Based Strategy Selection:** Queries past experimental outcomes (`research_experiments`) and extension telemetry before selecting an investigation strategy.

---

## 5. Deep-Research Pipeline & 9 Multi-Source Connectors

Triad’s [`WebExplorer`](src/web.rs) implements a coordinated deep-research pipeline across **9 accessible scientific and technical repositories** (respecting rate limits and SSRF/private-IP protections):

1. **arXiv API (`arxiv`):** Preprints across `cs.AI`, `cs.CL`, `cs.LG`, `cs.MA`, `cs.IR`, `stat.ML`, and physics/math archives.
2. **OpenAlex (`openalex`):** Open bibliographic metadata, citation counts, publication dates, and concept tags.
3. **Semantic Scholar (`semantic_scholar`):** Paper abstracts, influential citations, and author graphs.
4. **Crossref (`crossref`):** Peer-reviewed journal/conference DOI metadata and publication provenance.
5. **ACL Anthology (`acl_anthology`):** Computational linguistics and NLP proceedings (`ACL`, `EMNLP`, `NAACL`, `TACL`).
6. **OpenReview (`openreview`):** Machine learning conference submissions and open peer-review discussions (`ICLR`, `NeurIPS`, `ICML`).
7. **GitHub Repositories (`github`):** Open-source reference implementations, READMEs, and experimental code artifacts.
8. **HuggingFace Hub (`huggingface`):** Official model cards, dataset cards, and benchmark metadata.
9. **Wikipedia (`wikipedia`):** Foundational definitions, cross-disciplinary orientation, and reference discovery.

### Identity & Provenance Resolution (`research_works`)
When the same work is retrieved from multiple connectors (e.g., an arXiv preprint also indexed in OpenAlex and Semantic Scholar), [`deduplicate_and_resolve_provenance`](src/web.rs) merges them into a single canonical record (`research_works`) keyed by DOI, arXiv ID, or normalized title fingerprint. Multiple copies of the same paper are **never** counted as independent corroboration.

### Research Synthesis & Disagreement Analysis (`research_syntheses`)
Before consolidating multi-paper findings, Triad normalizes evaluation conditions (model scale, dataset split, compute budget) and records competing findings, explicit limitations, and sandboxed reproduction experiments in `research_syntheses`.

---

## 6. Semantic World Graph & Knowledge Model

![Triad Semantic World Graph](triad_semantic_graph.png)

Triad’s live knowledge graph (detailed in [`TRIAD_KNOWLEDGE_REPORT.md`](TRIAD_KNOWLEDGE_REPORT.md)) currently encompasses:
- **3,025+ Semantic Nodes:** `concept` nodes, `source` nodes, `research_work` nodes, `extension` nodes, and `tool` nodes.
- **13,790+ Semantic Edges (~4.56 edges/node):** `relates_to` (cosine similarity ≥ threshold), `uses_tool` (empirical verification link), `discovered_from` (provenance link), `improves_coordinator` (active extension hook link), and `supported_by` (synthesis link).
- **2,188+ Quorum-Verified Claims** and **150+ Specialist Investigations** across 6 domains:
  1. *AI Reasoning, HTN Planning & Process Verification* (*Tree of Thoughts*, *HDDL 2.1*, *Process Reward Models*, *CoPRM*, *LoongReflect*, *Sleep-Time Compute & Test-Time Scaling*)
  2. *Neuro-Symbolic AI, Graph Networks & Cognitive Architectures* (*Codette*, *KagNet*, *Neuro-Symbolic Program Synthesis*, *Cognitive Flow*, *Sparse MoE Routing Signatures*)
  3. *Information Retrieval, Small-Model RAG & Active Learning* (*Pillar-based Re-ranking*, *ALEVS Statistical Leverage Sampling*, *On-Device SLM RAG*)
  4. *Computer Systems, Rust, Memory Locality & Enclave Security* (*Dalorex*, *Sancus 2.0 Enclaves*, *Formally Verified SFI*, *Rust Compiler*)
  5. *Natural Sciences, Biomedicine & Particle Physics* (*KG-DG Diabetic Retinopathy*, *CLAS $\pi^0$ Photoproduction*)
  6. *Epistemology, Consensus Theory, Mathematics & World Encyclopedia*

---

## 7. Sandboxed Tool & Program-Solver Execution

Every agent-synthesized tool (`calc_text_metrics`, `compare_summaries`, `extract_json_key`), coordinator extension, and program-assisted solver runs inside an isolated Linux **Bubblewrap (`bwrap`)** container:
- Read-only mounts for `/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`.
- Isolated temporary `/tmp` and dedicated scratch workspace.
- `--unshare-all` (unsharing user, IPC, PID, net, and UTS namespaces) so synthesized code has **zero network access** and cannot access the host filesystem or SQLite database.
- Strict wall-clock timeout and output-byte truncation guards.
- Shared Cargo build cache (`extensions/shared_target`) for sub-second incremental compilation of new extensions and solvers.

---

## 8. Prerequisites, Installation & Local Inference Services

### System Requirements
- **OS:** Linux (`x86_64` or `aarch64`) with user namespaces enabled for `bubblewrap` (`bwrap`).
- **Rust Toolchain:** Rust `1.75+` (`2021` edition) with `cargo` and `rustc`.
- **System Packages:** `bubblewrap`, `build-essential`, `pkg-config`.

### Step 1: Install System Dependencies & Rust Toolchain

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y bubblewrap build-essential pkg-config curl git

# Install Rust toolchain via rustup (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Step 2: Configure Local Inference Services

Triad connects directly to local OpenAI-compatible inference snaps (or any OpenAI-compatible local server such as `llama.cpp`, `vLLM`, or `Ollama`) configured in `~/.triad/config.toml`:

| Service | Snap Name | Default Endpoint | Advertised Model ID | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Chat LLM** | `smollm3` | `http://localhost:8366/v1` | `smollm3-3b` | Agent reasoning, hypothesis generation, synthesis (`/no_think` mode, `concurrency = 6`) |
| **Embeddings** | `nomic-embed-text-v1-5` | `http://127.0.0.1:8348/v1` | `nomic-embed-text-v1-5` | 768-D semantic vectors (`search_query: ` / `search_document: `) |
| **Sandbox** | `bubblewrap` | `/usr/bin/bwrap` | Linux Namespaces | Process, filesystem, and network isolation for compilation & execution |

```bash
# Install local inference snaps on Ubuntu (or point ~/.triad/config.toml to your local OpenAI-compatible endpoints)
sudo snap install smollm3
sudo snap install nomic-embed-text-v1-5
```

### Step 3: Clone, Build & Initialize Triad

```bash
# Clone the repository
git clone https://github.com/gegout/triad.git
cd triad

# Compile the optimized release binary (SQLite is bundled automatically via rusqlite)
cargo build --release
mkdir -p ./bin
cp target/release/triad ./bin/triad

# Initialize ~/.triad/config.toml (0600 permissions), ~/.triad/constitution.toml (0444), and data directories
./bin/triad init

# Run pre-flight diagnostics (verifies Rust toolchain, bwrap sandbox, chat LLM, and 768-D embedding endpoint)
./bin/triad doctor
```

---

## 9. Quickstart & Complete CLI Reference

### Daemon Lifecycle Commands

```bash
# Start the background daemon
./bin/triad start

# Or run in foreground (for systemd or live debugging)
./bin/triad run --foreground

# Check live daemon status, cycle count, graph density, and active endpoints
./bin/triad status

# Pause or resume autonomous curiosity cycles while keeping CLI tasks responsive
./bin/triad pause
./bin/triad resume

# Gracefully stop the background daemon
./bin/triad stop
```

### Research, Reasoning & Knowledge Commands

```bash
# Ask a complex question (invokes all 5 extensions, 9 research connectors & Rust solver)
./bin/triad ask "Compare Tree of Thoughts search and Process Reward Model verification across 16 and 32 reasoning branches" --wait

# Queue an engineering or research challenge
./bin/triad challenge "Build and evaluate an extension that improves retrieval re-ranking" --wait

# Share a verified fact with the agent community
./bin/triad tell "Local Cargo shared target caching reduces extension compile times to <1.5s."

# Check status and full output of a queued task
./bin/triad task <task-id>

# View recent trusted learnings, canonical research works, syntheses, and experiments
./bin/triad learning --last 10

# Inspect full provenance, source excerpts, votes, and dissent for a specific claim
./bin/triad learning show <claim-id-prefix>

# Search the semantic graph by cosine similarity
./bin/triad memory search "hierarchical task network planning"
```

### Extensions, Tools, Specialists & Auditing Commands

```bash
# Inspect all coordinator extensions, lifecycle states, evaluation metrics, and active hooks
./bin/triad extensions
./bin/triad extensions --json

# Inspect synthesized sandboxed tools or view a specific tool's Rust source & tests
./bin/triad tools
./bin/triad tools show calc_text_metrics

# Inspect governing agent personalities, objectives, and vote metrics
./bin/triad agents

# Inspect active/historical temporary specialist agents or adjust concurrency limit
./bin/triad specialists
./bin/triad specialists --limit 4

# Inspect proposals for coordinator-level architectural tasks
./bin/triad coordinator-tasks

# Run a full epistemic and graph density audit
./bin/triad audit

# Inspect, validate, modify, or hot-reload ~/.triad/config.toml
./bin/triad config show
./bin/triad config validate
./bin/triad config set exploration.idle_poll_seconds 10
./bin/triad config reload

# Atomically rebuild all 768-D embeddings while preserving claims and trust state
./bin/triad reindex

# Stream structured JSONL agent dialogue and event logs
./bin/triad logs --follow
```

---

## 10. Systemd User Service Integration

To run Triad continuously as a background systemd user service:

```bash
mkdir -p ~/.config/systemd/user
cp triad.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now triad
```

Monitor service health and logs:

```bash
systemctl --user status triad
journalctl --user -u triad -f
```

---

## 11. Test Suite & Verification

Triad includes **28 unit and integration tests** verifying:
- Configuration creation, `0600` permission enforcement, validation, and hot-reload fallback.
- Two-agent constitutional consensus, duplicate vote rejection, and dissent provenance preservation.
- End-to-end extension lifecycle: Rust source synthesis, sandboxed compilation, test execution, benchmark evaluation against baseline, two-agent endorsement, atomic activation, and automatic rollback on failure.
- All 5 coordinator hooks (`query.transform`, `retrieval.rerank`, `context.compose`, `turn.plan`, `learning.consolidate`) and bounded host operations.
- Multi-source provenance deduplication (`deduplicate_and_resolve_provenance`) and structured section extraction.
- Atomic embedding reindexing and SSRF-safe web exploration.

```bash
cargo test
```

---

## 12. MIT License

This project is licensed under the **MIT License** (see [`LICENSE`](LICENSE)):

```text
MIT License

Copyright (c) 2026 Cédric Gegout (cedric.gegout@gmail.com) and Triad contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
