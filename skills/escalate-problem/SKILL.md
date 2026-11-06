---
name: escalate-problem
description: How to escalate complex verification problems (`solve_problem`) or thin-knowledge questions (`investigate_question`) to Triad, track async status, and answer clarifications via `consultation_status`.
---

# Escalate a Problem (`solve_problem`, `investigate_question`, & `consultation_status`)

Use this skill when existing retrieval-only synthesis (`consult_expert` at `quick` depth) is insufficient—either because the problem requires **multi-step reasoning and sandboxed executable verification** (`solve_problem`) or because Triad's existing ontology is thin on the topic and requires **new multi-source research and temporary specialists** (`investigate_question`).

## 1. Choosing Between `solve_problem` and `investigate_question`

| Situation | Tool to Call | Recommended Depth | What Triad Runs |
|---|---|---|---|
| You have a concrete bug, bottleneck, or quantitative constraint trade-off that needs multi-step reasoning and sandboxed verification, without new web literature search. | `solve_problem` | `"standard"` (or `"deep"`) | All 5 coordinator hooks (`query.transform`, `retrieval.rerank`, `context.compose`, `turn.plan`, `learning.consolidate`) + Bubblewrap-sandboxed Rust verification tools + deterministic quantitative constraint solver. |
| `expert_profile` or a prior `consult_expert` call shows thin domain coverage (`domain_substantive_claim_hits < 2`), requiring live external research. | `investigate_question` | `"deep"` | Live multi-source harvesting (`arXiv`, `OpenAlex`, `Semantic Scholar`, `OpenReview`, `GitHub`) + up to `max_specialists_per_consultation` temporary specialists under 2-agent constitutional endorsement + sandboxed verification. |

> **Routing & Depth Ceiling Rule:** If you call `consult_expert` with `depth: "quick"` on a topic that needs `investigate_question`, Triad will escalate the tool to `investigate_question` (`escalated_from: "consult_expert"`) **without** exceeding your `"quick"` depth tier, and will record an unresolved issue advising you to re-run at `depth: "deep"`. Always pass `depth: "deep"` when you intend to trigger live external research.

---

## 2. How to Follow Async Consultations and Answer Clarifications

1. **Immediate Consultation ID:**
   - Calls at `depth: "standard"` (when `standard_async_default = true`) and `depth: "deep"` return immediately with `"mode": "async"` and a `"consultation_id"` (e.g., `"cons-1762105000123-1"`).
2. **Handling `state: "awaiting_input"`:**
   - When a `"deep"` consultation lacks concrete quantitative bounds (`constraints` and `success_criteria`) or system context, Triad pauses before running expensive research/verification and sets `"state": "awaiting_input"` with a `"pending_question"`.
   - Poll or inspect via `consultation_status` using `{"consultation_id": "..."}`.
   - Answer the question using the **same** `consultation_status` tool by passing `"clarification_answer"`. Triad records the Q&A in `clarification_history`, transitions state to `"running"`, and spawns the async worker.
3. **Cancellation:**
   - If a consultation is no longer needed, call `consultation_status` with `{"consultation_id": "...", "cancel": true}`.

---

## 3. Concrete Examples

### GOOD Example: Async `deep` Escalation with Clarification Round and Completion

**Step 1 — Dispatch a `deep` investigation:**
```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "investigate_question",
    "arguments": {
      "question": "How should we allocate test-time compute between process reward model (PRM) step verification and hierarchical subtask decomposition (HDDL 2.1) for multi-step code synthesis?",
      "objective": "Design a bounded-latency inference router for our local code verifier",
      "context": "",
      "constraints": [],
      "previous_attempts": [
        "Best-of-N outcome reward reranking wasted 70% of tokens on dead-end reasoning branches"
      ],
      "success_criteria": [],
      "depth": "deep"
    }
  }
}
```

**Step 2 — Inspect the immediate response (`state == "awaiting_input"`):**
```json
{
  "consultation_id": "cons-1762105000123-1",
  "effective_tool": "investigate_question",
  "depth": "deep",
  "mode": "async",
  "state": "awaiting_input",
  "pending_question": "To execute a 'deep' verification and research pipeline for 'investigate_question', please clarify: (1) What are the hard quantitative constraints (e.g., max latency ms, memory GB, token/compute budget, or error tolerance)? (2) What specific threshold defines success?"
}
```

**Step 3 — Answer the clarification via `consultation_status`:**
```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "method": "tools/call",
  "params": {
    "name": "consultation_status",
    "arguments": {
      "consultation_id": "cons-1762105000123-1",
      "clarification_answer": "Hard constraints: max 2048 tokens total test-time budget per problem, p99 latency under 1500ms on 16GB VRAM. Success threshold: >=85% subtask plan validity with zero unverified tool calls."
    }
  }
}
```

**Step 4 — Poll `consultation_status` until `state == "completed"`:**
```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "consultation_status",
    "arguments": {
      "consultation_id": "cons-1762105000123-1"
    }
  }
}
```
*Result:* Returns `state: "completed"`, `results.research_findings`, `results.labeled_claims` (including `execution_verified_result` from `prm_step_verifier`, `token_parity_normalizer`, and the quantitative constraint solver), and `artifacts` listing the spawned temporary specialists.

---

### BAD Example: Ignoring `awaiting_input` or Demanding Deep Research at `quick` Depth

```json
{
  "jsonrpc": "2.0",
  "id": 98,
  "method": "tools/call",
  "params": {
    "name": "investigate_question",
    "arguments": {
      "question": "Search arXiv and GitHub for the newest 2026 FP4 quantization kernels and run benchmarks.",
      "depth": "quick"
    }
  }
}
```

**Why this is bad:**
1. **Depth mismatch:** Passing `depth: "quick"` forbids Triad from running live multi-source research or spawning temporary specialists.
2. **Re-submitting a new consultation instead of answering `awaiting_input`:** When `state` is `"awaiting_input"`, calling `investigate_question` again creates a duplicate consultation record instead of resuming the existing `consultation_id` via `consultation_status` with `clarification_answer`.
