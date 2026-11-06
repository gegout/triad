---
name: consult-triad
description: When and how to consult Triad's domain expertise via MCP, how to package context and constraints, and how to interpret epistemic output labels.
---

# Consult Triad (`expert_profile` & `consult_expert`)

Use this skill when interacting with Triad's MCP consultation server (`stdio` or Streamable HTTP at `POST /mcp`) to answer technical questions grounded in Triad's accumulated domain ontology, research syntheses, and verified claims.

## 1. When This Skill Applies

1. **Always inspect coverage first (`expert_profile`)** when entering a new subtopic or domain:
   - Do **not** assume Triad has frontier expertise merely because its domain is configured as AI or because its knowledge graph has thousands of nodes.
   - Check `evaluated_strengths`, `known_weaknesses`, `research_freshness`, `latest_external_evaluation`, and `depth_configuration` in the `expert_profile` output.
2. **Use `consult_expert`** when:
   - Your question is within `domain.scope` and outside `domain.exclusions`.
   - `expert_profile` shows substantive claims and canonical research works covering the topic.
   - You want an answer synthesized from **existing** knowledge and reasoning without waiting for new external literature harvesting.
3. **Do not use `consult_expert`** when:
   - You need criticism of an architectural decision or RFC before committing (`use request-review` -> `review_proposal`).
   - You have a failing system or complex quantitative trade-off requiring sandboxed solver execution (`use escalate-problem` -> `solve_problem`).
   - `expert_profile` reports thin coverage (e.g., 0–6 claims on the subtopic) and you need fresh multi-source literature search (`use escalate-problem` -> `investigate_question`).

---

## 2. How to Package Context

Every Triad consultation tool accepts a standardized seven-field payload:
- `question` *(required)*: The specific technical question to answer.
- `objective`: What decision, implementation, or milestone this answer unlocks.
- `context`: System architecture, model sizes, dataset characteristics, and runtime environment.
- `constraints`: Quantitative bounds (latency p99, VRAM budget, query budget, safety/governance bounds).
- `previous_attempts`: What you have already tried or ruled out, and why it failed.
- `success_criteria`: Measurable conditions that a valid recommendation must satisfy.
- `depth`: `"quick"` (synchronous, <30s target), `"standard"` (adds coordinator hooks & sandboxed verification), or `"deep"` (adds live research & temporary specialists; always async).

> **Routing Rule Awareness:** If you call `consult_expert` with numerical constraints or on a thin topic, Triad's consultation engine may automatically escalate the call to `solve_problem` or `investigate_question` (indicated by `escalated_from: "consult_expert"` and `escalation_reason`). It will **never** exceed the `depth` tier you specified.

---

## 3. How to Read Epistemic Output Labels

Triad labels every claim in `labeled_claims` with one of three strict epistemic labels. Never treat them as interchangeable:

| Label | Meaning | How to Treat It |
|---|---|---|
| `retrieved_finding` | Directly retrieved from a 2-of-3 agent-promoted domain claim, semantic memory vector, or deduplicated canonical research work (`arXiv`, `OpenAlex`, `Semantic Scholar`). | Check `source_ids` against `sources` to inspect the exact paper/claim provenance and limitations excerpt. |
| `expert_synthesis` | Synthesized across Triad's permanent `Explorer`, `Sceptic`, and `Builder` agents from retrieved findings and caller constraints. | Treat as reasoned domain inference. Respect `applicability_conditions`, `assumptions`, and `uncertainty`. |
| `execution_verified_result` | Produced by running a compiled Rust verification tool inside a Bubblewrap (`bwrap`) sandbox or deterministic constraint solver. | Highest operational confidence for the tested inputs, but still bounded by the test payload assumptions. |

Always inspect `uncertainty` (`stated_confidence`, `calibration_note`, `thin_ontology_warnings`) and `unresolved_issues` before acting.

---

## 4. Concrete Examples

### GOOD Example: Inspecting Profile, Packaging Context, and Reading Labels

**Step 1 — Check `expert_profile` first:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "expert_profile",
    "arguments": {}
  }
}
```
*Observation from response:* `evaluated_strengths` shows strong coverage in `Active Learning via Statistical Leverage Sampling (ALEVS)` (`arXiv:1507.04155`) and `Hierarchical Task Network (HDDL 2.1)`. `depth_configuration.tiers.quick.target_seconds` is `30`.

**Step 2 — Call `consult_expert` with complete context and `depth: "quick"`:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "consult_expert",
    "arguments": {
      "question": "When should we switch from uniform random sampling to statistical leverage score sampling (ALEVS) for active learning in linear/kernelized regression, and what are its failure modes?",
      "objective": "Select the query selection policy for our offline sensor calibration pipeline",
      "context": "Streaming feature matrix X with n=250,000 candidate rows and d=64 features; heavy-tailed leverage distribution due to rare transient regimes",
      "constraints": [
        "Labeling budget capped at k=400 oracle queries per batch",
        "Selection overhead must stay under 2 seconds on CPU"
      ],
      "previous_attempts": [
        "Uniform random sampling missed rare high-leverage transient regimes, yielding high worst-case prediction error"
      ],
      "success_criteria": [
        "Bound worst-case L2 parameter estimation error under adversarial noise",
        "Identify exact regime where leverage sampling degrades"
      ],
      "depth": "quick"
    }
  }
}
```

**Step 3 — Properly interpreting the response:**
- Read `labeled_claims`:
  - Claims labeled `retrieved_finding` cite `1507.04155v1` (*Active Learning for Linear Regression via Statistical Leverage Score Sampling*).
  - Claims labeled `expert_synthesis` connect `d=64, k=400` (`k >= d log d`) to the applicability threshold.
- Check `applicability_conditions`: Verify that `k >= d log(d)` (`400 >= 64 * ln(64) ≈ 266`) holds in our system before deploying.

---

### BAD Example: Blindly Calling `consult_expert` with Vague Context and Ignoring Labels

```json
{
  "jsonrpc": "2.0",
  "id": 99,
  "method": "tools/call",
  "params": {
    "name": "consult_expert",
    "arguments": {
      "question": "What is the best sampling method for AI?",
      "depth": "quick"
    }
  }
}
```

**Why this is bad:**
1. **Skipped `expert_profile`:** The caller did not check what domain or subtopics Triad actually covers.
2. **Missing `objective`, `context`, `constraints`, `previous_attempts`, and `success_criteria`:** Without dimensionality `d`, query budget `k`, or noise model, Triad cannot evaluate whether `k >= d log d` holds.
3. **Misreading labels downstream:** Treating an `expert_synthesis` claim as an `execution_verified_result` or ignoring `thin_ontology_warnings` when `domain_substantive_claim_hits == 0` leads to ungrounded deployment decisions.
