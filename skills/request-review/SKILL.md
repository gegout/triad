---
name: request-review
description: When and how to seek adversarial criticism from Triad (`review_proposal`) before making a consequential architectural or research decision, and how to act on the findings.
---

# Request Review (`review_proposal`)

Use this skill before committing to a consequential architectural, algorithmic, or experimental design decision. Triad's `review_proposal` tool dispatches your proposal through its adversarial `Sceptic` agent, `Explorer` literature comparison, `Builder` verification pipeline, and deterministic flaw detector to identify concrete weaknesses, counterexamples, missing considerations, and actionable fixes.

## 1. When to Seek Criticism

Call `review_proposal` when:
1. **Designing multi-agent or consensus workflows:** e.g., relying on majority voting, self-consistency, or LLM-as-a-judge without sandboxed executable verification.
2. **Selecting training, routing, or active learning algorithms:** e.g., choosing uniform sampling vs. leverage sampling, outcome rewards vs. process reward models (PRMs), or static vs. hierarchical planning.
3. **Changing evaluation or promotion criteria:** e.g., promoting unverified observations or user feedback directly into trusted knowledge stores.
4. **Making hardware/latency trade-offs under tight budgets:** e.g., deploying a pipeline with unvalidated token budgets, missing fallback paths, or uncalibrated confidence thresholds.

---

## 2. How to Package a Proposal for Review

Provide the complete proposal in `question` and `context`, and explicitly list:
- `objective`: The production or research decision at stake.
- `constraints`: Hard resource, latency, memory, and safety boundaries.
- `previous_attempts`: Alternatives already rejected.
- `success_criteria`: How the proposal claims it will measure success.
- `depth`:
  - Use `"quick"` for rapid structural and epistemic critique (<30s).
  - Use `"standard"` (with `standard_async_default = false` or polling `consultation_status`) when you want sandboxed verification tools (`bayesian_claim_calibrator`, `token_parity_normalizer`, `prm_step_verifier`) to test the proposal's assumptions.

---

## 3. How to Act on `review_proposal` Output

A `review_proposal` response contains four structured critique arrays in addition to `labeled_claims`:
1. `weaknesses`: Structural, algorithmic, or epistemic flaws identified in your proposal.
   - **Action:** Treat every weakness backed by a `retrieved_finding` or `execution_verified_result` as a blocker until addressed.
2. `counterexamples`: Concrete regimes, inputs, or distribution shifts where your proposal fails.
   - **Action:** Add each counterexample as a regression test or boundary check in your implementation.
3. `missing_considerations`: Operational, calibration, or resource factors omitted from your spec.
   - **Action:** Update your design specification to explicitly bound each omitted factor.
4. `concrete_fixes`: Actionable architectural replacements proposed by Triad's `Builder` and `Sceptic` synthesis.
   - **Action:** Adopt the fixes, verify them in your environment, and close the loop with `outcome_feedback` via `consultation_status`.

---

## 4. Concrete Examples

### GOOD Example: Submitting a Detailed Architecture Proposal and Acting on Flaws Found

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "method": "tools/call",
  "params": {
    "name": "review_proposal",
    "arguments": {
      "question": "Proposal: We will replace our sandboxed code verifier with a 3-LLM majority vote (agreement >= 2/3) to mark generated Rust patches as trusted, using uniform random sampling over historical logs to select few-shot prompts and automatically promoting positive user thumbs-up into our trusted knowledge graph.",
      "objective": "Reduce verification latency in our automated patch synthesis pipeline",
      "context": "Local 3B parameter models generating Rust data-processing patches; historical logs have heavy-tailed error distributions",
      "constraints": [
        "Zero silent data corruption in promoted patches",
        "Must run within 16GB VRAM"
      ],
      "previous_attempts": [
        "Full test suite execution took 12 seconds per candidate patch"
      ],
      "success_criteria": [
        "95% patch correctness in production"
      ],
      "depth": "quick"
    }
  }
}
```

**What `review_proposal` catches:**
- **Weakness 1 (Multi-Agent Correlated Failure):** Multi-agent agreement without independent executable or empirical verification is vulnerable to correlated hallucination and shared pretraining biases (`smollm3-3b` agents agreeing on syntactically plausible but buggy code).
- **Weakness 2 (Suboptimal Sampling):** Uniform random sampling over heavy-tailed historical logs misses rare high-leverage edge cases compared to statistical leverage score sampling (`arXiv:1507.04155`).
- **Weakness 3 (Epistemic Contamination):** Automatically promoting unverified user feedback into the trusted knowledge graph violates epistemic governance and enables feedback poisoning.
- **Concrete Fixes Returned:** Keep fast Bubblewrap sandbox execution for compilation/unit-test verification; use leverage-weighted retrieval for prompt selection; store user feedback strictly as `attributable_evaluation_evidence`.

---

### BAD Example: Asking for Generic Approval After the Decision Is Already Made

```json
{
  "jsonrpc": "2.0",
  "id": 97,
  "method": "tools/call",
  "params": {
    "name": "review_proposal",
    "arguments": {
      "question": "We are using agents for our AI system. Looks good, right?",
      "depth": "quick"
    }
  }
}
```

**Why this is bad:**
1. **No falsifiable mechanism:** Without stating how consensus, sampling, verification, or memory promotion works, Triad cannot test failure boundaries or construct concrete counterexamples.
2. **Confirmation bias framing:** `review_proposal` is designed for adversarial critique before a consequential decision, not rubber-stamping vague descriptions.
