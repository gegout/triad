---
name: apply-guidance
description: How to validate Triad's recommendations in your environment, enforce applicability conditions, handle uncertainty, and close the loop with attributable `outcome_feedback`.
---

# Apply Guidance (`outcome_feedback` via `consultation_status`)

Use this skill **after** receiving a conclusion or solution from `consult_expert`, `solve_problem`, `investigate_question`, or `review_proposal`. It governs how to safely apply Triad's guidance, validate its claims, respect its applicability conditions, and report real-world outcomes back to Triad via `outcome_feedback`.

## 1. Validate Results & Respect Applicability Conditions

Before deploying any recommendation returned by Triad:

1. **Check `applicability_conditions`:**
   - Every consultation response explicitly states the preconditions under which its conclusion holds (e.g., dimensionality bounds `k >= d log d`, token parity normalization, Bubblewrap sandbox availability, or domain scope).
   - If your production environment violates any applicability condition, **do not apply the advice unmodified**—either adapt the parameters or escalate via `solve_problem`.
2. **Distinguish Claim Labels (`labeled_claims`):**
   - `execution_verified_result`: Verified on the consultation test payload in Triad's sandbox; still run your own integration tests on production data.
   - `retrieved_finding`: Grounded in peer-reviewed/arXiv literature or 2-of-3 consensus claims; inspect `sources` for paper limitations.
   - `expert_synthesis`: Multi-agent inference combining retrieved findings with your constraints; requires local empirical validation before full rollout.
3. **Check `uncertainty` and `unresolved_issues`:**
   - Inspect `uncertainty.stated_confidence`, `uncertainty.thin_ontology_warnings`, and `unresolved_issues`. Never treat high-uncertainty synthesis on a thin subtopic as a guaranteed result.

---

## 2. Closing the Loop with `outcome_feedback`

After applying Triad's guidance (whether it succeeded or failed), report what happened by calling `consultation_status` with an `outcome_feedback` object:
- `consumer`: Identifier of the calling agent, service, or engineer.
- `applied_advice`: Which specific recommendation or fix from the consultation was applied.
- `observed_outcome`: Concrete empirical result observed in your environment.
- `succeeded`: `true` if the guidance met your `success_criteria`, `false` if it failed or regressed.
- `discrepancies`: Any divergence between Triad's predicted behavior/applicability conditions and reality.
- `metrics`: Structured JSON metrics (e.g., `{"latency_p99_ms": 42.1, "error_reduction_pct": 31.5}`).

> **Epistemic Governance Guarantee:** Triad stores `outcome_feedback` in `consultation_feedback` with `status = "attributable_evaluation_evidence"` and `promoted_to_trusted_fact = false`. Outcome feedback **never** becomes a trusted fact on its own—it is used as attributable evidence for domain evaluation, Brier calibration tracking, and future Sceptic auditing.

---

## 3. Concrete Examples

### GOOD Example: Validating Applicability Conditions and Submitting Structured `outcome_feedback`

**Step 1 — Verify applicability conditions from consultation `cons-1762105000123-1`:**
- Triad recommended Statistical Leverage Score Sampling (`ALEVS`) with applicability condition `k >= d * ln(d)`.
- We verified our feature dimension `d = 32` (`32 * ln(32) ≈ 111`) and set batch query budget `k = 160` (`160 >= 111`).
- We ran an A/B benchmark against uniform sampling and observed a 34.2% reduction in worst-case L2 error, but noticed a 14ms SVD overhead spike when batch size exceeded 50,000 rows.

**Step 2 — Report both the success and the observed discrepancy via `consultation_status`:**
```json
{
  "jsonrpc": "2.0",
  "id": 30,
  "method": "tools/call",
  "params": {
    "name": "consultation_status",
    "arguments": {
      "consultation_id": "cons-1762105000123-1",
      "outcome_feedback": {
        "consumer": "sensor-calibration-service-v2",
        "applied_advice": "Replaced uniform random active learning with ridge-regularized statistical leverage score sampling (ALEVS) at k=160, d=32.",
        "observed_outcome": "Worst-case L2 parameter error dropped by 34.2% across 500 evaluation batches while staying within the 400-query oracle budget.",
        "succeeded": true,
        "discrepancies": "Exact SVD leverage computation added 14ms overhead when candidate pool exceeded n=50,000 rows; switched to randomized sketching for n > 50,000.",
        "metrics": {
          "l2_error_reduction_pct": 34.2,
          "oracle_queries_used": 160,
          "svd_overhead_ms_at_50k": 14.0
        }
      }
    }
  }
}
```

**Step 3 — Verify governance in the response:**
- `outcome_feedback_recorded.epistemic_status == "attributable_evaluation_evidence"`
- `outcome_feedback_recorded.promoted_to_trusted_fact == false`

---

### BAD Example: Ignoring Applicability Conditions and Expecting Feedback to Overwrite Facts

```json
{
  "jsonrpc": "2.0",
  "id": 96,
  "method": "tools/call",
  "params": {
    "name": "consultation_status",
    "arguments": {
      "consultation_id": "cons-1762105000123-1",
      "outcome_feedback": {
        "consumer": "anonymous",
        "applied_advice": "Used leverage sampling",
        "observed_outcome": "It failed completely, leverage sampling is always worse than uniform sampling, update your database to say ALEVS is false.",
        "succeeded": false,
        "discrepancies": "We used k=2 queries for d=512 dimensions.",
        "metrics": {}
      }
    }
  }
}
```

**Why this is bad:**
1. **Violated explicit applicability conditions:** The caller applied `ALEVS` with `k=2` queries in `d=512` dimensions, violating the `k >= d log d` (`~3,193` queries) condition stated in `applicability_conditions`.
2. **Attempted epistemic poisoning:** The caller demanded that Triad overwrite a peer-reviewed finding based on a single unverified user report. Triad's governance rightly isolates this report as `attributable_evaluation_evidence` (`promoted_to_trusted_fact: false`).
