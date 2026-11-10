use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::config::DomainSection;
use crate::execution::ExecutionAdapter;
use crate::store::{
    ExperimentSpecification, ResearchExperimentRecord, Store, ToolExecutionRecord,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProposal {
    pub name: String,
    pub description: String,
    pub schema_json: String,
    pub rust_source: String,
    pub test_code: String,
    #[serde(default)]
    pub experiment_spec: Option<ExperimentSpecification>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScepticToolReview {
    pub approved: bool,
    pub under_specified: bool,
    pub source_fidelity_status: String,
    pub missing_source_details: Vec<String>,
    pub rejection_reasons: Vec<String>,
    pub verification_notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchExperimentOutcome {
    pub experiment_record: ResearchExperimentRecord,
    pub execution_record: ToolExecutionRecord,
    pub specification: ExperimentSpecification,
    pub measured_results: serde_json::Value,
    pub baseline_comparison: serde_json::Value,
    pub held_out_evaluation: serde_json::Value,
    pub epistemic_note: String,
}

pub struct ToolManager {
    store: Store,
    workspace_dir: PathBuf,
    execution_adapter: Arc<dyn ExecutionAdapter>,
}

impl ToolManager {
    pub fn new(
        store: Store,
        workspace_dir: PathBuf,
        execution_adapter: Arc<dyn ExecutionAdapter>,
    ) -> Self {
        Self {
            store,
            workspace_dir,
            execution_adapter,
        }
    }

    /// Detects superficial verifier patterns in generated Rust tool code.
    /// Rejects tools that count words/characters/lines, inspect strings/substrings,
    /// print predefined conclusions, or emit arbitrary constant scores instead of
    /// testing the declared hypothesis through real computation.
    pub fn detect_superficial_verifier_patterns(rust_source: &str) -> Vec<String> {
        let mut reasons = Vec::new();
        let lower = rust_source.to_lowercase();

        // 1. Reject tools that count words, tokens, or characters as verification
        if lower.contains("split_whitespace().count()")
            || lower.contains("split_ascii_whitespace().count()")
            || (lower.contains("word_count") && !lower.contains("matrix"))
            || (lower.contains(".chars().count()") && lower.contains("verif"))
        {
            reasons.push(
                "Superficial verifier rejected: tool counts words or characters (split_whitespace().count() / word_count) instead of performing real hypothesis computation".to_string(),
            );
        }

        let normalized = lower.replace("\\\"", "\"");

        // 2. Reject tools that inspect strings/substrings for keywords or citation tags
        let string_inspection_markers = [
            "contains(\"arxiv\")",
            "contains(\"doi\")",
            "contains(\"http\")",
            "contains(\"verified\")",
            "contains(\"proven\")",
            "contains(\"hypothesis\")",
            "score_citations",
            "string_match_verifier",
            "substring_check",
        ];
        for marker in string_inspection_markers {
            if normalized.contains(marker) {
                reasons.push(format!(
                    "Superficial verifier rejected: tool inspects strings/substrings (`{}`) instead of executing an empirical or mathematical test",
                    marker
                ));
                break;
            }
        }

        // 3. Reject tools that print predefined conclusions
        let predefined_conclusion_markers = [
            "predefined_conclusion",
            "hypothesis_confirmed_by_default",
            "println!(\"hypothesis confirmed\")",
            "println!(\"verified: true\")",
            "println!(\"qed\")",
            "\"conclusion\": \"proven\"",
        ];
        for marker in predefined_conclusion_markers {
            if normalized.contains(marker) {
                reasons.push(format!(
                    "Superficial verifier rejected: tool prints or emits a predefined conclusion (`{}`)",
                    marker
                ));
                break;
            }
        }

        // 4. Reject tools that emit arbitrary constant scores
        let arbitrary_score_markers = [
            "routing_entropy_bits: 2.85",
            "\"routing_entropy_bits\": 2.85",
            "step_consistency_score: 0.92",
            "\"step_consistency_score\": 0.92",
            "sha256_mock_len: 64",
            "\"sha256_mock_len\": 64",
            "arbitrary_score",
            "hardcoded_score",
            "let score = 0.95;",
            "let accuracy = 0.99;",
        ];
        for marker in arbitrary_score_markers {
            if normalized.contains(marker) {
                reasons.push(format!(
                    "Superficial verifier rejected: tool emits an arbitrary or hardcoded constant score (`{}`)",
                    marker
                ));
                break;
            }
        }

        reasons
    }

    /// Validates that an `ExperimentSpecification` contains all required pre-implementation fields:
    /// research question, falsifiable hypothesis, mechanism under test, inputs, assumptions,
    /// baseline, controls, metrics with declared result keys and predefined success criteria,
    /// budget, seeds, reproducibility requirements, expected outputs, and known limitations.
    /// Also checks domain-configured executable check kinds and source fidelity requirements.
    pub fn validate_experiment_specification(
        spec: &ExperimentSpecification,
        domain_cfg: Option<&DomainSection>,
    ) -> Vec<String> {
        let mut errors = Vec::new();

        if spec.research_question.trim().is_empty() {
            errors.push("Experiment specification missing research_question".to_string());
        }
        if spec.falsifiable_hypothesis.trim().is_empty() {
            errors.push("Experiment specification missing falsifiable_hypothesis".to_string());
        }
        if spec.mechanism_under_test.trim().is_empty() {
            errors.push("Experiment specification missing mechanism_under_test".to_string());
        }
        if spec.inputs_description.trim().is_empty() {
            errors.push("Experiment specification missing inputs_description".to_string());
        }
        if spec.assumptions.is_empty() || spec.assumptions.iter().all(|s| s.trim().is_empty()) {
            errors.push("Experiment specification requires at least one explicit assumption".to_string());
        }
        if spec.baseline_description.trim().is_empty() {
            errors.push("Experiment specification missing baseline_description for comparison".to_string());
        }
        if spec.controls.is_empty() || spec.controls.iter().all(|s| s.trim().is_empty()) {
            errors.push("Experiment specification requires at least one experimental control".to_string());
        }
        if spec.metrics.is_empty() {
            errors.push("Experiment specification requires at least one declared metric with result_key and success_criterion".to_string());
        } else {
            for (idx, m) in spec.metrics.iter().enumerate() {
                if m.name.trim().is_empty() {
                    errors.push(format!("Metric #{} has an empty name", idx + 1));
                }
                if m.result_key.trim().is_empty() {
                    errors.push(format!(
                        "Metric '{}' has an empty result_key",
                        m.name
                    ));
                }
                if m.success_criterion.trim().is_empty() {
                    errors.push(format!(
                        "Metric '{}' (key '{}') is missing a predefined success_criterion",
                        m.name, m.result_key
                    ));
                }
            }
        }
        if spec.budget_max_runtime_secs == 0 {
            errors.push("Experiment specification budget_max_runtime_secs must be > 0".to_string());
        }
        if spec.budget_max_memory_mb == 0 {
            errors.push("Experiment specification budget_max_memory_mb must be > 0".to_string());
        }
        if spec.seeds.is_empty() {
            errors.push("Experiment specification requires explicit random/evaluation seeds for reproducibility".to_string());
        }
        if spec.reproducibility_requirements.trim().is_empty() {
            errors.push("Experiment specification missing reproducibility_requirements".to_string());
        }
        if spec.expected_outputs.is_empty() {
            errors.push("Experiment specification missing expected_outputs".to_string());
        }
        if spec.known_limitations.is_empty() {
            errors.push("Experiment specification missing known_limitations".to_string());
        }

        // Check executable_check_kind against domain config (keeping the coordinator domain-independent)
        let check_kind = spec.executable_check_kind.trim();
        if check_kind.is_empty() {
            errors.push("Experiment specification missing executable_check_kind".to_string());
        } else if let Some(dcfg) = domain_cfg {
            if !dcfg.executable_checks.is_empty()
                && !dcfg
                    .executable_checks
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(check_kind))
            {
                errors.push(format!(
                    "Executable check kind '{}' is not enabled in domain.executable_checks ({:?})",
                    check_kind, dcfg.executable_checks
                ));
            }
        }

        // If flagged under_specified due to abstract-only or low-fidelity sources, missing_source_details must be populated
        if spec.under_specified && spec.missing_source_details.is_empty() {
            errors.push(
                "Experiment is flagged under_specified (abstract-only or low-fidelity source), but missing_source_details does not specify which method/algorithm/equation details are missing".to_string(),
            );
        }

        errors
    }

    /// Builder implements; Sceptic verifies the computation actually tests the hypothesis.
    /// Their approval supplements, never replaces, execution evidence.
    pub fn review_research_tool(
        proposal: &ToolProposal,
        spec: &ExperimentSpecification,
        domain_cfg: Option<&DomainSection>,
    ) -> ScepticToolReview {
        let mut rejection_reasons = Self::detect_superficial_verifier_patterns(&proposal.rust_source);
        rejection_reasons.extend(Self::validate_experiment_specification(spec, domain_cfg));

        let src = &proposal.rust_source;
        // Verify the tool writes a structured results JSON file (`results.json` or `TRIAD_RESULTS_FILE`)
        let writes_results_file = src.contains("TRIAD_RESULTS_FILE") || src.contains("results.json");
        let writes_to_fs = src.contains("fs::write")
            || src.contains("std::fs::write")
            || src.contains("File::create")
            || src.contains("OpenOptions");
        if !writes_results_file || !writes_to_fs {
            rejection_reasons.push(
                "Generated research tool must write a structured JSON results file (`results.json` / `TRIAD_RESULTS_FILE`); stdout-only output is rejected".to_string(),
            );
        }

        // Verify every declared metric `result_key` in the specification is produced by the tool code
        for metric in &spec.metrics {
            let rk = metric.result_key.trim();
            if !rk.is_empty() && !src.contains(rk) {
                rejection_reasons.push(format!(
                    "Generated tool source does not emit declared specification metric result_key '{}'",
                    rk
                ));
            }
        }

        // Verify the tool performs substantive computation (iteration, mathematical/numerical operations, or data evaluation)
        let has_computation = src.contains("for ")
            || src.contains("while ")
            || src.contains(".iter()")
            || src.contains(".map(")
            || src.contains(".fold(");
        let has_numeric_ops = src.contains("f64")
            || src.contains("usize")
            || src.contains("i64")
            || src.contains("u64");
        if !has_computation || !has_numeric_ops {
            rejection_reasons.push(
                "Generated research tool lacks substantive iterative or numerical computation to test the declared hypothesis".to_string(),
            );
        }

        let approved = rejection_reasons.is_empty();
        let verification_notes = if approved {
            if spec.under_specified {
                format!(
                    "Sceptic verified tool '{}' implements '{}' computation and writes declared metric keys {:?} to results.json, but flagged experiment UNDER-SPECIFIED ({}) with missing details: {:?}",
                    proposal.name,
                    spec.executable_check_kind,
                    spec.metrics.iter().map(|m| m.result_key.as_str()).collect::<Vec<_>>(),
                    spec.source_fidelity_status,
                    spec.missing_source_details
                )
            } else {
                format!(
                    "Sceptic verified tool '{}' tests hypothesis via '{}' against baseline ('{}') and writes declared metric keys {:?} to results.json (source_fidelity={}). Approval supplements, never replaces, execution evidence.",
                    proposal.name,
                    spec.executable_check_kind,
                    spec.baseline_description,
                    spec.metrics.iter().map(|m| m.result_key.as_str()).collect::<Vec<_>>(),
                    spec.source_fidelity_status
                )
            }
        } else {
            format!(
                "Sceptic rejected tool '{}': {}",
                proposal.name,
                rejection_reasons.join(" | ")
            )
        };

        ScepticToolReview {
            approved,
            under_specified: spec.under_specified,
            source_fidelity_status: spec.source_fidelity_status.clone(),
            missing_source_details: spec.missing_source_details.clone(),
            rejection_reasons,
            verification_notes,
        }
    }

    /// Complete lifecycle for a spec-backed research tool:
    /// Validate & Register Specification (including source fidelity check) -> Sceptic Review ->
    /// Generate Source -> Compile -> Run Tests -> Endorse -> Register with Spec ID.
    pub async fn build_and_register_research_tool(
        &self,
        proposal: &ToolProposal,
        mut spec: ExperimentSpecification,
        builder_agent: &str,
        sceptic_agent: &str,
        domain_cfg: Option<&DomainSection>,
    ) -> Result<(i64, ExperimentSpecification, ScepticToolReview)> {
        let valid_agents = ["Explorer", "Sceptic", "Builder"];
        if !valid_agents.contains(&builder_agent) || !valid_agents.contains(&sceptic_agent) {
            bail!("Constitutional invariant violation: endorsers must be among Explorer, Sceptic, Builder");
        }
        if builder_agent == sceptic_agent {
            bail!("Constitutional invariant violation: tool endorsement requires two distinct agent identities");
        }

        spec.builder_agent = builder_agent.to_string();
        spec.sceptic_agent = sceptic_agent.to_string();

        // 1. Validate source fidelity and persist the ExperimentSpecification
        let validated_spec = self.store.register_and_validate_experiment_spec(&mut spec)?;

        // 2. Sceptic verifies the computation actually tests the hypothesis
        let review = Self::review_research_tool(proposal, &validated_spec, domain_cfg);
        let review_status = if review.approved {
            if review.under_specified {
                "approved_under_specified"
            } else {
                "approved"
            }
        } else {
            "rejected"
        };
        let _ = self.store.update_experiment_spec_review(
            &validated_spec.id,
            review_status,
            &review.verification_notes,
        );

        if !review.approved {
            bail!(
                "Sceptic review failed for research tool '{}': {}",
                proposal.name,
                review.rejection_reasons.join("; ")
            );
        }

        if !self.execution_adapter.is_available() {
            bail!("Cannot build or register tools: execution isolation adapter is unavailable");
        }

        let tools_root = self.workspace_dir.join("tools");
        let tools_ws = tools_root.join(&proposal.name);
        let shared_target = tools_root.join("shared_target");
        fs::create_dir_all(&tools_ws)
            .with_context(|| format!("Failed to create tool workspace at {}", tools_ws.display()))?;
        fs::create_dir_all(&shared_target)
            .with_context(|| format!("Failed to create shared tool target at {}", shared_target.display()))?;

        let cargo_toml = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

[profile.release]
strip = true
"#,
            proposal.name
        );

        let manifest_path = tools_ws.join("Cargo.toml");
        fs::write(&manifest_path, cargo_toml)?;
        let src_dir = tools_ws.join("src");
        fs::create_dir_all(&src_dir)?;

        let full_source = format!(
            "{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{}\n}}\n",
            proposal.rust_source, proposal.test_code
        );
        fs::write(src_dir.join("main.rs"), &full_source)?;

        let mut hasher = Sha256::new();
        hasher.update(full_source.as_bytes());
        let hash = hasher.finalize();
        let source_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

        let manifest_str = manifest_path.to_str().unwrap_or_default();
        let target_str = shared_target.to_str().unwrap_or_default();

        let test_res = self
            .execution_adapter
            .execute_isolated(
                "cargo",
                &["test", "--release", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
                None,
                &tools_root,
                Duration::from_secs(120),
            )
            .await;

        let test_passed = match test_res {
            Ok(ref res) => {
                if res.exit_code != 0 {
                    eprintln!(
                        "Tool test exited with code {}:\nSTDOUT:\n{}\nSTDERR:\n{}",
                        res.exit_code, res.stdout, res.stderr
                    );
                }
                res.exit_code == 0
            }
            Err(ref e) => {
                eprintln!("Tool test failed execution: {}", e);
                false
            }
        };

        if !test_passed {
            bail!("Tool tests failed; registration aborted. Repair required.");
        }

        let build_res = self
            .execution_adapter
            .execute_isolated(
                "cargo",
                &["build", "--release", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
                None,
                &tools_root,
                Duration::from_secs(120),
            )
            .await?;

        if build_res.exit_code != 0 {
            bail!("Tool compilation failed: {}", build_res.stderr);
        }

        let shared_bin_path = shared_target.join("release").join(&proposal.name);
        let local_release_dir = tools_ws.join("target").join("release");
        fs::create_dir_all(&local_release_dir)?;
        let bin_path = local_release_dir.join(&proposal.name);
        if shared_bin_path.exists() {
            let _ = fs::copy(&shared_bin_path, &bin_path);
            let _ = fs::remove_file(&shared_bin_path);
            let _ = fs::remove_file(shared_target.join("release").join(format!("{}.d", proposal.name)));
        }
        prune_crate_artifacts_from_shared_target(&shared_target, &proposal.name);
        let _ = fs::remove_dir_all(tools_ws.join("target").join("debug"));
        let _ = fs::remove_dir_all(local_release_dir.join("deps"));
        let _ = fs::remove_dir_all(local_release_dir.join("build"));
        let _ = fs::remove_dir_all(local_release_dir.join("incremental"));
        let bin_str = bin_path.to_str().unwrap_or_default();

        let next_version = match self.store.get_tool_by_name(&proposal.name)? {
            Some(existing) => existing.version + 1,
            None => 1,
        };

        let tool_id = self.store.register_tool_with_spec(
            &proposal.name,
            next_version,
            &proposal.description,
            &proposal.schema_json,
            &full_source,
            bin_str,
            &source_hash,
            Some(builder_agent),
            Some(sceptic_agent),
            test_passed,
            Some(&validated_spec.id),
            Some(&review.verification_notes),
        )?;

        let final_spec = self
            .store
            .get_experiment_spec(&validated_spec.id)?
            .unwrap_or(validated_spec);
        Ok((tool_id, final_spec, review))
    }

    /// Complete lifecycle:
    /// Contract -> Sceptic Superficial-Verifier Check -> Generate Source -> Compile -> Run Tests -> Endorse -> Register -> Execute
    pub async fn build_and_register_tool(
        &self,
        proposal: &ToolProposal,
        endorser_1: &str,
        endorser_2: &str,
    ) -> Result<i64> {
        // Always reject superficial verifier tools (word counting, string inspection, predefined conclusions, arbitrary constant scores)
        let superficial_reasons = Self::detect_superficial_verifier_patterns(&proposal.rust_source);
        if !superficial_reasons.is_empty() {
            bail!(
                "Sceptic rejected superficial verifier tool '{}': {}",
                proposal.name,
                superficial_reasons.join("; ")
            );
        }

        // If an ExperimentSpecification is attached to the proposal, run the full research-tool review and registration pipeline
        if let Some(ref spec) = proposal.experiment_spec {
            let (tool_id, _, _) = self
                .build_and_register_research_tool(proposal, spec.clone(), endorser_1, endorser_2, None)
                .await?;
            return Ok(tool_id);
        }

        let valid_agents = ["Explorer", "Sceptic", "Builder"];
        if !valid_agents.contains(&endorser_1) || !valid_agents.contains(&endorser_2) {
            bail!("Constitutional invariant violation: endorsers must be among Explorer, Sceptic, Builder");
        }
        if endorser_1 == endorser_2 {
            bail!("Constitutional invariant violation: tool endorsement requires two distinct agent identities");
        }

        if !self.execution_adapter.is_available() {
            bail!("Cannot build or register tools: execution isolation adapter is unavailable");
        }

        let tools_root = self.workspace_dir.join("tools");
        let tools_ws = tools_root.join(&proposal.name);
        let shared_target = tools_root.join("shared_target");
        fs::create_dir_all(&tools_ws)
            .with_context(|| format!("Failed to create tool workspace at {}", tools_ws.display()))?;
        fs::create_dir_all(&shared_target)
            .with_context(|| format!("Failed to create shared tool target at {}", shared_target.display()))?;

        // Write Cargo.toml for standalone tool
        let cargo_toml = format!(r#"[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

[profile.release]
strip = true
"#, proposal.name);

        let manifest_path = tools_ws.join("Cargo.toml");
        fs::write(&manifest_path, cargo_toml)?;
        let src_dir = tools_ws.join("src");
        fs::create_dir_all(&src_dir)?;

        // Combine rust source with tests
        let full_source = format!("{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{}\n}}\n",
            proposal.rust_source, proposal.test_code);
        fs::write(src_dir.join("main.rs"), &full_source)?;

        // Compute source hash
        let mut hasher = Sha256::new();
        hasher.update(full_source.as_bytes());
        let hash = hasher.finalize();
        let source_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

        let manifest_str = manifest_path.to_str().unwrap_or_default();
        let target_str = shared_target.to_str().unwrap_or_default();

        // Run cargo test --release in isolated container sharing target directory
        let test_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["test", "--release", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
            None,
            &tools_root,
            Duration::from_secs(120),
        ).await;

        let test_passed = match test_res {
            Ok(ref res) => {
                if res.exit_code != 0 {
                    eprintln!("Tool test exited with code {}:\nSTDOUT:\n{}\nSTDERR:\n{}", res.exit_code, res.stdout, res.stderr);
                }
                res.exit_code == 0
            }
            Err(ref e) => {
                eprintln!("Tool test failed execution: {}", e);
                false
            }
        };

        if !test_passed {
            bail!("Tool tests failed; registration aborted. Repair required.");
        }

        // Build release binary sharing target directory
        let build_res = self.execution_adapter.execute_isolated(
            "cargo",
            &["build", "--release", "--quiet", "--manifest-path", manifest_str, "--target-dir", target_str],
            None,
            &tools_root,
            Duration::from_secs(120),
        ).await?;

        if build_res.exit_code != 0 {
            bail!("Tool compilation failed: {}", build_res.stderr);
        }

        let shared_bin_path = shared_target.join("release").join(&proposal.name);
        let local_release_dir = tools_ws.join("target").join("release");
        fs::create_dir_all(&local_release_dir)?;
        let bin_path = local_release_dir.join(&proposal.name);
        if shared_bin_path.exists() {
            let _ = fs::copy(&shared_bin_path, &bin_path);
            let _ = fs::remove_file(&shared_bin_path);
            let _ = fs::remove_file(shared_target.join("release").join(format!("{}.d", proposal.name)));
        }
        prune_crate_artifacts_from_shared_target(&shared_target, &proposal.name);
        let _ = fs::remove_dir_all(tools_ws.join("target").join("debug"));
        let _ = fs::remove_dir_all(local_release_dir.join("deps"));
        let _ = fs::remove_dir_all(local_release_dir.join("build"));
        let _ = fs::remove_dir_all(local_release_dir.join("incremental"));
        let bin_str = bin_path.to_str().unwrap_or_default();

        let next_version = match self.store.get_tool_by_name(&proposal.name)? {
            Some(existing) => existing.version + 1,
            None => 1,
        };

        let tool_id = self.store.register_tool(
            &proposal.name,
            next_version,
            &proposal.description,
            &proposal.schema_json,
            &full_source,
            bin_str,
            &source_hash,
            Some(endorser_1),
            Some(endorser_2),
            test_passed,
        )?;

        Ok(tool_id)
    }

    pub async fn execute_tool(&self, name: &str, input_json: &str) -> Result<String> {
        let rec = self
            .execute_tool_with_evidence(name, input_json, None, None)
            .await?;
        if rec.exit_code == 0 && rec.error.is_none() {
            Ok(rec.output_json.unwrap_or_default())
        } else {
            bail!(
                "Tool exited with code {}: {}",
                rec.exit_code,
                rec.error.unwrap_or_default()
            )
        }
    }

    /// Executes a registered tool in the isolated sandbox with an explicit timeout and records
    /// an immutable `ToolExecutionRecord` in `tool_executions`.
    /// When the tool has an associated `ExperimentSpecification`, the host strictly requires
    /// the tool to write a valid JSON `results.json` file whose keys include every metric
    /// `result_key` declared in the specification. Missing, malformed, or incomplete `results.json`
    /// files are rejected by the host!
    pub async fn execute_tool_with_evidence(
        &self,
        name: &str,
        input_json: &str,
        seed: Option<i64>,
        specialist_id: Option<&str>,
    ) -> Result<ToolExecutionRecord> {
        let tool = self
            .store
            .get_tool_by_name(name)?
            .context(format!("Tool '{}' not found in registry", name))?;

        let exp_spec: Option<ExperimentSpecification> = tool
            .experiment_spec_id
            .as_deref()
            .and_then(|sid| self.store.get_experiment_spec(sid).ok().flatten());

        let timeout_secs = exp_spec
            .as_ref()
            .map(|s| s.budget_max_runtime_secs.max(1))
            .unwrap_or(30);

        self.execute_tool_with_timeout(
            name,
            input_json,
            seed,
            specialist_id,
            Duration::from_secs(timeout_secs),
        )
        .await
    }

    pub async fn execute_tool_with_timeout(
        &self,
        name: &str,
        input_json: &str,
        seed: Option<i64>,
        specialist_id: Option<&str>,
        timeout: Duration,
    ) -> Result<ToolExecutionRecord> {
        let tool = self
            .store
            .get_tool_by_name(name)?
            .context(format!("Tool '{}' not found in registry", name))?;

        if tool.status != "active" {
            bail!("Tool '{}' is not active (status: {})", name, tool.status);
        }

        let bin_path = Path::new(&tool.binary_path);
        if !bin_path.exists() {
            bail!("Tool binary does not exist at {}", tool.binary_path);
        }

        let exp_spec: Option<ExperimentSpecification> = tool
            .experiment_spec_id
            .as_deref()
            .and_then(|sid| self.store.get_experiment_spec(sid).ok().flatten());

        let effective_seed = seed
            .or_else(|| {
                serde_json::from_str::<serde_json::Value>(input_json)
                    .ok()
                    .and_then(|v| v.get("seed").and_then(|s| s.as_i64()))
            })
            .or_else(|| {
                exp_spec
                    .as_ref()
                    .and_then(|s| s.seeds.first().copied())
            });

        let dependencies_json = r#"["serde@1.0","serde_json@1.0"]"#;

        // Create an isolated per-run workspace directory so `results.json` is never stale
        let run_nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let run_ws = self
            .workspace_dir
            .join("runs")
            .join(format!("{}_{}", name, run_nonce));
        fs::create_dir_all(&run_ws)
            .with_context(|| format!("Failed to create per-run workspace at {}", run_ws.display()))?;
        let results_file_path = run_ws.join("results.json");

        let start = std::time::Instant::now();
        let res = self
            .execution_adapter
            .execute_isolated(&tool.binary_path, &[], Some(input_json), &run_ws, timeout)
            .await;

        let elapsed = start.elapsed().as_millis() as u64;

        match res {
            Ok(out) => {
                let mut artifact_entries = vec![serde_json::json!({
                    "kind": "tool_binary",
                    "path": tool.binary_path,
                    "source_hash": tool.source_hash
                })];

                // Inspect structured `results.json` written by the tool
                let results_file_exists = results_file_path.exists();
                let results_file_raw = if results_file_exists {
                    fs::read_to_string(&results_file_path).ok()
                } else {
                    None
                };

                if results_file_exists {
                    artifact_entries.push(serde_json::json!({
                        "kind": "structured_results_json",
                        "path": results_file_path.to_string_lossy()
                    }));
                }

                // Validate `results.json` against `ExperimentSpecification` if present
                let mut validation_error: Option<String> = None;
                let mut measured_json = "{}".to_string();

                if let Some(ref spec) = exp_spec {
                    match results_file_raw {
                        None => {
                            validation_error = Some(format!(
                                "RESULTS_FILE_MISSING: Tool '{}' did not write required structured results file '{}' for specification '{}'",
                                name,
                                results_file_path.display(),
                                spec.id
                            ));
                        }
                        Some(ref raw_content) => {
                            match serde_json::from_str::<serde_json::Value>(raw_content.trim()) {
                                Err(e) => {
                                    validation_error = Some(format!(
                                        "RESULTS_FILE_MALFORMED: Tool '{}' wrote malformed JSON in '{}': {}",
                                        name,
                                        results_file_path.display(),
                                        e
                                    ));
                                }
                                Ok(val) => {
                                    if let Some(obj) = val.as_object() {
                                        let mut missing_keys = Vec::new();
                                        for m in &spec.metrics {
                                            let rk = m.result_key.trim();
                                            if !rk.is_empty() && !obj.contains_key(rk) {
                                                missing_keys.push(rk.to_string());
                                            }
                                        }
                                        if !missing_keys.is_empty() {
                                            validation_error = Some(format!(
                                                "RESULTS_FILE_MISSING_DECLARED_KEYS: Tool '{}' results.json lacks declared metric keys {:?} required by specification '{}'",
                                                name, missing_keys, spec.id
                                            ));
                                        }
                                        measured_json = val.to_string();
                                    } else {
                                        validation_error = Some(format!(
                                            "RESULTS_FILE_NOT_OBJECT: Tool '{}' results.json must be a JSON object",
                                            name
                                        ));
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(ref raw_content) = results_file_raw {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(raw_content.trim()) {
                        if val.is_object() {
                            measured_json = val.to_string();
                        }
                    }
                }

                let artifacts_json = serde_json::Value::Array(artifact_entries).to_string();
                let effective_exit_code = if validation_error.is_some() && out.exit_code == 0 {
                    -2
                } else {
                    out.exit_code
                };
                let combined_error = match (validation_error, out.stderr.trim().is_empty()) {
                    (Some(verr), true) => Some(verr),
                    (Some(verr), false) => Some(format!("{} | stderr: {}", verr, out.stderr.trim())),
                    (None, false) if out.exit_code != 0 => Some(out.stderr),
                    _ => None,
                };

                let rec = self.store.record_immutable_tool_execution(
                    name,
                    tool.version,
                    &tool.source_hash,
                    input_json,
                    effective_seed,
                    dependencies_json,
                    Some(&out.stdout),
                    combined_error.as_deref(),
                    effective_exit_code,
                    elapsed,
                    &artifacts_json,
                    &measured_json,
                    specialist_id,
                )?;
                let _ = fs::remove_dir_all(&run_ws);
                Ok(rec)
            }
            Err(e) => {
                let err_msg = e.to_string();
                let is_timeout = err_msg.contains("SANDBOX_TIMED_OUT") || err_msg.contains("timed out");
                let exit_code = if is_timeout { -124 } else { -1 };
                let artifacts_json = serde_json::json!([
                    {
                        "kind": "tool_binary",
                        "path": tool.binary_path,
                        "source_hash": tool.source_hash
                    }
                ])
                .to_string();
                let measured_json = serde_json::json!({
                    "exit_code": exit_code,
                    "timed_out": is_timeout
                })
                .to_string();
                let rec = self.store.record_immutable_tool_execution(
                    name,
                    tool.version,
                    &tool.source_hash,
                    input_json,
                    effective_seed,
                    dependencies_json,
                    None,
                    Some(&err_msg),
                    exit_code,
                    elapsed,
                    &artifacts_json,
                    &measured_json,
                    specialist_id,
                )?;
                let _ = fs::remove_dir_all(&run_ws);
                Ok(rec)
            }
        }
    }

    /// Runs a spec-backed research experiment, validates its `results.json` against the declared
    /// `ExperimentSpecification`, extracts baseline comparison and held-out generalization results,
    /// and records both the immutable `ToolExecutionRecord` and `ResearchExperimentRecord`.
    /// Also records informative failures and timeouts accurately.
    pub async fn execute_research_experiment(
        &self,
        cycle_id: i64,
        tool_name: &str,
        input_json: &str,
        seed: Option<i64>,
        specialist_id: Option<&str>,
        timeout_override: Option<Duration>,
    ) -> Result<ResearchExperimentOutcome> {
        let tool = self
            .store
            .get_tool_by_name(tool_name)?
            .context(format!("Research tool '{}' not found in registry", tool_name))?;

        let spec_id = tool.experiment_spec_id.clone().context(format!(
            "Tool '{}' does not have an ExperimentSpecification; research experiments require a specification before execution",
            tool_name
        ))?;
        let spec = self
            .store
            .get_experiment_spec(&spec_id)?
            .context(format!("ExperimentSpecification '{}' not found", spec_id))?;

        let timeout = timeout_override
            .unwrap_or_else(|| Duration::from_secs(spec.budget_max_runtime_secs.max(1)));

        let exec_rec = self
            .execute_tool_with_timeout(tool_name, input_json, seed, specialist_id, timeout)
            .await?;

        let is_timeout = exec_rec.exit_code == -124
            || exec_rec
                .error
                .as_deref()
                .map(|s| s.contains("SANDBOX_TIMED_OUT"))
                .unwrap_or(false);
        let is_rejected_results = exec_rec.exit_code == -2
            || exec_rec
                .error
                .as_deref()
                .map(|s| s.contains("RESULTS_FILE_"))
                .unwrap_or(false);
        let is_failure = exec_rec.exit_code != 0 || exec_rec.error.is_some();

        let measured_val: serde_json::Value =
            serde_json::from_str(&exec_rec.measured_results_json).unwrap_or_else(|_| serde_json::json!({}));

        let baseline_comparison = measured_val
            .get("baseline_comparison")
            .cloned()
            .unwrap_or_else(|| {
                serde_json::json!({
                    "baseline_description": spec.baseline_description,
                    "baseline_value": measured_val.get("baseline_value").or_else(|| measured_val.get("baseline_accuracy")).or_else(|| measured_val.get("baseline_loss")),
                    "candidate_value": spec.metrics.first().and_then(|m| measured_val.get(&m.result_key)),
                })
            });

        let held_out_evaluation = measured_val
            .get("held_out_evaluation")
            .cloned()
            .unwrap_or_else(|| {
                serde_json::json!({
                    "held_out_cases": spec.held_out_cases,
                    "held_out_score": measured_val.get("held_out_accuracy").or_else(|| measured_val.get("held_out_loss")).or_else(|| measured_val.get("held_out_generalization_gap")),
                })
            });

        let outcome_status = if is_timeout {
            "timed_out"
        } else if is_rejected_results {
            "rejected_invalid_results_file"
        } else if is_failure {
            "failed"
        } else if spec.under_specified {
            "completed_under_specified"
        } else {
            "completed"
        };

        let observed_outcome = if is_failure {
            format!(
                "exec_id={} status={} exit_code={} error={}",
                exec_rec.id,
                outcome_status,
                exec_rec.exit_code,
                exec_rec.error.as_deref().unwrap_or("unknown error")
            )
        } else {
            format!(
                "exec_id={} status={} measured_results={}",
                exec_rec.id, outcome_status, exec_rec.measured_results_json
            )
        };

        let lessons_learned = if is_timeout {
            format!(
                "Tool '{}' exceeded timeout ({:?}); sandbox process group was killed and confirmed exited.",
                tool_name, timeout
            )
        } else if is_rejected_results {
            format!(
                "Host rejected tool '{}' output because results.json was missing, malformed, or lacked a declared metric key.",
                tool_name
            )
        } else if is_failure {
            format!(
                "Informative failure recorded for hypothesis '{}': {}",
                spec.falsifiable_hypothesis,
                exec_rec.error.as_deref().unwrap_or("non-zero exit")
            )
        } else {
            format!(
                "Evaluated mechanism '{}' against baseline '{}'. Known limitations: {}. Numerical agreement supports an experiment; it is not a proof.",
                spec.mechanism_under_test,
                spec.baseline_description,
                spec.known_limitations.join("; ")
            )
        };

        let exp_id = self.store.record_structured_research_experiment(
            cycle_id,
            Some(&spec.id),
            Some(tool_name),
            Some(exec_rec.id),
            &spec.falsifiable_hypothesis,
            input_json,
            &spec.mechanism_under_test,
            &baseline_comparison.to_string(),
            &held_out_evaluation.to_string(),
            &exec_rec.measured_results_json,
            &spec.source_fidelity_status,
            spec.under_specified,
            &spec.missing_source_details,
            &observed_outcome,
            outcome_status,
            &lessons_learned,
            is_failure,
            if is_failure {
                Some("Inspect execution error or missing source details before retrying")
            } else {
                None
            },
        )?;

        let exp_rec = self
            .store
            .list_research_experiments(20)?
            .into_iter()
            .find(|r| r.id == exp_id)
            .context("Failed to load recorded ResearchExperimentRecord")?;

        if is_rejected_results {
            bail!(
                "Host rejected tool '{}' execution #{}: {}",
                tool_name,
                exec_rec.id,
                exec_rec.error.unwrap_or_default()
            );
        }

        Ok(ResearchExperimentOutcome {
            experiment_record: exp_rec,
            execution_record: exec_rec,
            specification: spec,
            measured_results: measured_val,
            baseline_comparison,
            held_out_evaluation,
            epistemic_note: "Numerical agreement supports an experiment; it is not a proof."
                .to_string(),
        })
    }
}

/// Removes intermediate test/binary artifacts (`<crate_name>-*`) from `shared_target/release/deps`
/// and removes `shared_target/debug` and `shared_target/release/incremental` while keeping
/// shared library dependencies (`lib*.rlib`, `lib*.so`, `lib*.rmeta`) intact for sub-second builds.
pub fn prune_crate_artifacts_from_shared_target(shared_target: &Path, crate_name: &str) {
    let _ = fs::remove_dir_all(shared_target.join("debug"));
    let _ = fs::remove_dir_all(shared_target.join("release").join("incremental"));
    let prefix = format!("{}-", crate_name);
    for sub in ["deps", ".fingerprint"] {
        let dir = shared_target.join("release").join(sub);
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(fname) = entry.file_name().to_str() {
                    if fname.starts_with(&prefix) {
                        let p = entry.path();
                        if p.is_dir() {
                            let _ = fs::remove_dir_all(&p);
                        } else {
                            let _ = fs::remove_file(&p);
                        }
                    }
                }
            }
        }
    }
}

/// Purges redundant compilation caches (`shared_target/debug`, duplicate `release/deps/<bin>-*`
/// test binaries, duplicate `tools/shared_target/release/<tool_name>`, temporary `workspace/runs`
/// directories, and obsolete pre-amendment DB backup) without affecting any active tool/extension
/// binary, shared library dependency cache (`lib*.rlib`/`lib*.so`), or SQLite WAL database.
pub fn purge_workspace_build_bloat(workspace_dir: &Path, data_dir: &Path) -> (usize, u64) {
    let mut removed_items = 0usize;
    let mut reclaimed_bytes = 0u64;

    fn dir_size(p: &Path) -> u64 {
        let mut total = 0u64;
        if let Ok(rd) = fs::read_dir(p) {
            for e in rd.flatten() {
                let path = e.path();
                if let Ok(meta) = e.metadata() {
                    if meta.is_dir() {
                        total += dir_size(&path);
                    } else {
                        total += meta.len();
                    }
                }
            }
        }
        total
    }

    // 1. Remove `shared_target/debug` and `shared_target/release/incremental` in both tools & extensions
    for sub in ["tools", "extensions"] {
        let shared = workspace_dir.join(sub).join("shared_target");
        for dir_to_rm in [
            shared.join("debug"),
            shared.join("release").join("incremental"),
        ] {
            if dir_to_rm.exists() {
                reclaimed_bytes += dir_size(&dir_to_rm);
                if fs::remove_dir_all(&dir_to_rm).is_ok() {
                    removed_items += 1;
                }
            }
        }

        // 2. In `shared_target/release/deps`, remove non-library test/bin artifacts (`!fname.starts_with("lib")`)
        let rel_deps = shared.join("release").join("deps");
        if let Ok(entries) = fs::read_dir(&rel_deps) {
            for entry in entries.flatten() {
                if let Some(fname) = entry.file_name().to_str() {
                    if !fname.starts_with("lib") {
                        let p = entry.path();
                        if let Ok(m) = entry.metadata() {
                            if m.is_file() {
                                reclaimed_bytes += m.len();
                                if fs::remove_file(&p).is_ok() {
                                    removed_items += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. In `tools/shared_target/release`, remove top-level duplicate tool binaries (already copied to `tools/<name>/target/release/<name>`)
    let tools_rel = workspace_dir.join("tools").join("shared_target").join("release");
    if let Ok(entries) = fs::read_dir(&tools_rel) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Ok(m) = entry.metadata() {
                if m.is_file() {
                    reclaimed_bytes += m.len();
                    if fs::remove_file(&p).is_ok() {
                        removed_items += 1;
                    }
                }
            }
        }
    }

    // 4. Clean temporary `workspace/runs` directories
    let runs_dir = workspace_dir.join("runs");
    if runs_dir.exists() {
        reclaimed_bytes += dir_size(&runs_dir);
        if fs::remove_dir_all(&runs_dir).is_ok() {
            removed_items += 1;
            let _ = fs::create_dir_all(&runs_dir);
        }
    }

    // 5. Remove obsolete `triad.db.bak_pre_amendment` if present
    let old_bak = data_dir.join("triad.db.bak_pre_amendment");
    if old_bak.exists() {
        if let Ok(m) = fs::metadata(&old_bak) {
            reclaimed_bytes += m.len();
        }
        if fs::remove_file(&old_bak).is_ok() {
            removed_items += 1;
        }
    }

    (removed_items, reclaimed_bytes)
}
