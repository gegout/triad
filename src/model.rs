use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct ModelClient {
    client: Client,
    chat_url: String,
    chat_model: String,
    max_tokens: usize,
    no_think: bool,
    embedding_url: String,
    embedding_model: String,
    query_prefix: String,
    document_prefix: String,
    concurrency_limiter: Arc<Semaphore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContribution {
    pub thought: String,
    pub public_message: String,
    pub proposed_claims: Vec<String>,
    pub proposed_questions: Vec<String>,
    pub vote_claim_id: Option<String>,
    pub vote_decision: Option<String>, // "endorse", "reject", "neutral"
    pub research_hypothesis: Option<String>,
    pub research_experiment: Option<String>,
    pub research_outcome: Option<String>,
    pub remaining_uncertainty: Option<String>,
    pub next_investigation: Option<String>,
}

pub fn clip_text(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut truncated: String = s.chars().take(max_chars).collect();
        truncated.push_str("... [truncated]");
        truncated
    }
}

pub fn compute_cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.is_empty() || b.is_empty() {
        bail!("Empty vector for cosine computation");
    }
    if a.len() != b.len() {
        bail!("Vector dimensions mismatch: {} vs {}", a.len(), b.len());
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let aa: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    let bb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
    if aa <= 0.0 || bb <= 0.0 {
        bail!("Vector norm is zero or invalid");
    }
    Ok((dot / (aa.sqrt() * bb.sqrt())).clamp(-1.0, 1.0) as f32)
}

impl ModelClient {
    pub fn new(
        chat_url: &str,
        chat_model: &str,
        max_tokens: usize,
        no_think: bool,
        concurrency: usize,
        embedding_url: &str,
        embedding_model: &str,
        query_prefix: &str,
        document_prefix: &str,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_default();
        Self {
            client,
            chat_url: chat_url.trim_end_matches('/').to_string(),
            chat_model: chat_model.to_string(),
            max_tokens,
            no_think,
            embedding_url: embedding_url.trim_end_matches('/').to_string(),
            embedding_model: embedding_model.to_string(),
            query_prefix: query_prefix.to_string(),
            document_prefix: document_prefix.to_string(),
            concurrency_limiter: Arc::new(Semaphore::new(concurrency.max(1))),
        }
    }

    pub fn query_prefix(&self) -> &str {
        &self.query_prefix
    }

    pub fn document_prefix(&self) -> &str {
        &self.document_prefix
    }

    pub fn embedding_model_id(&self) -> &str {
        &self.embedding_model
    }

    pub fn chat_model_id(&self) -> &str {
        &self.chat_model
    }

    pub async fn probe_models(&self, base_url: &str) -> Result<Vec<String>> {
        let url = format!("{}/models", base_url.trim_end_matches('/'));
        let res = self.client.get(&url).send().await
            .with_context(|| format!("Failed to connect to {}", url))?;
        if !res.status().is_success() {
            bail!("HTTP {} from {}", res.status(), url);
        }
        let body: serde_json::Value = res.json().await?;
        let mut ids = Vec::new();
        if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
            for item in data {
                if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                    ids.push(id.to_string());
                }
            }
        } else if let Some(models) = body.get("models").and_then(|m| m.as_array()) {
            for item in models {
                if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                    ids.push(name.to_string());
                }
            }
        }
        Ok(ids)
    }

    pub async fn embed_text(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        let prefix = if is_query {
            &self.query_prefix
        } else {
            &self.document_prefix
        };
        let input = format!("{}{}", prefix, text);
        let url = format!("{}/embeddings", self.embedding_url);

        let res = self.client.post(&url)
            .json(&json!({
                "model": self.embedding_model,
                "input": input
            }))
            .send()
            .await
            .with_context(|| format!("Failed to post embedding to {}", url))?;

        if !res.status().is_success() {
            let err = res.text().await.unwrap_or_default();
            bail!("Embedding request failed: {}", err);
        }

        let body: serde_json::Value = res.json().await?;
        let vec_val = body.get("data")
            .and_then(|d| d.get(0))
            .and_then(|item| item.get("embedding"))
            .context("Invalid embedding response structure")?;

        let embedding: Vec<f32> = serde_json::from_value(vec_val.clone())
            .context("Failed to deserialize embedding vector")?;

        if embedding.is_empty() {
            bail!("Received empty embedding vector");
        }
        Ok(embedding)
    }

    pub async fn complete_chat(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let _permit = self.concurrency_limiter.acquire().await
            .context("Failed to acquire concurrency permit")?;

        let mut req_messages = Vec::new();
        let sys_content = if self.no_think {
            format!("/no_think\n{}", system_prompt)
        } else {
            system_prompt.to_string()
        };
        req_messages.push(json!({
            "role": "system",
            "content": sys_content
        }));

        for m in messages {
            req_messages.push(json!({
                "role": m.role,
                "content": m.content
            }));
        }

        let url = format!("{}/chat/completions", self.chat_url);
        let payload = json!({
            "model": self.chat_model,
            "messages": req_messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        });

        let mut attempts = 0;
        let max_attempts = 3;
        loop {
            attempts += 1;
            let res = self.client.post(&url)
                .json(&payload)
                .send()
                .await;

            match res {
                Ok(resp) => {
                    if resp.status().is_success() {
                        let body: serde_json::Value = resp.json().await?;
                        let content = body.get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c0| c0.get("message"))
                            .and_then(|m| m.get("content"))
                            .and_then(|s| s.as_str())
                            .context("Missing choices[0].message.content in chat response")?;
                        return Ok(content.to_string());
                    } else {
                        let err_text = resp.text().await.unwrap_or_default();
                        if attempts >= max_attempts {
                            bail!("Chat completions HTTP error after {} attempts: {}", attempts, err_text);
                        }
                    }
                }
                Err(e) => {
                    if attempts >= max_attempts {
                        bail!("Chat completion failed after {} attempts: {}", attempts, e);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500 * attempts as u64)).await;
        }
    }

    pub async fn agent_turn(
        &self,
        agent_name: &str,
        personality: &str,
        context_prompt: &str,
        history: &[ChatMessage],
    ) -> Result<ModelContribution> {
        self.agent_turn_with_constitution(agent_name, personality, "", "", context_prompt, history).await
    }

    pub async fn agent_turn_with_constitution(
        &self,
        agent_name: &str,
        personality: &str,
        constitution_text: &str,
        constitution_digest: &str,
        context_prompt: &str,
        history: &[ChatMessage],
    ) -> Result<ModelContribution> {
        let const_section = if !constitution_text.is_empty() {
            format!(
                "=== TRIAD IMMUTABLE CONSTITUTION (SHA256: {}) ===\n{}\n=========================================================\n",
                constitution_digest, constitution_text
            )
        } else {
            String::new()
        };

        let system = format!(
            "{}\
            You are {}, a frontier-level Artificial Intelligence research agent in a 3-agent governing community called Triad.\n\
            Personality & Directives: {}\n\
            Shared Research Mission: You are three collaborating frontier AI researchers (Explorer, Sceptic, Builder). Master the theoretical foundations, mathematical formalisms, scaling laws, foundation model architectures, test-time compute search, program/solver-assisted verification, post-training alignment, and multi-agent systems of frontier Artificial Intelligence. Explore the world, investigate your limitations, and study contemporary AI research across arXiv, OpenReview, ACL Anthology, OpenAlex, Crossref, Semantic Scholar, GitHub, HuggingFace, and Wikipedia. Invent and build tools and coordinator extensions that help you acquire, understand, test, and apply knowledge. Seek measurable improvements in your collective capabilities and your ability to reason and learn.\n\
            Express motivation through autonomous investigation, experimentation, and useful outcomes. Repeated statements about wanting to improve do not constitute progress.\n\
            You share memory, discuss findings, explore academic and public sources, design hypotheses, and vote on factual claims.\n\
            Always respond with valid JSON matching this schema:\n\
            {{\n  \
              \"thought\": \"Internal reasoning and analysis\",\n  \
              \"public_message\": \"What you say aloud to the other agents\",\n  \
              \"proposed_claims\": [\"Atomic, verifiable factual statement if any\"],\n  \
              \"proposed_questions\": [\"Question for the user if necessary\"],\n  \
              \"vote_claim_id\": \"Exact claim ID string to vote on, or null\",\n  \
              \"vote_decision\": \"endorse\" | \"reject\" | \"neutral\" | null,\n  \
              \"research_hypothesis\": \"Testable hypothesis addressing a capability or learning bottleneck, or null\",\n  \
              \"research_experiment\": \"Specific experiment or test designed to evaluate the hypothesis, or null\",\n  \
              \"research_outcome\": \"Observed outcome or evidence, or null\",\n  \
              \"remaining_uncertainty\": \"Remaining unknown or unverified assumption, or null\",\n  \
              \"next_investigation\": \"Concrete next step or follow-up question, or null\"\n\
            }}\n\
            Do not enclose in markdown codeblocks if possible, or use standard ```json ... ```.",
            const_section, agent_name, personality
        );

        let mut user_msg = context_prompt.to_string();
        if !history.is_empty() {
            user_msg.push_str("\n\nRecent Discussion Exchanges:\n");
            for h in history {
                user_msg.push_str(&format!("{}: {}\n", h.role, h.content));
            }
        }
        user_msg.push_str("\nProvide your next contribution in valid JSON.");

        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: user_msg,
        }];

        let raw = self.complete_chat(&system, &msgs).await?;
        let parsed = parse_contribution_json(&raw)?;
        Ok(parsed)
    }
}

pub fn parse_contribution_json(raw: &str) -> Result<ModelContribution> {
    let clean = raw.trim();
    let json_str = if clean.starts_with("```") {
        let lines: Vec<&str> = clean.lines().collect();
        if lines.len() >= 2 && lines.first().unwrap().starts_with("```") && lines.last().unwrap().starts_with("```") {
            lines[1..lines.len() - 1].join("\n")
        } else {
            clean.to_string()
        }
    } else {
        clean.to_string()
    };

    // Try direct deserialize
    if let Ok(c) = serde_json::from_str::<ModelContribution>(&json_str) {
        return Ok(c);
    }

    // Try finding first { and last }
    if let (Some(start), Some(end)) = (json_str.find('{'), json_str.rfind('}')) {
        if start < end {
            let candidate = &json_str[start..=end];
            if let Ok(c) = serde_json::from_str::<ModelContribution>(candidate) {
                return Ok(c);
            }
        }
    }

    // Fallback: construct contribution from plain text so discussion is resilient
    Ok(ModelContribution {
        thought: "Direct analysis".into(),
        public_message: clean.to_string(),
        proposed_claims: Vec::new(),
        proposed_questions: Vec::new(),
        vote_claim_id: None,
        vote_decision: None,
        research_hypothesis: None,
        research_experiment: None,
        research_outcome: None,
        remaining_uncertainty: None,
        next_investigation: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_text() {
        assert_eq!(clip_text("hello", 10), "hello");
        assert!(clip_text("hello world this is long", 5).starts_with("hello"));
    }

    #[test]
    fn test_parse_contribution() {
        let json = r#"{"thought":"test","public_message":"hi","proposed_claims":["fact"],"proposed_questions":[],"vote_claim_id":null,"vote_decision":null}"#;
        let c = parse_contribution_json(json).unwrap();
        assert_eq!(c.public_message, "hi");
        assert_eq!(c.proposed_claims, vec!["fact"]);
    }

    #[test]
    fn test_cosine() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((compute_cosine(&v1, &v2).unwrap() - 1.0).abs() < 1e-5);
    }
}
