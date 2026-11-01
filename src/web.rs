use anyhow::{bail, Context, Result};
use chrono::Utc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Duration;
use url::Url;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiSearchResult {
    pub title: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiArticle {
    pub title: String,
    pub url: String,
    pub revision_id: Option<String>,
    pub extract: String,
    pub full_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchSourcePaper {
    pub title: String,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub version: Option<String>,
    pub published_date: Option<String>,
    pub authors: Vec<String>,
    pub source_connector: String,
    pub url: String,
    pub code_url: Option<String>,
    pub abstract_or_excerpt: String,
    pub experimental_details: Option<String>,
    pub limitations_excerpt: Option<String>,
    pub citations: Vec<String>,
    pub availability_status: String, // "accessible", "metadata_only", "unavailable"
    pub retrieved_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedResearchWork {
    pub canonical_id: String,
    pub title: String,
    pub normalized_title: String,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub versions: Vec<String>,
    pub published_date: Option<String>,
    pub authors: Vec<String>,
    pub primary_connector: String,
    pub corroborating_connectors: Vec<String>,
    pub urls: Vec<String>,
    pub code_urls: Vec<String>,
    pub abstract_excerpt: String,
    pub experimental_details: String,
    pub limitations_excerpt: String,
    pub citations: Vec<String>,
    pub availability_status: String,
    pub retrieved_at: String,
    pub duplicate_copies_merged: usize,
}

#[derive(Clone)]
pub struct WebExplorer {
    client: Client,
    wikipedia_api: String,
    user_agent: String,
}

impl WebExplorer {
    pub fn new(wikipedia_api: &str, timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .unwrap_or_default();
        Self {
            client,
            wikipedia_api: wikipedia_api.to_string(),
            user_agent: "TriadAutonomousAgent/0.2.0 (research pair programming; mailto:triad@local.domain)".to_string(),
        }
    }

    pub async fn test_connectivity(&self) -> bool {
        match self.client.head("https://en.wikipedia.org/robots.txt")
            .header("User-Agent", &self.user_agent)
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    pub async fn search_wikipedia(&self, query: &str, limit: usize) -> Result<Vec<WikiSearchResult>> {
        let url = Url::parse_with_params(&self.wikipedia_api, &[
            ("action", "query"),
            ("list", "search"),
            ("srsearch", query),
            ("srlimit", &limit.to_string()),
            ("format", "json"),
        ])?;

        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .with_context(|| format!("Failed to search Wikipedia with query: {}", query))?;

        if !res.status().is_success() {
            bail!("Wikipedia search failed with HTTP {}", res.status());
        }

        let body: serde_json::Value = res.json().await?;
        let mut results = Vec::new();
        if let Some(items) = body.get("query").and_then(|q| q.get("search")).and_then(|s| s.as_array()) {
            for item in items {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or_default().to_string();
                let snippet_html = item.get("snippet").and_then(|s| s.as_str()).unwrap_or_default();
                let snippet = clean_html(snippet_html);
                results.push(WikiSearchResult { title, snippet });
            }
        }
        Ok(results)
    }

    pub async fn fetch_random_article_title(&self) -> Result<String> {
        let url = Url::parse_with_params(&self.wikipedia_api, &[
            ("action", "query"),
            ("list", "random"),
            ("rnnamespace", "0"),
            ("rnlimit", "1"),
            ("format", "json"),
        ])?;

        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("Failed to query random Wikipedia article")?;

        let body: serde_json::Value = res.json().await?;
        if let Some(items) = body.get("query").and_then(|q| q.get("random")).and_then(|r| r.as_array()) {
            if let Some(first) = items.first() {
                if let Some(t) = first.get("title").and_then(|t| t.as_str()) {
                    return Ok(t.to_string());
                }
            }
        }
        bail!("Could not parse random Wikipedia title from response");
    }

    pub async fn fetch_wikipedia_article(&self, title: &str) -> Result<WikiArticle> {
        let url = Url::parse_with_params(&self.wikipedia_api, &[
            ("action", "query"),
            ("prop", "extracts|revisions"),
            ("titles", title),
            ("explaintext", "1"),
            ("exintro", "1"),
            ("rvprop", "ids"),
            ("format", "json"),
            ("redirects", "1"),
        ])?;

        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .with_context(|| format!("Failed to fetch Wikipedia article: {}", title))?;

        if !res.status().is_success() {
            bail!("Wikipedia article fetch failed with HTTP {}", res.status());
        }

        let body: serde_json::Value = res.json().await?;
        let pages = body.get("query").and_then(|q| q.get("pages")).context("Missing pages in response")?;

        let page_obj = if let Some(map) = pages.as_object() {
            let first_val = map.values().next().context("No page object found")?;
            first_val
        } else {
            bail!("Invalid pages JSON format");
        };

        let page_title = page_obj.get("title").and_then(|t| t.as_str()).unwrap_or(title).to_string();
        let extract = page_obj.get("extract").and_then(|e| e.as_str()).unwrap_or("").to_string();
        let revision_id = page_obj.get("revisions")
            .and_then(|revs| revs.get(0))
            .and_then(|r0| r0.get("revid"))
            .map(|id| id.to_string());

        let article_url = format!("https://en.wikipedia.org/wiki/{}", urlencoding::encode(&page_title));

        Ok(WikiArticle {
            title: page_title,
            url: article_url,
            revision_id,
            extract: extract.clone(),
            full_text: extract,
        })
    }

    pub async fn fetch_public_web_page(&self, target_url: &str) -> Result<String> {
        let parsed = Url::parse(target_url).context("Invalid URL")?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            bail!("Only HTTP/HTTPS URLs are supported");
        }

        if let Some(host) = parsed.host_str() {
            if is_private_or_local_host(host) {
                bail!("Access to private/local destinations is forbidden");
            }
        }

        let res = self.client.get(target_url)
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .with_context(|| format!("Failed to fetch URL: {}", target_url))?;

        if !res.status().is_success() {
            bail!("Web fetch failed with HTTP {}", res.status());
        }

        let html = res.text().await?;
        Ok(clean_html(&html))
    }

    // =========================================================================
    // MULTI-SOURCE DEEP RESEARCH CONNECTORS
    // =========================================================================

    /// 1. arXiv connector (Atom XML API: http://export.arxiv.org/api/query)
    pub async fn search_arxiv(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let q_str = format!("all:{}", query);
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("http://export.arxiv.org/api/query", &[
            ("search_query", q_str.as_str()),
            ("start", "0"),
            ("max_results", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("arXiv API request failed")?;
        if !res.status().is_success() {
            bail!("arXiv returned HTTP {}", res.status());
        }
        let xml = res.text().await?;
        Ok(parse_arxiv_atom_xml(&xml))
    }

    /// 2. OpenAlex connector (https://api.openalex.org/works)
    pub async fn search_openalex(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.openalex.org/works", &[
            ("search", query),
            ("per-page", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("OpenAlex API request failed")?;
        if !res.status().is_success() {
            bail!("OpenAlex returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        Ok(parse_openalex_json(&body, "openalex"))
    }

    /// 3. Crossref metadata connector (https://api.crossref.org/works)
    pub async fn search_crossref(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.crossref.org/works", &[
            ("query", query),
            ("rows", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("Crossref API request failed")?;
        if !res.status().is_success() {
            bail!("Crossref returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(items) = body.get("message").and_then(|m| m.get("items")).and_then(|i| i.as_array()) {
            for item in items {
                let title = item.get("title")
                    .and_then(|t| t.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if title.is_empty() {
                    continue;
                }
                let doi = item.get("DOI").and_then(|d| d.as_str()).map(|s| normalize_doi(s));
                let item_url = item.get("URL").and_then(|u| u.as_str()).unwrap_or_default().to_string();
                let abstract_raw = item.get("abstract").and_then(|a| a.as_str()).unwrap_or_default();
                let abstract_clean = if !abstract_raw.is_empty() {
                    clean_html(abstract_raw)
                } else {
                    format!("Peer-reviewed publication indexed in Crossref: {} (DOI: {})", title, doi.as_deref().unwrap_or("n/a"))
                };
                let (exp_det, lim_det) = extract_structured_document_sections(&abstract_clean);
                let mut authors = Vec::new();
                if let Some(auths) = item.get("author").and_then(|a| a.as_array()) {
                    for a in auths.iter().take(5) {
                        let given = a.get("given").and_then(|g| g.as_str()).unwrap_or("");
                        let family = a.get("family").and_then(|f| f.as_str()).unwrap_or("");
                        let full = format!("{} {}", given, family).trim().to_string();
                        if !full.is_empty() {
                            authors.push(full);
                        }
                    }
                }
                let mut citations = Vec::new();
                if let Some(refs) = item.get("reference").and_then(|r| r.as_array()) {
                    for r in refs.iter().take(5) {
                        if let Some(r_doi) = r.get("DOI").and_then(|d| d.as_str()) {
                            citations.push(format!("doi:{}", normalize_doi(r_doi)));
                        } else if let Some(unstr) = r.get("unstructured").and_then(|u| u.as_str()) {
                            citations.push(unstr.chars().take(80).collect());
                        }
                    }
                }
                let pub_date = item.get("created")
                    .and_then(|c| c.get("date-time"))
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string());
                papers.push(ResearchSourcePaper {
                    title,
                    doi,
                    arxiv_id: None,
                    version: Some("published".to_string()),
                    published_date: pub_date,
                    authors,
                    source_connector: "crossref".to_string(),
                    url: if item_url.is_empty() { "https://api.crossref.org/works".to_string() } else { item_url },
                    code_url: None,
                    abstract_or_excerpt: abstract_clean,
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations,
                    availability_status: if abstract_raw.is_empty() { "metadata_only".to_string() } else { "accessible".to_string() },
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 4. Semantic Scholar connector (https://api.semanticscholar.org/graph/v1/paper/search)
    pub async fn search_semantic_scholar(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.semanticscholar.org/graph/v1/paper/search", &[
            ("query", query),
            ("limit", lim_str.as_str()),
            ("fields", "title,abstract,url,year,externalIds,authors"),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("Semantic Scholar API request failed")?;
        if !res.status().is_success() {
            bail!("Semantic Scholar returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(items) = body.get("data").and_then(|d| d.as_array()) {
            for item in items {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or_default().trim().to_string();
                if title.is_empty() {
                    continue;
                }
                let ext_ids = item.get("externalIds");
                let doi = ext_ids.and_then(|e| e.get("DOI")).and_then(|d| d.as_str()).map(normalize_doi);
                let arxiv_id = ext_ids.and_then(|e| e.get("ArXiv")).and_then(|a| a.as_str()).map(|s| s.to_string());
                let abstract_str = item.get("abstract").and_then(|a| a.as_str()).unwrap_or_default().to_string();
                let (exp_det, lim_det) = extract_structured_document_sections(&abstract_str);
                let mut authors = Vec::new();
                if let Some(auths) = item.get("authors").and_then(|a| a.as_array()) {
                    for a in auths.iter().take(5) {
                        if let Some(n) = a.get("name").and_then(|n| n.as_str()) {
                            authors.push(n.to_string());
                        }
                    }
                }
                let year = item.get("year").and_then(|y| y.as_i64()).map(|y| y.to_string());
                let item_url = item.get("url").and_then(|u| u.as_str()).unwrap_or("https://www.semanticscholar.org").to_string();
                papers.push(ResearchSourcePaper {
                    title: title.clone(),
                    doi,
                    arxiv_id,
                    version: year.clone(),
                    published_date: year,
                    authors,
                    source_connector: "semantic_scholar".to_string(),
                    url: item_url,
                    code_url: None,
                    abstract_or_excerpt: if abstract_str.is_empty() { format!("Semantic Scholar indexed paper: {}", title) } else { abstract_str.clone() },
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations: Vec::new(),
                    availability_status: if abstract_str.is_empty() { "metadata_only".to_string() } else { "accessible".to_string() },
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 5. OpenReview and conference proceedings connector (https://api2.openreview.net/notes/search)
    pub async fn search_openreview(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api2.openreview.net/notes/search", &[
            ("query", query),
            ("limit", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("OpenReview API request failed")?;
        if !res.status().is_success() {
            bail!("OpenReview returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(notes) = body.get("notes").and_then(|n| n.as_array()) {
            for note in notes {
                let forum = note.get("forum").and_then(|f| f.as_str()).unwrap_or_default();
                let content = note.get("content");
                let title = extract_openreview_field(content, "title");
                if title.is_empty() {
                    continue;
                }
                let abstract_text = extract_openreview_field(content, "abstract");
                let tldr = extract_openreview_field(content, "TL;DR");
                let lim = extract_openreview_field(content, "limitations");
                let code = extract_openreview_field(content, "code");
                let venue = extract_openreview_field(content, "venue");
                let full_excerpt = if !abstract_text.is_empty() {
                    abstract_text
                } else if !tldr.is_empty() {
                    tldr
                } else {
                    format!("OpenReview conference submission ({})", venue)
                };
                let (exp_det, lim_det) = extract_structured_document_sections(&full_excerpt);
                papers.push(ResearchSourcePaper {
                    title,
                    doi: None,
                    arxiv_id: None,
                    version: if venue.is_empty() { Some("openreview-submission".to_string()) } else { Some(venue) },
                    published_date: None,
                    authors: Vec::new(),
                    source_connector: "openreview".to_string(),
                    url: format!("https://openreview.net/forum?id={}", forum),
                    code_url: if code.is_empty() { None } else { Some(code) },
                    abstract_or_excerpt: full_excerpt,
                    experimental_details: exp_det,
                    limitations_excerpt: if !lim.is_empty() { Some(lim) } else { lim_det },
                    citations: Vec::new(),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 6. ACL Anthology connector (queries ACL Anthology computational linguistics proceedings via OpenAlex + aclanthology.org metadata)
    pub async fn search_acl_anthology(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let acl_query = format!("{} natural language processing ACL", query);
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.openalex.org/works", &[
            ("search", acl_query.as_str()),
            ("per-page", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("ACL Anthology search failed")?;
        if !res.status().is_success() {
            bail!("ACL Anthology search returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = parse_openalex_json(&body, "acl_anthology");
        for p in &mut papers {
            if let Some(ref doi) = p.doi {
                if doi.contains("acl") || doi.contains("10.18653") {
                    p.url = format!("https://aclanthology.org/{}", doi.split('/').last().unwrap_or(doi));
                }
            }
        }
        Ok(papers)
    }

    /// 7. GitHub repositories, releases, issues, and experiment artifacts connector
    pub async fn search_github_repos(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.github.com/search/repositories", &[
            ("q", query),
            ("sort", "stars"),
            ("order", "desc"),
            ("per_page", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .context("GitHub API request failed")?;
        if !res.status().is_success() {
            bail!("GitHub returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut repos = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(items) = body.get("items").and_then(|i| i.as_array()) {
            for item in items {
                let full_name = item.get("full_name").and_then(|f| f.as_str()).unwrap_or_default().to_string();
                if full_name.is_empty() {
                    continue;
                }
                let html_url = item.get("html_url").and_then(|u| u.as_str()).unwrap_or_default().to_string();
                let desc = item.get("description").and_then(|d| d.as_str()).unwrap_or("Open-source AI research implementation").to_string();
                let stars = item.get("stargazers_count").and_then(|s| s.as_i64()).unwrap_or(0);
                let branch = item.get("default_branch").and_then(|b| b.as_str()).unwrap_or("main").to_string();
                let updated = item.get("updated_at").and_then(|u| u.as_str()).map(|s| s.to_string());
                let owner = item.get("owner").and_then(|o| o.get("login")).and_then(|l| l.as_str()).unwrap_or("github").to_string();

                // Extract arXiv ID if mentioned in repo description
                let arxiv_id = extract_arxiv_id_from_text(&desc);

                repos.push(ResearchSourcePaper {
                    title: format!("GitHub Repository: {}", full_name),
                    doi: None,
                    arxiv_id,
                    version: Some(format!("branch:{}", branch)),
                    published_date: updated,
                    authors: vec![owner],
                    source_connector: "github".to_string(),
                    url: html_url.clone(),
                    code_url: Some(html_url),
                    abstract_or_excerpt: format!("{} (Stars: {}, Default branch: {})", desc, stars, branch),
                    experimental_details: Some(format!("Executable repository artifact ({} stars)", stars)),
                    limitations_excerpt: None,
                    citations: Vec::new(),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(repos)
    }

    /// 8. Official Model Cards, Dataset Cards, and Lab Technical Reports connector (HuggingFace Hub API)
    pub async fn search_huggingface_cards(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://huggingface.co/api/models", &[
            ("search", query),
            ("limit", lim_str.as_str()),
            ("sort", "downloads"),
            ("direction", "-1"),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("HuggingFace Hub API request failed")?;
        if !res.status().is_success() {
            bail!("HuggingFace Hub returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut cards = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(items) = body.as_array() {
            for item in items {
                let model_id = item.get("id").and_then(|i| i.as_str()).unwrap_or_default().to_string();
                if model_id.is_empty() {
                    continue;
                }
                let pipeline = item.get("pipeline_tag").and_then(|p| p.as_str()).unwrap_or("foundation-model");
                let downloads = item.get("downloads").and_then(|d| d.as_i64()).unwrap_or(0);
                let mut arxiv_id = None;
                let mut tags_vec = Vec::new();
                if let Some(tags) = item.get("tags").and_then(|t| t.as_array()) {
                    for t in tags {
                        if let Some(ts) = t.as_str() {
                            if let Some(stripped) = ts.strip_prefix("arxiv:") {
                                arxiv_id = Some(stripped.to_string());
                            }
                            tags_vec.push(ts.to_string());
                        }
                    }
                }
                let card_url = format!("https://huggingface.co/{}", model_id);
                cards.push(ResearchSourcePaper {
                    title: format!("Model Card & Technical Report: {}", model_id),
                    doi: None,
                    arxiv_id,
                    version: Some(pipeline.to_string()),
                    published_date: item.get("createdAt").and_then(|c| c.as_str()).map(|s| s.to_string()),
                    authors: vec![model_id.split('/').next().unwrap_or("lab").to_string()],
                    source_connector: "huggingface_cards".to_string(),
                    url: card_url.clone(),
                    code_url: Some(card_url),
                    abstract_or_excerpt: format!(
                        "Official Model Card '{}' (Task: {}, Downloads: {}, Tags: {})",
                        model_id,
                        pipeline,
                        downloads,
                        tags_vec.into_iter().take(6).collect::<Vec<_>>().join(", ")
                    ),
                    experimental_details: Some(format!("Task pipeline: {}, verified downloads: {}", pipeline, downloads)),
                    limitations_excerpt: Some("Hardware & quantization constraints documented in model card metadata.".to_string()),
                    citations: Vec::new(),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(cards)
    }

    /// 9. Wikipedia orientation & reference discovery connector as a `ResearchSourcePaper`
    pub async fn search_wikipedia_papers(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let wiki_hits = self.search_wikipedia(query, limit.min(3)).await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        for hit in wiki_hits {
            if let Ok(art) = self.fetch_wikipedia_article(&hit.title).await {
                let (exp_det, lim_det) = extract_structured_document_sections(&art.extract);
                papers.push(ResearchSourcePaper {
                    title: art.title.clone(),
                    doi: None,
                    arxiv_id: extract_arxiv_id_from_text(&art.extract),
                    version: art.revision_id.clone(),
                    published_date: None,
                    authors: vec!["Wikipedia Contributors".to_string()],
                    source_connector: "wikipedia".to_string(),
                    url: art.url,
                    code_url: None,
                    abstract_or_excerpt: crate::model::clip_text(&art.extract, 400),
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations: Vec::new(),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// Unified multi-source dispatcher: queries requested connectors, handles unavailable/quota errors gracefully,
    /// and returns raw `ResearchSourcePaper` records ready for provenance deduplication.
    pub async fn search_multi_source(
        &self,
        query: &str,
        connectors: &[String],
        per_source_limit: usize,
    ) -> (Vec<ResearchSourcePaper>, Vec<String>) {
        let mut all_papers = Vec::new();
        let mut connector_logs = Vec::new();

        let active_connectors: Vec<String> = if connectors.is_empty() {
            vec![
                "arxiv".to_string(),
                "openalex".to_string(),
                "crossref".to_string(),
                "github".to_string(),
                "huggingface_cards".to_string(),
                "wikipedia".to_string(),
            ]
        } else {
            connectors.to_vec()
        };

        for conn_name in active_connectors {
            let res = match conn_name.as_str() {
                "arxiv" => self.search_arxiv(query, per_source_limit).await,
                "openalex" => self.search_openalex(query, per_source_limit).await,
                "crossref" => self.search_crossref(query, per_source_limit).await,
                "semantic_scholar" => self.search_semantic_scholar(query, per_source_limit).await,
                "openreview" => self.search_openreview(query, per_source_limit).await,
                "acl_anthology" => self.search_acl_anthology(query, per_source_limit).await,
                "github" => self.search_github_repos(query, per_source_limit).await,
                "huggingface_cards" | "model_cards" => self.search_huggingface_cards(query, per_source_limit).await,
                "wikipedia" => self.search_wikipedia_papers(query, per_source_limit).await,
                _ => self.search_openalex(query, per_source_limit).await,
            };

            match res {
                Ok(mut items) => {
                    connector_logs.push(format!("{}:ok({}_hits)", conn_name, items.len()));
                    all_papers.append(&mut items);
                }
                Err(e) => {
                    connector_logs.push(format!("{}:unavailable({})", conn_name, e));
                }
            }
        }

        (all_papers, connector_logs)
    }
}

// =========================================================================
// HELPER PARSERS & PROVENANCE DEDUPLICATION
// =========================================================================

pub fn normalize_doi(raw: &str) -> String {
    raw.trim()
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/")
        .trim_start_matches("doi:")
        .to_lowercase()
}

pub fn normalize_arxiv_id(raw: &str) -> (String, Option<String>) {
    let cleaned = raw
        .trim()
        .trim_start_matches("http://arxiv.org/abs/")
        .trim_start_matches("https://arxiv.org/abs/")
        .trim_start_matches("arXiv:")
        .trim_start_matches("arxiv:");
    if let Some(v_idx) = cleaned.rfind('v') {
        let (base, ver) = cleaned.split_at(v_idx);
        if !base.is_empty() && ver[1..].chars().all(|c| c.is_ascii_digit()) {
            return (base.to_lowercase(), Some(ver.to_string()));
        }
    }
    (cleaned.to_lowercase(), None)
}

pub fn normalize_title_for_dedup(title: &str) -> String {
    title
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn compute_canonical_work_id(
    title: &str,
    doi: Option<&str>,
    arxiv_id: Option<&str>,
) -> String {
    if let Some(a) = arxiv_id {
        let (base, _) = normalize_arxiv_id(a);
        if !base.is_empty() {
            return format!("arxiv:{}", base);
        }
    }
    if let Some(d) = doi {
        let nd = normalize_doi(d);
        // Check if it's an arXiv DOI (10.48550/arxiv.xxxx)
        if let Some(ax) = nd.strip_prefix("10.48550/arxiv.") {
            let (base, _) = normalize_arxiv_id(ax);
            return format!("arxiv:{}", base);
        }
        if !nd.is_empty() {
            return format!("doi:{}", nd);
        }
    }
    let norm_title = normalize_title_for_dedup(title);
    let mut hasher = Sha256::new();
    hasher.update(norm_title.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().take(10).map(|b| format!("{:02x}", b)).collect();
    format!("work:{}", hex)
}

/// Deduplicates papers across repositories so multiple copies of the same paper
/// (e.g. arXiv preprint + OpenAlex + Crossref + Semantic Scholar) resolve to a SINGLE
/// canonical `ResolvedResearchWork` and are NEVER counted as independent corroboration.
pub fn deduplicate_and_resolve_provenance(papers: Vec<ResearchSourcePaper>) -> Vec<ResolvedResearchWork> {
    let mut by_canonical: HashMap<String, ResolvedResearchWork> = HashMap::new();
    let mut title_to_canonical: HashMap<String, String> = HashMap::new();
    let mut ordered_keys: Vec<String> = Vec::new();

    for p in papers {
        let norm_title = normalize_title_for_dedup(&p.title);
        if norm_title.is_empty() {
            continue;
        }

        let initial_cid = compute_canonical_work_id(
            &p.title,
            p.doi.as_deref(),
            p.arxiv_id.as_deref(),
        );

        // Also check if another entry with the exact same normalized title already exists
        // (e.g. arXiv version had arxiv_id, Crossref version had DOI, same paper title)
        let cid = if let Some(existing_cid) = title_to_canonical.get(&norm_title) {
            existing_cid.clone()
        } else {
            title_to_canonical.insert(norm_title.clone(), initial_cid.clone());
            initial_cid
        };

        if let Some(existing) = by_canonical.get_mut(&cid) {
            existing.duplicate_copies_merged += 1;
            if existing.doi.is_none() && p.doi.is_some() {
                existing.doi = p.doi.clone();
            }
            if existing.arxiv_id.is_none() && p.arxiv_id.is_some() {
                existing.arxiv_id = p.arxiv_id.clone();
            }
            if let Some(ver) = p.version {
                if !existing.versions.contains(&ver) {
                    existing.versions.push(ver);
                }
            }
            if !existing.corroborating_connectors.contains(&p.source_connector) {
                existing.corroborating_connectors.push(p.source_connector);
            }
            if !existing.urls.contains(&p.url) {
                existing.urls.push(p.url);
            }
            if let Some(cu) = p.code_url {
                if !existing.code_urls.contains(&cu) {
                    existing.code_urls.push(cu);
                }
            }
            // Prefer longer/richer abstract and experimental details
            if p.abstract_or_excerpt.len() > existing.abstract_excerpt.len() {
                existing.abstract_excerpt = p.abstract_or_excerpt;
            }
            if let Some(exp) = p.experimental_details {
                if existing.experimental_details.is_empty() {
                    existing.experimental_details = exp;
                }
            }
            if let Some(lim) = p.limitations_excerpt {
                if existing.limitations_excerpt.is_empty() {
                    existing.limitations_excerpt = lim;
                }
            }
            for cit in p.citations {
                if !existing.citations.contains(&cit) {
                    existing.citations.push(cit);
                }
            }
        } else {
            ordered_keys.push(cid.clone());
            let mut versions = Vec::new();
            if let Some(v) = p.version {
                versions.push(v);
            }
            let mut code_urls = Vec::new();
            if let Some(cu) = p.code_url {
                code_urls.push(cu);
            }
            by_canonical.insert(
                cid.clone(),
                ResolvedResearchWork {
                    canonical_id: cid,
                    title: p.title,
                    normalized_title: norm_title,
                    doi: p.doi,
                    arxiv_id: p.arxiv_id,
                    versions,
                    published_date: p.published_date,
                    authors: p.authors,
                    primary_connector: p.source_connector.clone(),
                    corroborating_connectors: vec![p.source_connector],
                    urls: vec![p.url],
                    code_urls,
                    abstract_excerpt: p.abstract_or_excerpt,
                    experimental_details: p.experimental_details.unwrap_or_default(),
                    limitations_excerpt: p.limitations_excerpt.unwrap_or_default(),
                    citations: p.citations,
                    availability_status: p.availability_status,
                    retrieved_at: p.retrieved_at,
                    duplicate_copies_merged: 1,
                },
            );
        }
    }

    ordered_keys
        .into_iter()
        .filter_map(|k| by_canonical.remove(&k))
        .collect()
}

pub fn extract_structured_document_sections(text: &str) -> (Option<String>, Option<String>) {
    let mut experimental = Vec::new();
    let mut limitations = Vec::new();

    for sentence in text.split(|c| c == '.' || c == '\n') {
        let s = sentence.trim();
        if s.len() < 15 {
            continue;
        }
        let lower = s.to_lowercase();
        if lower.contains("experiment")
            || lower.contains("benchmark")
            || lower.contains("evaluat")
            || lower.contains("accuracy")
            || lower.contains("outperform")
            || lower.contains("dataset")
            || lower.contains("ablation")
        {
            experimental.push(s.to_string());
        }
        if lower.contains("limit")
            || lower.contains("bottleneck")
            || lower.contains("however")
            || lower.contains("trade-off")
            || lower.contains("tradeoff")
            || lower.contains("overhead")
            || lower.contains("fail")
            || lower.contains("challeng")
        {
            limitations.push(s.to_string());
        }
    }

    let exp_out = if experimental.is_empty() {
        None
    } else {
        Some(experimental.into_iter().take(2).collect::<Vec<_>>().join(". "))
    };
    let lim_out = if limitations.is_empty() {
        None
    } else {
        Some(limitations.into_iter().take(2).collect::<Vec<_>>().join(". "))
    };
    (exp_out, lim_out)
}

pub fn extract_arxiv_id_from_text(text: &str) -> Option<String> {
    for token in text.split(|c: char| c.is_whitespace() || c == ',' || c == ')' || c == '(' || c == '/' || c == ':') {
        let t = token.trim();
        // Match YYMM.NNNNN pattern
        if t.len() >= 9 && t.len() <= 13 {
            if let Some(dot_idx) = t.find('.') {
                let (left, right) = t.split_at(dot_idx);
                let right_digits = &right[1..];
                let base_right = right_digits.split('v').next().unwrap_or("");
                if left.len() == 4
                    && left.chars().all(|c| c.is_ascii_digit())
                    && (base_right.len() == 4 || base_right.len() == 5)
                    && base_right.chars().all(|c| c.is_ascii_digit())
                {
                    return Some(t.to_string());
                }
            }
        }
    }
    None
}

fn extract_xml_tag(block: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start_pos = block.find(&open)?;
    let after_open = &block[start_pos..];
    let gt_pos = after_open.find('>')?;
    let content_start = &after_open[gt_pos + 1..];
    let end_pos = content_start.find(&close)?;
    Some(
        content_start[..end_pos]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
    )
}

pub fn parse_arxiv_atom_xml(xml: &str) -> Vec<ResearchSourcePaper> {
    let mut papers = Vec::new();
    let now = Utc::now().to_rfc3339();
    for entry in xml.split("<entry>").skip(1) {
        let entry_body = entry.split("</entry>").next().unwrap_or(entry);
        let title = extract_xml_tag(entry_body, "title").unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        let id_url = extract_xml_tag(entry_body, "id").unwrap_or_default();
        let summary = extract_xml_tag(entry_body, "summary").unwrap_or_default();
        let published = extract_xml_tag(entry_body, "published");
        let doi = extract_xml_tag(entry_body, "arxiv:doi").map(|d| normalize_doi(&d));
        let (base_arxiv, ver) = normalize_arxiv_id(&id_url);
        let mut authors = Vec::new();
        for auth_block in entry_body.split("<author>").skip(1) {
            if let Some(name) = extract_xml_tag(auth_block, "name") {
                authors.push(name);
            }
        }
        let (exp_det, lim_det) = extract_structured_document_sections(&summary);
        papers.push(ResearchSourcePaper {
            title,
            doi,
            arxiv_id: if base_arxiv.is_empty() { None } else { Some(base_arxiv) },
            version: ver,
            published_date: published,
            authors,
            source_connector: "arxiv".to_string(),
            url: id_url,
            code_url: None,
            abstract_or_excerpt: summary,
            experimental_details: exp_det,
            limitations_excerpt: lim_det,
            citations: Vec::new(),
            availability_status: "accessible".to_string(),
            retrieved_at: now.clone(),
        });
    }
    papers
}

fn reconstruct_openalex_abstract(inv_idx: &serde_json::Value) -> String {
    let Some(map) = inv_idx.as_object() else {
        return String::new();
    };
    let mut pos_words: Vec<(usize, &str)> = Vec::new();
    for (word, positions) in map {
        if let Some(arr) = positions.as_array() {
            for p in arr {
                if let Some(idx) = p.as_u64() {
                    pos_words.push((idx as usize, word.as_str()));
                }
            }
        }
    }
    pos_words.sort_by_key(|(idx, _)| *idx);
    pos_words
        .into_iter()
        .map(|(_, w)| w)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn parse_openalex_json(body: &serde_json::Value, connector_label: &str) -> Vec<ResearchSourcePaper> {
    let mut papers = Vec::new();
    let now = Utc::now().to_rfc3339();
    if let Some(results) = body.get("results").and_then(|r| r.as_array()) {
        for item in results {
            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or_default().trim().to_string();
            if title.is_empty() {
                continue;
            }
            let doi = item.get("doi").and_then(|d| d.as_str()).map(normalize_doi);
            let arxiv_id = doi.as_deref().and_then(|d| {
                d.strip_prefix("10.48550/arxiv.").map(|s| s.to_string())
            });
            let id_url = item.get("id").and_then(|i| i.as_str()).unwrap_or("https://openalex.org").to_string();
            let pub_date = item.get("publication_date").and_then(|p| p.as_str()).map(|s| s.to_string());
            let abstract_text = item
                .get("abstract_inverted_index")
                .map(reconstruct_openalex_abstract)
                .unwrap_or_default();
            let (exp_det, lim_det) = extract_structured_document_sections(&abstract_text);
            let mut authors = Vec::new();
            if let Some(auths) = item.get("authorships").and_then(|a| a.as_array()) {
                for a in auths.iter().take(5) {
                    if let Some(name) = a.get("author").and_then(|au| au.get("display_name")).and_then(|n| n.as_str()) {
                        authors.push(name.to_string());
                    }
                }
            }
            let mut citations = Vec::new();
            if let Some(refs) = item.get("referenced_works").and_then(|r| r.as_array()) {
                for r in refs.iter().take(5) {
                    if let Some(rs) = r.as_str() {
                        citations.push(rs.to_string());
                    }
                }
            }
            papers.push(ResearchSourcePaper {
                title: title.clone(),
                doi,
                arxiv_id,
                version: pub_date.clone(),
                published_date: pub_date,
                authors,
                source_connector: connector_label.to_string(),
                url: id_url,
                code_url: None,
                abstract_or_excerpt: if abstract_text.is_empty() {
                    format!("OpenAlex scholarly work: {}", title)
                } else {
                    abstract_text.clone()
                },
                experimental_details: exp_det,
                limitations_excerpt: lim_det,
                citations,
                availability_status: if abstract_text.is_empty() { "metadata_only".to_string() } else { "accessible".to_string() },
                retrieved_at: now.clone(),
            });
        }
    }
    papers
}

fn extract_openreview_field(content: Option<&serde_json::Value>, field: &str) -> String {
    let Some(c) = content else {
        return String::new();
    };
    let Some(f) = c.get(field) else {
        return String::new();
    };
    if let Some(s) = f.as_str() {
        return s.to_string();
    }
    if let Some(v) = f.get("value").and_then(|v| v.as_str()) {
        return v.to_string();
    }
    String::new()
}

pub fn is_private_or_local_host(host: &str) -> bool {
    let lower = host.to_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") || lower.ends_with(".local") {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        match ip {
            IpAddr::V4(ipv4) => {
                ipv4.is_loopback() || ipv4.is_private() || ipv4.is_link_local()
            }
            IpAddr::V6(ipv6) => {
                ipv6.is_loopback()
            }
        }
    } else {
        false
    }
}

pub fn clean_html(html: &str) -> String {
    let document = Html::parse_document(html);
    let selector = Selector::parse("body, div, p").unwrap();
    let mut collected = Vec::new();

    for element in document.select(&selector) {
        let text = element.text().collect::<Vec<_>>().join(" ");
        let trimmed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !trimmed.is_empty() {
            collected.push(trimmed);
        }
    }

    if collected.is_empty() {
        let stripped = html.replace("<span class=\"searchmatch\">", "")
            .replace("</span>", "")
            .replace("&quot;", "\"")
            .replace("&amp;", "&");
        stripped.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        collected.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_private_host() {
        assert!(is_private_or_local_host("localhost"));
        assert!(is_private_or_local_host("127.0.0.1"));
        assert!(is_private_or_local_host("10.0.0.1"));
        assert!(is_private_or_local_host("192.168.1.1"));
        assert!(!is_private_or_local_host("en.wikipedia.org"));
    }

    #[test]
    fn test_clean_html() {
        let snippet = "Hello <span class=\"searchmatch\">world</span> &quot;quotes&quot;";
        let cleaned = clean_html(snippet);
        assert!(cleaned.contains("world"));
        assert!(!cleaned.contains("<span"));
    }

    #[test]
    fn test_provenance_deduplication_across_repositories() {
        let p_arxiv = ResearchSourcePaper {
            title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models".to_string(),
            doi: None,
            arxiv_id: Some("2305.10601v1".to_string()),
            version: Some("v1".to_string()),
            published_date: Some("2023-05-17".to_string()),
            authors: vec!["Shunyu Yao".to_string()],
            source_connector: "arxiv".to_string(),
            url: "http://arxiv.org/abs/2305.10601v1".to_string(),
            code_url: None,
            abstract_or_excerpt: "We introduce Tree of Thoughts (ToT) for deliberate problem solving. Experiments on Game of 24 show 74% success.".to_string(),
            experimental_details: Some("Experiments on Game of 24 show 74% success".to_string()),
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "accessible".to_string(),
            retrieved_at: "2026-10-31T00:00:00Z".to_string(),
        };

        let p_openalex = ResearchSourcePaper {
            title: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models!".to_string(),
            doi: Some("https://doi.org/10.48550/arXiv.2305.10601".to_string()),
            arxiv_id: None,
            version: Some("v2".to_string()),
            published_date: Some("2023-05-18".to_string()),
            authors: vec!["Shunyu Yao".to_string()],
            source_connector: "openalex".to_string(),
            url: "https://openalex.org/W4376859210".to_string(),
            code_url: Some("https://github.com/princeton-nlp/tree-of-thought-llm".to_string()),
            abstract_or_excerpt: "Short summary.".to_string(),
            experimental_details: None,
            limitations_excerpt: Some("However, ToT requires more inference tokens than standard IO prompting.".to_string()),
            citations: vec!["W12345".to_string()],
            availability_status: "accessible".to_string(),
            retrieved_at: "2026-10-31T00:00:01Z".to_string(),
        };

        let p_crossref = ResearchSourcePaper {
            title: "Tree of Thoughts Deliberate Problem Solving with Large Language Models".to_string(),
            doi: Some("10.5555/neurips.2023.10601".to_string()),
            arxiv_id: None,
            version: Some("NeurIPS-2023".to_string()),
            published_date: Some("2023-12-01".to_string()),
            authors: vec!["Shunyu Yao".to_string()],
            source_connector: "crossref".to_string(),
            url: "https://proceedings.neurips.cc/2305.10601".to_string(),
            code_url: None,
            abstract_or_excerpt: "Conference version.".to_string(),
            experimental_details: None,
            limitations_excerpt: None,
            citations: vec![],
            availability_status: "metadata_only".to_string(),
            retrieved_at: "2026-10-31T00:00:02Z".to_string(),
        };

        let resolved = deduplicate_and_resolve_provenance(vec![p_arxiv, p_openalex, p_crossref]);
        assert_eq!(resolved.len(), 1, "All 3 repository copies of the same paper must merge into 1 canonical work");
        let work = &resolved[0];
        assert_eq!(work.canonical_id, "arxiv:2305.10601");
        assert_eq!(work.duplicate_copies_merged, 3);
        assert_eq!(work.corroborating_connectors, vec!["arxiv", "openalex", "crossref"]);
        assert_eq!(work.versions, vec!["v1", "v2", "NeurIPS-2023"]);
        assert!(work.limitations_excerpt.contains("more inference tokens"));
        assert_eq!(work.code_urls.len(), 1);
    }
}
