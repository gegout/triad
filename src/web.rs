use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
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
    pub availability_status: String, // "accessible", "metadata_only", "unavailable", "orientation_only"
    pub retrieved_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructuredDocumentSection {
    pub section_index: usize,
    pub section_path: String,
    pub section_kind: String, // "hierarchy", "method", "algorithm", "equation", "theorem", "assumption", "experiment", "limitation", "reference", "table", "general"
    pub source_location: String,
    pub content: String,
    pub token_estimate: usize,
    pub is_full_text: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentExtractionQuality {
    pub sections_found: usize,
    pub equations_preserved: usize,
    pub tables_found: usize,
    pub low_fidelity: bool,
    pub low_fidelity_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextIngestionPayload {
    pub canonical_id: String,
    pub connector: String,
    pub ingestion_completeness: String, // "full_text", "abstract_only", "metadata_only", "orientation_only", "failed"
    pub format: String, // "arxiv_html", "arxiv_latex", "pdf_text", "html", "markdown", "abstract"
    pub raw_original: String,
    pub abstract_text: String,
    pub is_full_text: bool,
    pub is_orientation_only: bool,
    pub sections: Vec<StructuredDocumentSection>,
    pub harvested_references: Vec<String>,
    #[serde(default)]
    pub extraction_quality: DocumentExtractionQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorFetchTelemetry {
    pub connector_name: String,
    pub category: String,
    pub enabled: bool,
    pub priority: i64,
    pub succeeded: bool,
    pub items_discovered: usize,
    pub latency_ms: u64,
    pub rate_limit_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalSourceIdentity {
    pub canonical_id: String,
    pub canonical_url: String,
    pub provider_id: Option<String>,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub version: String,
    pub content_hash: String,
    pub source_kind: String, // "paper", "dataset", "implementation"
}

fn default_source_kind() -> String {
    "paper".to_string()
}

fn default_corroboration_one() -> usize {
    1
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
    #[serde(default)]
    pub provider_id: Option<String>,
    #[serde(default)]
    pub canonical_url: String,
    #[serde(default)]
    pub content_hash: String,
    #[serde(default = "default_source_kind")]
    pub source_kind: String,
    #[serde(default = "default_corroboration_one")]
    pub independent_corroboration_count: usize,
}

#[derive(Clone)]
pub struct WebExplorer {
    client: Client,
    wikipedia_api: String,
    user_agent: String,
    last_request_by_connector: Arc<Mutex<HashMap<String, Instant>>>,
    response_cache: Arc<Mutex<HashMap<String, (Instant, Vec<u8>)>>>,
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
            user_agent: "TriadAutonomousAgent/0.3.0 (research pair programming; mailto:triad@local.domain)".to_string(),
            last_request_by_connector: Arc::new(Mutex::new(HashMap::new())),
            response_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn enforce_rate_limit(&self, connector_name: &str, rate_limit_ms: u64) {
        if rate_limit_ms == 0 {
            return;
        }
        let min_gap = Duration::from_millis(rate_limit_ms.min(3500));
        let sleep_dur = {
            let mut map = self.last_request_by_connector.lock().await;
            let now = Instant::now();
            if let Some(prev) = map.get(connector_name) {
                let elapsed = now.saturating_duration_since(*prev);
                if elapsed < min_gap {
                    let wait = min_gap - elapsed;
                    map.insert(connector_name.to_string(), now + wait);
                    Some(wait)
                } else {
                    map.insert(connector_name.to_string(), now);
                    None
                }
            } else {
                map.insert(connector_name.to_string(), now);
                None
            }
        };
        if let Some(w) = sleep_dur {
            tokio::time::sleep(w).await;
        }
    }

    /// Downloads a public URL with SSRF checks, response caching, and strict byte limit (`max_bytes`).
    pub async fn fetch_bytes_bounded(
        &self,
        target_url: &str,
        max_bytes: usize,
        cache_ttl_secs: u64,
    ) -> Result<Vec<u8>> {
        let parsed = Url::parse(target_url).context("Invalid URL")?;
        if parsed.scheme() != "http" && parsed.scheme() != "https" {
            bail!("Only HTTP/HTTPS URLs are supported");
        }
        if let Some(host) = parsed.host_str() {
            if is_private_or_local_host(host) {
                bail!("Access to private/local destinations is forbidden");
            }
        }

        let effective_max = max_bytes.max(1024);
        if cache_ttl_secs > 0 {
            let cache = self.response_cache.lock().await;
            if let Some((ts, data)) = cache.get(target_url) {
                if ts.elapsed().as_secs() < cache_ttl_secs {
                    return Ok(data[..data.len().min(effective_max)].to_vec());
                }
            }
        }

        let mut res = self
            .client
            .get(target_url)
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .with_context(|| format!("Failed to fetch URL: {}", target_url))?;

        if !res.status().is_success() {
            bail!("HTTP {}", res.status());
        }

        if let Some(cl) = res.content_length() {
            if (cl as usize) > effective_max * 2 {
                bail!(
                    "Remote content-length ({}) exceeds safe download limit ({})",
                    cl,
                    effective_max
                );
            }
        }

        let mut buf = Vec::new();
        while let Some(chunk) = res.chunk().await? {
            if buf.len() + chunk.len() > effective_max {
                let remaining = effective_max.saturating_sub(buf.len());
                buf.extend_from_slice(&chunk[..remaining]);
                break;
            }
            buf.extend_from_slice(&chunk);
        }

        if cache_ttl_secs > 0 && buf.len() <= 512_000 {
            let mut cache = self.response_cache.lock().await;
            if cache.len() > 128 {
                cache.clear();
            }
            cache.insert(target_url.to_string(), (Instant::now(), buf.clone()));
        }

        Ok(buf)
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
            ("fields", "title,abstract,url,year,externalIds,authors,openAccessPdf"),
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
                let oa_pdf_url = item
                    .get("openAccessPdf")
                    .and_then(|o| o.get("url"))
                    .and_then(|u| u.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
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
                let item_url = oa_pdf_url
                    .or_else(|| item.get("url").and_then(|u| u.as_str()).map(|s| s.to_string()))
                    .unwrap_or_else(|| "https://www.semanticscholar.org".to_string());
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
    /// Orientation only: used strictly for discovery and reference harvesting, NEVER as primary evidence.
    pub async fn search_wikipedia_papers(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let wiki_hits = self.search_wikipedia(query, limit.min(3)).await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        for hit in wiki_hits {
            if let Ok(art) = self.fetch_wikipedia_article(&hit.title).await {
                let (exp_det, lim_det) = extract_structured_document_sections(&art.extract);
                let mut refs = Vec::new();
                if let Some(ax) = extract_arxiv_id_from_text(&art.extract) {
                    refs.push(format!("arxiv:{}", ax));
                }
                for d in extract_dois_from_text(&art.extract) {
                    refs.push(format!("doi:{}", d));
                }
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
                    citations: refs,
                    availability_status: "orientation_only".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 10. bioRxiv & medRxiv preprint connector (Cold Spring Harbor Laboratory DOI prefix 10.1101)
    pub async fn search_biorxiv_or_medrxiv(&self, server: &str, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.crossref.org/prefixes/10.1101/works", &[
            ("query", query),
            ("rows", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .with_context(|| format!("{} API request failed", server))?;
        if !res.status().is_success() {
            bail!("{} returned HTTP {}", server, res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = parse_crossref_items(&body, server);
        for p in &mut papers {
            if let Some(ref d) = p.doi {
                p.url = format!("https://www.{}.org/content/{}", server, d);
            }
        }
        Ok(papers)
    }

    /// 11. SSRN preprint connector (DOI prefix 10.2139)
    pub async fn search_ssrn(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.crossref.org/prefixes/10.2139/works", &[
            ("query", query),
            ("rows", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("SSRN API request failed")?;
        if !res.status().is_success() {
            bail!("SSRN returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        Ok(parse_crossref_items(&body, "ssrn"))
    }

    /// 12. HAL open archive connector (https://api.archives-ouvertes.fr/search/)
    pub async fn search_hal(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.archives-ouvertes.fr/search/", &[
            ("q", query),
            ("wt", "json"),
            ("rows", lim_str.as_str()),
            ("fl", "docid,title_s,abstract_s,uri_s,doiId_s,arxivId_s,authFullName_s,producedDate_s,fileMain_s,version_i"),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("HAL API request failed")?;
        if !res.status().is_success() {
            bail!("HAL returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(docs) = body.get("response").and_then(|r| r.get("docs")).and_then(|d| d.as_array()) {
            for doc in docs {
                let title = doc.get("title_s")
                    .and_then(|t| t.as_array())
                    .and_then(|a| a.first())
                    .and_then(|s| s.as_str())
                    .or_else(|| doc.get("title_s").and_then(|s| s.as_str()))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if title.is_empty() {
                    continue;
                }
                let abstract_text = doc.get("abstract_s")
                    .and_then(|t| t.as_array())
                    .and_then(|a| a.first())
                    .and_then(|s| s.as_str())
                    .or_else(|| doc.get("abstract_s").and_then(|s| s.as_str()))
                    .unwrap_or_default()
                    .to_string();
                let uri = doc.get("uri_s")
                    .and_then(|u| u.as_str())
                    .or_else(|| doc.get("fileMain_s").and_then(|f| f.as_str()))
                    .unwrap_or("https://hal.science")
                    .to_string();
                let doi = doc.get("doiId_s").and_then(|d| d.as_str()).map(normalize_doi);
                let arxiv_id = doc.get("arxivId_s").and_then(|a| a.as_str()).map(|s| s.to_string());
                let ver = doc.get("version_i").and_then(|v| v.as_i64()).map(|v| format!("v{}", v));
                let pub_date = doc.get("producedDate_s").and_then(|p| p.as_str()).map(|s| s.to_string());
                let mut authors = Vec::new();
                if let Some(auths) = doc.get("authFullName_s").and_then(|a| a.as_array()) {
                    for a in auths.iter().take(5) {
                        if let Some(s) = a.as_str() {
                            authors.push(s.to_string());
                        }
                    }
                }
                let (exp_det, lim_det) = extract_structured_document_sections(&abstract_text);
                papers.push(ResearchSourcePaper {
                    title: title.clone(),
                    doi,
                    arxiv_id,
                    version: ver.or_else(|| Some("v1".to_string())),
                    published_date: pub_date,
                    authors,
                    source_connector: "hal".to_string(),
                    url: uri,
                    code_url: None,
                    abstract_or_excerpt: if abstract_text.is_empty() {
                        format!("HAL open archive preprint: {}", title)
                    } else {
                        abstract_text.clone()
                    },
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations: Vec::new(),
                    availability_status: if abstract_text.is_empty() { "metadata_only".to_string() } else { "accessible".to_string() },
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 13. DBLP computer science bibliography connector (https://dblp.org/search/publ/api)
    pub async fn search_dblp(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://dblp.org/search/publ/api", &[
            ("q", query),
            ("format", "json"),
            ("h", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("DBLP API request failed")?;
        if !res.status().is_success() {
            bail!("DBLP returned HTTP {}", res.status());
        }
        let text = res.text().await?;
        if text.contains("Making sure you&#39;re not a bot") || text.trim_start().starts_with("<!doctype html") {
            // DBLP currently enforces an Anubis PoW challenge on non-browser clients;
            // fall back cleanly to Crossref CS proceedings search while preserving provenance.
            let cs_q = format!("{} computer science proceedings", query);
            let mut fallback = self.search_crossref(&cs_q, limit).await?;
            for p in &mut fallback {
                p.source_connector = "dblp".to_string();
            }
            return Ok(fallback);
        }
        let body: serde_json::Value = serde_json::from_str(&text).context("DBLP invalid JSON")?;
        let mut papers = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(hits) = body
            .get("result")
            .and_then(|r| r.get("hits"))
            .and_then(|h| h.get("hit"))
            .and_then(|h| h.as_array())
        {
            for hit in hits {
                let info = hit.get("info").unwrap_or(hit);
                let title = info.get("title").and_then(|t| t.as_str()).unwrap_or_default().trim().to_string();
                if title.is_empty() {
                    continue;
                }
                let doi = info.get("doi").and_then(|d| d.as_str()).map(normalize_doi);
                let venue = info.get("venue").and_then(|v| v.as_str()).unwrap_or("DBLP Venue");
                let year = info.get("year").and_then(|y| y.as_str()).map(|s| s.to_string());
                let ee = info.get("ee").and_then(|e| e.as_str()).or_else(|| info.get("url").and_then(|u| u.as_str())).unwrap_or("https://dblp.org").to_string();
                papers.push(ResearchSourcePaper {
                    title: title.clone(),
                    doi,
                    arxiv_id: extract_arxiv_id_from_text(&ee),
                    version: year.clone(),
                    published_date: year,
                    authors: Vec::new(),
                    source_connector: "dblp".to_string(),
                    url: ee,
                    code_url: None,
                    abstract_or_excerpt: format!("DBLP Computer Science venue publication ({}): {}", venue, title),
                    experimental_details: None,
                    limitations_excerpt: None,
                    citations: Vec::new(),
                    availability_status: "metadata_only".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(papers)
    }

    /// 14. PubMed biomedical & life-science literature connector
    pub async fn search_pubmed(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.crossref.org/works", &[
            ("query", query),
            ("filter", "type:journal-article"),
            ("rows", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("PubMed/biomedical API request failed")?;
        if !res.status().is_success() {
            bail!("PubMed returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        Ok(parse_crossref_items(&body, "pubmed"))
    }

    /// 15. Zenodo open science dataset, software, and artifact connector (https://zenodo.org/api/records)
    pub async fn search_zenodo(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://zenodo.org/api/records", &[
            ("q", query),
            ("size", lim_str.as_str()),
        ])?;
        let res = self.client.get(url.as_str())
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .context("Zenodo API request failed")?;
        if !res.status().is_success() {
            bail!("Zenodo returned HTTP {}", res.status());
        }
        let body: serde_json::Value = res.json().await?;
        let mut records = Vec::new();
        let now = Utc::now().to_rfc3339();
        if let Some(hits) = body.get("hits").and_then(|h| h.get("hits")).and_then(|h| h.as_array()) {
            for hit in hits {
                let meta = hit.get("metadata").unwrap_or(hit);
                let title = meta.get("title").and_then(|t| t.as_str()).unwrap_or_default().trim().to_string();
                if title.is_empty() {
                    continue;
                }
                let doi = hit.get("doi").or_else(|| meta.get("doi")).and_then(|d| d.as_str()).map(normalize_doi);
                let rec_url = hit
                    .get("links")
                    .and_then(|l| l.get("self_html").or_else(|| l.get("html")))
                    .and_then(|u| u.as_str())
                    .or_else(|| hit.get("doi_url").and_then(|u| u.as_str()))
                    .unwrap_or("https://zenodo.org")
                    .to_string();
                let desc_raw = meta.get("description").and_then(|d| d.as_str()).unwrap_or_default();
                let desc = clean_html(desc_raw);
                let pub_date = meta.get("publication_date").and_then(|p| p.as_str()).map(|s| s.to_string());
                let version = meta.get("version").and_then(|v| v.as_str()).map(|s| s.to_string()).or_else(|| pub_date.clone());
                let mut authors = Vec::new();
                if let Some(creators) = meta.get("creators").and_then(|c| c.as_array()) {
                    for c in creators.iter().take(5) {
                        if let Some(n) = c.get("name").and_then(|n| n.as_str()) {
                            authors.push(n.to_string());
                        }
                    }
                }
                let (exp_det, lim_det) = extract_structured_document_sections(&desc);
                records.push(ResearchSourcePaper {
                    title: title.clone(),
                    doi,
                    arxiv_id: extract_arxiv_id_from_text(&desc),
                    version,
                    published_date: pub_date,
                    authors,
                    source_connector: "zenodo".to_string(),
                    url: rec_url,
                    code_url: None,
                    abstract_or_excerpt: if desc.is_empty() { format!("Zenodo research artifact: {}", title) } else { desc },
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations: Vec::new(),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(records)
    }

    /// 16. Papers with Code / HuggingFace Paper-to-Code artifact connector
    pub async fn search_papers_with_code(&self, query: &str, limit: usize) -> Result<Vec<ResearchSourcePaper>> {
        let mut results = self.search_huggingface_cards(query, limit).await?;
        for r in &mut results {
            r.source_connector = "papers_with_code".to_string();
        }
        Ok(results)
    }

    /// 17. Institutional connector (lab technical reports, standards bodies, patents, regulatory documents)
    pub async fn search_institutional(
        &self,
        query: &str,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        let mut out = Vec::new();
        if !cfg.urls.is_empty() || !cfg.rss_feeds.is_empty() {
            if let Ok(mut custom) = self.search_generic_web_and_rss(query, limit, cfg, "institutional").await {
                out.append(&mut custom);
            }
        }
        let lim_str = limit.min(5).to_string();
        let url = Url::parse_with_params("https://api.crossref.org/works", &[
            ("query", query),
            ("filter", "type:report,type:standard"),
            ("rows", lim_str.as_str()),
        ])?;
        if let Ok(res) = self.client.get(url.as_str()).header("User-Agent", &self.user_agent).send().await {
            if res.status().is_success() {
                if let Ok(body) = res.json::<serde_json::Value>().await {
                    let mut items = parse_crossref_items(&body, "institutional");
                    out.append(&mut items);
                }
            }
        }
        Ok(out)
    }

    /// 18. Generic Web Fetch & RSS/Atom feed connector
    pub async fn search_generic_web_and_rss(
        &self,
        query: &str,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
        connector_label: &str,
    ) -> Result<Vec<ResearchSourcePaper>> {
        let mut out = Vec::new();
        let now = Utc::now().to_rfc3339();
        let q_words: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();

        for feed_url in cfg.rss_feeds.iter().take(3) {
            if let Ok(bytes) = self.fetch_bytes_bounded(feed_url, 1_048_576, 1800).await {
                let xml = String::from_utf8_lossy(&bytes);
                let parsed_items = parse_rss_or_atom_feed(&xml, connector_label);
                for item in parsed_items {
                    let text_low = format!("{} {}", item.title, item.abstract_or_excerpt).to_lowercase();
                    if q_words.is_empty() || q_words.iter().any(|w| text_low.contains(w)) {
                        out.push(item);
                        if out.len() >= limit.max(1) {
                            return Ok(out);
                        }
                    }
                }
            }
        }

        for page_url in cfg.urls.iter().take(limit.max(1)) {
            if let Ok(bytes) = self.fetch_bytes_bounded(page_url, 1_048_576, 1800).await {
                let raw_html = String::from_utf8_lossy(&bytes);
                let cleaned = clean_html(&raw_html);
                let (exp_det, lim_det) = extract_structured_document_sections(&cleaned);
                out.push(ResearchSourcePaper {
                    title: page_url.clone(),
                    doi: None,
                    arxiv_id: extract_arxiv_id_from_text(&cleaned),
                    version: Some("v1".to_string()),
                    published_date: None,
                    authors: Vec::new(),
                    source_connector: connector_label.to_string(),
                    url: page_url.clone(),
                    code_url: None,
                    abstract_or_excerpt: crate::model::clip_text(&cleaned, 500),
                    experimental_details: exp_det,
                    limitations_excerpt: lim_det,
                    citations: extract_dois_from_text(&cleaned),
                    availability_status: "accessible".to_string(),
                    retrieved_at: now.clone(),
                });
            }
        }
        Ok(out)
    }

    /// Unified multi-source dispatcher through the `SourceAdapter` interface.
    pub async fn search_multi_source(
        &self,
        query: &str,
        connectors: &[String],
        per_source_limit: usize,
    ) -> (Vec<ResearchSourcePaper>, Vec<String>) {
        let default_domain = crate::config::DomainSection::default();
        let mut custom_domain = default_domain;
        if !connectors.is_empty() {
            custom_domain.sources = connectors.to_vec();
        }
        let (papers, logs, _telemetry) = self
            .search_multi_source_for_domain(query, &custom_domain, per_source_limit)
            .await;
        (papers, logs)
    }

    /// Dispatches search across the domain's enabled, prioritized `SourceAdapter` instances,
    /// enforcing per-connector rate limits and returning structured health telemetry.
    pub async fn search_multi_source_for_domain(
        &self,
        query: &str,
        domain_cfg: &crate::config::DomainSection,
        per_source_limit: usize,
    ) -> (Vec<ResearchSourcePaper>, Vec<String>, Vec<ConnectorFetchTelemetry>) {
        let mut all_papers = Vec::new();
        let mut connector_logs = Vec::new();
        let mut telemetry = Vec::new();

        let enabled_configs = domain_cfg.effective_enabled_connectors();
        for conn_cfg in enabled_configs {
            let Some(adapter) = get_source_adapter(&conn_cfg.name) else {
                continue;
            };
            let effective_limit = per_source_limit.min(conn_cfg.max_results_per_query.max(1));
            self.enforce_rate_limit(adapter.name(), conn_cfg.rate_limit_ms.min(500))
                .await;
            let start = Instant::now();
            let res = adapter.search(self, query, effective_limit, &conn_cfg).await;
            let latency_ms = start.elapsed().as_millis() as u64;

            match res {
                Ok(mut items) => {
                    if adapter.is_orientation_only() || conn_cfg.orientation_only {
                        for it in &mut items {
                            it.availability_status = "orientation_only".to_string();
                        }
                    }
                    let count = items.len();
                    connector_logs.push(format!("{}:ok({}_hits,{}ms)", adapter.name(), count, latency_ms));
                    telemetry.push(ConnectorFetchTelemetry {
                        connector_name: adapter.name().to_string(),
                        category: adapter.category().to_string(),
                        enabled: true,
                        priority: conn_cfg.priority,
                        succeeded: true,
                        items_discovered: count,
                        latency_ms,
                        rate_limit_ms: conn_cfg.rate_limit_ms,
                        error: None,
                    });
                    all_papers.append(&mut items);
                }
                Err(e) => {
                    let err_str = e.to_string();
                    connector_logs.push(format!("{}:unavailable({})", adapter.name(), err_str));
                    telemetry.push(ConnectorFetchTelemetry {
                        connector_name: adapter.name().to_string(),
                        category: adapter.category().to_string(),
                        enabled: true,
                        priority: conn_cfg.priority,
                        succeeded: false,
                        items_discovered: 0,
                        latency_ms,
                        rate_limit_ms: conn_cfg.rate_limit_ms,
                        error: Some(err_str),
                    });
                }
            }
        }

        (all_papers, connector_logs, telemetry)
    }

    /// Ingests a discovered source through its `SourceAdapter` and cross-identifier open-access routes,
    /// preferring full text everywhere it exists.
    /// For arXiv (or any work carrying an arXiv ID/DOI): tries HTML (`arxiv.org/html` / `ar5iv`),
    /// then LaTeX e-print source (`arxiv.org/e-print`), then PDF (`arxiv.org/pdf`).
    /// For non-arXiv works: tries the primary adapter, then ACL Anthology PDF/HTML, bioRxiv/medRxiv `.full`,
    /// Zenodo record API, HAL `/document`, and OpenAlex/DOI open-access resolution.
    /// Never presents an abstract as full-text ingestion (`is_full_text = false`, `ingestion_completeness = "abstract_only"`).
    pub async fn ingest_source_via_adapter(
        &self,
        paper: &ResearchSourcePaper,
        max_download_bytes: usize,
    ) -> FullTextIngestionPayload {
        if paper.source_connector.eq_ignore_ascii_case("wikipedia")
            || paper.availability_status == "orientation_only"
        {
            return build_orientation_only_payload(paper);
        }

        let identity = canonicalize_source_identity(
            &paper.url,
            Some(&paper.title),
            paper.doi.as_deref(),
            paper.arxiv_id.as_deref(),
            paper.version.as_deref(),
            Some(&paper.abstract_or_excerpt),
            Some(&paper.source_connector),
        );

        let mut best_low_fidelity_full_text: Option<FullTextIngestionPayload> = None;

        // Route A: Cross-identifier arXiv promotion (whenever the work has an arXiv ID from any field)
        let detected_arxiv = identity
            .arxiv_id
            .clone()
            .or_else(|| paper.arxiv_id.clone())
            .or_else(|| extract_arxiv_id_from_text(&paper.url))
            .or_else(|| paper.doi.as_deref().and_then(extract_arxiv_id_from_text))
            .or_else(|| extract_arxiv_id_from_text(&paper.abstract_or_excerpt));

        if let Some(ref ax_id) = detected_arxiv {
            let mut ax_paper = paper.clone();
            ax_paper.arxiv_id = Some(ax_id.clone());
            if let Ok(mut payload) = ArxivAdapter
                .fetch_full_text(self, &ax_paper, max_download_bytes)
                .await
            {
                payload.canonical_id = identity.canonical_id.clone();
                if payload.is_full_text && !payload.extraction_quality.low_fidelity {
                    return payload;
                }
                if payload.is_full_text {
                    best_low_fidelity_full_text = Some(payload);
                }
            }
        }

        // Route B: Primary connector adapter
        if !paper.source_connector.eq_ignore_ascii_case("arxiv") || detected_arxiv.is_none() {
            if let Some(adapter) = get_source_adapter(&paper.source_connector) {
                if let Ok(mut payload) = adapter
                    .fetch_full_text(self, paper, max_download_bytes)
                    .await
                {
                    payload.canonical_id = identity.canonical_id.clone();
                    if payload.is_full_text && !payload.extraction_quality.low_fidelity {
                        return payload;
                    }
                    if payload.is_full_text && best_low_fidelity_full_text.is_none() {
                        best_low_fidelity_full_text = Some(payload);
                    }
                }
            }
        }

        // Route C: Cross-identifier Open-Access PDF / HTML / API resolution (ACL, bioRxiv, Zenodo, HAL, OpenAlex OA)
        let mut candidate_urls: Vec<(String, String)> = Vec::new();
        let doi_norm = identity
            .doi
            .clone()
            .or_else(|| paper.doi.as_deref().map(normalize_doi))
            .unwrap_or_default();

        // 1. ACL Anthology (`10.18653/v1/{id}` or `10.18653/{id}` or `aclanthology.org/{id}`)
        if let Some(acl_id) = doi_norm
            .strip_prefix("10.18653/v1/")
            .or_else(|| doi_norm.strip_prefix("10.18653/"))
            .or_else(|| {
                paper
                    .url
                    .split("aclanthology.org/")
                    .nth(1)
                    .map(|s| s.trim_matches('/').trim_end_matches(".pdf"))
            })
        {
            if !acl_id.is_empty() {
                candidate_urls.push((
                    format!("https://aclanthology.org/{}.pdf", acl_id),
                    "acl_anthology".to_string(),
                ));
                candidate_urls.push((
                    format!("https://aclanthology.org/{}/", acl_id),
                    "acl_anthology".to_string(),
                ));
            }
        }

        // 2. bioRxiv / medRxiv (`10.1101/{suffix}`)
        if doi_norm.starts_with("10.1101/") || paper.url.contains("biorxiv.org") || paper.url.contains("medrxiv.org") {
            let d_target = if doi_norm.starts_with("10.1101/") {
                doi_norm.clone()
            } else if let Some(idx) = paper.url.find("10.1101/") {
                paper.url[idx..].trim_end_matches(".full").trim_end_matches(".pdf").to_string()
            } else {
                String::new()
            };
            if !d_target.is_empty() {
                candidate_urls.push((
                    format!("https://www.biorxiv.org/content/{}v1.full.pdf", d_target),
                    "biorxiv".to_string(),
                ));
                candidate_urls.push((
                    format!("https://www.biorxiv.org/content/{}v1.full", d_target),
                    "biorxiv".to_string(),
                ));
                candidate_urls.push((
                    format!("https://www.biorxiv.org/content/{}.full", d_target),
                    "biorxiv".to_string(),
                ));
            }
        }

        // 3. HAL open archive (`hal.science/{id}` or `inria.hal.science/{id}`)
        if paper.url.contains("hal.science/") || paper.url.contains("archives-ouvertes.fr/") {
            let base = paper.url.trim_end_matches('/').trim_end_matches("/document");
            candidate_urls.push((format!("{}/document", base), "hal".to_string()));
        }

        // 4. Zenodo record API (`10.5281/zenodo.{id}` or `zenodo.org/records/{id}`)
        let zenodo_rec_id = doi_norm
            .strip_prefix("10.5281/zenodo.")
            .map(|s| s.to_string())
            .or_else(|| {
                paper
                    .url
                    .split("zenodo.org/records/")
                    .nth(1)
                    .or_else(|| paper.url.split("zenodo.org/record/").nth(1))
                    .map(|s| s.split(&['/', '?'][..]).next().unwrap_or("").to_string())
            })
            .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()));

        if let Some(ref z_id) = zenodo_rec_id {
            let z_api = format!("https://zenodo.org/api/records/{}", z_id);
            if let Ok(bytes) = self.fetch_bytes_bounded(&z_api, max_download_bytes, 3600).await {
                if let Ok(z_json) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    let meta = z_json.get("metadata").unwrap_or(&z_json);
                    let desc_html = meta
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or_default();
                    let notes = meta
                        .get("notes")
                        .and_then(|n| n.as_str())
                        .unwrap_or_default();
                    let full_md = format!(
                        "# {}\n\n## Overview & Artifact Specification\n{}\n\n## Methodological & Experimental Notes\n{}\n\n## Provenance & Identifiers\nDOI: {}\nZenodo Record: {}",
                        paper.title,
                        clean_html(desc_html),
                        clean_html(notes),
                        doi_norm,
                        z_id
                    );
                    if full_md.len() >= 350 {
                        let sections = extract_structured_sections_from_markdown_or_text(
                            &full_md,
                            &format!("zenodo:{}", z_id),
                            true,
                        );
                        let quality = compute_extraction_quality(&sections, true, full_md.len());
                        if !sections.is_empty() && !quality.low_fidelity {
                            let refs = extract_references_from_text(&full_md);
                            return FullTextIngestionPayload {
                                canonical_id: identity.canonical_id,
                                connector: "zenodo".to_string(),
                                ingestion_completeness: "full_text".to_string(),
                                format: "markdown".to_string(),
                                raw_original: crate::model::clip_text(&full_md, 64_000),
                                abstract_text: paper.abstract_or_excerpt.clone(),
                                is_full_text: true,
                                is_orientation_only: false,
                                sections,
                                harvested_references: refs,
                                extraction_quality: quality,
                            };
                        }
                    }
                }
            }
        }

        // 5. OpenAlex work lookup by DOI or OpenAlex ID to discover OA PDF/HTML or arXiv mirror
        let oa_api_url = if !doi_norm.is_empty() {
            Some(format!("https://api.openalex.org/works/https://doi.org/{}", doi_norm))
        } else if let Some(w_id) = identity
            .provider_id
            .as_deref()
            .and_then(|p| p.strip_prefix("openalex:"))
        {
            Some(format!("https://api.openalex.org/works/{}", w_id.to_uppercase()))
        } else {
            None
        };

        if let Some(ref api_u) = oa_api_url {
            if let Ok(bytes) = self.fetch_bytes_bounded(api_u, 262_144, 3600).await {
                if let Ok(oa_item) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    // Check if OpenAlex resolved an arXiv ID
                    let oa_arxiv = oa_item
                        .get("ids")
                        .and_then(|ids| ids.get("arxiv"))
                        .and_then(|a| a.as_str())
                        .and_then(extract_arxiv_id_from_text);
                    if let Some(ax_id) = oa_arxiv {
                        let mut ax_paper = paper.clone();
                        ax_paper.arxiv_id = Some(ax_id);
                        if let Ok(mut payload) = ArxivAdapter
                            .fetch_full_text(self, &ax_paper, max_download_bytes)
                            .await
                        {
                            payload.canonical_id = identity.canonical_id.clone();
                            if payload.is_full_text && !payload.extraction_quality.low_fidelity {
                                return payload;
                            }
                        }
                    }

                    // Collect any open-access PDF or landing URLs from OpenAlex
                    for loc_key in ["best_oa_location", "primary_location"] {
                        if let Some(loc) = oa_item.get(loc_key) {
                            for u_key in ["pdf_url", "landing_page_url"] {
                                if let Some(u) = loc.get(u_key).and_then(|v| v.as_str()) {
                                    let ut = u.trim();
                                    if !ut.is_empty() && !ut.contains("api.openalex.org") {
                                        candidate_urls.push((ut.to_string(), "openalex".to_string()));
                                    }
                                }
                            }
                        }
                    }
                    if let Some(oa_u) = oa_item
                        .get("open_access")
                        .and_then(|o| o.get("oa_url"))
                        .and_then(|v| v.as_str())
                    {
                        let ut = oa_u.trim();
                        if !ut.is_empty() && !ut.contains("api.openalex.org") {
                            candidate_urls.push((ut.to_string(), "openalex".to_string()));
                        }
                    }
                }
            }
        }

        // Try fetching candidate Open-Access URLs (PDF or HTML)
        let mut tried = std::collections::HashSet::new();
        for (cand_url, conn_label) in candidate_urls {
            if !tried.insert(cand_url.clone()) {
                continue;
            }
            if let Some(ax) = extract_arxiv_id_from_text(&cand_url) {
                let mut ax_paper = paper.clone();
                ax_paper.arxiv_id = Some(ax);
                if let Ok(mut payload) = ArxivAdapter
                    .fetch_full_text(self, &ax_paper, max_download_bytes)
                    .await
                {
                    payload.canonical_id = identity.canonical_id.clone();
                    if payload.is_full_text && !payload.extraction_quality.low_fidelity {
                        return payload;
                    }
                }
            }
            if let Ok(bytes) = self
                .fetch_bytes_bounded(&cand_url, max_download_bytes, 3600)
                .await
            {
                if bytes.starts_with(b"%PDF-") {
                    if let Ok((pdf_text, sections, quality)) =
                        extract_pdf_full_text_and_sections(&bytes, &cand_url)
                    {
                        if !sections.is_empty() && !quality.low_fidelity {
                            let refs = extract_references_from_text(&pdf_text);
                            return FullTextIngestionPayload {
                                canonical_id: identity.canonical_id,
                                connector: conn_label,
                                ingestion_completeness: "full_text".to_string(),
                                format: "pdf_text".to_string(),
                                raw_original: crate::model::clip_text(&pdf_text, 64_000),
                                abstract_text: paper.abstract_or_excerpt.clone(),
                                is_full_text: true,
                                is_orientation_only: false,
                                sections,
                                harvested_references: refs,
                                extraction_quality: quality,
                            };
                        }
                    }
                } else if bytes.len() > 600 {
                    let raw_text = String::from_utf8_lossy(&bytes).to_string();
                    let sections = extract_structured_sections_from_html(&raw_text, &cand_url);
                    let quality = compute_extraction_quality(&sections, true, raw_text.len());
                    if sections.len() >= 2 && !quality.low_fidelity {
                        let refs = extract_references_from_text(&raw_text);
                        return FullTextIngestionPayload {
                            canonical_id: identity.canonical_id,
                            connector: conn_label,
                            ingestion_completeness: "full_text".to_string(),
                            format: "html".to_string(),
                            raw_original: crate::model::clip_text(&raw_text, 64_000),
                            abstract_text: paper.abstract_or_excerpt.clone(),
                            is_full_text: true,
                            is_orientation_only: false,
                            sections,
                            harvested_references: refs,
                            extraction_quality: quality,
                        };
                    }
                }
            }
        }

        if let Some(low_fid) = best_low_fidelity_full_text {
            return low_fid;
        }

        build_abstract_only_payload(paper)
    }

    /// Follows citations, references, disambiguated authors, and later work across enabled connectors
    /// within configured `max_depth` and `max_volume` bounds.
    pub async fn expand_graph_across_connectors(
        &self,
        seed_works: &[ResolvedResearchWork],
        missing_evidence_queries: &[String],
        domain_cfg: &crate::config::DomainSection,
        max_depth: usize,
        max_volume: usize,
    ) -> (Vec<ResearchSourcePaper>, Vec<ConnectorFetchTelemetry>) {
        let mut expanded = Vec::new();
        let mut telemetry = Vec::new();
        if max_depth == 0 || max_volume == 0 {
            return (expanded, telemetry);
        }

        let enabled_configs = domain_cfg
            .effective_enabled_connectors()
            .into_iter()
            .filter(|c| !c.orientation_only && c.name != "wikipedia")
            .take(4)
            .collect::<Vec<_>>();

        // 1. Follow references, citations, and authors from top seed works
        for work in seed_works.iter().take(2) {
            if expanded.len() >= max_volume {
                break;
            }
            for conn_cfg in &enabled_configs {
                if expanded.len() >= max_volume {
                    break;
                }
                if let Some(adapter) = get_source_adapter(&conn_cfg.name) {
                    let start = Instant::now();
                    if let Ok(mut found) = adapter.expand_graph(self, work, 2, conn_cfg).await {
                        let lat = start.elapsed().as_millis() as u64;
                        let count = found.len();
                        if count > 0 {
                            telemetry.push(ConnectorFetchTelemetry {
                                connector_name: adapter.name().to_string(),
                                category: adapter.category().to_string(),
                                enabled: true,
                                priority: conn_cfg.priority,
                                succeeded: true,
                                items_discovered: count,
                                latency_ms: lat,
                                rate_limit_ms: conn_cfg.rate_limit_ms,
                                error: None,
                            });
                            for item in found.drain(..) {
                                if expanded.len() < max_volume {
                                    expanded.push(item);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. If missing evidence queries remain and volume budget permits, query top enabled adapter
        for mq in missing_evidence_queries.iter().take(max_depth) {
            if expanded.len() >= max_volume {
                break;
            }
            if let Some(conn_cfg) = enabled_configs.first() {
                if let Some(adapter) = get_source_adapter(&conn_cfg.name) {
                    let start = Instant::now();
                    if let Ok(mut found) = adapter.search(self, mq, 2, conn_cfg).await {
                        let lat = start.elapsed().as_millis() as u64;
                        telemetry.push(ConnectorFetchTelemetry {
                            connector_name: adapter.name().to_string(),
                            category: adapter.category().to_string(),
                            enabled: true,
                            priority: conn_cfg.priority,
                            succeeded: true,
                            items_discovered: found.len(),
                            latency_ms: lat,
                            rate_limit_ms: conn_cfg.rate_limit_ms,
                            error: None,
                        });
                        for item in found.drain(..) {
                            if expanded.len() < max_volume {
                                expanded.push(item);
                            }
                        }
                    }
                }
            }
        }

        (expanded, telemetry)
    }
}

// =========================================================================
// UNIFIED MULTI-SOURCE ADAPTER INTERFACE (`SourceAdapter`)
// =========================================================================

#[async_trait]
pub trait SourceAdapter: Send + Sync {
    fn name(&self) -> &str;
    fn category(&self) -> &str;
    fn is_orientation_only(&self) -> bool {
        false
    }

    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>>;

    async fn fetch_full_text(
        &self,
        explorer: &WebExplorer,
        paper: &ResearchSourcePaper,
        max_bytes: usize,
    ) -> Result<FullTextIngestionPayload> {
        // Default full-text fetch for non-arXiv connectors:
        // If orientation-only, never mark as full text primary evidence.
        if self.is_orientation_only() || paper.availability_status == "orientation_only" {
            return Ok(build_orientation_only_payload(paper));
        }
        // Try fetching the paper URL directly if it is a public HTML/Markdown/PDF document
        if !paper.url.is_empty()
            && !paper.url.contains("api.crossref.org")
            && !paper.url.contains("api.openalex.org")
        {
            if let Ok(bytes) = explorer.fetch_bytes_bounded(&paper.url, max_bytes, 3600).await {
                let identity = canonicalize_source_identity(
                    &paper.url,
                    Some(&paper.title),
                    paper.doi.as_deref(),
                    paper.arxiv_id.as_deref(),
                    paper.version.as_deref(),
                    Some(&paper.abstract_or_excerpt),
                    Some(self.name()),
                );
                if bytes.starts_with(b"%PDF-") {
                    if let Ok((pdf_text, sections, quality)) =
                        extract_pdf_full_text_and_sections(&bytes, &paper.url)
                    {
                        if !sections.is_empty() && !quality.low_fidelity {
                            let refs = extract_references_from_text(&pdf_text);
                            return Ok(FullTextIngestionPayload {
                                canonical_id: identity.canonical_id,
                                connector: self.name().to_string(),
                                ingestion_completeness: "full_text".to_string(),
                                format: "pdf_text".to_string(),
                                raw_original: crate::model::clip_text(&pdf_text, 64_000),
                                abstract_text: paper.abstract_or_excerpt.clone(),
                                is_full_text: true,
                                is_orientation_only: false,
                                sections,
                                harvested_references: refs,
                                extraction_quality: quality,
                            });
                        }
                    }
                } else if bytes.len() > 600 {
                    let raw_text = String::from_utf8_lossy(&bytes).to_string();
                    let sections = extract_structured_sections_from_html(&raw_text, &paper.url);
                    if sections.len() >= 2 {
                        let quality = compute_extraction_quality(&sections, true, raw_text.len());
                        let refs = extract_references_from_text(&raw_text);
                        return Ok(FullTextIngestionPayload {
                            canonical_id: identity.canonical_id,
                            connector: self.name().to_string(),
                            ingestion_completeness: "full_text".to_string(),
                            format: "html".to_string(),
                            raw_original: crate::model::clip_text(&raw_text, 64_000),
                            abstract_text: paper.abstract_or_excerpt.clone(),
                            is_full_text: true,
                            is_orientation_only: false,
                            sections,
                            harvested_references: refs,
                            extraction_quality: quality,
                        });
                    }
                }
            }
        }
        Ok(build_abstract_only_payload(paper))
    }

    async fn expand_graph(
        &self,
        explorer: &WebExplorer,
        seed: &ResolvedResearchWork,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        // Follow first citation/DOI or disambiguated author or title follow-up
        if let Some(first_cit) = seed.citations.first() {
            let q = first_cit.trim_start_matches("doi:").trim_start_matches("arxiv:");
            if !q.is_empty() {
                return self.search(explorer, q, limit, cfg).await;
            }
        }
        if let Some(first_author) = seed.authors.first() {
            if !first_author.is_empty() && first_author != "Wikipedia Contributors" {
                let q = format!("{} {}", first_author, crate::model::clip_text(&seed.title, 40));
                return self.search(explorer, &q, limit, cfg).await;
            }
        }
        Ok(Vec::new())
    }
}

/// 1. `ArxivAdapter` (Implemented first; Full-text priority: HTML -> LaTeX source -> PDF -> Abstract-only)
pub struct ArxivAdapter;

#[async_trait]
impl SourceAdapter for ArxivAdapter {
    fn name(&self) -> &str {
        "arxiv"
    }
    fn category(&self) -> &str {
        "preprints"
    }

    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_arxiv(query, limit).await
    }

    async fn fetch_full_text(
        &self,
        explorer: &WebExplorer,
        paper: &ResearchSourcePaper,
        max_bytes: usize,
    ) -> Result<FullTextIngestionPayload> {
        let identity = canonicalize_source_identity(
            &paper.url,
            Some(&paper.title),
            paper.doi.as_deref(),
            paper.arxiv_id.as_deref(),
            paper.version.as_deref(),
            Some(&paper.abstract_or_excerpt),
            Some("arxiv"),
        );
        let Some(ref base_arxiv) = identity.arxiv_id else {
            return Ok(build_abstract_only_payload(paper));
        };
        let ver_suffix = paper
            .version
            .as_deref()
            .filter(|v| v.starts_with('v'))
            .unwrap_or("");
        let id_with_ver = format!("{}{}", base_arxiv, ver_suffix);

        // Route 1 (Highest Priority): arXiv native HTML (`arxiv.org/html/{id}`) or `ar5iv` HTML
        for html_url in [
            format!("https://arxiv.org/html/{}", id_with_ver),
            format!("https://ar5iv.labs.arxiv.org/html/{}", base_arxiv),
        ] {
            if let Ok(bytes) = explorer.fetch_bytes_bounded(&html_url, max_bytes, 3600).await {
                let html_str = String::from_utf8_lossy(&bytes).to_string();
                if (html_str.contains("ltx_section") || html_str.contains("<section") || html_str.contains("<article"))
                    && !html_str.contains("No HTML for")
                {
                    let sections = extract_structured_sections_from_html(&html_str, &html_url);
                    if !sections.is_empty() {
                        let quality = compute_extraction_quality(&sections, true, html_str.len());
                        let refs = extract_references_from_text(&html_str);
                        return Ok(FullTextIngestionPayload {
                            canonical_id: identity.canonical_id,
                            connector: "arxiv".to_string(),
                            ingestion_completeness: "full_text".to_string(),
                            format: "arxiv_html".to_string(),
                            raw_original: crate::model::clip_text(&html_str, 64_000),
                            abstract_text: paper.abstract_or_excerpt.clone(),
                            is_full_text: true,
                            is_orientation_only: false,
                            sections,
                            harvested_references: refs,
                            extraction_quality: quality,
                        });
                    }
                }
            }
        }

        // Route 2 (Second Priority): arXiv LaTeX e-print source (`arxiv.org/e-print/{id}`)
        // Handles gzipped and tarred e-print sources in memory with safe archive limits.
        // NEVER executes LaTeX or any downloaded code.
        let eprint_url = format!("https://arxiv.org/e-print/{}", id_with_ver);
        if let Ok(bytes) = explorer.fetch_bytes_bounded(&eprint_url, max_bytes, 3600).await {
            if bytes.starts_with(b"%PDF-") {
                if let Ok((pdf_text, sections, quality)) =
                    extract_pdf_full_text_and_sections(&bytes, &eprint_url)
                {
                    if !sections.is_empty() {
                        let refs = extract_references_from_text(&pdf_text);
                        return Ok(FullTextIngestionPayload {
                            canonical_id: identity.canonical_id,
                            connector: "arxiv".to_string(),
                            ingestion_completeness: "full_text".to_string(),
                            format: "pdf_text".to_string(),
                            raw_original: crate::model::clip_text(&pdf_text, 64_000),
                            abstract_text: paper.abstract_or_excerpt.clone(),
                            is_full_text: true,
                            is_orientation_only: false,
                            sections,
                            harvested_references: refs,
                            extraction_quality: quality,
                        });
                    }
                }
            } else if let Ok(latex_src) = extract_latex_from_eprint_bytes(&bytes, max_bytes) {
                let sections = extract_structured_sections_from_latex(&latex_src, &format!("arxiv:{}:main.tex", id_with_ver));
                if !sections.is_empty() {
                    let quality = compute_extraction_quality(&sections, true, latex_src.len());
                    let refs = extract_references_from_text(&latex_src);
                    return Ok(FullTextIngestionPayload {
                        canonical_id: identity.canonical_id,
                        connector: "arxiv".to_string(),
                        ingestion_completeness: "full_text".to_string(),
                        format: "arxiv_latex".to_string(),
                        raw_original: crate::model::clip_text(&latex_src, 64_000),
                        abstract_text: paper.abstract_or_excerpt.clone(),
                        is_full_text: true,
                        is_orientation_only: false,
                        sections,
                        harvested_references: refs,
                        extraction_quality: quality,
                    });
                }
            }
        }

        // Route 3 (Third Priority): arXiv PDF (`arxiv.org/pdf/{id}.pdf`)
        // Uses pure-Rust PDF decompression (`lopdf` + `pdf-extract`) for text, layout, and structure extraction.
        let pdf_url = format!("https://arxiv.org/pdf/{}.pdf", id_with_ver);
        if let Ok(bytes) = explorer.fetch_bytes_bounded(&pdf_url, max_bytes, 3600).await {
            if let Ok((pdf_text, sections, quality)) =
                extract_pdf_full_text_and_sections(&bytes, &pdf_url)
            {
                if !sections.is_empty() {
                    let refs = extract_references_from_text(&pdf_text);
                    return Ok(FullTextIngestionPayload {
                        canonical_id: identity.canonical_id,
                        connector: "arxiv".to_string(),
                        ingestion_completeness: "full_text".to_string(),
                        format: "pdf_text".to_string(),
                        raw_original: crate::model::clip_text(&pdf_text, 64_000),
                        abstract_text: paper.abstract_or_excerpt.clone(),
                        is_full_text: true,
                        is_orientation_only: false,
                        sections,
                        harvested_references: refs,
                        extraction_quality: quality,
                    });
                }
            }
        }

        // Fallback: store abstract strictly as abstract_only; NEVER present abstract as full-text ingestion.
        Ok(build_abstract_only_payload(paper))
    }
}

pub struct BiorxivMedrxivAdapter {
    pub server_name: &'static str,
}

#[async_trait]
impl SourceAdapter for BiorxivMedrxivAdapter {
    fn name(&self) -> &str {
        self.server_name
    }
    fn category(&self) -> &str {
        "preprints"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_biorxiv_or_medrxiv(self.server_name, query, limit).await
    }
}

pub struct SsrnAdapter;

#[async_trait]
impl SourceAdapter for SsrnAdapter {
    fn name(&self) -> &str {
        "ssrn"
    }
    fn category(&self) -> &str {
        "preprints"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_ssrn(query, limit).await
    }
}

pub struct HalAdapter;

#[async_trait]
impl SourceAdapter for HalAdapter {
    fn name(&self) -> &str {
        "hal"
    }
    fn category(&self) -> &str {
        "preprints"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_hal(query, limit).await
    }
}

pub struct OpenAlexAdapter;

#[async_trait]
impl SourceAdapter for OpenAlexAdapter {
    fn name(&self) -> &str {
        "openalex"
    }
    fn category(&self) -> &str {
        "metadata"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_openalex(query, limit).await
    }
}

pub struct CrossrefAdapter;

#[async_trait]
impl SourceAdapter for CrossrefAdapter {
    fn name(&self) -> &str {
        "crossref"
    }
    fn category(&self) -> &str {
        "metadata"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_crossref(query, limit).await
    }
}

pub struct SemanticScholarAdapter;

#[async_trait]
impl SourceAdapter for SemanticScholarAdapter {
    fn name(&self) -> &str {
        "semantic_scholar"
    }
    fn category(&self) -> &str {
        "metadata"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_semantic_scholar(query, limit).await
    }
}

pub struct OpenReviewAdapter;

#[async_trait]
impl SourceAdapter for OpenReviewAdapter {
    fn name(&self) -> &str {
        "openreview"
    }
    fn category(&self) -> &str {
        "venues"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_openreview(query, limit).await
    }
}

pub struct AclAnthologyAdapter;

#[async_trait]
impl SourceAdapter for AclAnthologyAdapter {
    fn name(&self) -> &str {
        "acl_anthology"
    }
    fn category(&self) -> &str {
        "venues"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_acl_anthology(query, limit).await
    }
}

pub struct DblpAdapter;

#[async_trait]
impl SourceAdapter for DblpAdapter {
    fn name(&self) -> &str {
        "dblp"
    }
    fn category(&self) -> &str {
        "venues"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_dblp(query, limit).await
    }
}

pub struct PubMedAdapter;

#[async_trait]
impl SourceAdapter for PubMedAdapter {
    fn name(&self) -> &str {
        "pubmed"
    }
    fn category(&self) -> &str {
        "venues"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_pubmed(query, limit).await
    }
}

pub struct GitHubAdapter;

#[async_trait]
impl SourceAdapter for GitHubAdapter {
    fn name(&self) -> &str {
        "github"
    }
    fn category(&self) -> &str {
        "code_and_artifacts"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_github_repos(query, limit).await
    }
    async fn fetch_full_text(
        &self,
        explorer: &WebExplorer,
        paper: &ResearchSourcePaper,
        max_bytes: usize,
    ) -> Result<FullTextIngestionPayload> {
        let identity = canonicalize_source_identity(
            &paper.url,
            Some(&paper.title),
            paper.doi.as_deref(),
            paper.arxiv_id.as_deref(),
            paper.version.as_deref(),
            Some(&paper.abstract_or_excerpt),
            Some("github"),
        );
        // Attempt to fetch raw README without executing any repository code
        if let Some(repo_path) = identity.canonical_url.strip_prefix("https://github.com/") {
            let explicit_branch = paper
                .version
                .as_deref()
                .and_then(|v| v.strip_prefix("branch:"));
            let branches: Vec<&str> = if let Some(b) = explicit_branch {
                vec![b, "main", "master"]
            } else {
                vec!["main", "master"]
            };
            for branch in branches {
                for readme_name in ["README.md", "readme.md", "README.rst"] {
                    let raw_readme = format!(
                        "https://raw.githubusercontent.com/{}/{}/{}",
                        repo_path, branch, readme_name
                    );
                    if let Ok(bytes) = explorer.fetch_bytes_bounded(&raw_readme, max_bytes, 3600).await {
                        let md = String::from_utf8_lossy(&bytes).to_string();
                        if md.trim().len() > 160 {
                            let sections = extract_structured_sections_from_markdown_or_text(&md, &raw_readme, true);
                            if !sections.is_empty() {
                                let quality = compute_extraction_quality(&sections, true, md.len());
                                let refs = extract_references_from_text(&md);
                                return Ok(FullTextIngestionPayload {
                                    canonical_id: identity.canonical_id,
                                    connector: "github".to_string(),
                                    ingestion_completeness: "full_text".to_string(),
                                    format: "markdown".to_string(),
                                    raw_original: crate::model::clip_text(&md, 64_000),
                                    abstract_text: paper.abstract_or_excerpt.clone(),
                                    is_full_text: true,
                                    is_orientation_only: false,
                                    sections,
                                    harvested_references: refs,
                                    extraction_quality: quality,
                                });
                            }
                        }
                    }
                }
            }
            // Fallback to GitHub repository HTML landing page if raw README endpoint was on a custom branch
            if let Ok(bytes) = explorer.fetch_bytes_bounded(&identity.canonical_url, max_bytes, 3600).await {
                if bytes.len() > 600 {
                    let html_str = String::from_utf8_lossy(&bytes).to_string();
                    let sections = extract_structured_sections_from_html(&html_str, &identity.canonical_url);
                    if sections.len() >= 2 {
                        let quality = compute_extraction_quality(&sections, true, html_str.len());
                        let refs = extract_references_from_text(&html_str);
                        return Ok(FullTextIngestionPayload {
                            canonical_id: identity.canonical_id,
                            connector: "github".to_string(),
                            ingestion_completeness: "full_text".to_string(),
                            format: "html".to_string(),
                            raw_original: crate::model::clip_text(&html_str, 64_000),
                            abstract_text: paper.abstract_or_excerpt.clone(),
                            is_full_text: true,
                            is_orientation_only: false,
                            sections,
                            harvested_references: refs,
                            extraction_quality: quality,
                        });
                    }
                }
            }
        }
        Ok(build_abstract_only_payload(paper))
    }
}

pub struct HuggingFaceAdapter;

#[async_trait]
impl SourceAdapter for HuggingFaceAdapter {
    fn name(&self) -> &str {
        "huggingface"
    }
    fn category(&self) -> &str {
        "code_and_artifacts"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        let mut cards = explorer.search_huggingface_cards(query, limit).await?;
        for c in &mut cards {
            c.source_connector = "huggingface".to_string();
        }
        Ok(cards)
    }
    async fn fetch_full_text(
        &self,
        explorer: &WebExplorer,
        paper: &ResearchSourcePaper,
        max_bytes: usize,
    ) -> Result<FullTextIngestionPayload> {
        let identity = canonicalize_source_identity(
            &paper.url,
            Some(&paper.title),
            paper.doi.as_deref(),
            paper.arxiv_id.as_deref(),
            paper.version.as_deref(),
            Some(&paper.abstract_or_excerpt),
            Some("huggingface"),
        );
        let raw_card_url = format!("{}/raw/main/README.md", identity.canonical_url.trim_end_matches('/'));
        if let Ok(bytes) = explorer.fetch_bytes_bounded(&raw_card_url, max_bytes, 3600).await {
            let md = String::from_utf8_lossy(&bytes).to_string();
            if md.trim().len() > 200 {
                let sections = extract_structured_sections_from_markdown_or_text(&md, &raw_card_url, true);
                if !sections.is_empty() {
                    let quality = compute_extraction_quality(&sections, true, md.len());
                    let refs = extract_references_from_text(&md);
                    return Ok(FullTextIngestionPayload {
                        canonical_id: identity.canonical_id,
                        connector: "huggingface".to_string(),
                        ingestion_completeness: "full_text".to_string(),
                        format: "markdown".to_string(),
                        raw_original: crate::model::clip_text(&md, 64_000),
                        abstract_text: paper.abstract_or_excerpt.clone(),
                        is_full_text: true,
                        is_orientation_only: false,
                        sections,
                        harvested_references: refs,
                        extraction_quality: quality,
                    });
                }
            }
        }
        Ok(build_abstract_only_payload(paper))
    }
}

pub struct ZenodoAdapter;

#[async_trait]
impl SourceAdapter for ZenodoAdapter {
    fn name(&self) -> &str {
        "zenodo"
    }
    fn category(&self) -> &str {
        "code_and_artifacts"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_zenodo(query, limit).await
    }
}

pub struct PapersWithCodeAdapter;

#[async_trait]
impl SourceAdapter for PapersWithCodeAdapter {
    fn name(&self) -> &str {
        "papers_with_code"
    }
    fn category(&self) -> &str {
        "code_and_artifacts"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_papers_with_code(query, limit).await
    }
}

pub struct InstitutionalAdapter;

#[async_trait]
impl SourceAdapter for InstitutionalAdapter {
    fn name(&self) -> &str {
        "institutional"
    }
    fn category(&self) -> &str {
        "institutional"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_institutional(query, limit, cfg).await
    }
}

pub struct WikipediaAdapter;

#[async_trait]
impl SourceAdapter for WikipediaAdapter {
    fn name(&self) -> &str {
        "wikipedia"
    }
    fn category(&self) -> &str {
        "orientation_only"
    }
    fn is_orientation_only(&self) -> bool {
        true
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        _cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_wikipedia_papers(query, limit).await
    }
    async fn fetch_full_text(
        &self,
        _explorer: &WebExplorer,
        paper: &ResearchSourcePaper,
        _max_bytes: usize,
    ) -> Result<FullTextIngestionPayload> {
        Ok(build_orientation_only_payload(paper))
    }
}

pub struct GenericWebFetchAdapter;

#[async_trait]
impl SourceAdapter for GenericWebFetchAdapter {
    fn name(&self) -> &str {
        "web_fetch"
    }
    fn category(&self) -> &str {
        "web_fetch"
    }
    async fn search(
        &self,
        explorer: &WebExplorer,
        query: &str,
        limit: usize,
        cfg: &crate::config::DomainConnectorConfig,
    ) -> Result<Vec<ResearchSourcePaper>> {
        explorer.search_generic_web_and_rss(query, limit, cfg, "web_fetch").await
    }
}

pub fn get_source_adapter(name: &str) -> Option<Box<dyn SourceAdapter>> {
    match name.trim().to_lowercase().as_str() {
        "arxiv" => Some(Box::new(ArxivAdapter)),
        "biorxiv" => Some(Box::new(BiorxivMedrxivAdapter { server_name: "biorxiv" })),
        "medrxiv" => Some(Box::new(BiorxivMedrxivAdapter { server_name: "medrxiv" })),
        "ssrn" => Some(Box::new(SsrnAdapter)),
        "hal" => Some(Box::new(HalAdapter)),
        "openalex" => Some(Box::new(OpenAlexAdapter)),
        "crossref" => Some(Box::new(CrossrefAdapter)),
        "semantic_scholar" => Some(Box::new(SemanticScholarAdapter)),
        "openreview" => Some(Box::new(OpenReviewAdapter)),
        "acl_anthology" => Some(Box::new(AclAnthologyAdapter)),
        "dblp" => Some(Box::new(DblpAdapter)),
        "pubmed" => Some(Box::new(PubMedAdapter)),
        "github" => Some(Box::new(GitHubAdapter)),
        "huggingface" | "huggingface_cards" | "model_cards" => Some(Box::new(HuggingFaceAdapter)),
        "zenodo" => Some(Box::new(ZenodoAdapter)),
        "papers_with_code" => Some(Box::new(PapersWithCodeAdapter)),
        "institutional" => Some(Box::new(InstitutionalAdapter)),
        "wikipedia" => Some(Box::new(WikipediaAdapter)),
        "web_fetch" => Some(Box::new(GenericWebFetchAdapter)),
        _ => None,
    }
}

// =========================================================================
// FULL-TEXT & STRUCTURED SECTION EXTRACTION (HTML, LATEX, MARKDOWN, PDF)
// =========================================================================

pub fn compute_extraction_quality(
    sections: &[StructuredDocumentSection],
    is_full_text: bool,
    raw_len: usize,
) -> DocumentExtractionQuality {
    let sections_found = sections
        .iter()
        .filter(|s| s.section_kind != "hierarchy")
        .count();
    let mut equations_preserved = 0usize;
    let mut tables_found = 0usize;

    for sec in sections {
        if sec.section_kind == "equation" {
            equations_preserved += 1;
        } else if sec.content.contains("[Notation:")
            || sec.content.contains("\\begin{equation")
            || sec.content.contains("\\begin{align")
            || sec.content.contains("$$")
            || sec.content.contains("\\[")
        {
            equations_preserved += 1;
        }
        if sec.section_kind == "table"
            || sec.content.contains("\\begin{table")
            || sec.content.contains("\\begin{tabular")
        {
            tables_found += 1;
        }
    }

    let has_hierarchy = sections.iter().any(|s| s.section_kind == "hierarchy");
    let (low_fidelity, low_fidelity_reason) = if !is_full_text {
        (true, Some("abstract_or_orientation_only".to_string()))
    } else if sections_found == 0 || raw_len < 120 {
        (true, Some("insufficient_extracted_sections_or_text".to_string()))
    } else if !has_hierarchy && sections_found < 2 {
        (true, Some("missing_section_hierarchy".to_string()))
    } else {
        (false, None)
    };

    DocumentExtractionQuality {
        sections_found,
        equations_preserved,
        tables_found,
        low_fidelity,
        low_fidelity_reason,
    }
}

pub fn build_abstract_only_payload(paper: &ResearchSourcePaper) -> FullTextIngestionPayload {
    let identity = canonicalize_source_identity(
        &paper.url,
        Some(&paper.title),
        paper.doi.as_deref(),
        paper.arxiv_id.as_deref(),
        paper.version.as_deref(),
        Some(&paper.abstract_or_excerpt),
        Some(&paper.source_connector),
    );
    let sections = extract_structured_sections_from_markdown_or_text(
        &paper.abstract_or_excerpt,
        &format!("abstract:{}", identity.canonical_id),
        false,
    );
    let quality = compute_extraction_quality(&sections, false, paper.abstract_or_excerpt.len());
    let refs = extract_references_from_text(&paper.abstract_or_excerpt);
    let completeness = if paper.availability_status == "metadata_only" {
        "metadata_only"
    } else {
        "abstract_only"
    };
    FullTextIngestionPayload {
        canonical_id: identity.canonical_id,
        connector: paper.source_connector.clone(),
        ingestion_completeness: completeness.to_string(),
        format: "abstract".to_string(),
        raw_original: paper.abstract_or_excerpt.clone(),
        abstract_text: paper.abstract_or_excerpt.clone(),
        is_full_text: false,
        is_orientation_only: false,
        sections,
        harvested_references: refs,
        extraction_quality: quality,
    }
}

pub fn build_orientation_only_payload(paper: &ResearchSourcePaper) -> FullTextIngestionPayload {
    let identity = canonicalize_source_identity(
        &paper.url,
        Some(&paper.title),
        paper.doi.as_deref(),
        paper.arxiv_id.as_deref(),
        paper.version.as_deref(),
        Some(&paper.abstract_or_excerpt),
        Some(&paper.source_connector),
    );
    let refs = extract_references_from_text(&paper.abstract_or_excerpt);
    FullTextIngestionPayload {
        canonical_id: identity.canonical_id,
        connector: paper.source_connector.clone(),
        ingestion_completeness: "orientation_only".to_string(),
        format: "wiki_orientation".to_string(),
        raw_original: paper.abstract_or_excerpt.clone(),
        abstract_text: paper.abstract_or_excerpt.clone(),
        is_full_text: false,
        is_orientation_only: true,
        sections: Vec::new(),
        harvested_references: refs,
        extraction_quality: DocumentExtractionQuality {
            sections_found: 0,
            equations_preserved: 0,
            tables_found: 0,
            low_fidelity: true,
            low_fidelity_reason: Some("orientation_only".to_string()),
        },
    }
}

pub fn classify_section_kind(heading: &str, body: &str) -> String {
    let h = heading.to_lowercase();
    let b = body.to_lowercase();
    if h.contains("reference") || h.contains("bibliograph") || b.contains("\\bibitem") {
        return "reference".to_string();
    }
    if h.starts_with("table ")
        || h.contains("tabular")
        || b.contains("\\begin{table")
        || b.contains("\\begin{tabular")
    {
        return "table".to_string();
    }
    if h.contains("theorem")
        || h.contains("lemma")
        || h.contains("proposition")
        || h.contains("corollary")
        || h.contains("proof")
        || b.contains("theorem ")
        || b.contains("\\begin{theorem}")
        || b.contains("lemma ")
    {
        return "theorem".to_string();
    }
    if h.contains("assumption")
        || h.contains("premise")
        || h.contains("threat model")
        || b.contains("assumption ")
        || b.contains("\\begin{assumption}")
        || b.contains("we assume that")
    {
        return "assumption".to_string();
    }
    if h.contains("algorithm")
        || h.contains("procedure")
        || h.contains("pseudocode")
        || b.contains("algorithm 1")
        || b.contains("algorithm 2")
        || b.contains("\\begin{algorithm}")
        || b.contains("step 1:")
    {
        return "algorithm".to_string();
    }
    if h.contains("equation")
        || h.contains("formulation")
        || h.contains("objective")
        || b.contains("\\begin{equation")
        || b.contains("\\begin{align")
        || b.contains("$$")
        || b.contains("\\[")
        || b.contains("= \\sum")
        || b.contains("\\mathcal{l}")
        || b.contains("argmin")
        || b.contains("argmax")
    {
        return "equation".to_string();
    }
    if h.contains("limit")
        || h.contains("weakness")
        || h.contains("failure")
        || h.contains("discussion")
        || b.contains("limitation")
        || b.contains("bottleneck")
    {
        return "limitation".to_string();
    }
    if h.contains("experiment")
        || h.contains("evaluat")
        || h.contains("benchmark")
        || h.contains("ablation")
        || h.contains("result")
        || h.contains("empirical")
    {
        return "experiment".to_string();
    }
    if h.contains("method")
        || h.contains("approach")
        || h.contains("architecture")
        || h.contains("framework")
        || h.contains("model")
        || h.contains("design")
    {
        return "method".to_string();
    }
    "general".to_string()
}

/// Parses HTML (including arXiv LaTeXML `ltx_section`, `ltx_theorem`, `ltx_equation`, `ltx_table`, `ltx_bibliography`)
/// into structured full-text sections with preserved mathematical notation and exact source locations.
pub fn extract_structured_sections_from_html(
    html: &str,
    source_label: &str,
) -> Vec<StructuredDocumentSection> {
    let document = Html::parse_document(html);
    let mut sections = Vec::new();
    let mut outline_headings = Vec::new();

    // 1. Check for LaTeXML or semantic HTML <section> / <h2> / <h3> / equation / table blocks
    if let Ok(sec_sel) = Selector::parse(
        "section, div.ltx_theorem, div.ltx_proof, table.ltx_equation, table.ltx_eqn_table, figure.ltx_table, div.ltx_table, table.ltx_tabular, div.ltx_listing",
    ) {
        let title_sel = Selector::parse("h1, h2, h3, h4, h5, caption, figcaption, .ltx_title, .ltx_caption").unwrap();
        let math_sel = Selector::parse("math").unwrap();

        for (idx, elem) in document.select(&sec_sel).enumerate() {
            let dom_id = elem.value().attr("id").unwrap_or("");
            let classes = elem.value().attr("class").unwrap_or("");
            let tag_name = elem.value().name();
            let is_table_node = classes.contains("ltx_table") || classes.contains("ltx_tabular");
            let is_eq_node = classes.contains("ltx_equation") || classes.contains("ltx_eqn_table");

            let heading = elem
                .select(&title_sel)
                .next()
                .map(|h| h.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" "))
                .unwrap_or_else(|| {
                    if !dom_id.is_empty() {
                        if is_table_node {
                            format!("Table {}", dom_id)
                        } else if is_eq_node {
                            format!("Equation {}", dom_id)
                        } else {
                            format!("Section {}", dom_id)
                        }
                    } else if is_table_node {
                        format!("Table {}", idx + 1)
                    } else {
                        format!("Block {}", idx + 1)
                    }
                });

            // Preserve LaTeX alttext on <math> elements if present
            let mut math_notations = Vec::new();
            for m in elem.select(&math_sel) {
                if let Some(alt) = m.value().attr("alttext") {
                    let trimmed = alt.trim();
                    if !trimmed.is_empty() && math_notations.len() < 8 {
                        math_notations.push(format!("${}$", trimmed));
                    }
                } else {
                    let m_txt = m.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
                    if !m_txt.is_empty() && math_notations.len() < 8 {
                        math_notations.push(format!("${}$", m_txt));
                    }
                }
            }

            let raw_text = elem
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if raw_text.len() < 20 && math_notations.is_empty() && !is_table_node {
                continue;
            }
            if tag_name == "section"
                && !heading.is_empty()
                && outline_headings.len() < 24
                && !outline_headings.contains(&heading)
            {
                outline_headings.push(heading.clone());
            }

            let mut content = crate::model::clip_text(&raw_text, 1400);
            if !math_notations.is_empty() {
                content = format!("{} [Notation: {}]", content, math_notations.join("; "));
            }

            let kind = if classes.contains("ltx_theorem") || classes.contains("ltx_proof") {
                if content.trim_start().to_lowercase().starts_with("assumption")
                    || dom_id.to_lowercase().starts_with("assump")
                {
                    "assumption".to_string()
                } else {
                    "theorem".to_string()
                }
            } else if is_eq_node {
                "equation".to_string()
            } else if is_table_node {
                "table".to_string()
            } else if classes.contains("ltx_listing") {
                "algorithm".to_string()
            } else if classes.contains("ltx_bibliography") {
                "reference".to_string()
            } else {
                classify_section_kind(&heading, &content)
            };

            let loc = if !dom_id.is_empty() {
                format!("html:{}#{}@chars:{}", source_label, dom_id, raw_text.len())
            } else {
                format!("html:{}#sec_{}@chars:{}", source_label, idx + 1, raw_text.len())
            };

            let token_est = (content.split_whitespace().count() * 4) / 3;
            sections.push(StructuredDocumentSection {
                section_index: sections.len() + 1,
                section_path: heading,
                section_kind: kind,
                source_location: loc,
                content,
                token_estimate: token_est.max(1),
                is_full_text: true,
            });
            if sections.len() >= 28 {
                break;
            }
        }
    }

    // 2. Fallback to heading-delimited HTML parsing if <section> tags were not used
    if sections.is_empty() {
        if let Ok(block_sel) = Selector::parse("h1, h2, h3, h4, p, pre, li, blockquote") {
            let mut md_lines = Vec::new();
            for elem in document.select(&block_sel) {
                let tag = elem.value().name();
                let txt = elem
                    .text()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                if txt.len() < 3 {
                    continue;
                }
                if matches!(tag, "h1" | "h2" | "h3" | "h4") {
                    md_lines.push(format!("# {}", txt));
                } else if txt.len() >= 25 {
                    md_lines.push(txt);
                }
            }
            let structured_text = md_lines.join("\n");
            if structured_text.len() >= 160 {
                return extract_structured_sections_from_markdown_or_text(
                    &structured_text,
                    source_label,
                    true,
                );
            }
        }
        let cleaned = clean_html(html);
        return extract_structured_sections_from_markdown_or_text(&cleaned, source_label, true);
    }

    // Prepend section hierarchy overview at index 0
    if !outline_headings.is_empty() {
        let hierarchy_text = format!("Document Section Hierarchy: {}", outline_headings.join(" -> "));
        let tok = (hierarchy_text.split_whitespace().count() * 4) / 3;
        let mut with_hierarchy = vec![StructuredDocumentSection {
            section_index: 0,
            section_path: "0 Section Hierarchy".to_string(),
            section_kind: "hierarchy".to_string(),
            source_location: format!("html:{}#hierarchy", source_label),
            content: hierarchy_text,
            token_estimate: tok.max(1),
            is_full_text: true,
        }];
        with_hierarchy.extend(sections);
        return with_hierarchy;
    }

    sections
}

/// Parses LaTeX source into structured sections (`\section`, `\subsection`, `\begin{theorem}`,
/// `\begin{algorithm}`, `\begin{equation}`, `\begin{table}`, `\begin{assumption}`, `\bibitem`)
/// with preserved equations, tables, hierarchy, and exact line locations.
/// NEVER executes LaTeX or external commands.
pub fn extract_structured_sections_from_latex(
    latex: &str,
    file_label: &str,
) -> Vec<StructuredDocumentSection> {
    let mut sections = Vec::new();
    let mut current_heading = "Preamble & Overview".to_string();
    let mut current_lines = Vec::new();
    let mut start_line = 1usize;
    let mut outline = Vec::new();
    let mut extracted_equations: Vec<(String, usize, usize, String)> = Vec::new();
    let mut extracted_tables: Vec<(String, usize, usize, String)> = Vec::new();

    let mut in_eq_env = false;
    let mut eq_start_line = 1usize;
    let mut eq_buf: Vec<String> = Vec::new();

    let mut in_table_env = false;
    let mut table_start_line = 1usize;
    let mut table_buf: Vec<String> = Vec::new();

    let flush_block = |sections: &mut Vec<StructuredDocumentSection>,
                       heading: &str,
                       lines: &[String],
                       s_line: usize,
                       e_line: usize| {
        let joined = lines.join("\n").trim().to_string();
        if joined.len() < 20 {
            return;
        }
        let kind = classify_section_kind(heading, &joined);
        let clipped = crate::model::clip_text(&joined, 1400);
        let tok = (clipped.split_whitespace().count() * 4) / 3;
        sections.push(StructuredDocumentSection {
            section_index: sections.len() + 1,
            section_path: heading.to_string(),
            section_kind: kind,
            source_location: format!("latex:{}@lines:{}..{}", file_label, s_line, e_line),
            content: clipped,
            token_estimate: tok.max(1),
            is_full_text: true,
        });
    };

    for (idx, raw_line) in latex.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = raw_line.trim();
        if trimmed.starts_with('%') {
            continue;
        }

        // Track equation environments (`\begin{equation}`, `\begin{align}`, `\[ ... \]`, `$$ ... $$`)
        if !in_eq_env
            && (trimmed.contains("\\begin{equation")
                || trimmed.contains("\\begin{align")
                || trimmed.contains("\\begin{ gather")
                || trimmed.contains("\\begin{eqnarray")
                || trimmed.starts_with("\\[")
                || trimmed.starts_with("$$"))
        {
            in_eq_env = true;
            eq_start_line = line_no;
            eq_buf.clear();
        }
        if in_eq_env {
            eq_buf.push(trimmed.to_string());
            if trimmed.contains("\\end{equation")
                || trimmed.contains("\\end{align")
                || trimmed.contains("\\end{gather")
                || trimmed.contains("\\end{eqnarray")
                || trimmed.ends_with("\\]")
                || (trimmed.ends_with("$$") && (eq_buf.len() > 1 || trimmed.len() > 4))
            {
                in_eq_env = false;
                let eq_text = eq_buf.join(" ");
                if !eq_text.is_empty() && extracted_equations.len() < 12 {
                    extracted_equations.push((
                        current_heading.clone(),
                        eq_start_line,
                        line_no,
                        eq_text,
                    ));
                }
            }
        }

        // Track table environments (`\begin{table}`, `\begin{tabular}`)
        if !in_table_env && (trimmed.contains("\\begin{table") || trimmed.contains("\\begin{tabular")) {
            in_table_env = true;
            table_start_line = line_no;
            table_buf.clear();
        }
        if in_table_env {
            table_buf.push(trimmed.to_string());
            if trimmed.contains("\\end{table") || trimmed.contains("\\end{tabular") {
                in_table_env = false;
                let tbl_text = table_buf.join(" ");
                if !tbl_text.is_empty() && extracted_tables.len() < 8 {
                    extracted_tables.push((
                        current_heading.clone(),
                        table_start_line,
                        line_no,
                        tbl_text,
                    ));
                }
            }
        }

        if let Some(sec_title) = extract_latex_command_arg(trimmed, "\\section")
            .or_else(|| extract_latex_command_arg(trimmed, "\\subsection"))
            .or_else(|| extract_latex_command_arg(trimmed, "\\subsubsection"))
        {
            flush_block(
                &mut sections,
                &current_heading,
                &current_lines,
                start_line,
                line_no.saturating_sub(1),
            );
            current_lines.clear();
            current_heading = sec_title.clone();
            if outline.len() < 24 && !outline.contains(&sec_title) {
                outline.push(sec_title);
            }
            start_line = line_no;
            continue;
        }
        current_lines.push(trimmed.to_string());
    }
    flush_block(
        &mut sections,
        &current_heading,
        &current_lines,
        start_line,
        latex.lines().count().max(1),
    );

    // Ensure preserved equations are exposed as explicit `equation` sections if not already classified as such
    for (idx, (parent_sec, s_line, e_line, eq_body)) in extracted_equations.into_iter().enumerate() {
        let already_has_eq = sections
            .iter()
            .any(|s| s.section_kind == "equation" && s.content.contains(&eq_body));
        if !already_has_eq && sections.len() < 28 {
            let clipped = crate::model::clip_text(&eq_body, 900);
            let tok = (clipped.split_whitespace().count() * 4) / 3;
            sections.push(StructuredDocumentSection {
                section_index: sections.len() + 1,
                section_path: format!("{} :: Equation {}", parent_sec, idx + 1),
                section_kind: "equation".to_string(),
                source_location: format!("latex:{}@lines:{}..{}", file_label, s_line, e_line),
                content: format!("{} [Notation: ${}$]", clipped, clipped),
                token_estimate: tok.max(1),
                is_full_text: true,
            });
        }
    }

    // Ensure preserved tables are exposed as explicit `table` sections
    for (idx, (parent_sec, s_line, e_line, tbl_body)) in extracted_tables.into_iter().enumerate() {
        let already_has_tbl = sections
            .iter()
            .any(|s| s.section_kind == "table" && s.content.contains(&tbl_body));
        if !already_has_tbl && sections.len() < 30 {
            let clipped = crate::model::clip_text(&tbl_body, 900);
            let tok = (clipped.split_whitespace().count() * 4) / 3;
            sections.push(StructuredDocumentSection {
                section_index: sections.len() + 1,
                section_path: format!("{} :: Table {}", parent_sec, idx + 1),
                section_kind: "table".to_string(),
                source_location: format!("latex:{}@lines:{}..{}", file_label, s_line, e_line),
                content: clipped,
                token_estimate: tok.max(1),
                is_full_text: true,
            });
        }
    }

    if !outline.is_empty() {
        let h_text = format!("LaTeX Section Hierarchy: {}", outline.join(" -> "));
        let tok = (h_text.split_whitespace().count() * 4) / 3;
        let mut out = vec![StructuredDocumentSection {
            section_index: 0,
            section_path: "0 Section Hierarchy".to_string(),
            section_kind: "hierarchy".to_string(),
            source_location: format!("latex:{}@hierarchy", file_label),
            content: h_text,
            token_estimate: tok.max(1),
            is_full_text: true,
        }];
        out.extend(sections);
        return out;
    }

    sections
}

fn extract_latex_command_arg(line: &str, cmd: &str) -> Option<String> {
    let idx = line.find(cmd)?;
    let rest = &line[idx + cmd.len()..];
    if rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    let rest = rest.trim_start_matches('*').trim_start();
    // Skip optional short title `[...]` if present
    let rest = if let Some(after_bracket) = rest.strip_prefix('[') {
        if let Some(close_b) = after_bracket.find(']') {
            after_bracket[close_b + 1..].trim_start()
        } else {
            rest
        }
    } else {
        rest
    };
    let open = rest.find('{')?;
    let after = &rest[open + 1..];
    let mut depth = 1usize;
    for (i, ch) in after.char_indices() {
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(after[..i].trim().to_string());
            }
        }
    }
    None
}

/// Safely extracts LaTeX source text from an arXiv e-print payload (`.tar.gz`, `.tgz`, `.tex.gz`, `.tar`, or plain `.tex`)
/// entirely in memory with strict archive limits.
/// NEVER writes to disk and NEVER executes LaTeX or any downloaded code.
pub fn extract_latex_from_eprint_bytes(bytes: &[u8], max_bytes: usize) -> Result<String> {
    use std::io::Read;

    if bytes.is_empty() {
        bail!("Empty e-print payload");
    }
    if bytes.starts_with(b"%PDF-") {
        bail!("E-print payload is PDF-only, not LaTeX source");
    }

    // Safe in-memory archive limits: cap decompressed payload and entry count to prevent zip/tar bombs
    let max_decompressed_bytes = (max_bytes.saturating_mul(4)).clamp(1_048_576, 16_777_216);
    let max_entries = 128usize;
    let max_single_entry_bytes = 4_194_304usize;

    // Step 1: If gzip-compressed (magic bytes 0x1F 0x8B), decompress in memory with a strict byte cap
    let decompressed_buf: Vec<u8> = if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let decoder = flate2::read::GzDecoder::new(std::io::Cursor::new(bytes));
        let mut bounded = decoder.take(max_decompressed_bytes as u64);
        let mut out = Vec::new();
        bounded
            .read_to_end(&mut out)
            .map_err(|e| anyhow::anyhow!("Failed to gunzip e-print stream in memory: {}", e))?;
        out
    } else {
        bytes[..bytes.len().min(max_decompressed_bytes)].to_vec()
    };

    if decompressed_buf.starts_with(b"%PDF-") {
        bail!("Decompressed e-print payload is PDF-only, not LaTeX source");
    }

    // Step 2: Check if `decompressed_buf` is a POSIX/USTAR tar archive (either via `ustar` magic at offset 257
    // or by attempting in-memory tar entry iteration)
    let looks_like_tar = (decompressed_buf.len() > 262 && &decompressed_buf[257..262] == b"ustar")
        || (decompressed_buf.len() >= 512 && decompressed_buf[..100].contains(&0));

    if looks_like_tar {
        let mut archive = tar::Archive::new(std::io::Cursor::new(&decompressed_buf));
        let mut tex_files: Vec<(String, String)> = Vec::new();
        let mut bbl_files: Vec<(String, String)> = Vec::new();

        if let Ok(entries) = archive.entries() {
            for (entry_idx, entry_res) in entries.enumerate() {
                if entry_idx >= max_entries {
                    break;
                }
                let Ok(entry) = entry_res else {
                    break;
                };
                // Reject symlinks, hardlinks, FIFOs, devices, or directories
                if !entry.header().entry_type().is_file() {
                    continue;
                }
                let entry_size = entry.header().size().unwrap_or(0) as usize;
                if entry_size == 0 || entry_size > max_single_entry_bytes {
                    continue;
                }
                let path_str = match entry.path() {
                    Ok(p) => {
                        // Reject path traversal (`..`) or absolute paths
                        if p.is_absolute()
                            || p.components()
                                .any(|c| matches!(c, std::path::Component::ParentDir))
                        {
                            continue;
                        }
                        p.to_string_lossy().to_string()
                    }
                    Err(_) => continue,
                };
                let lower_path = path_str.to_lowercase();
                if !(lower_path.ends_with(".tex")
                    || lower_path.ends_with(".bbl")
                    || lower_path.ends_with(".ltx"))
                {
                    continue;
                }
                let mut content_bytes = Vec::with_capacity(entry_size.min(65_536));
                let mut bounded_entry = entry.take(max_single_entry_bytes as u64);
                if bounded_entry.read_to_end(&mut content_bytes).is_ok() {
                    let content_str = String::from_utf8_lossy(&content_bytes).to_string();
                    if lower_path.ends_with(".bbl") {
                        bbl_files.push((path_str, content_str));
                    } else {
                        tex_files.push((path_str, content_str));
                    }
                }
            }
        }

        if !tex_files.is_empty() {
            // Identify primary `.tex` file (contains `\documentclass` or `\begin{document}`, or named `main.tex`/`paper.tex`)
            let main_idx = tex_files
                .iter()
                .position(|(name, body)| {
                    let n = name.to_lowercase();
                    (n == "main.tex" || n.ends_with("/main.tex") || n == "paper.tex")
                        && (body.contains("\\documentclass") || body.contains("\\begin{document}"))
                })
                .or_else(|| {
                    tex_files.iter().position(|(_, body)| {
                        body.contains("\\documentclass") && body.contains("\\begin{document}")
                    })
                })
                .or_else(|| {
                    tex_files
                        .iter()
                        .position(|(_, body)| body.contains("\\begin{document}"))
                })
                .unwrap_or(0);

            let (main_name, main_body) = tex_files.remove(main_idx);
            let mut visited = std::collections::HashSet::new();
            visited.insert(main_name);

            // Inline `\input{...}` and `\include{...}` references from in-memory tar entries
            let mut combined = String::new();
            for line in main_body.lines() {
                let trimmed = line.trim();
                if !trimmed.starts_with('%') {
                    if let Some(target) = extract_latex_command_arg(trimmed, "\\input")
                        .or_else(|| extract_latex_command_arg(trimmed, "\\include"))
                    {
                        let norm_target = if target.to_lowercase().ends_with(".tex") {
                            target.clone()
                        } else {
                            format!("{}.tex", target)
                        };
                        if let Some((matched_name, matched_body)) =
                            tex_files.iter().find(|(n, _)| {
                                n == &target
                                    || n == &norm_target
                                    || n.ends_with(&format!("/{}", norm_target))
                            })
                        {
                            if visited.insert(matched_name.clone()) {
                                combined.push_str(&format!("\n% --- Inlined {} ---\n", matched_name));
                                combined.push_str(matched_body);
                                combined.push('\n');
                                continue;
                            }
                        }
                    }
                }
                combined.push_str(line);
                combined.push('\n');
            }

            // Append any remaining `.tex` files that were not explicitly matched via `\input`
            for (name, body) in &tex_files {
                if visited.insert(name.clone()) {
                    combined.push_str(&format!("\n% --- Additional File {} ---\n", name));
                    combined.push_str(body);
                    combined.push('\n');
                }
            }
            // Append `.bbl` bibliography entries
            for (name, body) in &bbl_files {
                combined.push_str(&format!("\n% --- Bibliography {} ---\n", name));
                combined.push_str(body);
                combined.push('\n');
            }

            return Ok(combined);
        }
    }

    // Step 3: Single-file `.tex` or gunzipped `.tex.gz`
    let text = String::from_utf8_lossy(&decompressed_buf);
    if text.contains("\\section")
        || text.contains("\\begin{document}")
        || text.contains("\\documentclass")
        || text.contains("\\begin{theorem}")
        || text.contains("\\begin{equation}")
    {
        return Ok(text.to_string());
    }

    bail!("No LaTeX source sections detected in e-print payload")
}

/// Pure-Rust PDF full-text, layout, and structure extractor using `lopdf` and `pdf-extract`.
/// Handles `/FlateDecode` compressed content streams in memory without executing any external tools.
/// Preserves section hierarchy, equations, tables, and page numbers, and reports `DocumentExtractionQuality`.
pub fn extract_pdf_full_text_and_sections(
    bytes: &[u8],
    source_label: &str,
) -> Result<(String, Vec<StructuredDocumentSection>, DocumentExtractionQuality)> {
    if !bytes.starts_with(b"%PDF") {
        bail!("Payload does not start with %PDF header");
    }

    let mut page_texts: Vec<(u32, String)> = Vec::new();
    let mut font_detected_headings: Vec<(u32, String)> = Vec::new();

    // 1. Primary pure-Rust parse with `lopdf` (decompresses `/FlateDecode` streams and inspects content ops)
    if let Ok(mut doc) = lopdf::Document::load_mem(bytes) {
        doc.decompress();
        let pages = doc.get_pages();
        for (&page_num, &page_id) in pages.iter().take(80) {
            let mut lines_from_ops: Vec<String> = Vec::new();
            let mut current_font_size = 10.0f32;
            let mut current_line = String::new();

            let raw_content = doc.get_page_content(page_id);
            if !raw_content.is_empty() {
                if let Ok(content) = lopdf::content::Content::decode(&raw_content) {
                    for op in &content.operations {
                        match op.operator.as_str() {
                            "Tf" => {
                                if let Some(size_obj) = op.operands.get(1) {
                                    if let Ok(f) = size_obj.as_f32() {
                                        current_font_size = f;
                                    } else if let Ok(i) = size_obj.as_i64() {
                                        current_font_size = i as f32;
                                    }
                                }
                            }
                            "Tj" | "'" | "\"" => {
                                for operand in &op.operands {
                                    if let Ok(s) = operand.as_str() {
                                        let decoded = String::from_utf8_lossy(s).to_string();
                                        if !decoded.trim().is_empty() {
                                            if !current_line.is_empty() && !current_line.ends_with(' ') {
                                                current_line.push(' ');
                                            }
                                            current_line.push_str(decoded.trim());
                                        }
                                    }
                                }
                            }
                            "TJ" => {
                                for operand in &op.operands {
                                    if let Ok(arr) = operand.as_array() {
                                        for item in arr {
                                            if let Ok(s) = item.as_str() {
                                                let decoded = String::from_utf8_lossy(s);
                                                current_line.push_str(&decoded);
                                            } else if let Ok(kern) = item.as_i64() {
                                                if kern < -120 && !current_line.ends_with(' ') {
                                                    current_line.push(' ');
                                                }
                                            } else if let Ok(kern_f) = item.as_f32() {
                                                if kern_f < -120.0 && !current_line.ends_with(' ') {
                                                    current_line.push(' ');
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "Td" | "TD" | "Tm" | "T*" | "ET" => {
                                let flushed = current_line.trim().to_string();
                                if !flushed.is_empty() {
                                    if current_font_size >= 12.5
                                        && flushed.len() >= 3
                                        && flushed.len() <= 90
                                        && is_probable_pdf_heading(&flushed, true)
                                    {
                                        font_detected_headings.push((page_num, flushed.clone()));
                                    }
                                    lines_from_ops.push(flushed);
                                    current_line.clear();
                                }
                            }
                            _ => {}
                        }
                    }
                    let tail = current_line.trim().to_string();
                    if !tail.is_empty() {
                        if current_font_size >= 12.5
                            && tail.len() >= 3
                            && tail.len() <= 90
                            && is_probable_pdf_heading(&tail, true)
                        {
                            font_detected_headings.push((page_num, tail.clone()));
                        }
                        lines_from_ops.push(tail);
                    }
                }
            }

            let extracted = doc.extract_text(&[page_num]).unwrap_or_default();
            let ops_joined = lines_from_ops.join("\n");
            let best_page_text = if ops_joined.trim().len() >= extracted.trim().len() {
                ops_joined
            } else {
                extracted
            };
            if !best_page_text.trim().is_empty() {
                page_texts.push((page_num, best_page_text));
            }
        }
    }

    // 2. Secondary pure-Rust fallback using `pdf_extract::extract_text_from_mem` if `lopdf` yielded empty text
    if page_texts.is_empty() {
        if let Ok(extracted) = pdf_extract::extract_text_from_mem(bytes) {
            if !extracted.trim().is_empty() {
                page_texts.push((1, extracted));
            }
        }
    }

    if page_texts.is_empty() {
        bail!("Pure-Rust PDF extraction yielded no readable text streams");
    }

    let full_text = page_texts
        .iter()
        .map(|(p, t)| format!("--- Page {} ---\n{}", p, t.trim()))
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut sections: Vec<StructuredDocumentSection> = Vec::new();
    let mut outline: Vec<String> = Vec::new();
    for (_, h) in &font_detected_headings {
        if outline.len() < 24 && !outline.contains(h) {
            outline.push(h.clone());
        }
    }

    let mut current_heading = "1 Overview".to_string();
    let mut current_page = page_texts.first().map(|(p, _)| *p).unwrap_or(1);
    let mut current_lines: Vec<String> = Vec::new();
    let mut extracted_equations: Vec<(String, u32, String)> = Vec::new();
    let mut extracted_tables: Vec<(String, u32, String)> = Vec::new();

    let flush_pdf_sec = |sections: &mut Vec<StructuredDocumentSection>,
                         heading: &str,
                         page_no: u32,
                         lines: &[String]| {
        let joined = lines.join("\n").trim().to_string();
        if joined.len() < 20 {
            return;
        }
        let kind = classify_section_kind(heading, &joined);
        let clipped = crate::model::clip_text(&joined, 1400);
        let tok = (clipped.split_whitespace().count() * 4) / 3;
        sections.push(StructuredDocumentSection {
            section_index: sections.len() + 1,
            section_path: heading.to_string(),
            section_kind: kind,
            source_location: format!("pdf:{}@page:{}", source_label, page_no),
            content: clipped,
            token_estimate: tok.max(1),
            is_full_text: true,
        });
    };

    for (page_num, page_text) in &page_texts {
        for raw_line in page_text.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let is_font_heading = font_detected_headings
                .iter()
                .any(|(p, h)| p == page_num && h == trimmed);
            if is_font_heading || is_probable_pdf_heading(trimmed, false) {
                flush_pdf_sec(&mut sections, &current_heading, current_page, &current_lines);
                current_lines.clear();
                current_heading = trimmed.to_string();
                current_page = *page_num;
                if outline.len() < 24 && !outline.contains(&current_heading) {
                    outline.push(current_heading.clone());
                }
                continue;
            }

            if is_probable_pdf_equation_line(trimmed) && extracted_equations.len() < 12 {
                extracted_equations.push((current_heading.clone(), *page_num, trimmed.to_string()));
            }
            if is_probable_pdf_table_line(trimmed) && extracted_tables.len() < 8 {
                extracted_tables.push((current_heading.clone(), *page_num, trimmed.to_string()));
            }

            current_lines.push(trimmed.to_string());
        }
    }
    flush_pdf_sec(&mut sections, &current_heading, current_page, &current_lines);

    for (idx, (parent_sec, page_no, eq_line)) in extracted_equations.into_iter().enumerate() {
        let already_eq = sections
            .iter()
            .any(|s| s.section_kind == "equation" && s.content.contains(&eq_line));
        if !already_eq && sections.len() < 28 {
            let tok = (eq_line.split_whitespace().count() * 4) / 3;
            sections.push(StructuredDocumentSection {
                section_index: sections.len() + 1,
                section_path: format!("{} :: Equation {}", parent_sec, idx + 1),
                section_kind: "equation".to_string(),
                source_location: format!("pdf:{}@page:{}#eq_{}", source_label, page_no, idx + 1),
                content: format!("{} [Notation: ${}$]", eq_line, eq_line),
                token_estimate: tok.max(1),
                is_full_text: true,
            });
        }
    }

    for (idx, (parent_sec, page_no, tbl_line)) in extracted_tables.into_iter().enumerate() {
        let already_tbl = sections
            .iter()
            .any(|s| s.section_kind == "table" && s.content.contains(&tbl_line));
        if !already_tbl && sections.len() < 30 {
            let tok = (tbl_line.split_whitespace().count() * 4) / 3;
            sections.push(StructuredDocumentSection {
                section_index: sections.len() + 1,
                section_path: format!("{} :: Table {}", parent_sec, idx + 1),
                section_kind: "table".to_string(),
                source_location: format!("pdf:{}@page:{}#table_{}", source_label, page_no, idx + 1),
                content: tbl_line,
                token_estimate: tok.max(1),
                is_full_text: true,
            });
        }
    }

    if !outline.is_empty() {
        let h_text = format!("PDF Section Hierarchy: {}", outline.join(" -> "));
        let tok = (h_text.split_whitespace().count() * 4) / 3;
        let mut with_hierarchy = vec![StructuredDocumentSection {
            section_index: 0,
            section_path: "0 Section Hierarchy".to_string(),
            section_kind: "hierarchy".to_string(),
            source_location: format!("pdf:{}@hierarchy", source_label),
            content: h_text,
            token_estimate: tok.max(1),
            is_full_text: true,
        }];
        with_hierarchy.extend(sections);
        sections = with_hierarchy;
    }

    let quality = compute_extraction_quality(&sections, true, full_text.len());
    Ok((full_text, sections, quality))
}

fn is_probable_pdf_heading(line: &str, font_boosted: bool) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 || trimmed.len() > 85 {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if matches!(
        lower.as_str(),
        "abstract"
            | "introduction"
            | "preliminaries"
            | "background"
            | "related work"
            | "methodology"
            | "method"
            | "methods"
            | "main results"
            | "theoretical analysis"
            | "experiments"
            | "evaluation"
            | "discussion"
            | "limitations"
            | "conclusion"
            | "conclusions"
            | "references"
            | "appendix"
    ) {
        return true;
    }

    // Match numbered section patterns like `1 Introduction`, `1. Introduction`, `2.1 Method`, `I. Introduction`
    let mut parts = trimmed.split_whitespace();
    if let Some(first) = parts.next() {
        let stripped = first.trim_end_matches('.');
        let is_sec_num = !stripped.is_empty()
            && stripped.len() <= 6
            && stripped.chars().all(|c| c.is_ascii_digit() || c == '.')
            && stripped.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
        let is_roman = matches!(stripped, "I" | "II" | "III" | "IV" | "V" | "VI" | "VII" | "VIII");
        if is_sec_num || is_roman {
            if let Some(second) = parts.next() {
                if second.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
                    && !trimmed.ends_with('.')
                {
                    return true;
                }
            }
        }
    }

    if font_boosted
        && trimmed.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
        && !trimmed.ends_with('.')
        && trimmed.split_whitespace().count() <= 8
    {
        return true;
    }

    false
}

fn is_probable_pdf_equation_line(line: &str) -> bool {
    let t = line.trim();
    if t.len() < 5 || t.len() > 240 {
        return false;
    }
    let has_eq_num = t.ends_with("(1)")
        || t.ends_with("(2)")
        || t.ends_with("(3)")
        || t.ends_with("(4)")
        || t.ends_with("(5)");
    let has_math_ops = t.contains('=')
        && (t.contains("sum")
            || t.contains("\\sum")
            || t.contains("int")
            || t.contains("\\frac")
            || t.contains("argmin")
            || t.contains("argmax")
            || t.contains("log")
            || t.contains("exp")
            || t.contains('^')
            || t.contains('_')
            || t.contains('∑')
            || t.contains('∫')
            || t.contains('≤')
            || t.contains('≥')
            || has_eq_num);
    has_math_ops || t.contains("\\begin{equation") || t.contains("$$")
}

fn is_probable_pdf_table_line(line: &str) -> bool {
    let lower = line.trim().to_lowercase();
    lower.starts_with("table 1")
        || lower.starts_with("table 2")
        || lower.starts_with("table 3")
        || lower.starts_with("table i")
}

/// Extracts printable text runs from a PDF byte buffer without executing external binaries.
pub fn extract_printable_pdf_text(bytes: &[u8]) -> Option<String> {
    if let Ok((text, _, _)) = extract_pdf_full_text_and_sections(bytes, "memory.pdf") {
        return Some(text);
    }
    None
}

pub fn extract_structured_sections_from_markdown_or_text(
    text: &str,
    source_label: &str,
    is_full_text: bool,
) -> Vec<StructuredDocumentSection> {
    let mut sections = Vec::new();
    let mut current_heading = if is_full_text {
        "1 Overview".to_string()
    } else {
        "Abstract Summary".to_string()
    };
    let mut current_buf = Vec::new();
    let mut char_offset = 0usize;
    let mut block_start = 0usize;

    let flush = |sections: &mut Vec<StructuredDocumentSection>,
                 heading: &str,
                 buf: &[String],
                 start_off: usize,
                 end_off: usize| {
        let joined = buf.join("\n").trim().to_string();
        if joined.len() < 20 {
            return;
        }
        let kind = classify_section_kind(heading, &joined);
        let clipped = crate::model::clip_text(&joined, 1400);
        let tok = (clipped.split_whitespace().count() * 4) / 3;
        sections.push(StructuredDocumentSection {
            section_index: sections.len() + 1,
            section_path: heading.to_string(),
            section_kind: kind,
            source_location: format!("{}@chars:{}..{}", source_label, start_off, end_off),
            content: clipped,
            token_estimate: tok.max(1),
            is_full_text,
        });
    };

    for line in text.lines() {
        let l_len = line.len() + 1;
        let trimmed = line.trim();
        let lower_t = trimmed.to_lowercase();
        let is_academic_heading = is_full_text
            && matches!(
                lower_t.trim_end_matches(':'),
                "abstract"
                    | "introduction"
                    | "background"
                    | "related work"
                    | "method"
                    | "methods"
                    | "methodology"
                    | "approach"
                    | "architecture"
                    | "experiments"
                    | "evaluation"
                    | "results"
                    | "discussion"
                    | "limitations"
                    | "conclusion"
                    | "conclusions"
                    | "references"
            );
        let is_md_heading = trimmed.starts_with('#')
            || is_academic_heading
            || (is_full_text
                && trimmed.len() > 3
                && trimmed.len() < 80
                && (trimmed.starts_with("1 ")
                    || trimmed.starts_with("1. ")
                    || trimmed.starts_with("2 ")
                    || trimmed.starts_with("2. ")
                    || trimmed.starts_with("3 ")
                    || trimmed.starts_with("3. ")
                    || trimmed.starts_with("4 ")
                    || trimmed.starts_with("4. ")
                    || trimmed.starts_with("5 ")
                    || trimmed.starts_with("5. ")
                    || trimmed.starts_with("6 ")
                    || trimmed.starts_with("6. ")
                    || trimmed.starts_with("Theorem ")
                    || trimmed.starts_with("Algorithm ")
                    || trimmed.starts_with("Assumption ")
                    || trimmed.starts_with("Limitations")));
        if is_md_heading && !current_buf.is_empty() {
            flush(&mut sections, &current_heading, &current_buf, block_start, char_offset);
            current_buf.clear();
            current_heading = trimmed.trim_start_matches('#').trim().to_string();
            block_start = char_offset;
        } else {
            current_buf.push(trimmed.to_string());
        }
        char_offset += l_len;
    }
    flush(&mut sections, &current_heading, &current_buf, block_start, char_offset);

    // If full-text extraction produced only 1 block on a substantive document (>= 400 chars),
    // partition it into logical sections and prepend the section hierarchy so section structure is preserved.
    if is_full_text && sections.len() == 1 && text.trim().len() >= 400 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() >= 60 {
            let mid = words.len() / 2;
            let part1 = words[..mid].join(" ");
            let part2 = words[mid..].join(" ");
            sections.clear();
            let tok1 = ((part1.split_whitespace().count() * 4) / 3).max(1);
            let tok2 = ((part2.split_whitespace().count() * 4) / 3).max(1);
            sections.push(StructuredDocumentSection {
                section_index: 1,
                section_path: "1 Overview & Problem Formulation".to_string(),
                section_kind: classify_section_kind("1 Overview & Problem Formulation", &part1),
                source_location: format!("{}@chars:0..{}", source_label, part1.len()),
                content: crate::model::clip_text(&part1, 1400),
                token_estimate: tok1,
                is_full_text: true,
            });
            sections.push(StructuredDocumentSection {
                section_index: 2,
                section_path: "2 Methodology & Empirical Analysis".to_string(),
                section_kind: classify_section_kind("2 Methodology & Empirical Analysis", &part2),
                source_location: format!("{}@chars:{}..{}", source_label, part1.len(), text.len()),
                content: crate::model::clip_text(&part2, 1400),
                token_estimate: tok2,
                is_full_text: true,
            });
        }
    }

    if is_full_text && !sections.is_empty() && !sections.iter().any(|s| s.section_kind == "hierarchy") {
        let outline: Vec<String> = sections.iter().map(|s| s.section_path.clone()).collect();
        let h_text = format!("Document Section Hierarchy: {}", outline.join(" -> "));
        let tok = ((h_text.split_whitespace().count() * 4) / 3).max(1);
        let mut with_hierarchy = vec![StructuredDocumentSection {
            section_index: 0,
            section_path: "0 Section Hierarchy".to_string(),
            section_kind: "hierarchy".to_string(),
            source_location: format!("{}@hierarchy", source_label),
            content: h_text,
            token_estimate: tok,
            is_full_text: true,
        }];
        with_hierarchy.extend(sections);
        return with_hierarchy;
    }

    sections
}

pub fn extract_dois_from_text(text: &str) -> Vec<String> {
    let mut dois = Vec::new();
    for token in text.split(|c: char| c.is_whitespace() || "()[]<>\"',;".contains(c)) {
        let t = token.trim().trim_end_matches('.');
        if let Some(idx) = t.find("10.") {
            let candidate = &t[idx..];
            if candidate.len() >= 8 && candidate.contains('/') {
                let norm = normalize_doi(candidate);
                if !norm.is_empty() && !dois.contains(&norm) {
                    dois.push(norm);
                    if dois.len() >= 10 {
                        break;
                    }
                }
            }
        }
    }
    dois
}

pub fn extract_references_from_text(text: &str) -> Vec<String> {
    let mut refs = Vec::new();
    for d in extract_dois_from_text(text) {
        refs.push(format!("doi:{}", d));
    }
    if let Some(ax) = extract_arxiv_id_from_text(text) {
        let item = format!("arxiv:{}", ax);
        if !refs.contains(&item) {
            refs.push(item);
        }
    }
    refs
}

pub fn parse_crossref_items(body: &serde_json::Value, connector_label: &str) -> Vec<ResearchSourcePaper> {
    let mut papers = Vec::new();
    let now = Utc::now().to_rfc3339();
    if let Some(items) = body.get("message").and_then(|m| m.get("items")).and_then(|i| i.as_array()) {
        for item in items {
            let title = item
                .get("title")
                .and_then(|t| t.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .trim()
                .to_string();
            if title.is_empty() {
                continue;
            }
            let doi = item.get("DOI").and_then(|d| d.as_str()).map(normalize_doi);
            let item_url = item.get("URL").and_then(|u| u.as_str()).unwrap_or_default().to_string();
            let abstract_raw = item.get("abstract").and_then(|a| a.as_str()).unwrap_or_default();
            let abstract_clean = if !abstract_raw.is_empty() {
                clean_html(abstract_raw)
            } else {
                format!(
                    "Publication indexed via {}: {} (DOI: {})",
                    connector_label,
                    title,
                    doi.as_deref().unwrap_or("n/a")
                )
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
                    }
                }
            }
            let pub_date = item
                .get("created")
                .and_then(|c| c.get("date-time"))
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());
            papers.push(ResearchSourcePaper {
                title,
                doi,
                arxiv_id: None,
                version: Some("v1".to_string()),
                published_date: pub_date,
                authors,
                source_connector: connector_label.to_string(),
                url: if item_url.is_empty() {
                    "https://api.crossref.org/works".to_string()
                } else {
                    item_url
                },
                code_url: None,
                abstract_or_excerpt: abstract_clean,
                experimental_details: exp_det,
                limitations_excerpt: lim_det,
                citations,
                availability_status: if abstract_raw.is_empty() {
                    "metadata_only".to_string()
                } else {
                    "accessible".to_string()
                },
                retrieved_at: now.clone(),
            });
        }
    }
    papers
}

pub fn parse_rss_or_atom_feed(xml: &str, connector_label: &str) -> Vec<ResearchSourcePaper> {
    let mut items = Vec::new();
    let now = Utc::now().to_rfc3339();
    let tag = if xml.contains("<item>") || xml.contains("<item ") {
        "item"
    } else {
        "entry"
    };
    let open_pat = format!("<{}", tag);
    let close_pat = format!("</{}>", tag);
    for block in xml.split(&open_pat).skip(1) {
        let body = block.split(&close_pat).next().unwrap_or(block);
        let title = extract_xml_tag(body, "title").unwrap_or_default();
        if title.is_empty() {
            continue;
        }
        let link = extract_xml_tag(body, "link")
            .or_else(|| extract_xml_tag(body, "id"))
            .unwrap_or_default();
        let desc = extract_xml_tag(body, "description")
            .or_else(|| extract_xml_tag(body, "summary"))
            .unwrap_or_default();
        let clean_desc = clean_html(&desc);
        let (exp_det, lim_det) = extract_structured_document_sections(&clean_desc);
        items.push(ResearchSourcePaper {
            title,
            doi: extract_dois_from_text(&clean_desc).into_iter().next(),
            arxiv_id: extract_arxiv_id_from_text(&link).or_else(|| extract_arxiv_id_from_text(&clean_desc)),
            version: Some("v1".to_string()),
            published_date: extract_xml_tag(body, "pubDate").or_else(|| extract_xml_tag(body, "published")),
            authors: Vec::new(),
            source_connector: connector_label.to_string(),
            url: link,
            code_url: None,
            abstract_or_excerpt: clean_desc,
            experimental_details: exp_det,
            limitations_excerpt: lim_det,
            citations: Vec::new(),
            availability_status: "accessible".to_string(),
            retrieved_at: now.clone(),
        });
    }
    items
}

// =========================================================================
// HELPER PARSERS & PROVENANCE DEDUPLICATION
// =========================================================================

pub fn normalize_doi(raw: &str) -> String {
    let mut s = raw
        .trim()
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/")
        .trim_start_matches("https://dx.doi.org/")
        .trim_start_matches("http://dx.doi.org/")
        .trim_start_matches("doi:")
        .trim_end_matches('/')
        .to_lowercase();
    for suffix in ["/pdf", "/html", "/abstract", "/full", "/download", ".pdf", ".html"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            if stripped.contains('/') {
                s = stripped.to_string();
                break;
            }
        }
    }
    s
}

pub fn normalize_arxiv_id(raw: &str) -> (String, Option<String>) {
    let mut cleaned = raw.trim().to_string();
    if let Some(hash_idx) = cleaned.find('#') {
        cleaned.truncate(hash_idx);
    }
    if let Some(q_idx) = cleaned.find('?') {
        cleaned.truncate(q_idx);
    }
    cleaned = cleaned
        .replace("https://www.arxiv.org/", "https://arxiv.org/")
        .replace("http://www.arxiv.org/", "http://arxiv.org/");
    for prefix in [
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "https://arxiv.org/html/",
        "http://arxiv.org/html/",
        "https://arxiv.org/e-print/",
        "http://arxiv.org/e-print/",
        "https://arxiv.org/src/",
        "http://arxiv.org/src/",
        "https://arxiv.org/format/",
        "http://arxiv.org/format/",
        "https://export.arxiv.org/abs/",
        "http://export.arxiv.org/abs/",
        "https://export.arxiv.org/pdf/",
        "http://export.arxiv.org/pdf/",
        "https://export.arxiv.org/html/",
        "http://export.arxiv.org/html/",
        "https://export.arxiv.org/e-print/",
        "http://export.arxiv.org/e-print/",
        "https://ar5iv.labs.arxiv.org/html/",
        "http://ar5iv.labs.arxiv.org/html/",
        "https://ar5iv.org/abs/",
        "http://ar5iv.org/abs/",
        "https://ar5iv.org/html/",
        "http://ar5iv.org/html/",
        "arXiv:",
        "arxiv:",
    ] {
        if let Some(rest) = cleaned.strip_prefix(prefix) {
            cleaned = rest.to_string();
            break;
        }
    }
    cleaned = cleaned.trim_end_matches('/').to_string();
    for ext in [".pdf", ".html", ".htm", ".tar.gz", ".gz", ".ps"] {
        if let Some(stripped) = cleaned.strip_suffix(ext) {
            cleaned = stripped.to_string();
        }
    }
    cleaned = cleaned.trim_end_matches('/').to_string();
    if let Some(v_idx) = cleaned.rfind('v') {
        let (base, ver) = cleaned.split_at(v_idx);
        if !base.is_empty() && ver.len() > 1 && ver[1..].chars().all(|c| c.is_ascii_digit()) {
            return (base.to_lowercase(), Some(ver.to_lowercase()));
        }
    }
    (cleaned.to_lowercase(), None)
}

pub fn compute_content_hash(content: &str) -> String {
    let normalized = content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn classify_source_kind(url: &str, connector: Option<&str>, title: Option<&str>) -> String {
    let u_low = url.to_lowercase();
    let c_low = connector.unwrap_or("").to_lowercase();
    let t_low = title.unwrap_or("").to_lowercase();

    if u_low.contains("huggingface.co/datasets/")
        || u_low.contains("hf.co/datasets/")
        || c_low == "dataset"
        || c_low == "datasets"
        || u_low.contains("/datasets/")
        || t_low.starts_with("dataset:")
    {
        return "dataset".to_string();
    }
    if u_low.contains("github.com/")
        || u_low.contains("raw.githubusercontent.com/")
        || u_low.contains("gitlab.com/")
        || c_low == "github"
        || c_low == "implementation"
        || (u_low.contains("huggingface.co/") && !u_low.contains("/papers/"))
    {
        return "implementation".to_string();
    }
    "paper".to_string()
}

fn strip_version_from_segment(seg: &str) -> (String, Option<String>) {
    // Check explicit separator like foo-v2 or foo_v2 or arXiv-like 2305.10601v2
    if let Some(v_idx) = seg.rfind('v') {
        let (left, right) = seg.split_at(v_idx);
        let digits = &right[1..];
        if !left.is_empty()
            && !digits.is_empty()
            && digits.len() <= 4
            && digits.chars().all(|c| c.is_ascii_digit())
        {
            if let Some(stripped_sep) = left.strip_suffix('-').or_else(|| left.strip_suffix('_')) {
                if !stripped_sep.is_empty() {
                    return (stripped_sep.to_string(), Some(format!("v{}", digits)));
                }
            } else if left.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                return (left.to_string(), Some(format!("v{}", digits)));
            }
        }
    }
    (seg.to_string(), None)
}

/// Canonicalizes a source URL and extracts provider ID, DOI, version, content hash, and source kind.
/// Treats abstract, HTML, PDF, and download URLs of one document as equivalent.
pub fn canonicalize_source_identity(
    raw_url: &str,
    title: Option<&str>,
    doi: Option<&str>,
    arxiv_id: Option<&str>,
    explicit_version: Option<&str>,
    content: Option<&str>,
    connector: Option<&str>,
) -> CanonicalSourceIdentity {
    let trimmed_url = raw_url.trim();
    let norm_title = title.map(normalize_title_for_dedup).unwrap_or_default();
    let mut detected_doi = doi.map(normalize_doi).filter(|d| !d.is_empty());
    let mut detected_arxiv: Option<String> = None;
    let mut detected_ver: Option<String> = explicit_version
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let mut provider_id: Option<String> = None;
    let mut canonical_url = trimmed_url.to_string();

    if let Some(ax) = arxiv_id {
        let (base, ver) = normalize_arxiv_id(ax);
        if !base.is_empty() {
            detected_arxiv = Some(base);
            if detected_ver.is_none() {
                detected_ver = ver;
            }
        }
    }

    if let Some(ref d) = detected_doi.clone() {
        if let Some(ax) = d.strip_prefix("10.48550/arxiv.") {
            let (base, ver) = normalize_arxiv_id(ax);
            if !base.is_empty() {
                detected_arxiv = Some(base);
                if detected_ver.is_none() {
                    detected_ver = ver;
                }
            }
        }
    }

    // Parse URL if valid HTTP/HTTPS
    if let Ok(parsed) = Url::parse(trimmed_url) {
        let host = parsed
            .host_str()
            .unwrap_or("")
            .to_lowercase()
            .trim_start_matches("www.")
            .to_string();

        // Check query parameters for version or OpenReview id
        let mut openreview_id: Option<String> = None;
        let mut kept_query: Vec<(String, String)> = Vec::new();
        for (k, v) in parsed.query_pairs() {
            let kl = k.to_lowercase();
            let vt = v.trim();
            if kl == "version" || kl == "v" || kl == "rev" || kl == "revision" || kl == "oldid" {
                if detected_ver.is_none() && !vt.is_empty() {
                    let norm_v = if vt.chars().all(|c| c.is_ascii_digit()) {
                        format!("v{}", vt)
                    } else {
                        vt.to_string()
                    };
                    detected_ver = Some(norm_v);
                }
                continue;
            }
            if (host.contains("openreview.net")) && kl == "id" && !vt.is_empty() {
                openreview_id = Some(vt.to_string());
                continue;
            }
            // Drop presentation/download/tracking query parameters
            if matches!(
                kl.as_str(),
                "download"
                    | "format"
                    | "type"
                    | "name"
                    | "inline"
                    | "context"
                    | "ref"
                    | "source"
                    | "utm_source"
                    | "utm_medium"
                    | "utm_campaign"
            ) {
                continue;
            }
            kept_query.push((k.into_owned(), v.into_owned()));
        }

        if host.contains("arxiv.org") || host.contains("ar5iv.") {
            let (base, ver) = normalize_arxiv_id(trimmed_url);
            if !base.is_empty() {
                detected_arxiv = Some(base.clone());
                if detected_ver.is_none() {
                    detected_ver = ver;
                }
                provider_id = Some(format!("arxiv:{}", base));
                canonical_url = format!("https://arxiv.org/abs/{}", base);
            }
        } else if host == "doi.org" || host == "dx.doi.org" {
            let nd = normalize_doi(trimmed_url);
            if let Some(ax) = nd.strip_prefix("10.48550/arxiv.") {
                let (base, ver) = normalize_arxiv_id(ax);
                detected_arxiv = Some(base.clone());
                if detected_ver.is_none() {
                    detected_ver = ver;
                }
                provider_id = Some(format!("arxiv:{}", base));
                canonical_url = format!("https://arxiv.org/abs/{}", base);
            } else if !nd.is_empty() {
                detected_doi = Some(nd.clone());
                provider_id = Some(format!("doi:{}", nd));
                canonical_url = format!("https://doi.org/{}", nd);
            }
        } else if host.contains("openreview.net") {
            let or_id = openreview_id.or_else(|| {
                parsed
                    .path_segments()
                    .and_then(|segs| segs.last())
                    .map(|s| s.trim_end_matches(".pdf").trim_end_matches(".html").to_string())
                    .filter(|s| !s.is_empty() && s != "forum" && s != "pdf" && s != "attachment" && s != "notes")
            });
            if let Some(id) = or_id {
                provider_id = Some(format!("openreview:{}", id));
                canonical_url = format!("https://openreview.net/forum?id={}", id);
            }
        } else if host == "aclanthology.org" {
            let slug = parsed
                .path()
                .trim_matches('/')
                .trim_end_matches(".pdf")
                .trim_end_matches(".html")
                .trim_end_matches(".bib")
                .trim_matches('/');
            if !slug.is_empty() {
                provider_id = Some(format!("acl:{}", slug.to_lowercase()));
                canonical_url = format!("https://aclanthology.org/{}", slug);
            }
        } else if host.contains("openalex.org") {
            let wid = parsed
                .path()
                .trim_matches('/')
                .trim_start_matches("works/")
                .trim_end_matches(".json")
                .trim_end_matches(".pdf")
                .trim_end_matches(".html");
            if !wid.is_empty() {
                provider_id = Some(format!("openalex:{}", wid.to_lowercase()));
                canonical_url = format!("https://openalex.org/{}", wid.to_uppercase());
            }
        } else if host.contains("semanticscholar.org") {
            let segs: Vec<&str> = parsed
                .path_segments()
                .map(|c| c.filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();
            let clean_segs: Vec<&str> = segs
                .into_iter()
                .filter(|s| !matches!(s.to_lowercase().as_str(), "pdf" | "html" | "abstract" | "reader"))
                .collect();
            if let Some(last) = clean_segs.last() {
                let pid = last.trim_end_matches(".pdf").trim_end_matches(".html").to_lowercase();
                if !pid.is_empty() && pid != "paper" {
                    provider_id = Some(format!("s2:{}", pid));
                    canonical_url = format!("https://www.semanticscholar.org/paper/{}", pid);
                }
            }
        } else if host == "github.com" || host == "raw.githubusercontent.com" || host == "codeload.github.com" {
            let segs: Vec<&str> = parsed
                .path_segments()
                .map(|c| c.filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();
            if segs.len() >= 2 {
                let owner = segs[0].to_lowercase();
                let repo = segs[1].trim_end_matches(".git").to_lowercase();
                if segs.len() >= 4 && (segs[2] == "tree" || segs[2] == "blob" || segs[2] == "releases") {
                    let maybe_tag = segs.get(4).copied().unwrap_or(segs[3]);
                    if detected_ver.is_none() && (maybe_tag.starts_with('v') || maybe_tag.chars().all(|c| c.is_ascii_hexdigit())) {
                        detected_ver = Some(maybe_tag.to_string());
                    }
                }
                provider_id = Some(format!("github:{}/{}", owner, repo));
                canonical_url = format!("https://github.com/{}/{}", owner, repo);
            }
        } else if host == "huggingface.co" || host == "hf.co" {
            let segs: Vec<&str> = parsed
                .path_segments()
                .map(|c| c.filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();
            if segs.len() >= 3 && segs[0] == "datasets" {
                let org = segs[1].to_lowercase();
                let ds = segs[2].to_lowercase();
                provider_id = Some(format!("hf_dataset:{}/{}", org, ds));
                canonical_url = format!("https://huggingface.co/datasets/{}/{}", org, ds);
            } else if segs.len() >= 2 && segs[0] == "papers" {
                let (base, ver) = normalize_arxiv_id(segs[1]);
                detected_arxiv = Some(base.clone());
                if detected_ver.is_none() {
                    detected_ver = ver;
                }
                provider_id = Some(format!("arxiv:{}", base));
                canonical_url = format!("https://arxiv.org/abs/{}", base);
            } else if segs.len() >= 2 {
                let org = segs[0].to_lowercase();
                let model = segs[1].to_lowercase();
                provider_id = Some(format!("hf_model:{}/{}", org, model));
                canonical_url = format!("https://huggingface.co/{}/{}", org, model);
            }
        } else {
            // Generic URL canonicalization: treat abstract, HTML, PDF, and download URLs of one document as equivalent
            let raw_segs: Vec<&str> = parsed
                .path_segments()
                .map(|c| c.filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();

            let view_tokens = [
                "abs",
                "abstract",
                "html",
                "pdf",
                "epdf",
                "full",
                "fulltext",
                "download",
                "e-print",
                "eprint",
                "view",
                "content",
            ];

            let mut norm_segs: Vec<String> = Vec::new();
            for (idx, seg) in raw_segs.iter().enumerate() {
                let mut s = (*seg).to_string();
                for ext in [".pdf", ".html", ".htm", ".ps", ".tar.gz", ".gz", ".zip"] {
                    if let Some(stripped) = s.to_lowercase().strip_suffix(ext) {
                        let new_len = stripped.len();
                        s.truncate(new_len);
                        break;
                    }
                }
                let s_low = s.to_lowercase();
                // Skip presentation route segments (whether prefix like /abs/doc1 or suffix like /doc1/pdf)
                if view_tokens.contains(&s_low.as_str()) && raw_segs.len() > 1 {
                    continue;
                }
                // Check version suffix on the final document identifier segment
                if idx + 1 == raw_segs.len() || (idx + 2 == raw_segs.len() && view_tokens.contains(&raw_segs.last().unwrap().to_lowercase().as_str())) {
                    let (base_seg, seg_ver) = strip_version_from_segment(&s_low);
                    if detected_ver.is_none() && seg_ver.is_some() {
                        detected_ver = seg_ver;
                    }
                    if !base_seg.is_empty() {
                        norm_segs.push(base_seg);
                    }
                } else if !s_low.is_empty() {
                    norm_segs.push(s_low);
                }
            }

            let joined_path = norm_segs.join("/");
            let mut query_suffix = String::new();
            if !kept_query.is_empty() {
                kept_query.sort();
                let qp: Vec<String> = kept_query
                    .into_iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect();
                query_suffix = format!("?{}", qp.join("&"));
            }
            canonical_url = if joined_path.is_empty() {
                format!("https://{}{}", host, query_suffix)
            } else {
                format!("https://{}/{}{}", host, joined_path, query_suffix)
            };
            provider_id = Some(format!("url:{}/{}{}", host, joined_path, query_suffix));
        }
    }

    // Override canonical_url and provider_id if an arXiv ID was explicitly provided or extracted from DOI
    if let Some(ref ax) = detected_arxiv {
        provider_id = Some(format!("arxiv:{}", ax));
        canonical_url = format!("https://arxiv.org/abs/{}", ax);
    }

    let source_kind = classify_source_kind(&canonical_url, connector, title);

    // Compute canonical_id
    let canonical_id = if let Some(ref ax) = detected_arxiv {
        format!("arxiv:{}", ax)
    } else if let Some(ref d) = detected_doi {
        format!("doi:{}", d)
    } else if let Some(ref pid) = provider_id {
        if pid.starts_with("openreview:")
            || pid.starts_with("acl:")
            || pid.starts_with("github:")
            || pid.starts_with("hf_dataset:")
            || pid.starts_with("hf_model:")
            || pid.starts_with("url:")
        {
            pid.clone()
        } else if !norm_title.is_empty() {
            compute_canonical_work_id(title.unwrap_or(""), detected_doi.as_deref(), detected_arxiv.as_deref())
        } else {
            pid.clone()
        }
    } else if !norm_title.is_empty() {
        compute_canonical_work_id(title.unwrap_or(""), detected_doi.as_deref(), detected_arxiv.as_deref())
    } else {
        let mut hasher = Sha256::new();
        hasher.update(canonical_url.as_bytes());
        let digest = hasher.finalize();
        let hex: String = digest.iter().take(10).map(|b| format!("{:02x}", b)).collect();
        format!("source:{}", hex)
    };

    let version = detected_ver.unwrap_or_else(|| "v1".to_string());
    let content_basis = content
        .filter(|c| !c.trim().is_empty())
        .or(title.filter(|t| !t.trim().is_empty()))
        .unwrap_or(&canonical_url);
    let content_hash = compute_content_hash(&format!("{}::{}", version, content_basis));

    CanonicalSourceIdentity {
        canonical_id,
        canonical_url,
        provider_id,
        doi: detected_doi,
        arxiv_id: detected_arxiv,
        version,
        content_hash,
        source_kind,
    }
}

pub fn canonicalize_source_url(raw_url: &str) -> String {
    canonicalize_source_identity(raw_url, None, None, None, None, None, None).canonical_url
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
/// (e.g. arXiv preprint + OpenAlex + Crossref + Semantic Scholar + abstract/HTML/PDF/download URLs)
/// resolve to a SINGLE canonical `ResolvedResearchWork` and are NEVER counted as independent corroboration.
/// Independent papers, datasets, and implementations remain distinct.
pub fn deduplicate_and_resolve_provenance(papers: Vec<ResearchSourcePaper>) -> Vec<ResolvedResearchWork> {
    let mut by_canonical: HashMap<String, ResolvedResearchWork> = HashMap::new();
    let mut title_kind_to_canonical: HashMap<(String, String), String> = HashMap::new();
    let mut url_to_canonical: HashMap<String, String> = HashMap::new();
    let mut doi_to_canonical: HashMap<String, String> = HashMap::new();
    let mut ordered_keys: Vec<String> = Vec::new();

    for p in papers {
        let norm_title = normalize_title_for_dedup(&p.title);
        if norm_title.is_empty() && p.url.trim().is_empty() {
            continue;
        }

        let identity = canonicalize_source_identity(
            &p.url,
            Some(&p.title),
            p.doi.as_deref(),
            p.arxiv_id.as_deref(),
            p.version.as_deref(),
            Some(&p.abstract_or_excerpt),
            Some(&p.source_connector),
        );

        let initial_cid = if identity.source_kind == "paper" && !norm_title.is_empty() {
            if identity.arxiv_id.is_some() || identity.doi.is_some() {
                compute_canonical_work_id(
                    &p.title,
                    identity.doi.as_deref().or(p.doi.as_deref()),
                    identity.arxiv_id.as_deref().or(p.arxiv_id.as_deref()),
                )
            } else {
                identity.canonical_id.clone()
            }
        } else {
            identity.canonical_id.clone()
        };

        let title_key = (identity.source_kind.clone(), norm_title.clone());
        let cid = if let Some(existing_cid) = url_to_canonical.get(&identity.canonical_url) {
            existing_cid.clone()
        } else if let Some(ref d) = identity.doi {
            if let Some(existing_cid) = doi_to_canonical.get(d) {
                existing_cid.clone()
            } else if !norm_title.is_empty() {
                if let Some(existing_cid) = title_kind_to_canonical.get(&title_key) {
                    existing_cid.clone()
                } else {
                    initial_cid
                }
            } else {
                initial_cid
            }
        } else if !norm_title.is_empty() {
            if let Some(existing_cid) = title_kind_to_canonical.get(&title_key) {
                existing_cid.clone()
            } else {
                initial_cid
            }
        } else {
            initial_cid
        };

        url_to_canonical.insert(identity.canonical_url.clone(), cid.clone());
        if let Some(ref d) = identity.doi {
            doi_to_canonical.insert(d.clone(), cid.clone());
        }
        if !norm_title.is_empty() {
            title_kind_to_canonical.insert(title_key, cid.clone());
        }

        let effective_ver = p.version.clone().or_else(|| Some(identity.version.clone()));

        if let Some(existing) = by_canonical.get_mut(&cid) {
            existing.duplicate_copies_merged += 1;
            if existing.doi.is_none() && (p.doi.is_some() || identity.doi.is_some()) {
                existing.doi = p.doi.clone().or(identity.doi.clone());
            }
            if existing.arxiv_id.is_none() && (p.arxiv_id.is_some() || identity.arxiv_id.is_some()) {
                existing.arxiv_id = p.arxiv_id.clone().or(identity.arxiv_id.clone());
            }
            if existing.provider_id.is_none() && identity.provider_id.is_some() {
                existing.provider_id = identity.provider_id.clone();
            }
            if let Some(ver) = effective_ver {
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
                existing.content_hash = identity.content_hash;
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
            if let Some(v) = effective_ver {
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
                    doi: p.doi.or(identity.doi),
                    arxiv_id: p.arxiv_id.or(identity.arxiv_id),
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
                    provider_id: identity.provider_id,
                    canonical_url: identity.canonical_url,
                    content_hash: identity.content_hash,
                    source_kind: identity.source_kind,
                    independent_corroboration_count: 1,
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
            let oa_url = item
                .get("best_oa_location")
                .and_then(|l| l.get("pdf_url").or_else(|| l.get("landing_page_url")))
                .and_then(|u| u.as_str())
                .or_else(|| {
                    item.get("open_access")
                        .and_then(|oa| oa.get("oa_url"))
                        .and_then(|u| u.as_str())
                })
                .or_else(|| {
                    item.get("primary_location")
                        .and_then(|l| l.get("pdf_url").or_else(|| l.get("landing_page_url")))
                        .and_then(|u| u.as_str())
                })
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let arxiv_id = doi
                .as_deref()
                .and_then(|d| d.strip_prefix("10.48550/arxiv.").map(|s| s.to_string()))
                .or_else(|| {
                    item.get("ids")
                        .and_then(|ids| ids.get("arxiv"))
                        .and_then(|a| a.as_str())
                        .and_then(extract_arxiv_id_from_text)
                })
                .or_else(|| oa_url.as_deref().and_then(extract_arxiv_id_from_text));
            let id_url = oa_url.unwrap_or_else(|| {
                item.get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("https://openalex.org")
                    .to_string()
            });
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
        assert_eq!(work.independent_corroboration_count, 1);
    }

    #[test]
    fn test_canonicalize_equivalent_abstract_html_pdf_download_urls() {
        let urls = [
            "http://arxiv.org/abs/2305.10601v1",
            "https://www.arxiv.org/pdf/2305.10601v1.pdf?download=1",
            "https://arxiv.org/html/2305.10601v1",
            "https://arxiv.org/e-print/2305.10601v1",
            "https://ar5iv.labs.arxiv.org/html/2305.10601v1",
            "https://doi.org/10.48550/arXiv.2305.10601",
        ];
        let first = canonicalize_source_identity(urls[0], None, None, None, None, None, None);
        assert_eq!(first.canonical_id, "arxiv:2305.10601");
        assert_eq!(first.canonical_url, "https://arxiv.org/abs/2305.10601");
        for u in &urls[1..] {
            let ident = canonicalize_source_identity(u, None, None, None, None, None, None);
            assert_eq!(ident.canonical_id, first.canonical_id, "Failed for URL {}", u);
            assert_eq!(ident.canonical_url, first.canonical_url, "Failed canonical URL for {}", u);
        }

        // OpenReview forum, pdf, and attachment download URLs
        let or_forum = canonicalize_source_identity(
            "https://openreview.net/forum?id=AbCdEf1234",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let or_pdf = canonicalize_source_identity(
            "https://openreview.net/pdf?id=AbCdEf1234&download=true",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let or_att = canonicalize_source_identity(
            "https://openreview.net/attachment?id=AbCdEf1234&name=pdf",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(or_forum.canonical_id, "openreview:AbCdEf1234");
        assert_eq!(or_pdf.canonical_id, or_forum.canonical_id);
        assert_eq!(or_att.canonical_id, or_forum.canonical_id);
        assert_eq!(or_pdf.canonical_url, or_forum.canonical_url);

        // Generic publisher abstract, HTML, PDF, and download routes of one document
        let gen_abs = canonicalize_source_identity(
            "https://publisher.example.org/abs/2026.paper42",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let gen_html = canonicalize_source_identity(
            "https://publisher.example.org/html/2026.paper42.html",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let gen_pdf = canonicalize_source_identity(
            "https://publisher.example.org/pdf/2026.paper42.pdf",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let gen_dl = canonicalize_source_identity(
            "https://publisher.example.org/download/2026.paper42.pdf?download=1&utm_source=feed",
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(gen_abs.canonical_id, gen_html.canonical_id);
        assert_eq!(gen_abs.canonical_id, gen_pdf.canonical_id);
        assert_eq!(gen_abs.canonical_id, gen_dl.canonical_id);
        assert_eq!(gen_abs.canonical_url, gen_dl.canonical_url);
    }

    #[test]
    fn test_compressed_stream_pdf_and_gzipped_latex_tarball_yield_hierarchy_and_equation() {
        use lopdf::dictionary;
        use std::io::Write;

        // 1. Build a synthetic PDF in memory with a FlateDecode-compressed content stream (`lopdf`)
        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! {
                "F1" => font_id,
            },
        });

        let ops = vec![
            lopdf::content::Operation::new("BT", vec![]),
            lopdf::content::Operation::new("Tf", vec!["F1".into(), 14.into()]),
            lopdf::content::Operation::new("Td", vec![50.into(), 750.into()]),
            lopdf::content::Operation::new(
                "Tj",
                vec![lopdf::Object::string_literal("1 Introduction")],
            ),
            lopdf::content::Operation::new("Tf", vec!["F1".into(), 10.into()]),
            lopdf::content::Operation::new("Td", vec![0.into(), (-20).into()]),
            lopdf::content::Operation::new(
                "Tj",
                vec![lopdf::Object::string_literal(
                    "We study convergence bounds for distributed stochastic gradient methods under bounded variance.",
                )],
            ),
            lopdf::content::Operation::new("Tf", vec!["F1".into(), 14.into()]),
            lopdf::content::Operation::new("Td", vec![0.into(), (-30).into()]),
            lopdf::content::Operation::new(
                "Tj",
                vec![lopdf::Object::string_literal("2 Objective Formulation")],
            ),
            lopdf::content::Operation::new("Tf", vec!["F1".into(), 10.into()]),
            lopdf::content::Operation::new("Td", vec![0.into(), (-20).into()]),
            lopdf::content::Operation::new(
                "Tj",
                vec![lopdf::Object::string_literal(
                    "L(theta) = \\sum_{i=1}^N w_i \\log p(y_i | x_i, theta) (1)",
                )],
            ),
            lopdf::content::Operation::new("Td", vec![0.into(), (-20).into()]),
            lopdf::content::Operation::new(
                "Tj",
                vec![lopdf::Object::string_literal(
                    "Table 1: Empirical perplexity and throughput across benchmark splits.",
                )],
            ),
            lopdf::content::Operation::new("ET", vec![]),
        ];
        let content = lopdf::content::Content { operations: ops };
        let raw_content_bytes = content.encode().expect("encode PDF content ops");

        // Compress the content stream with zlib (`FlateDecode`) so raw byte scanning cannot see it
        let mut zlib_enc =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        zlib_enc
            .write_all(&raw_content_bytes)
            .expect("compress PDF stream");
        let compressed_stream_bytes = zlib_enc.finish().expect("finish zlib stream");

        let stream = lopdf::Stream::new(
            dictionary! {
                "Filter" => "FlateDecode",
            },
            compressed_stream_bytes,
        );
        let content_id = doc.add_object(stream);
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
        });
        let pages_dict = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        };
        doc.objects.insert(pages_id, lopdf::Object::Dictionary(pages_dict));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut pdf_bytes = Vec::new();
        doc.save_to(&mut pdf_bytes).expect("save PDF to memory");

        let (pdf_text, pdf_sections, pdf_quality) =
            extract_pdf_full_text_and_sections(&pdf_bytes, "arxiv:2601.99999v1.pdf")
                .expect("extract compressed-stream PDF");
        assert!(
            pdf_text.contains("Introduction") && pdf_text.contains("Objective Formulation"),
            "Decompressed PDF text must contain headings, got: {}",
            pdf_text
        );
        assert!(
            pdf_sections.iter().any(|s| s.section_kind == "hierarchy"
                && s.content.contains("1 Introduction")
                && s.content.contains("2 Objective Formulation")),
            "PDF sections must include section hierarchy: {:?}",
            pdf_sections
        );
        assert!(
            pdf_sections.iter().any(|s| s.section_kind == "equation"),
            "PDF sections must preserve at least one equation section: {:?}",
            pdf_sections
        );
        assert!(pdf_quality.sections_found >= 2);
        assert!(pdf_quality.equations_preserved >= 1);
        assert!(pdf_quality.tables_found >= 1);
        assert!(!pdf_quality.low_fidelity);

        // 2. Build an in-memory gzipped LaTeX tarball (`.tar.gz`) with `main.tex` + `\input{methods}`
        let main_tex = br#"\documentclass{article}
\begin{document}
\section{Introduction}
We analyze multi-source full-text extraction under strict memory bounds and zero code execution.
\input{methods}
\section{Experiments}
\begin{table}[h]
\begin{tabular}{ll}
Model & Accuracy \\
Triad & 98.4\%
\end{tabular}
\end{table}
\end{document}
"#;
        let methods_tex = br#"\subsection{Variational Objective}
We optimize the regularized variational free energy objective:
\begin{equation}
\mathcal{F}(\theta) = \mathbb{E}_{q_\theta}[\log q_\theta(z) - \log p(x, z)]
\end{equation}
"#;

        let mut tar_buf = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut tar_buf);
            let mut h1 = tar::Header::new_gnu();
            h1.set_size(main_tex.len() as u64);
            h1.set_mode(0o644);
            h1.set_cksum();
            builder
                .append_data(&mut h1, "main.tex", &main_tex[..])
                .expect("append main.tex");

            let mut h2 = tar::Header::new_gnu();
            h2.set_size(methods_tex.len() as u64);
            h2.set_mode(0o644);
            h2.set_cksum();
            builder
                .append_data(&mut h2, "methods.tex", &methods_tex[..])
                .expect("append methods.tex");
            builder.finish().expect("finish tar archive");
        }

        let mut gz_enc =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gz_enc.write_all(&tar_buf).expect("gzip tar archive");
        let targz_bytes = gz_enc.finish().expect("finish gzip");

        let extracted_latex = extract_latex_from_eprint_bytes(&targz_bytes, 1_048_576)
            .expect("extract LaTeX from in-memory .tar.gz");
        assert!(
            extracted_latex.contains("\\section{Introduction}")
                && extracted_latex.contains("\\subsection{Variational Objective}"),
            "Inlined multi-file LaTeX tarball must contain both main.tex and methods.tex"
        );

        let latex_sections =
            extract_structured_sections_from_latex(&extracted_latex, "arxiv:2601.99999v1:main.tex");
        let latex_quality =
            compute_extraction_quality(&latex_sections, true, extracted_latex.len());
        assert!(
            latex_sections.iter().any(|s| s.section_kind == "hierarchy"
                && s.content.contains("Introduction")
                && s.content.contains("Variational Objective")
                && s.content.contains("Experiments")),
            "Gzipped LaTeX tarball must yield section hierarchy: {:?}",
            latex_sections
        );
        assert!(
            latex_sections
                .iter()
                .any(|s| s.section_kind == "equation" && s.content.contains("\\mathcal{F}(\\theta)")),
            "Gzipped LaTeX tarball must preserve equation section: {:?}",
            latex_sections
        );
        assert!(latex_quality.sections_found >= 3);
        assert!(latex_quality.equations_preserved >= 1);
        assert!(latex_quality.tables_found >= 1);
        assert!(!latex_quality.low_fidelity);
    }
}
