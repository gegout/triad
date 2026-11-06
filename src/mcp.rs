use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::consultation::{ConsultationEngine, ConsultationInput, ConsultationStatusInput};

pub const MCP_PROTOCOL_VERSION: &str = "2025-03-26";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>, data: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data,
            }),
        }
    }
}

/// Returns the JSON Schema shared by `consult_expert`, `solve_problem`, `investigate_question`, and `review_proposal`.
fn standard_consultation_input_schema() -> Value {
    json!({
        "type": "object",
        "required": ["question"],
        "properties": {
            "question": {
                "type": "string",
                "description": "The primary question, problem statement, or proposal to evaluate."
            },
            "objective": {
                "type": "string",
                "description": "What decision or outcome the caller is trying to achieve."
            },
            "context": {
                "type": "string",
                "description": "System architecture, workload characteristics, or relevant background."
            },
            "constraints": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Hard boundaries such as latency budgets, memory/VRAM limits, token budgets, or safety invariants."
            },
            "previous_attempts": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Approaches already tried and how they failed or fell short."
            },
            "success_criteria": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Measurable conditions that a valid answer or solution must satisfy."
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": "Depth tier: 'quick' (synchronous retrieval + synthesis, <30s target), 'standard' (adds reasoning & verification extensions, async), 'deep' (adds live multi-source research & temporary specialists, always async)."
            }
        }
    })
}

/// Returns the list of six MCP tools over Triad's consultation engine.
pub fn mcp_tools_list() -> Value {
    let consultation_schema = standard_consultation_input_schema();
    json!({
        "tools": [
            {
                "name": "expert_profile",
                "description": "Use when caller needs to know what Triad covers. Returns configured domain, scope, exclusions, research freshness, evaluated strengths, known weaknesses, and depth tier defaults/ceilings.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "consult_expert",
                "description": "Use when answerable from existing knowledge and reasoning (no new research). May escalate to 'solve_problem' or 'investigate_question' without exceeding caller's depth tier. Returns conclusion, applicability conditions, recommendations, labeled evidence, sources, assumptions, uncertainty, and unresolved issues.",
                "inputSchema": consultation_schema
            },
            {
                "name": "solve_problem",
                "description": "Use when problem needs executable verification or multi-step reasoning (no new research). Returns diagnosis, solution, alternatives, sandboxed verification results, remaining gaps, and labeled claims.",
                "inputSchema": consultation_schema
            },
            {
                "name": "investigate_question",
                "description": "Use when existing knowledge is insufficient and new multi-source research is required. Returns research findings synthesized into an expert conclusion with labeled claims and provenance.",
                "inputSchema": consultation_schema
            },
            {
                "name": "review_proposal",
                "description": "Use when caller wants rigorous criticism before a consequential decision. Returns weaknesses, counterexamples, missing considerations, concrete fixes, and labeled claims.",
                "inputSchema": consultation_schema
            },
            {
                "name": "consultation_status",
                "description": "Use when following a long-running consultation, answering an 'awaiting_input' clarification question, cancelling a consultation, or submitting attributable 'outcome_feedback' after applying advice.",
                "inputSchema": {
                    "type": "object",
                    "required": ["consultation_id"],
                    "properties": {
                        "consultation_id": {
                            "type": "string",
                            "description": "Unique ID of the consultation returned by consult_expert, solve_problem, investigate_question, or review_proposal."
                        },
                        "clarification_answer": {
                            "type": "string",
                            "description": "Caller's answer when the consultation is in state 'awaiting_input'."
                        },
                        "cancel": {
                            "type": "boolean",
                            "description": "Set to true to cancel an active or awaiting_input consultation."
                        },
                        "outcome_feedback": {
                            "type": "object",
                            "description": "Report what happened after applying Triad's advice. Stored as attributable evaluation evidence; never becomes a trusted fact on its own.",
                            "required": ["applied_advice", "observed_outcome", "succeeded"],
                            "properties": {
                                "consumer": { "type": "string" },
                                "applied_advice": { "type": "string" },
                                "observed_outcome": { "type": "string" },
                                "succeeded": { "type": "boolean" },
                                "discrepancies": { "type": "string" },
                                "metrics": { "type": "object" }
                            }
                        }
                    }
                }
            }
        ]
    })
}

/// Processes a single JSON-RPC 2.0 request against `ConsultationEngine`.
/// Returns `None` for notifications (requests without an `id`).
pub async fn handle_mcp_request(
    engine: &ConsultationEngine,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    if req.id.is_none() || req.method.starts_with("notifications/") {
        return None;
    }
    let id = req.id.clone();

    match req.method.as_str() {
        "initialize" => {
            let client_version = req
                .params
                .as_ref()
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or(MCP_PROTOCOL_VERSION);
            let negotiated = if client_version == "2024-11-05" || client_version == "2025-03-26" {
                client_version
            } else {
                MCP_PROTOCOL_VERSION
            };
            Some(JsonRpcResponse::success(
                id,
                json!({
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "triad-expert-consultation-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "instructions": "Triad Expert Consultation MCP server. Call 'expert_profile' first to inspect domain scope, exclusions, evaluated strengths, known weaknesses, and depth ceilings. Every consultation claim is epistemically labeled as 'retrieved_finding', 'expert_synthesis', or 'execution_verified_result'."
                }),
            ))
        }
        "ping" => Some(JsonRpcResponse::success(id, json!({}))),
        "tools/list" => Some(JsonRpcResponse::success(id, mcp_tools_list())),
        "tools/call" => {
            let Some(params) = req.params else {
                return Some(JsonRpcResponse::error(
                    id,
                    -32602,
                    "Missing params for tools/call",
                    None,
                ));
            };
            let Some(tool_name) = params.get("name").and_then(|v| v.as_str()) else {
                return Some(JsonRpcResponse::error(
                    id,
                    -32602,
                    "Missing tool 'name' in tools/call",
                    None,
                ));
            };
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

            let tool_res: Result<Value> = match tool_name {
                "expert_profile" => engine.expert_profile().await,
                "consult_expert" | "solve_problem" | "investigate_question" | "review_proposal" => {
                    match serde_json::from_value::<ConsultationInput>(args) {
                        Ok(input) => engine.start_consultation(tool_name, input).await,
                        Err(e) => Err(anyhow::anyhow!("Invalid consultation arguments: {}", e)),
                    }
                }
                "consultation_status" => {
                    match serde_json::from_value::<ConsultationStatusInput>(args) {
                        Ok(status_in) => engine.consultation_status(status_in).await,
                        Err(e) => Err(anyhow::anyhow!(
                            "Invalid consultation_status arguments: {}",
                            e
                        )),
                    }
                }
                other => {
                    return Some(JsonRpcResponse::error(
                        id,
                        -32601,
                        format!("Unknown tool '{}'", other),
                        None,
                    ));
                }
            };

            match tool_res {
                Ok(payload) => {
                    let text = serde_json::to_string_pretty(&payload)
                        .unwrap_or_else(|_| payload.to_string());
                    Some(JsonRpcResponse::success(
                        id,
                        json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": text
                                }
                            ],
                            "structuredContent": payload,
                            "isError": false
                        }),
                    ))
                }
                Err(err) => Some(JsonRpcResponse::success(
                    id,
                    json!({
                        "content": [
                            {
                                "type": "text",
                                "text": serde_json::to_string_pretty(&json!({
                                    "error": err.to_string(),
                                    "tool": tool_name
                                })).unwrap_or_else(|_| err.to_string())
                            }
                        ],
                        "isError": true
                    }),
                )),
            }
        }
        other => Some(JsonRpcResponse::error(
            id,
            -32601,
            format!("Method not found: {}", other),
            None,
        )),
    }
}

/// Processes a raw JSON-RPC payload (either a single object or a batch array) and returns the JSON response string (if any).
pub async fn process_raw_jsonrpc_payload(
    engine: &ConsultationEngine,
    raw: &str,
) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let parsed: Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            let resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e), None);
            return serde_json::to_string(&resp).ok();
        }
    };

    if let Some(arr) = parsed.as_array() {
        let mut responses = Vec::new();
        for item in arr {
            match serde_json::from_value::<JsonRpcRequest>(item.clone()) {
                Ok(req) => {
                    if let Some(resp) = handle_mcp_request(engine, req).await {
                        responses.push(resp);
                    }
                }
                Err(e) => {
                    responses.push(JsonRpcResponse::error(
                        None,
                        -32600,
                        format!("Invalid Request: {}", e),
                        None,
                    ));
                }
            }
        }
        if responses.is_empty() {
            None
        } else {
            serde_json::to_string(&responses).ok()
        }
    } else {
        match serde_json::from_value::<JsonRpcRequest>(parsed) {
            Ok(req) => {
                let resp = handle_mcp_request(engine, req).await?;
                serde_json::to_string(&resp).ok()
            }
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    None,
                    -32600,
                    format!("Invalid Request: {}", e),
                    None,
                );
                serde_json::to_string(&resp).ok()
            }
        }
    }
}

/// Runs the MCP server over `stdio`.
/// If a live Triad daemon Unix socket is present at `sock_path`, forwards every JSON-RPC message
/// through the running daemon's single `Coordinator` instance so no duplicate permanent agents or
/// competing research loops are created.
pub async fn run_mcp_stdio(
    fallback_engine: Arc<ConsultationEngine>,
    sock_path: Option<std::path::PathBuf>,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    while reader.read_line(&mut line).await? > 0 {
        let trimmed = line.trim().to_string();
        line.clear();
        if trimmed.is_empty() {
            continue;
        }

        // First attempt to dispatch through the live daemon socket if running
        let mut handled_via_daemon = None;
        if let Some(ref sp) = sock_path {
            if sp.exists() {
                if let Ok(crate::ipc::IpcResponse::McpJsonRpcResult { payload }) =
                    crate::server::send_ipc_request(
                        &crate::ipc::IpcRequest::McpJsonRpc {
                            payload: trimmed.clone(),
                        },
                        sp,
                    )
                    .await
                {
                    handled_via_daemon = Some(payload);
                }
            }
        }

        let response_opt = match handled_via_daemon {
            Some(opt) => opt,
            None => process_raw_jsonrpc_payload(&fallback_engine, &trimmed).await,
        };

        if let Some(resp_str) = response_opt {
            stdout.write_all(resp_str.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

/// Runs the Streamable HTTP MCP server (`POST /mcp`, `GET /mcp`, `DELETE /mcp`) on `bind_addr`.
pub async fn run_mcp_http_server(bind_addr: &str, engine: Arc<ConsultationEngine>) -> Result<()> {
    let listener = TcpListener::bind(bind_addr).await?;
    loop {
        let (mut stream, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        let eng = engine.clone();
        tokio::spawn(async move {
            let _ = handle_http_connection(&mut stream, eng).await;
        });
    }
}

async fn handle_http_connection(
    stream: &mut tokio::net::TcpStream,
    engine: Arc<ConsultationEngine>,
) -> Result<()> {
    let mut buf_reader = BufReader::new(stream);
    let mut request_line = String::new();
    if buf_reader.read_line(&mut request_line).await? == 0 {
        return Ok(());
    }
    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0].to_uppercase();
    let path = parts[1].to_string();

    let mut content_length: usize = 0;
    let mut accept_header = String::new();
    let mut session_id = String::new();

    loop {
        let mut header_line = String::new();
        if buf_reader.read_line(&mut header_line).await? == 0 {
            break;
        }
        let h_trim = header_line.trim();
        if h_trim.is_empty() {
            break;
        }
        if let Some((k, v)) = h_trim.split_once(':') {
            let key = k.trim().to_lowercase();
            let val = v.trim();
            if key == "content-length" {
                content_length = val.parse::<usize>().unwrap_or(0);
            } else if key == "accept" {
                accept_header = val.to_lowercase();
            } else if key == "mcp-session-id" {
                session_id = val.to_string();
            }
        }
    }

    if session_id.is_empty() {
        session_id = format!(
            "triad-mcp-session-{}",
            chrono::Utc::now().timestamp_millis()
        );
    }

    let mut body_bytes = vec![0u8; content_length];
    if content_length > 0 {
        buf_reader.read_exact(&mut body_bytes).await?;
    }
    let body_str = String::from_utf8_lossy(&body_bytes).to_string();
    let inner_stream = buf_reader.into_inner();

    if method == "OPTIONS" {
        let resp = format!(
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Accept, Mcp-Session-Id\r\nMcp-Session-Id: {}\r\nContent-Length: 0\r\n\r\n",
            session_id
        );
        inner_stream.write_all(resp.as_bytes()).await?;
        return Ok(());
    }

    if method == "DELETE" {
        let resp = format!(
            "HTTP/1.1 200 OK\r\nMcp-Session-Id: {}\r\nContent-Length: 0\r\n\r\n",
            session_id
        );
        inner_stream.write_all(resp.as_bytes()).await?;
        return Ok(());
    }

    if method == "GET" && (path.starts_with("/mcp") || path == "/") {
        if accept_header.contains("text/event-stream") {
            let sse_payload = format!(
                "event: endpoint\ndata: {{\"session_id\":\"{}\",\"protocolVersion\":\"{}\"}}\n\n",
                session_id, MCP_PROTOCOL_VERSION
            );
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nMcp-Session-Id: {}\r\nContent-Length: {}\r\n\r\n{}",
                session_id,
                sse_payload.len(),
                sse_payload
            );
            inner_stream.write_all(resp.as_bytes()).await?;
        } else {
            let info = json!({
                "server": "triad-expert-consultation-mcp",
                "transport": "Streamable HTTP",
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "session_id": session_id,
                "tools": ["expert_profile", "consult_expert", "solve_problem", "investigate_question", "review_proposal", "consultation_status"]
            });
            let body = serde_json::to_string(&info)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: {}\r\nContent-Length: {}\r\n\r\n{}",
                session_id,
                body.len(),
                body
            );
            inner_stream.write_all(resp.as_bytes()).await?;
        }
        return Ok(());
    }

    if method == "POST" && (path.starts_with("/mcp") || path == "/") {
        match process_raw_jsonrpc_payload(&engine, &body_str).await {
            Some(json_resp) => {
                // Streamable HTTP: if client explicitly requests SSE only or prefers SSE framing
                let wants_sse = accept_header.contains("text/event-stream")
                    && !accept_header.contains("application/json");
                if wants_sse {
                    let sse_body = format!("event: message\ndata: {}\n\n", json_resp);
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nMcp-Session-Id: {}\r\nContent-Length: {}\r\n\r\n{}",
                        session_id,
                        sse_body.len(),
                        sse_body
                    );
                    inner_stream.write_all(resp.as_bytes()).await?;
                } else {
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: {}\r\nContent-Length: {}\r\n\r\n{}",
                        session_id,
                        json_resp.len(),
                        json_resp
                    );
                    inner_stream.write_all(resp.as_bytes()).await?;
                }
            }
            None => {
                // Notification (e.g. notifications/initialized) -> 202 Accepted
                let resp = format!(
                    "HTTP/1.1 202 Accepted\r\nMcp-Session-Id: {}\r\nContent-Length: 0\r\n\r\n",
                    session_id
                );
                inner_stream.write_all(resp.as_bytes()).await?;
            }
        }
        return Ok(());
    }

    let not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    inner_stream.write_all(not_found.as_bytes()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tools_manifest_has_all_six_tools() {
        let manifest = mcp_tools_list();
        let tools = manifest["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 6);
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "expert_profile",
                "consult_expert",
                "solve_problem",
                "investigate_question",
                "review_proposal",
                "consultation_status"
            ]
        );
    }
}
