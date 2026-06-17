#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs_lib::domain::config::McpConfig;
use agent_rs_lib::domain::mcp::McpTransportKind;
use agent_rs_lib::mcp::registry::McpRegistry;
use anyhow::Context;
use reqwest::header::{HeaderName, HeaderValue};
use serde_json::json;
use std::collections::HashMap;
use std::str::FromStr;

#[test]
fn parses_registry_from_mcp_json() {
    let registry = McpRegistry::from_str(
        r#"{
				"mcpServers": {
					"memory": {
						"command": "npx",
						"args": ["-y", "@modelcontextprotocol/server-memory"]
					},
					"remote": {
						"type": "http",
						"url": "http://localhost:3000/mcp",
						"headers": {
							"X-API-Key": "secret"
						}
					}
				}
			}"#,
    )
    .expect("registry should parse");

    assert!(registry.server("memory").is_some());
    assert!(registry.server("remote").is_some());
    assert_eq!(
        registry.resolved_server("memory").unwrap().transport.kind(),
        McpTransportKind::Stdio
    );
}

#[test]
fn normalizes_http_headers() {
    let headers = build_http_headers(&HashMap::from([
        ("X-API-Key".to_string(), "secret".to_string()),
        ("X-Trace-Id".to_string(), "abc123".to_string()),
    ]))
    .expect("headers should normalize");

    assert_eq!(headers.len(), 2);
    let api_key = HeaderName::from_static("x-api-key");
    assert_eq!(
        headers.get(&api_key).and_then(|value| value.to_str().ok()),
        Some("secret")
    );
}

#[test]
fn rejects_invalid_http_headers() {
    let err = build_http_headers(&HashMap::from([(
        "Bad Header".to_string(),
        "value".to_string(),
    )]))
    .expect_err("invalid header should fail");

    assert!(err.to_string().contains("invalid HTTP header name"));
}

#[test]
fn rejects_empty_config() {
    let registry = McpRegistry::new(McpConfig::default());
    let err = registry
        .validate()
        .expect_err("empty config should be rejected");

    assert!(err.to_string().contains("no MCP servers"));
}

#[test]
fn preserves_extra_fields_in_config() {
    let registry = McpRegistry::from_str(
        r#"{
				"mcpServers": {
					"memory": {
						"command": "npx",
						"args": ["-y", "@modelcontextprotocol/server-memory"],
						"custom": { "enabled": true }
					}
				}
			}"#,
    )
    .expect("registry should parse");

    let server = registry.server("memory").expect("server exists");
    assert_eq!(server.extra.get("custom"), Some(&json!({"enabled": true})));
}

fn build_http_headers(
    headers: &HashMap<String, String>,
) -> anyhow::Result<HashMap<HeaderName, HeaderValue>> {
    let mut normalized = HashMap::with_capacity(headers.len());

    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .with_context(|| format!("invalid HTTP header name `{name}`"))?;
        let header_value = HeaderValue::from_str(value)
            .with_context(|| format!("invalid HTTP header value for `{name}`"))?;
        normalized.insert(header_name, header_value);
    }

    Ok(normalized)
}
