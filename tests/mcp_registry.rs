#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::domain::config::McpConfig;
use agent_rs::domain::mcp::{McpServerDef, McpTransportKind, McpTransportSpec};
use agent_rs::mcp::registry::McpRegistry;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
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

#[test]
fn parses_stdio_server() {
    let registry = McpRegistry::from_str(
        r#"{
            "mcpServers": {
                "memory": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-memory"],
                    "env": {
                        "MEMORY_PATH": "/tmp/memory"
                    }
                }
            }
        }"#,
    )
    .expect("registry should parse");

    let server = registry.server("memory").expect("server exists");
    assert_eq!(server.transport_kind().unwrap(), McpTransportKind::Stdio);

    match server.transport_spec().unwrap() {
        McpTransportSpec::Stdio(spec) => {
            assert_eq!(spec.command, "npx");
            assert_eq!(spec.args, vec!["-y", "@modelcontextprotocol/server-memory"]);
            assert_eq!(
                spec.env.get("MEMORY_PATH").map(String::as_str),
                Some("/tmp/memory")
            );
        }
        _ => panic!("expected stdio transport"),
    }
}

#[test]
fn parses_streamable_http_server() {
    let registry = McpRegistry::from_str(
        r#"{
            "mcpServers": {
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

    let server = registry.server("remote").expect("server exists");
    assert_eq!(
        server.transport_kind().unwrap(),
        McpTransportKind::StreamableHttp
    );

    match server.transport_spec().unwrap() {
        McpTransportSpec::StreamableHttp(spec) => {
            assert_eq!(spec.url.as_str(), "http://localhost:3000/mcp");
            assert_eq!(
                spec.headers.get("X-API-Key").map(String::as_str),
                Some("secret")
            );
        }
        _ => panic!("expected streamable http transport"),
    }
}

#[test]
fn rejects_mixed_transport_fields() {
    let config = McpConfig::from_str(
        r#"{
            "mcpServers": {
                "broken": {
                    "command": "node",
                    "url": "http://localhost:3000/mcp"
                }
            }
        }"#,
    )
    .expect("config parsing should succeed");

    let registry = McpRegistry::new(config);
    let err = registry
        .validate()
        .expect_err("mixed config validation should fail");

    let message = err.to_string();
    assert!(
        message.contains("invalid MCP server `broken`") || message.contains("server definition")
    );
}

#[test]
fn builds_command_for_stdio_server() {
    let server = McpServerDef {
        transport_type: None,
        command: Some("python".to_string()),
        args: vec!["server.py".to_string()],
        env: HashMap::from([("A".to_string(), "B".to_string())]),
        cwd: Some(std::path::PathBuf::from("/tmp")),
        url: None,
        headers: HashMap::new(),
        extra: HashMap::from([("custom".to_string(), json!({"enabled": true}))]),
    };

    let command = server.build_stdio_command().expect("command should build");
    assert_eq!(command.get_program(), Path::new("python"));
}

// ---------------------------------------------------------------------------
// ToolRegistry + McpRegistryRuntime integration tests
// ---------------------------------------------------------------------------

/// McpRegistryRuntime::tool_boxes() returns the right count of boxed tools.
/// This test uses an empty runtime (no live server) to verify the accessor API.
#[test]
fn mcp_runtime_tool_boxes_count() {
    use agent_rs::mcp::registry::McpRegistryRuntime;

    let runtime = McpRegistryRuntime::default();
    assert_eq!(runtime.tool_boxes().len(), 0);
    assert!(runtime.tool_names().next().is_none());
}

/// ToolRegistryBuilder can register tools and produce active_tools.
#[test]
fn tool_registry_register_and_active_tools() {
    use agent_rs::agent::tools::ToolRegistryBuilder;

    let registry = ToolRegistryBuilder::new()
        .register("test", || Box::new(common::EchoTool))
        .expect("register should succeed")
        .enable(&["test"])
        .build();

    let tools = registry.active_tools();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "echo");
}

/// ToolRegistryBuilder rejects duplicate tool names.
#[test]
fn tool_registry_rejects_duplicate_names() {
    use agent_rs::agent::tools::ToolRegistryBuilder;

    let result = ToolRegistryBuilder::new()
        .register("a", || Box::new(common::EchoTool))
        .expect("first register should succeed")
        .register("b", || Box::new(common::EchoTool));

    assert!(result.is_err(), "duplicate tool name should be rejected");
}

/// ToolRegistry enable/disable groups.
#[test]
fn tool_registry_enable_disable_groups() {
    use agent_rs::agent::tools::ToolRegistryBuilder;

    let mut registry = ToolRegistryBuilder::new()
        .register("grp", || Box::new(common::EchoTool))
        .expect("register")
        .enable(&["grp"])
        .build();

    assert_eq!(registry.active_tools().len(), 1);
    registry.disable_group("grp");
    assert_eq!(registry.active_tools().len(), 0);
    registry.enable_group("grp");
    assert_eq!(registry.active_tools().len(), 1);
}
