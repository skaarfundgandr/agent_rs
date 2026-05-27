use agent_rs_lib::domain::config::McpConfig;
use agent_rs_lib::domain::mcp::{McpServerDef, McpTransportKind, McpTransportSpec};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[test]
fn parses_stdio_server() {
    let config = McpConfig::from_str(
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
    .expect("config should parse");

    let server = config.server("memory").expect("server exists");
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
    let config = McpConfig::from_str(
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
    .expect("config should parse");

    let server = config.server("remote").expect("server exists");
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

    let err = config.validate().expect_err("mixed config validation should fail");

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
        cwd: Some(PathBuf::from("/tmp")),
        url: None,
        headers: HashMap::new(),
        extra: HashMap::from([("custom".to_string(), json!({"enabled": true}))]),
    };

    let command = server.build_stdio_command().expect("command should build");
    assert_eq!(command.get_program(), Path::new("python"));
}
