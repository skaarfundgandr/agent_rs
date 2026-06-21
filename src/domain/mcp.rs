use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use url::Url;

/// The kind of transport used to communicate with an MCP server.
///
/// Auto-detected from the server definition when `type` is omitted:
/// stdio when `command` is present, streamable HTTP when `url` is present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpTransportKind {
    Stdio,
    #[serde(
        alias = "http",
        alias = "jsonrpc",
        alias = "json-rpc",
        alias = "streamable_http"
    )]
    StreamableHttp,
}

/// Raw server definition as it appears in `mcp.json`.
///
/// Supports both stdio (via `command`, `args`, `env`, `cwd`) and
/// streamable HTTP (via `url`, `headers`). Fields are validated at
/// parse/resolution time by the config module.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpServerDef {
    #[serde(rename = "type", default)]
    pub transport_type: Option<McpTransportKind>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(flatten, default)]
    pub extra: HashMap<String, Value>,
}

/// A server whose transport has been resolved into a concrete spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMcpServer {
    /// The server name as configured in `mcp.json`.
    pub name: String,
    /// The resolved transport specification (stdio or HTTP).
    pub transport: McpTransportSpec,
}

/// A resolved transport specification — either stdio or streamable HTTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransportSpec {
    /// Stdio-based subprocess transport.
    Stdio(McpStdioTransportSpec),
    /// HTTP(S)-based streamable transport.
    StreamableHttp(McpStreamableHttpTransportSpec),
}

/// Configuration for spawning an MCP server as a child process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStdioTransportSpec {
    /// The command to execute.
    pub command: String,
    /// Arguments passed to the command.
    pub args: Vec<String>,
    /// Environment variables set for the subprocess.
    pub env: HashMap<String, String>,
    /// Working directory for the subprocess (inherits parent if `None`).
    pub cwd: Option<PathBuf>,
}

/// Configuration for connecting to an MCP server over HTTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStreamableHttpTransportSpec {
    /// The MCP server endpoint URL.
    pub url: Url,
    /// HTTP headers to include in every request.
    pub headers: HashMap<String, String>,
}

/// An MCP server that has been connected and its tools discovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredMcpServer {
    /// The server name as configured in `mcp.json`.
    pub name: String,
    /// The transport spec the server was connected via.
    pub transport: McpTransportSpec,
    /// Names of tools exposed by this server.
    pub tool_names: Vec<String>,
}
