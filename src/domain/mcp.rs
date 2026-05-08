use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use url::Url;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMcpServer {
    pub name: String,
    pub transport: McpTransportSpec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransportSpec {
    Stdio(McpStdioTransportSpec),
    StreamableHttp(McpStreamableHttpTransportSpec),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStdioTransportSpec {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStreamableHttpTransportSpec {
    pub url: Url,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredMcpServer {
    pub name: String,
    pub transport: McpTransportSpec,
    pub tool_names: Vec<String>,
}
