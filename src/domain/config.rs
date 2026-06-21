use crate::domain::mcp::McpServerDef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parsed MCP server configuration loaded from an `mcp.json` file.
///
/// Each entry under `mcpServers` defines the transport and execution
/// parameters for one MCP server (stdio or HTTP).
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct McpConfig {
    /// Map of server names to their definitions (keyed from `"mcpServers"` in JSON).
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerDef>,
}
