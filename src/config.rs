use std::collections::HashMap;
use std::path::Path;
use anyhow::{bail, Context, Result};
use crate::mcp::client::{McpServerDef, ResolvedMcpServer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerDef>,
}

impl McpConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config_str = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MCP config from {}", path.display()))?;
        Self::from_str(&config_str)
    }

    pub fn from_str(config: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(config).context("failed to parse MCP config JSON")?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<()> {
        if self.mcp_servers.is_empty() {
            bail!("no MCP servers were defined under `mcpServers`");
        }

        for (name, server) in &self.mcp_servers {
            server.validate().with_context(|| format!("invalid MCP server `{name}`"))?;
        }

        Ok(())
    }

    pub fn server(&self, name: &str) -> Option<&McpServerDef> {
        self.mcp_servers.get(name)
    }

    pub fn resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        let server = self
            .server(name)
            .with_context(|| format!("MCP server `{name}` was not found in the config"))?;

        Ok(ResolvedMcpServer {
            name: name.to_owned(),
            transport: server.transport_spec()?,
        })
    }

    pub fn resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>> {
        self.mcp_servers
            .iter()
            .map(|(name, server)| {
                Ok(ResolvedMcpServer {
                    name: name.clone(),
                    transport: server.transport_spec()?,
                })
            })
            .collect()
    }
}