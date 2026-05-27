use crate::domain::config::McpConfig;
use crate::domain::mcp::{McpServerDef, ResolvedMcpServer};
use crate::mcp::registry::{McpRegistry, McpRegistryRuntime};
use anyhow::Result;
use rig::tool::ToolDyn;
use std::str::FromStr;

/// Client manager that holds MCP server configurations and handles connecting to them.
pub struct McpClient {
    config: McpConfig,
}

impl McpClient {
    /// Retrieve the underlying configuration reference.
    pub fn config(&self) -> &McpConfig {
        &self.config
    }

    /// Construct the client manager from the path to an `mcp.json` file.
    pub fn from_config_path(path: &str) -> Result<Self> {
        Ok(Self {
            config: McpConfig::from_path(path)?,
        })
    }

    /// Construct a new client manager from an initialized `McpConfig` struct.
    pub fn new(config: McpConfig) -> Self {
        Self { config }
    }

    /// Retrieve the server definition for a specific named server.
    pub fn get_server_def(&self, name: &str) -> Option<&McpServerDef> {
        self.config.mcp_servers.get(name)
    }

    /// Retrieve the resolved transport specification for a specific named server.
    pub fn get_resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        self.config.resolved_server(name)
    }

    /// Return an iterator over the names of all configured MCP servers.
    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.config.mcp_servers.keys().map(String::as_str)
    }

    /// Validate the configurations of all registered MCP servers.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()
    }

    /// Establish connections with all configured MCP servers and return a runtime registry.
    pub async fn connect(self) -> Result<McpRegistryRuntime> {
        let registry = McpRegistry::from_client(self);

        match registry.connect().await {
            Ok(runtime) => Ok(runtime),
            Err(e) => Err(e),
        }
    }

    /// Connect to all configured servers and return a collection of all exposed tools.
    pub async fn tools(self) -> Result<Vec<Box<dyn ToolDyn>>> {
        Ok(self.connect().await?.into_tools())
    }
}

impl FromStr for McpClient {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            config: McpConfig::from_str(s)?,
        })
    }
}
