use crate::domain::config::McpConfig;
use crate::domain::mcp::{McpServerDef, ResolvedMcpServer};
use crate::mcp::registry::{McpRegistry, McpRegistryRuntime};
use anyhow::Result;
use rig_core::tool::ToolDyn;
use std::str::FromStr;

/// Client manager that holds MCP server configurations and handles connecting to them.
pub struct McpClient {
    config: McpConfig,
}

impl McpClient {
    /// Retrieve the underlying configuration reference.
    ///
    /// # Returns
    ///
    /// Returns a reference to the `McpConfig`.
    pub fn config(&self) -> &McpConfig {
        &self.config
    }

    /// Construct the client manager from the path to an `mcp.json` file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to the JSON configuration file.
    ///
    /// # Returns
    ///
    /// Returns the initialized `McpClient`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or validated.
    pub fn from_config_path(path: &str) -> Result<Self> {
        Ok(Self {
            config: McpConfig::from_path(path)?,
        })
    }

    /// Construct a new client manager from an initialized `McpConfig` struct.
    ///
    /// # Arguments
    ///
    /// * `config` - The already parsed and validated `McpConfig` struct.
    ///
    /// # Returns
    ///
    /// Returns the initialized `McpClient`.
    pub fn new(config: McpConfig) -> Self {
        Self { config }
    }

    /// Retrieve the server definition for a specific named server.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the MCP server.
    ///
    /// # Returns
    ///
    /// Returns `Some(&McpServerDef)` if found, or `None` otherwise.
    pub fn get_server_def(&self, name: &str) -> Option<&McpServerDef> {
        self.config.mcp_servers.get(name)
    }

    /// Retrieve the resolved transport specification for a specific named server.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the MCP server.
    ///
    /// # Returns
    ///
    /// Returns the `ResolvedMcpServer` details.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is not found or fails to resolve.
    pub fn get_resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        self.config.resolved_server(name)
    }

    /// Return an iterator over the names of all configured MCP servers.
    ///
    /// # Returns
    ///
    /// Returns an iterator yielding the name string slices of all servers.
    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.config.mcp_servers.keys().map(String::as_str)
    }

    /// Validate the configurations of all registered MCP servers.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all configurations are valid.
    ///
    /// # Errors
    ///
    /// Returns an error if any server configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()
    }

    /// Establish connections with all configured MCP servers and return a runtime registry.
    ///
    /// # Returns
    ///
    /// Returns the active `McpRegistryRuntime` containing the connected servers.
    ///
    /// # Errors
    ///
    /// Returns an error if connection or validation fails.
    pub async fn connect(self) -> Result<McpRegistryRuntime> {
        let registry = McpRegistry::from_client(self);

        match registry.connect().await {
            Ok(runtime) => Ok(runtime),
            Err(e) => Err(e),
        }
    }

    /// Connect to all configured servers and return a collection of all exposed tools.
    ///
    /// # Returns
    ///
    /// Returns a vector of boxed dynamic Rig `ToolDyn` objects.
    ///
    /// # Errors
    ///
    /// Returns an error if connection or tool discovery fails.
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
