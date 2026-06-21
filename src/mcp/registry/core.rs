use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rig_core::tool::ToolDyn;

use crate::agent::permission::PermissionPolicy;
use crate::domain::config::McpConfig;
use crate::domain::mcp::{McpServerDef, ResolvedMcpServer};
use crate::mcp::client::McpClient;

use super::connect::{ConnectedMcpServer, connect_resolved_server};
use super::runtime::McpRegistryRuntime;

/// Registry that resolves MCP server definitions from `mcp.json` into Rig tools.
pub struct McpRegistry {
    client: McpClient,
}

impl McpRegistry {
    /// Create a registry from a validated config.
    ///
    /// # Arguments
    ///
    /// * `config` - The validated `McpConfig` instance.
    ///
    /// # Returns
    ///
    /// Returns the initialized `McpRegistry`.
    pub fn new(config: McpConfig) -> Self {
        Self {
            client: McpClient::new(config),
        }
    }

    /// Create a registry from a configuration file path.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to the JSON configuration file.
    ///
    /// # Returns
    ///
    /// Returns the initialized `McpRegistry`.
    ///
    /// # Errors
    ///
    /// Returns an error if the config file cannot be read, parsed, or validated.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(McpConfig::from_path(path)?))
    }

    /// Create a registry from an existing client wrapper.
    ///
    /// # Arguments
    ///
    /// * `client` - The existing `McpClient`.
    ///
    /// # Returns
    ///
    /// Returns the initialized `McpRegistry`.
    pub fn from_client(client: McpClient) -> Self {
        Self { client }
    }

    /// Access the underlying parsed config.
    ///
    /// # Returns
    ///
    /// Returns a reference to the `McpConfig`.
    pub fn config(&self) -> &McpConfig {
        self.client.config()
    }

    /// Validate the registry configuration.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration validation fails.
    pub fn validate(&self) -> Result<()> {
        self.client.validate()
    }

    /// Get a server definition by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the server.
    ///
    /// # Returns
    ///
    /// Returns `Some(&McpServerDef)` if found, or `None` otherwise.
    pub fn server(&self, name: &str) -> Option<&McpServerDef> {
        self.client.get_server_def(name)
    }

    /// Resolve a single MCP server into a normalized transport spec.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the server.
    ///
    /// # Returns
    ///
    /// Returns the resolved transport specifications.
    ///
    /// # Errors
    ///
    /// Returns an error if the server is not found or fails to resolve.
    pub fn resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        self.client.get_resolved_server(name)
    }

    /// Resolve every configured server into transport specs.
    ///
    /// # Returns
    ///
    /// Returns a vector of resolved transport specifications.
    ///
    /// # Errors
    ///
    /// Returns an error if any server configuration fails to resolve.
    pub fn resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>> {
        self.client.config().resolved_servers()
    }

    /// Return the configured server names.
    ///
    /// # Returns
    ///
    /// Returns an iterator yielding the name string slices of all configured servers.
    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.client.server_names()
    }

    /// Connect to all configured MCP servers and collect their tools.
    ///
    /// # Arguments
    ///
    /// * `policy` - The `PermissionPolicy` instance to evaluate permissions for discovered tools.
    ///
    /// # Returns
    ///
    /// Returns the active `McpRegistryRuntime`.
    ///
    /// # Errors
    ///
    /// Returns an error if connection or validation fails, or if a duplicate tool name is found across servers.
    pub async fn connect(&self, policy: PermissionPolicy) -> Result<McpRegistryRuntime> {
        self.validate()?;

        let mut connected_servers = Vec::new();
        let mut registered_tools = Vec::new();
        let mut seen_tool_names: HashMap<String, String> = HashMap::new();

        for resolved in self.resolved_servers()? {
            let ConnectedMcpServer { server, tools } =
                connect_resolved_server(resolved, policy.clone()).await?;

            for tool in &tools {
                if let Some(previous_server) =
                    seen_tool_names.insert(tool.tool_name.clone(), server.name.clone())
                {
                    anyhow::bail!(
                        "duplicate MCP tool `{}` returned by servers `{}` and `{}`",
                        tool.tool_name,
                        previous_server,
                        server.name
                    );
                }
            }

            connected_servers.push(server);
            registered_tools.extend(tools);
        }

        Ok(McpRegistryRuntime {
            servers: connected_servers,
            tools: registered_tools,
        })
    }

    /// Connect to all configured MCP servers and return Rig-compatible boxed tools.
    ///
    /// # Arguments
    ///
    /// * `policy` - The `PermissionPolicy` instance to evaluate permissions for discovered tools.
    ///
    /// # Returns
    ///
    /// Returns a vector of boxed dynamic Rig `ToolDyn` objects.
    ///
    /// # Errors
    ///
    /// Returns an error if connection, tool discovery, or instantiation fails.
    pub async fn tools(&self, policy: PermissionPolicy) -> Result<Vec<Box<dyn ToolDyn>>> {
        Ok(self.connect(policy).await?.into_tools())
    }
}
