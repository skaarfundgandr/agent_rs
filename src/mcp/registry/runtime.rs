use rig_core::tool::ToolDyn;

use crate::domain::mcp::RegisteredMcpServer;

use super::tool::RegisteredMcpTool;

/// Runtime registry returned after connecting to the MCP servers.
#[derive(Debug, Default)]
pub struct McpRegistryRuntime {
    pub(super) servers: Vec<RegisteredMcpServer>,
    pub(super) tools: Vec<RegisteredMcpTool>,
}

impl McpRegistryRuntime {
    /// Registered servers in connection order.
    ///
    /// # Returns
    ///
    /// Returns a slice of the connected `RegisteredMcpServer` instances.
    pub fn servers(&self) -> &[RegisteredMcpServer] {
        &self.servers
    }

    /// Look up a server by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique identifier name of the server.
    ///
    /// # Returns
    ///
    /// Returns `Some(&RegisteredMcpServer)` if found, or `None` otherwise.
    pub fn server(&self, name: &str) -> Option<&RegisteredMcpServer> {
        self.servers.iter().find(|server| server.name == name)
    }

    /// Registered tools in connection order.
    ///
    /// # Returns
    ///
    /// Returns a slice of the discovered `RegisteredMcpTool`s.
    pub fn tools(&self) -> &[RegisteredMcpTool] {
        &self.tools
    }

    /// Convenience iterator over tool names.
    ///
    /// # Returns
    ///
    /// Returns an iterator yielding the names of all discovered tools.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|tool| tool.tool_name.as_str())
    }

    /// Convert the runtime registry into boxed Rig tools.
    ///
    /// # Returns
    ///
    /// Returns a vector of boxed dynamic Rig `ToolDyn` objects.
    pub fn into_tools(self) -> Vec<Box<dyn ToolDyn>> {
        self.tools
            .into_iter()
            .map(|tool| Box::new(tool) as Box<dyn ToolDyn>)
            .collect()
    }
}
