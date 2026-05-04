use anyhow::Result;
use rig::tool::ToolDyn;
use crate::mcp::registry::{McpRegistry, McpRegistryRuntime};
use crate::domain::config::McpConfig;
use crate::domain::mcp::{McpServerDef, ResolvedMcpServer};

pub struct McpClient {
    pub config: McpConfig,
}

impl McpClient {
    pub fn from_config_path(path: &str) -> Result<Self> {
        Ok(Self {
            config: McpConfig::from_path(path)?,
        })
    }

    pub fn from_str(config: &str) -> Result<Self> {
        Ok(Self {
            config: McpConfig::from_str(config)?,
        })
    }

    pub fn new(config: McpConfig) -> Self {
        Self { config }
    }

    pub fn get_server_def(&self, name: &str) -> Option<&McpServerDef> {
        self.config.mcp_servers.get(name)
    }

    pub fn get_resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        self.config.resolved_server(name)
    }

    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.config.mcp_servers.keys().map(String::as_str)
    }

    pub fn validate(&self) -> Result<()> {
        self.config.validate()
    }

    pub async fn connect(self) -> Result<McpRegistryRuntime> {
        let registry = McpRegistry::from_client(self);

        match registry.connect().await {
            Ok(runtime) => Ok(runtime),
            Err(e) => Err(e),
        }
    }

    pub async fn tools(self) -> Result<Vec<Box<dyn ToolDyn>>> {
        Ok(self.connect().await?.into_tools())
    }
}
