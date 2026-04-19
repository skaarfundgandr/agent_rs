use anyhow::Result;
use rig::tool::rmcp::McpClientHandler;
use rig::tool::server::ToolServerHandle;
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::child_process::TokioChildProcess;
use tokio::process::Command;

pub type RigMcpService = RunningService<RoleClient, McpClientHandler>;

pub struct McpClient {
    client_info: ClientInfo,
    service: Option<RigMcpService>,
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl McpClient {
    pub fn new() -> Self {
        let client_info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new(
                env!("CARGO_PKG_NAME").to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            ),
        );

        Self {
            client_info,
            service: None,
        }
    }

    pub async fn connect(
        &mut self,
        tool_server_handle: ToolServerHandle,
        cmd: Command,
    ) -> Result<&RigMcpService> {
        let handler = McpClientHandler::new(self.client_info.clone(), tool_server_handle);

        let transport = TokioChildProcess::new(cmd)?;

        let mcp_service = handler.connect(transport).await?;

        self.service = Some(mcp_service);

        Ok(self.service.as_ref().unwrap())
    }

    pub fn service(&self) -> Option<&RigMcpService> {
        self.service.as_ref()
    }
}
