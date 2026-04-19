use crate::mcp::client::McpClient;
use std::sync::Arc;
use rig::client::ProviderClient;

pub struct Agent<P: ProviderClient> {
    provider: P,
    system_prompt: String,
    mcp_client: Option<Arc<McpClient>>,
}

impl<P: ProviderClient> Agent<P> {
    pub fn new(provider: P, system_prompt: String) -> Self {
        Self {
            provider,
            system_prompt,
            mcp_client: None,
        }
    }

    pub fn with_mcp(mut self, mcp_client: Arc<McpClient>) -> Self {
        self.mcp_client = Some(mcp_client);
        self
    }
}
