use std::sync::Arc;

use rig_core::tool::{ToolDyn, ToolError, rmcp::McpTool as RigMcpTool};
use rig_core::wasm_compat::WasmBoxedFuture;
use rmcp::service::{RoleClient, RunningService};
use tokio::sync::Mutex;

use crate::agent::permission::{PermissionPolicy, PermissionResult};
use crate::domain::errors::DocumentError;

static STDIN_MUTEX: Mutex<()> = Mutex::const_new(());

/// A Rig tool wrapper that keeps the underlying MCP server connection alive.
#[derive(Clone)]
pub struct RegisteredMcpTool {
    server_name: String,
    pub(super) tool_name: String,
    inner: RigMcpTool,
    _keepalive: Arc<RunningService<RoleClient, ()>>,
    policy: PermissionPolicy,
}

impl std::fmt::Debug for RegisteredMcpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisteredMcpTool")
            .field("server_name", &self.server_name)
            .field("tool_name", &self.tool_name)
            .finish_non_exhaustive()
    }
}

impl RegisteredMcpTool {
    pub(super) fn new(
        server_name: String,
        tool_name: String,
        inner: RigMcpTool,
        keepalive: Arc<RunningService<RoleClient, ()>>,
        policy: PermissionPolicy,
    ) -> Self {
        Self {
            server_name,
            tool_name,
            inner,
            _keepalive: keepalive,
            policy,
        }
    }

    /// Retrieve the name of the server this tool belongs to.
    ///
    /// # Returns
    ///
    /// Returns a string slice representing the server name.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Retrieve the name of the tool.
    ///
    /// # Returns
    ///
    /// Returns a string slice representing the tool name.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

impl ToolDyn for RegisteredMcpTool {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn description(&self) -> String {
        self.inner.description()
    }

    fn parameters(&self) -> serde_json::Value {
        self.inner.parameters()
    }

    fn call(&'_ self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        let policy = self.policy.clone();
        let tool_name = self.tool_name.clone();
        let args_owned = args.clone();
        Box::pin(async move {
            let desc = format!(
                "MCP tool {tool_name} called with args: {}",
                &args_owned[..args_owned.len().min(200)]
            );
            let _guard = STDIN_MUTEX.lock().await;
            match policy.evaluate(&tool_name, &desc).await {
                PermissionResult::Allow => {}
                PermissionResult::Deny { reason } => {
                    let err =
                        DocumentError::PermissionDenied(format!("MCP tool {tool_name}: {reason}"));
                    return Err(ToolError::ToolCallError(Box::new(err)));
                }
            }
            drop(_guard);
            self.inner.call(args_owned).await
        })
    }
}
