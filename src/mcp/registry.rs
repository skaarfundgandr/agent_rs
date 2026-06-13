use crate::domain::config::McpConfig;
use crate::domain::mcp::{
    McpServerDef, McpStdioTransportSpec, McpStreamableHttpTransportSpec, McpTransportSpec,
    RegisteredMcpServer, ResolvedMcpServer,
};
use crate::mcp::client::McpClient;
use anyhow::{Context, Result, bail};
use reqwest::header::{HeaderName, HeaderValue};
use rig_core::completion::ToolDefinition;
use rig_core::tool::{ToolDyn, ToolError, rmcp::McpTool as RigMcpTool};
use rig_core::wasm_compat::WasmBoxedFuture;
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use tokio::process::Command;

/// Registry that resolves MCP server definitions from `mcp.json` into Rig tools.
pub struct McpRegistry {
    client: McpClient,
}

impl McpRegistry {
    /// Create a registry from a validated config.
    pub fn new(config: McpConfig) -> Self {
        Self {
            client: McpClient::new(config),
        }
    }

    /// Create a registry from a configuration file path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(McpConfig::from_path(path)?))
    }

    /// Create a registry from an existing client wrapper.
    pub fn from_client(client: McpClient) -> Self {
        Self { client }
    }

    /// Access the underlying parsed config.
    pub fn config(&self) -> &McpConfig {
        self.client.config()
    }

    /// Validate the registry configuration.
    pub fn validate(&self) -> Result<()> {
        self.client.validate()
    }

    /// Get a server definition by name.
    pub fn server(&self, name: &str) -> Option<&McpServerDef> {
        self.client.get_server_def(name)
    }

    /// Resolve a single MCP server into a normalized transport spec.
    pub fn resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        self.client.get_resolved_server(name)
    }

    /// Resolve every configured server into transport specs.
    pub fn resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>> {
        self.client.config().resolved_servers()
    }

    /// Return the configured server names.
    pub fn server_names(&self) -> impl Iterator<Item = &str> {
        self.client.server_names()
    }

    /// Connect to all configured MCP servers and collect their tools.
    pub async fn connect(&self) -> Result<McpRegistryRuntime> {
        self.validate()?;

        let mut connected_servers = Vec::new();
        let mut registered_tools = Vec::new();
        let mut seen_tool_names: HashMap<String, String> = HashMap::new();

        for resolved in self.resolved_servers()? {
            let ConnectedMcpServer { server, tools } = connect_resolved_server(resolved).await?;

            for tool in &tools {
                if let Some(previous_server) =
                    seen_tool_names.insert(tool.tool_name.clone(), server.name.clone())
                {
                    bail!(
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
    pub async fn tools(&self) -> Result<Vec<Box<dyn ToolDyn>>> {
        Ok(self.connect().await?.into_tools())
    }
}

/// Runtime registry returned after connecting to the MCP servers.
#[derive(Debug, Default)]
pub struct McpRegistryRuntime {
    servers: Vec<RegisteredMcpServer>,
    tools: Vec<RegisteredMcpTool>,
}

impl McpRegistryRuntime {
    /// Registered servers in connection order.
    pub fn servers(&self) -> &[RegisteredMcpServer] {
        &self.servers
    }

    /// Look up a server by name.
    pub fn server(&self, name: &str) -> Option<&RegisteredMcpServer> {
        self.servers.iter().find(|server| server.name == name)
    }

    /// Registered tools in connection order.
    pub fn tools(&self) -> &[RegisteredMcpTool] {
        &self.tools
    }

    /// Convenience iterator over tool names.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|tool| tool.tool_name.as_str())
    }

    /// Convert the runtime registry into boxed Rig tools.
    pub fn into_tools(self) -> Vec<Box<dyn ToolDyn>> {
        self.tools
            .into_iter()
            .map(|tool| Box::new(tool) as Box<dyn ToolDyn>)
            .collect()
    }
}

struct ConnectedMcpServer {
    server: RegisteredMcpServer,
    tools: Vec<RegisteredMcpTool>,
}

/// A Rig tool wrapper that keeps the underlying MCP server connection alive.
pub struct RegisteredMcpTool {
    server_name: String,
    tool_name: String,
    inner: RigMcpTool,
    _keepalive: Arc<RunningService<RoleClient, ()>>,
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
    fn new(
        server_name: String,
        tool_name: String,
        inner: RigMcpTool,
        keepalive: Arc<RunningService<RoleClient, ()>>,
    ) -> Self {
        Self {
            server_name,
            tool_name,
            inner,
            _keepalive: keepalive,
        }
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

impl ToolDyn for RegisteredMcpTool {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn definition(&'_ self, prompt: String) -> WasmBoxedFuture<'_, ToolDefinition> {
        self.inner.definition(prompt)
    }

    fn call(&'_ self, args: String) -> WasmBoxedFuture<'_, Result<String, ToolError>> {
        self.inner.call(args)
    }
}

async fn connect_resolved_server(resolved: ResolvedMcpServer) -> Result<ConnectedMcpServer> {
    let ResolvedMcpServer { name, transport } = resolved;

    match transport {
        McpTransportSpec::Stdio(spec) => connect_stdio_server(name, spec).await,
        McpTransportSpec::StreamableHttp(spec) => connect_http_server(name, spec).await,
    }
}

async fn connect_stdio_server(
    name: String,
    spec: McpStdioTransportSpec,
) -> Result<ConnectedMcpServer> {
    if name.trim().is_empty() {
        bail!("MCP server names cannot be empty");
    }

    let command = build_stdio_command(&spec)?;
    let transport = TokioChildProcess::new(command).context("failed to spawn MCP stdio server")?;
    let service = Arc::new(
        ().serve(transport)
            .await
            .with_context(|| format!("failed to connect to MCP stdio server `{name}`"))?,
    );

    collect_registered_tools(name, spec.into(), service).await
}

async fn connect_http_server(
    name: String,
    spec: McpStreamableHttpTransportSpec,
) -> Result<ConnectedMcpServer> {
    if name.trim().is_empty() {
        bail!("MCP server names cannot be empty");
    }

    let headers = build_http_headers(&spec.headers)?;
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(spec.url.as_str()).custom_headers(headers),
    );
    let service = Arc::new(
        ().serve(transport)
            .await
            .with_context(|| format!("failed to connect to MCP HTTP server `{name}`"))?,
    );

    collect_registered_tools(name, spec.into(), service).await
}

async fn collect_registered_tools(
    name: String,
    transport: McpTransportSpec,
    service: Arc<RunningService<RoleClient, ()>>,
) -> Result<ConnectedMcpServer> {
    let sink = service.peer().clone();
    let tools = sink
        .list_all_tools()
        .await
        .with_context(|| format!("failed to list tools from MCP server `{name}`"))?;

    let mut registered_tools = Vec::with_capacity(tools.len());
    let mut tool_names = Vec::with_capacity(tools.len());
    let mut seen_local_names = HashSet::new();

    for tool in tools {
        if tool.name.trim().is_empty() {
            bail!("MCP server `{name}` returned a tool with an empty name");
        }

        let tool_name = tool.name.to_string();

        if !seen_local_names.insert(tool_name.clone()) {
            bail!(
                "MCP server `{name}` returned duplicate tool `{}`",
                tool_name
            );
        }

        tool_names.push(tool_name.clone());
        let inner = RigMcpTool::from_mcp_server(tool, sink.clone());
        registered_tools.push(RegisteredMcpTool::new(
            name.clone(),
            tool_name,
            inner,
            Arc::clone(&service),
        ));
    }

    Ok(ConnectedMcpServer {
        server: RegisteredMcpServer {
            name,
            transport,
            tool_names,
        },
        tools: registered_tools,
    })
}

fn build_stdio_command(spec: &McpStdioTransportSpec) -> Result<Command> {
    let mut process = Command::new(&spec.command);
    process
        .args(&spec.args)
        .envs(&spec.env)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    if let Some(cwd) = &spec.cwd {
        process.current_dir(cwd);
    }

    #[cfg(windows)]
    process.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

    Ok(process)
}

fn build_http_headers(
    headers: &HashMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>> {
    let mut normalized = HashMap::with_capacity(headers.len());

    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .with_context(|| format!("invalid HTTP header name `{name}`"))?;
        let header_value = HeaderValue::from_str(value)
            .with_context(|| format!("invalid HTTP header value for `{name}`"))?;
        normalized.insert(header_name, header_value);
    }

    Ok(normalized)
}

impl From<McpStdioTransportSpec> for McpTransportSpec {
    fn from(value: McpStdioTransportSpec) -> Self {
        Self::Stdio(value)
    }
}

impl From<McpStreamableHttpTransportSpec> for McpTransportSpec {
    fn from(value: McpStreamableHttpTransportSpec) -> Self {
        Self::StreamableHttp(value)
    }
}

impl FromStr for McpRegistry {
    type Err = anyhow::Error;

    /// Parse and validate an MCP configuration, returning an initialized registry.
    ///
    /// # Errors
    ///
    /// This will return an error if the configuration JSON is invalid or if the validation fails.
    fn from_str(config: &str) -> Result<Self, Self::Err> {
        let parsed = McpConfig::from_str(config)?;
        parsed.validate()?;
        Ok(Self::new(parsed))
    }
}

impl FromStr for McpConfig {
    type Err = anyhow::Error;

    /// Parse an MCP configuration from a JSON string.
    ///
    /// Note: This parses the JSON structure but does not validate the server definitions.
    /// Call `.validate()` to validate the configurations.
    fn from_str(config: &str) -> Result<Self, Self::Err> {
        let parsed: Self =
            serde_json::from_str(config).context("failed to parse MCP config JSON")?;
        Ok(parsed)
    }
}
