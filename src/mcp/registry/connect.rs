use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::agent::permission::PermissionPolicy;
use crate::domain::mcp::{
    McpStdioTransportSpec, McpStreamableHttpTransportSpec, McpTransportSpec, RegisteredMcpServer,
    ResolvedMcpServer,
};
use anyhow::{Context, Result, bail};
use reqwest::header::{HeaderName, HeaderValue};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};

use super::tool::RegisteredMcpTool;

pub(super) struct ConnectedMcpServer {
    pub(super) server: RegisteredMcpServer,
    pub(super) tools: Vec<RegisteredMcpTool>,
}

pub(super) async fn connect_resolved_server(
    resolved: ResolvedMcpServer,
    policy: PermissionPolicy,
) -> Result<ConnectedMcpServer> {
    let ResolvedMcpServer { name, transport } = resolved;

    match transport {
        McpTransportSpec::Stdio(spec) => connect_stdio_server(name, spec, policy).await,
        McpTransportSpec::StreamableHttp(spec) => connect_http_server(name, spec, policy).await,
    }
}

async fn connect_stdio_server(
    name: String,
    spec: McpStdioTransportSpec,
    policy: PermissionPolicy,
) -> Result<ConnectedMcpServer> {
    if name.trim().is_empty() {
        bail!("MCP server names cannot be empty");
    }

    let command: tokio::process::Command = super::stdio_cmd::build_stdio_command(&spec)?.into();
    let transport = TokioChildProcess::new(command).context("failed to spawn MCP stdio server")?;
    let service = Arc::new(
        ().serve(transport)
            .await
            .with_context(|| format!("failed to connect to MCP stdio server `{name}`"))?,
    );

    collect_registered_tools(name, spec.into(), service, policy).await
}

async fn connect_http_server(
    name: String,
    spec: McpStreamableHttpTransportSpec,
    policy: PermissionPolicy,
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

    collect_registered_tools(name, spec.into(), service, policy).await
}

async fn collect_registered_tools(
    name: String,
    transport: McpTransportSpec,
    service: Arc<RunningService<RoleClient, ()>>,
    policy: PermissionPolicy,
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
        let inner = rig_core::tool::rmcp::McpTool::from_mcp_server(tool, sink.clone());
        registered_tools.push(RegisteredMcpTool::new(
            name.clone(),
            tool_name,
            inner,
            Arc::clone(&service),
            policy.clone(),
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
