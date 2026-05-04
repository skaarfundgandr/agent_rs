use anyhow::{bail, Context, Result};
use std::path::Path;
use crate::domain::mcp::{McpServerDef, ResolvedMcpServer, McpTransportSpec, McpStdioTransportSpec, McpStreamableHttpTransportSpec};
pub use crate::domain::config::McpConfig;
use url::Url;

impl McpConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config_str = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MCP config from {}", path.display()))?;
        Self::from_str(&config_str)
    }

    pub fn from_str(config: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(config).context("failed to parse MCP config JSON")?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<()> {
        if self.mcp_servers.is_empty() {
            bail!("no MCP servers were defined under `mcpServers`");
        }

        for (name, server) in &self.mcp_servers {
            server.validate().with_context(|| format!("invalid MCP server `{name}`"))?;
        }

        Ok(())
    }

    pub fn server(&self, name: &str) -> Option<&McpServerDef> {
        self.mcp_servers.get(name)
    }

    pub fn resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        let server = self
            .server(name)
            .with_context(|| format!("MCP server `{name}` was not found in the config"))?;

        Ok(ResolvedMcpServer {
            name: name.to_owned(),
            transport: server.transport_spec()?,
        })
    }

    pub fn resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>> {
        self.mcp_servers
            .iter()
            .map(|(name, server)| {
                Ok(ResolvedMcpServer {
                    name: name.clone(),
                    transport: server.transport_spec()?,
                })
            })
            .collect()
    }
}

impl McpServerDef {
    pub fn transport_kind(&self) -> Result<crate::domain::mcp::McpTransportKind> {
        if let Some(transport_type) = self.transport_type {
            return Ok(transport_type);
        }

        match (self.command.is_some(), self.url.is_some()) {
            (true, false) => Ok(crate::domain::mcp::McpTransportKind::Stdio),
            (false, true) => Ok(crate::domain::mcp::McpTransportKind::StreamableHttp),
            (true, true) => bail!("server definition mixes stdio `command` fields with remote `url` fields"),
            (false, false) => bail!("server definition must define either a stdio `command` or a remote `url`"),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self.transport_kind()? {
            crate::domain::mcp::McpTransportKind::Stdio => {
                let command = self
                    .command
                    .as_ref()
                    .context("stdio servers must include a `command`")?;

                if self.url.is_some() {
                    bail!("stdio server `{command}` cannot also define a `url`");
                }
            }
            crate::domain::mcp::McpTransportKind::StreamableHttp => {
                let url = self.url.as_ref().context("HTTP MCP servers must include a `url`")?;
                Url::parse(url).with_context(|| format!("invalid MCP server URL `{url}`"))?;

                if self.command.is_some() {
                    bail!("remote MCP servers cannot also define a `command`");
                }
            }
        }

        Ok(())
    }

    pub fn transport_spec(&self) -> Result<McpTransportSpec> {
        match self.transport_kind()? {
            crate::domain::mcp::McpTransportKind::Stdio => {
                let command = self
                    .command
                    .clone()
                    .context("stdio servers must include a `command`")?;

                Ok(McpTransportSpec::Stdio(McpStdioTransportSpec {
                    command,
                    args: self.args.clone(),
                    env: self.env.clone(),
                    cwd: self.cwd.clone(),
                }))
            }
            crate::domain::mcp::McpTransportKind::StreamableHttp => {
                let url = self.url.clone().context("HTTP MCP servers must include a `url`")?;
                let parsed_url = Url::parse(&url).with_context(|| format!("invalid MCP server URL `{url}`"))?;

                Ok(McpTransportSpec::StreamableHttp(McpStreamableHttpTransportSpec {
                    url: parsed_url,
                    headers: self.headers.clone(),
                }))
            }
        }
    }

    pub fn build_stdio_command(&self) -> Result<std::process::Command> {
        let transport = match self.transport_spec()? {
            McpTransportSpec::Stdio(transport) => transport,
            McpTransportSpec::StreamableHttp(_) => bail!("remote MCP servers do not have a stdio command"),
        };

        let mut command = std::process::Command::new(&transport.command);
        command.args(&transport.args);
        command.envs(&transport.env);

        if let Some(cwd) = transport.cwd {
            command.current_dir(cwd);
        }

        Ok(command)
    }
}

impl McpTransportSpec {
    pub fn kind(&self) -> crate::domain::mcp::McpTransportKind {
        match self {
            Self::Stdio(_) => crate::domain::mcp::McpTransportKind::Stdio,
            Self::StreamableHttp(_) => crate::domain::mcp::McpTransportKind::StreamableHttp,
        }
    }
}
