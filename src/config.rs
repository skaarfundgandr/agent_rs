pub use crate::domain::config::McpConfig;
use crate::domain::mcp::{
    McpServerDef, McpStdioTransportSpec, McpStreamableHttpTransportSpec, McpTransportSpec,
    ResolvedMcpServer,
};
use anyhow::{Context, Result, bail};
use std::path::Path;
use std::str::FromStr;
use url::Url;

impl McpConfig {
    /// Load, parse, and validate an MCP configuration from a JSON file.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The file cannot be read from the given path.
    /// - The file contains invalid JSON structure.
    /// - The configuration fails validation (e.g. no servers are defined or server transport specs are invalid).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config_str = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MCP config from {}", path.display()))?;
        let config = Self::from_str(&config_str)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate the MCP configuration settings, checking for server transport specifications.
    pub fn validate(&self) -> Result<()> {
        if self.mcp_servers.is_empty() {
            bail!("no MCP servers were defined under `mcpServers`");
        }

        for (name, server) in &self.mcp_servers {
            server
                .validate()
                .with_context(|| format!("invalid MCP server `{name}`"))?;
        }

        Ok(())
    }

    /// Retrieve the server definition for a specific MCP server by name.
    pub fn server(&self, name: &str) -> Option<&McpServerDef> {
        self.mcp_servers.get(name)
    }

    /// Resolve transport specifications for a single named server.
    pub fn resolved_server(&self, name: &str) -> Result<ResolvedMcpServer> {
        let server = self
            .server(name)
            .with_context(|| format!("MCP server `{name}` was not found in the config"))?;

        Ok(ResolvedMcpServer {
            name: name.to_owned(),
            transport: server.transport_spec()?,
        })
    }

    /// Resolve transport specifications for all configured MCP servers.
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
    /// Detect the transport kind (Stdio or StreamableHttp) based on properties.
    pub fn transport_kind(&self) -> Result<crate::domain::mcp::McpTransportKind> {
        if let Some(transport_type) = self.transport_type {
            return Ok(transport_type);
        }

        match (self.command.is_some(), self.url.is_some()) {
            (true, false) => Ok(crate::domain::mcp::McpTransportKind::Stdio),
            (false, true) => Ok(crate::domain::mcp::McpTransportKind::StreamableHttp),
            (true, true) => {
                bail!("server definition mixes stdio `command` fields with remote `url` fields")
            }
            (false, false) => {
                bail!("server definition must define either a stdio `command` or a remote `url`")
            }
        }
    }

    /// Validate the server definition to check that configuration fields match the transport spec.
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
                let url = self
                    .url
                    .as_ref()
                    .context("HTTP MCP servers must include a `url`")?;
                Url::parse(url).with_context(|| format!("invalid MCP server URL `{url}`"))?;

                if self.command.is_some() {
                    bail!("remote MCP servers cannot also define a `command`");
                }
            }
        }

        Ok(())
    }

    /// Generate an `McpTransportSpec` based on configuration values.
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
                let url = self
                    .url
                    .clone()
                    .context("HTTP MCP servers must include a `url`")?;
                let parsed_url =
                    Url::parse(&url).with_context(|| format!("invalid MCP server URL `{url}`"))?;

                Ok(McpTransportSpec::StreamableHttp(
                    McpStreamableHttpTransportSpec {
                        url: parsed_url,
                        headers: self.headers.clone(),
                    },
                ))
            }
        }
    }

    /// Build a stdio process `Command` configured with environmental variables, arguments, and working directory.
    pub fn build_stdio_command(&self) -> Result<std::process::Command> {
        let transport = match self.transport_spec()? {
            McpTransportSpec::Stdio(transport) => transport,
            McpTransportSpec::StreamableHttp(_) => {
                bail!("remote MCP servers do not have a stdio command")
            }
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
    /// Return the transport kind corresponding to this specification.
    pub fn kind(&self) -> crate::domain::mcp::McpTransportKind {
        match self {
            Self::Stdio(_) => crate::domain::mcp::McpTransportKind::Stdio,
            Self::StreamableHttp(_) => crate::domain::mcp::McpTransportKind::StreamableHttp,
        }
    }
}
