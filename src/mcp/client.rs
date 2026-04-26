use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, path::{Path, PathBuf}, process::Command};
use url::Url;

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: HashMap<String, McpServerDef>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpTransportKind {
    Stdio,
    #[serde(alias = "http", alias = "jsonrpc", alias = "json-rpc", alias = "streamable_http")]
    StreamableHttp,
}

/// MCP server definition as used inside `mcpServers`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpServerDef {
    #[serde(rename = "type", default)]
    pub transport_type: Option<McpTransportKind>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(flatten, default)]
    pub extra: HashMap<String, Value>,
}

impl McpServerDef {
    pub fn transport_kind(&self) -> Result<McpTransportKind> {
        if let Some(transport_type) = self.transport_type {
            return Ok(transport_type);
        }

        match (self.command.is_some(), self.url.is_some()) {
            (true, false) => Ok(McpTransportKind::Stdio),
            (false, true) => Ok(McpTransportKind::StreamableHttp),
            (true, true) => bail!("server definition mixes stdio `command` fields with remote `url` fields"),
            (false, false) => bail!("server definition must define either a stdio `command` or a remote `url`"),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self.transport_kind()? {
            McpTransportKind::Stdio => {
                let command = self
                    .command
                    .as_ref()
                    .context("stdio servers must include a `command`")?;

                if self.url.is_some() {
                    bail!("stdio server `{command}` cannot also define a `url`");
                }
            }
            McpTransportKind::StreamableHttp => {
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
            McpTransportKind::Stdio => {
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
            McpTransportKind::StreamableHttp => {
                let url = self.url.clone().context("HTTP MCP servers must include a `url`")?;
                let parsed_url = Url::parse(&url).with_context(|| format!("invalid MCP server URL `{url}`"))?;

                Ok(McpTransportSpec::StreamableHttp(McpStreamableHttpTransportSpec {
                    url: parsed_url,
                    headers: self.headers.clone(),
                }))
            }
        }
    }

    pub fn build_stdio_command(&self) -> Result<Command> {
        let transport = match self.transport_spec()? {
            McpTransportSpec::Stdio(transport) => transport,
            McpTransportSpec::StreamableHttp(_) => bail!("remote MCP servers do not have a stdio command"),
        };

        let mut command = Command::new(&transport.command);
        command.args(&transport.args);
        command.envs(&transport.env);

        if let Some(cwd) = transport.cwd {
            command.current_dir(cwd);
        }

        Ok(command)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedMcpServer {
    pub name: String,
    pub transport: McpTransportSpec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTransportSpec {
    Stdio(McpStdioTransportSpec),
    StreamableHttp(McpStreamableHttpTransportSpec),
}

impl McpTransportSpec {
    pub fn kind(&self) -> McpTransportKind {
        match self {
            Self::Stdio(_) => McpTransportKind::Stdio,
            Self::StreamableHttp(_) => McpTransportKind::StreamableHttp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStdioTransportSpec {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpStreamableHttpTransportSpec {
    pub url: Url,
    pub headers: HashMap<String, String>,
}

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
}
