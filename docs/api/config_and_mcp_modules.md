# Config and MCP Modules

Provides configuration parser and client interfaces to load and connect to stdio or HTTP-based MCP servers.

> **Architecture reference:** See the [C4 architecture diagram](../diagrams/c4-architecture.md) for how these modules fit into the system, the [class diagram](../diagrams/class-diagram.md) for type relationships (`McpConfig`, `McpServerDef`, `McpTransportSpec`), and the [module dependency graph](../diagrams/module-dependency.md) for crate-level structure.

## `McpConfig`

Stores parsed configuration definitions for one or more MCP servers. Compatible with standard MCP `mcp.json` layouts.

### Methods
- **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Loads, parses, and validates an MCP configuration from a JSON file.
- **`validate(&self) -> Result<()>`**
  Validates that all configured servers have valid transport settings (e.g. valid URLs, stdio arguments, and no mixing of Stdio/HTTP configuration).
- **`server(&self, name: &str) -> Option<&McpServerDef>`**
  Retrieves the server definition for a specific MCP server by name.
- **`resolved_server(&self, name: &str) -> Result<ResolvedMcpServer>`**
  Resolves transport specifications for a single named server.
- **`resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>>`**
  Resolves transport specifications for all configured servers.

### Trait Implementations
- `FromStr` — parses MCP config from a JSON string (no validation; call `.validate()` separately).

---

## `McpServerDef`

Raw server definition as it appears in `mcp.json`. Supports both stdio and streamable HTTP transports. Fields are validated at resolution time.

### Methods
- **`transport_kind(&self) -> Result<McpTransportKind>`**
  Detects the transport kind based on properties (explicit `type`, or auto-detected from `command` vs `url`).
- **`validate(&self) -> Result<()>`**
  Validates that the server definition matches its transport type (e.g. no `url` on stdio servers, no `command` on HTTP servers).
- **`transport_spec(&self) -> Result<McpTransportSpec>`**
  Generates an `McpTransportSpec` from the configuration values.
- **`build_stdio_command(&self) -> Result<std::process::Command>`**
  Builds a stdio process `Command` configured with env, args, and working directory. Returns an error if the server uses HTTP transport.

---

## `McpTransportKind`

The kind of transport used to communicate with an MCP server.

```rust
pub enum McpTransportKind {
    Stdio,
    StreamableHttp,
}
```

---

## `McpTransportSpec`

A resolved transport specification — either stdio or streamable HTTP.

```rust
pub enum McpTransportSpec {
    Stdio(McpStdioTransportSpec),
    StreamableHttp(McpStreamableHttpTransportSpec),
}
```

### Methods
- **`kind(&self) -> McpTransportKind`**
  Returns the transport kind corresponding to this specification.

---

## `McpStdioTransportSpec`

Configuration for spawning an MCP server as a child process.

```rust
pub struct McpStdioTransportSpec {
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<PathBuf>,
}
```

---

## `McpStreamableHttpTransportSpec`

Configuration for connecting to an MCP server over HTTP.

```rust
pub struct McpStreamableHttpTransportSpec {
    pub url: Url,
    pub headers: HashMap<String, String>,
}
```

---

## `ResolvedMcpServer`

A server whose transport has been resolved into a concrete spec.

```rust
pub struct ResolvedMcpServer {
    pub name: String,
    pub transport: McpTransportSpec,
}
```

---

## `RegisteredMcpServer`

An MCP server that has been connected and its tools discovered.

```rust
pub struct RegisteredMcpServer {
    pub name: String,
    pub transport: McpTransportSpec,
    pub tool_names: Vec<String>,
}
```

---

## `McpClient`

Manages connections and tool listing for the configured MCP servers.

> **Lifecycle reference:** See the [MCP connection state diagram](../diagrams/state-diagram.md) for the full connection lifecycle and the [startup sequence diagram](../diagrams/sequence-diagram.md) for how `connect()` fits into application bootstrap.

### Methods
- **`from_config_path(path: &str) -> Result<Self>`**
  Initializes the client from a path to an `mcp.json` file.
- **`new(config: McpConfig) -> Self`**
  Constructs a new client using an existing configuration struct.
- **`config(&self) -> &McpConfig`**
  Returns a reference to the underlying `McpConfig`.
- **`get_server_def(&self, name: &str) -> Option<&McpServerDef>`**
  Retrieves the server definition for a specific named server.
- **`get_resolved_server(&self, name: &str) -> Result<ResolvedMcpServer>`**
  Retrieves the resolved transport specification for a specific named server.
- **`server_names(&self) -> impl Iterator<Item = &str>`**
  Returns an iterator over the names of all configured MCP servers.
- **`validate(&self) -> Result<()>`**
  Validates the configurations of all registered MCP servers.
- **`async connect(self, policy: PermissionPolicy) -> Result<McpRegistryRuntime>`**
  Establishes standard I/O processes or HTTP streams with all configured MCP servers. The provided policy is wrapped around each discovered tool.
- **`async tools(self, policy: PermissionPolicy) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all servers and returns all exposed endpoints as a list of dynamic Rig `ToolDyn` objects, with each tool wrapped in a permission policy check.

### Trait Implementations
- `FromStr` — parses MCP config from a JSON string.

---

## `McpRegistry`

Registry that resolves MCP server definitions from `mcp.json` into Rig tools and performs name deduplication.

### Methods
- **`new(config: McpConfig) -> Self`**
  Creates a registry from a validated configuration.
- **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Creates a registry from a configuration file path.
- **`from_client(client: McpClient) -> Self`**
  Creates a registry from an existing client manager.
- **`config(&self) -> &McpConfig`**
  Access the underlying parsed config.
- **`validate(&self) -> Result<()>`**
  Validates the registry configuration.
- **`server(&self, name: &str) -> Option<&McpServerDef>`**
  Gets a server definition by name.
- **`resolved_server(&self, name: &str) -> Result<ResolvedMcpServer>`**
  Resolves a single MCP server into a normalized transport spec.
- **`resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>>`**
  Resolves every configured server into transport specs.
- **`server_names(&self) -> impl Iterator<Item = &str>`**
  Returns the configured server names.
- **`async connect(&self, policy: PermissionPolicy) -> Result<McpRegistryRuntime>`**
  Connects to all configured MCP servers and collects their tools, wrapping each in the provided policy.
- **`async tools(&self, policy: PermissionPolicy) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all configured MCP servers and returns Rig-compatible boxed tools wrapped in the provided policy.

---

## `McpRegistryRuntime`

Runtime registry returned after connecting to the MCP servers, holding the active connections and resolved tools.

### Methods
- **`servers(&self) -> &[RegisteredMcpServer]`**
  Returns registered servers in connection order.
- **`server(&self, name: &str) -> Option<&RegisteredMcpServer>`**
  Looks up a server by name.
- **`tools(&self) -> &[RegisteredMcpTool]`**
  Returns registered tools in connection order.
- **`tool_names(&self) -> impl Iterator<Item = &str>`**
  Convenience iterator over tool names.
- **`into_tools(self) -> Vec<Box<dyn ToolDyn>>`**
  Converts the runtime registry into boxed Rig tools.

---

## `RegisteredMcpTool`

A Rig tool wrapper that keeps the underlying MCP server connection alive. Applies the configured `PermissionPolicy` on every call.

### Methods
- **`server_name(&self) -> &str`**
  Returns the name of the server this tool belongs to.
- **`tool_name(&self) -> &str`**
  Returns the name of the tool.

---

### Example Usage: Loading MCP Tools

```rust,no_run
use agent_rs_lib::mcp::client::McpClient;
use agent_rs_lib::agent::permission::PermissionPolicy;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Read and connect to MCP servers defined in mcp.json
    let client = McpClient::from_config_path("./mcp.json")?;

    // Connect and load tools with an AllowAll permission policy
    let mcp_tools = client.tools(PermissionPolicy::AllowAll).await?;

    println!("Loaded {} tools from MCP servers.", mcp_tools.len());
    Ok(())
}
```
