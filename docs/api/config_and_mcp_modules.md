# Config and MCP Modules

Provides configuration parser and client interfaces to load and connect to stdio or HTTP-based MCP servers.

> **Architecture reference:** See the [C4 architecture diagram](../diagrams/c4-architecture.md) for how these modules fit into the system, the [class diagram](../diagrams/class-diagram.md) for type relationships (`McpConfig`, `McpServerDef`, `McpTransportSpec`), and the [module dependency graph](../diagrams/module-dependency.md) for crate-level structure.

## `McpConfig`

Stores parsed configuration definitions for one or more MCP servers. Compatible with standard MCP `mcp.json` layouts.

### Methods
* **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Loads and parses an MCP configuration from a JSON file.
* **`validate(&self) -> Result<()>`**
  Validates that all configured servers have valid transport settings (e.g. valid URLs, stdio arguments, and no mixing of Stdio/HTTP configuration).
* **`resolved_server(&self, name: &str) -> Result<ResolvedMcpServer>`**
  Resolves transport specifications for a single named server.
* **`resolved_servers(&self) -> Result<Vec<ResolvedMcpServer>>`**
  Resolves transport specifications for all configured servers.

---

## `McpClient`

Manages connections and tool listing for the configured MCP servers.

> **Lifecycle reference:** See the [MCP connection state diagram](../diagrams/state-diagram.md) for the full connection lifecycle and the [startup sequence diagram](../diagrams/sequence-diagram.md) for how `connect()` fits into application bootstrap.

### Methods
* **`from_config_path(path: &str) -> Result<Self>`**
  Initializes the client from a path to an `mcp.json` file.
* **`new(config: McpConfig) -> Self`**
  Constructs a new client using an existing configuration struct.
* **`async connect(self, policy: PermissionPolicy) -> Result<McpRegistryRuntime>`**
  Establishes standard I/O processes or HTTP streams with all configured MCP servers. The provided policy is wrapped around each discovered tool.
* **`async tools(self, policy: PermissionPolicy) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all servers and returns all exposed endpoints as a list of dynamic Rig `ToolDyn` objects, with each tool wrapped in a permission policy check.

---

## `McpRegistry`

Registry that resolves MCP server definitions from `mcp.json` into Rig tools and performs name deduplication.

### Methods
* **`new(config: McpConfig) -> Self`**
  Creates a registry from a validated configuration.
* **`from_path(path: impl AsRef<Path>) -> Result<Self>`**
  Creates a registry from a configuration file path.
* **`from_client(client: McpClient) -> Self`**
  Creates a registry from an existing client manager.
* **`async connect(&self, policy: PermissionPolicy) -> Result<McpRegistryRuntime>`**
  Connects to all configured MCP servers and collects their tools, wrapping each in the provided policy.
* **`async tools(&self, policy: PermissionPolicy) -> Result<Vec<Box<dyn ToolDyn>>>`**
  Connects to all configured MCP servers and returns Rig-compatible boxed tools wrapped in the provided policy.

---

## `McpRegistryRuntime`

Runtime registry returned after connecting to the MCP servers, holding the active connections and resolved tools.

### Methods
* **`servers(&self) -> &[RegisteredMcpServer]`**
  Returns registered servers in connection order.
* **`server(&self, name: &str) -> Option<&RegisteredMcpServer>`**
  Looks up a server by name.
* **`tools(&self) -> &[RegisteredMcpTool]`**
  Returns registered tools in connection order.
* **`tool_names(&self) -> impl Iterator<Item = &str>`**
  Convenience iterator over tool names.
* **`into_tools(self) -> Vec<Box<dyn ToolDyn>>`**
  Converts the runtime registry into boxed Rig tools.

---

### Example Usage: Loading MCP Tools

```rust
use agent_rs_lib::config::McpConfig;
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
