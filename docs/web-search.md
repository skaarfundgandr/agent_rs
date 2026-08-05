# Web Search Integration

Enables web search for the Deep Research System **without any library changes** — web search is implemented as an MCP server, not a library feature (per report §1.4).

> **Architecture reference:** See the [MCP modules reference](api/config_and_mcp_modules.md) for the full `McpRegistry` → `McpRegistryRuntime` lifecycle, and the [agent tools reference](api/agent_tools.md) for the `ToolRegistryBuilder` API.

---

## Overview

AgentRS provides a `ToolRegistryBuilder` that accepts MCP runtimes via `register_mcp()`. A web search capability is wired by running a search-provider MCP server (e.g. Tavily, Serper, Brave Search) as a child process, connecting to it through the MCP registry, and registering its tools under a named group.

Two options are available:

| | Option A | Option B |
|---|---|---|
| **Approach** | Use an existing community MCP server | Build a custom MCP server |
| **When to use** | The provider has a published server (most cases) | No community server exists, or you need full control |
| **Effort** | Low (config only) | Medium (write + deploy a server) |

---

## Option A: Use an Existing Community MCP Server

### Recommended: Tavily

[Tavily](https://tavily.com/) is a search API built for AI agents. The community [`tavily-mcp`](https://github.com/tavily-ai/tavily-mcp) server exposes a `tavily_search` tool over MCP.

#### 1. Get an API key

Sign up at [tavily.com](https://tavily.com/) and create an API key.

#### 2. Add the server to `mcp.json`

```json
{
  "mcpServers": {
    "tavily-search": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "tavily-mcp"],
      "env": {
        "TAVILY_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

> **Other community servers:** [`serper-mcp`](https://github.com/nicholaswatertank/serper-mcp) (Google Search), [`brave-search-mcp`](https://github.com/nicholaswatertank/brave-search-mcp) (Brave Search), [`exa-mcp`](https://github.com/nicholaswatertank/exa-mcp) (Exa). Substitute the server name, command, and env vars as needed. The wiring pattern below is identical for all providers.

#### 3. Wire the server into the ToolRegistry

```rust
use agent_rs::mcp::registry::McpRegistry;
use agent_rs::agent::tools::ToolRegistryBuilder;
use agent_rs::agent::permission::PermissionPolicy;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load MCP config (includes the tavily-search server)
    let registry = McpRegistry::from_path("./mcp.json")?;

    // 2. Connect — spawns the server process, discovers tools
    let mcp_runtime = registry.connect(PermissionPolicy::AllowAll).await?;

    // 3. Register MCP tools under a named group
    let tool_registry = ToolRegistryBuilder::new()
        .register_mcp("research", &mcp_runtime)?  // search tools → "research" group
        .enable(&["research"])
        .build();

    // 4. Use active_tools() when building the agent
    let tools = tool_registry.active_tools();  // Vec<Box<dyn ToolDyn>>

    println!("Loaded {} research tools", tools.len());
    Ok(())
}
```

The search tool (e.g. `tavily_search`) is now available to any agent or ReAct loop that calls `tool_registry.active_tools()`.

---

## Option B: Custom MCP Server

When no community server covers the desired provider, build a custom MCP server wrapping the Tavily, Serper, or Brave Search HTTP API. The server exposes one or more tools over stdio or HTTP, and AgentRS connects to it identically.

A minimal custom server:
1. Receives a tool-call request (tool name + JSON arguments).
2. Calls the provider's REST API (e.g. `POST https://api.tavily.com/search`).
3. Returns the results as structured JSON.

Use the official [MCP SDK](https://github.com/modelcontextprotocol) for TypeScript or Python to scaffold the server, then add it to `mcp.json` as a stdio entry (same pattern as Option A).

The AgentRS consumer-side code is identical — only the `mcp.json` entry changes.

---

## Environment Variables

| Variable | Server | Description |
|---|---|---|
| `TAVILY_API_KEY` | Tavily | API key from [tavily.com](https://tavily.com/) |
| `SERPER_API_KEY` | Serper | API key from [serper.dev](https://serper.dev/) |
| `BRAVE_API_KEY` | Brave Search | API key from [brave.com/search/api](https://brave.com/search/api/) |

Set these in your `.env` file or export them in the shell before running the application. The MCP server process inherits the environment from the application.
