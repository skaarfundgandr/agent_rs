# Sequence Diagrams

## Startup Flow

```mermaid
sequenceDiagram
    participant Main as main.rs
    participant DotEnv as .env
    participant RigClient as Rig Client
    participant Embed as EmbeddingService
    participant McpClient as McpClient
    participant McpReg as McpRegistry
    participant McpSrv as MCP Server
    participant Tools as Internal Tools
    participant Pdf as PdfLoader
    participant Splitter as WordSplitter
    participant Rag as RagPipeline
    participant Agent as Rig Agent

    Main->>DotEnv: load .env
    Main->>RigClient: new(http://127.0.0.1:1234/v1)
    Main->>Embed: new(client.embedding_model())
    Main->>McpClient: from_path(./mcp.json)
    activate McpClient
    McpClient->>McpClient: parse McpConfig
    Main->>McpReg: new(client)
    Main->>McpReg: connect()
    activate McpReg
    McpReg->>McpSrv: for each server: spawn/connect
    activate McpSrv
    McpSrv-->>McpReg: list_all_tools()
    McpReg->>McpReg: wrap in RegisteredMcpTool
    deactivate McpSrv
    McpReg-->>Main: McpRegistryRuntime
    deactivate McpReg
    deactivate McpClient
    Main->>McpReg: into_tools() as ToolDyn list

    Main->>Tools: create internal tools
    Main->>Main: merge MCP + internal + CompactTool

    Main->>Pdf: load(./docs/sample.pdf)
    activate Pdf
    Pdf-->>Main: Document
    deactivate Pdf
    Main->>Splitter: new(220, 40)
    Main->>Splitter: split(document)
    activate Splitter
    Splitter-->>Main: chunk list
    deactivate Splitter
    Main->>Rag: new().add_document(doc, splitter)
    Main->>Rag: build_index(embedding_service)
    activate Rag
    Rag->>Embed: embed_texts(chunks)
    Embed-->>Rag: embeddings
    Rag->>Rag: InMemoryVectorStore to Index
    Rag-->>Main: VectorIndex
    deactivate Rag

    Main->>Agent: new(client, tools, preamble, context(index))
    Main->>Main: ChatBotBuilder run()
```

## Runtime Chat Loop

```mermaid
sequenceDiagram
    participant User as User (CLI)
    participant Agent as Rig Agent
    participant LLM as LLM Server
    participant MCP as MCP Tools
    participant Internal as Internal Tools
    participant Compact as CompactTool

    loop Chat Loop
        User->>Agent: text input
        activate Agent
        Agent->>LLM: generate(prompt + history)
        activate LLM
        LLM-->>Agent: response (text or tool_calls)
        deactivate LLM

        alt Tool Call
            Agent->>Agent: dispatch to ToolDyn
            alt MCP Tool
                Agent->>MCP: execute remote tool
                MCP-->>Agent: tool result
            else Internal Tool
                Agent->>Internal: read/write/grep/glob/list
                Internal-->>Agent: tool result
            else Compact Tool
                Agent->>Compact: summarize text
                activate Compact
                Compact->>LLM: prompt(summarize)
                LLM-->>Compact: summary
                deactivate Compact
                Compact-->>Agent: summary
            end
            Agent->>LLM: tool result as context
            activate LLM
            LLM-->>Agent: final text response
            deactivate LLM
        end

        Agent-->>User: text response
        deactivate Agent

        Note over Agent: ContextManagedAgent auto-compacts<br/>history when threshold exceeded
    end
```
