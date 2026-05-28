# Module Dependency Graph

```mermaid
graph LR
    subgraph Binary
        main_rs["main.rs"]
    end

    subgraph lib["lib.rs (reexports)"]
        lib_rs["agent, config, domain, mcp"]
    end

    subgraph domain["src/domain/"]
        domain_config["config.rs<br/>McpConfig (raw JSON)"]
        domain_mcp["mcp.rs<br/>McpTransportKind, McpServerDef<br/>McpTransportSpec, ResolvedMcpServer"]
        domain_rag["rag.rs<br/>Document, Chunk, ChunkingOptions"]
        domain_errors["errors.rs<br/>DocumentError, CompactError"]
    end

    subgraph config["src/config.rs"]
        config_rs["McpConfig impl<br/>validation, resolution, transport detection"]
    end

    subgraph agent["src/agent/"]
        agent_mod["mod.rs"]
        embeddings["embeddings.rs<br/>EmbeddingService&lt;M&gt;"]
        rag["rag.rs<br/>PdfLoader, TextLoader<br/>WordSplitter, RagPipeline"]

        subgraph memory["memory/"]
            context["context.rs<br/>ContextManagedAgent&lt;M,C&gt;<br/>AgentContextExt"]
        end

        subgraph tools["tools/"]
            document["document.rs<br/>ReadDocumentTool, WriteDocumentTool<br/>validate_sandboxed_path"]
            search["search.rs<br/>GrepSearchTool"]
            glob["glob.rs<br/>GlobSearchTool"]
            directory["directory.rs<br/>ListDirectoryTool"]
            compact["context.rs<br/>CompactTool&lt;M&gt;"]
        end
    end

    subgraph mcp["src/mcp/"]
        client["client.rs<br/>McpClient"]
        registry["registry.rs<br/>McpRegistry, McpRegistryRuntime<br/>RegisteredMcpTool"]
    end

    main_rs --> lib_rs
    lib_rs --> config_rs
    lib_rs --> domain
    lib_rs --> agent
    lib_rs --> mcp

    config_rs --> domain_config
    config_rs --> domain_mcp

    agent_mod --> embeddings
    agent_mod --> rag
    agent_mod --> memory
    agent_mod --> tools

    rag --> embeddings
    rag --> domain_rag

    document --> domain_errors
    document --> rag
    search --> document
    glob --> document
    directory --> document
    compact --> domain_errors

    client --> domain_config
    client --> domain_mcp
    client --> registry

    registry --> domain_config
    registry --> domain_mcp
    registry --> client
```
