# Class / Type Diagram

To make the codebase structure easier to navigate, the type definitions are broken down into four thematic, logical diagrams:
1. [Crate Configuration & Transport Models](#1-crate-configuration--transport-models)
2. [MCP Registry & Connection Runtime](#2-mcp-registry--connection-runtime)
3. [RAG & Document Ingestion Pipeline](#3-rag--document-ingestion-pipeline)
4. [Agent Memory & Internal Tools](#4-agent-memory--internal-tools)

---

## 1. Crate Configuration & Transport Models

This diagram represents the parsing of MCP configuration (`mcp.json`) and the abstraction of STDIO or HTTP connection parameters.

```mermaid
classDiagram
    direction TB
    class McpConfig {
        +HashMap~String, McpServerDef~ mcp_servers
        +from_path(path) Result~McpConfig~
        +validate() Result
        +resolved_servers() List~ResolvedMcpServer~
    }

    class McpServerDef {
        +Option~McpTransportKind~ transport_type
        +Option~String~ command
        +Vec~String~ args
        +HashMap~String, String~ env
        +Option~PathBuf~ cwd
        +Option~String~ url
        +HashMap~String, String~ headers
        +HashMap~String, Value~ extra (serde flatten)
        +transport_spec() Result~McpTransportSpec~
    }

    class McpTransportKind {
        <<enumeration>>
        +Stdio
        +StreamableHttp
    }

    class McpTransportSpec {
        <<enumeration>>
        +Stdio(McpStdioTransportSpec)
        +StreamableHttp(McpStreamableHttpTransportSpec)
    }

    class McpStdioTransportSpec {
        +String command
        +Vec~String~ args
        +HashMap~String, String~ env
        +Option~PathBuf~ cwd
    }

    class McpStreamableHttpTransportSpec {
        +Url url
        +HashMap~String, String~ headers
    }

    class ResolvedMcpServer {
        +String name
        +McpTransportSpec transport
    }

    McpConfig "1" *--> "many" McpServerDef
    McpServerDef --> McpTransportKind : has
    McpServerDef ..> McpTransportSpec : resolves to
    McpTransportSpec "1" *--> McpStdioTransportSpec : variant
    McpTransportSpec "1" *--> McpStreamableHttpTransportSpec : variant
    ResolvedMcpServer --> McpTransportSpec : has
```

---

## 2. MCP Registry & Connection Runtime

This diagram illustrates how connections to MCP servers are managed, spawned, and wrapped into Rig-compatible tools.

```mermaid
classDiagram
    direction TB
    class McpRegistry {
        -McpConfig config
        +from_path(path) Result~Self~
        +connect(policy PermissionPolicy) Result~McpRegistryRuntime~
        +tools(policy PermissionPolicy) Result~List~ToolDyn~~
    }

    class McpRegistryRuntime {
        -Vec~RegisteredMcpServer~ servers
        -Vec~RegisteredMcpTool~ tools
        +servers() List~RegisteredMcpServer~
        +tools() List~RegisteredMcpTool~
        +into_tools() List~ToolDyn~
        +tool_boxes() List~ToolDyn~
    }

    class RegisteredMcpServer {
        +String name
        +McpTransportSpec transport
        +Vec~String~ tool_names
    }

    class RegisteredMcpTool {
        -String server_name
        -String tool_name
        -RigMcpTool inner
        -Arc~ArcService~ _keepalive
    }

    McpRegistry ..> McpRegistryRuntime : creates
    McpRegistryRuntime *--> RegisteredMcpServer : manages
    McpRegistryRuntime *--> RegisteredMcpTool : manages
```

---

## 3. RAG & Document Ingestion Pipeline

This diagram shows how documents (PDF, Text, Markdown) are loaded, split into chunks, and loaded into the vector index using embedding models.

```mermaid
classDiagram
    direction TB
    class Document {
        +String content
        +HashMap~String, String~ metadata
    }

    class Chunk {
        +String text
        +ChunkMetadata metadata
    }

    class ChunkingOptions {
        +usize chunk_words
        +usize chunk_overlap_words
    }

    class DocumentLoader {
        <<interface>>
        +async load(path) Result~Document~
    }

    class PdfLoader {
        +async load(path) Result~Document~
    }

    class TextLoader {
        +async load(path) Result~Document~
    }

    class TextSplitter {
        <<interface>>
        +split(document) Vec~Chunk~
    }

    class WordSplitter {
        +new(chunk_words, overlap) Self
        +split(document) Vec~Chunk~
    }

    class RagPipeline {
        -turbo SharedTurboIndex
        -store Arc~DocumentStore~
        +builder() RagPipelineBuilder
        +add_source(path, service) Result~usize~
        +add_source_dyn(path, embedder) Result~usize~
        +remove_source(name) Result~usize~
        +build(embedder) TurboVectorIndex
        +save(path) Result
        +chunk_count() Result~i64~
    }

    class RagPipelineBuilder {
        +embedder(service) Self
        +db_path(path) Self
        +index_path(path) Self
        +extensions(exts) Self
        +chunk_words(n) Self
        +chunk_overlap_words(n) Self
        +bit_width(n) Self
        +sandbox(s) Self
        +build() BuiltRag
    }

    class BuiltRag {
        +TurboVectorIndex vector_index
        +RagIndexer indexer
    }

    class RagIndexer {
        +add(path) Result~usize~
        +remove(path) Result~usize~
        +reindex(path) Result~usize~
        +list() List~RagSource~
        +tool(policy) ManageRagTool
    }

    class EmbeddingService~M~ {
        -model M
        +new(model) Self
        +ndims() usize
        +max_documents() usize
        +embed_text(text) Result~Embedding~
        +embed_texts(texts) Result~Vec~Embedding~~
        +embed_documents(docs) Result~Vec~(T, OneOrMany~Embedding~)~~
    }

    class DocumentError {
        <<enumeration>>
        +Io(io::Error)
        +Pdf(String)
        +UnsupportedExtension(String)
        +SandboxEscape(String)
        +Sandbox(String)
        +PermissionDenied(String)
        +Rag(String)
    }

    DocumentLoader <|.. PdfLoader
    DocumentLoader <|.. TextLoader
    DocumentLoader ..> Document : creates
    DocumentLoader ..> DocumentError : throws
    
    TextSplitter <|.. WordSplitter
    TextSplitter ..> Chunk : creates

    RagPipeline ..> RagPipelineBuilder : builder()
    RagPipelineBuilder ..> BuiltRag : build()
    BuiltRag *--> TurboVectorIndex : vector_index
    BuiltRag *--> RagIndexer : indexer
    RagIndexer *--> RagPipeline : owns (Arc)

    RagPipeline --> EmbeddingService : uses
    RagPipeline --> TextSplitter : uses
    RagPipeline *--> Chunk : stores

    note for RagPipeline "SharedTurboIndex = Arc<RwLock<TurboIndex>>\nopen_or_create / from_parts are pub(crate)"
```

---

## 4. Agent Memory & Internal Tools

This diagram outlines the context-managed agent wrapper for conversation compaction and the set of sandboxed filesystem utilities.

```mermaid
classDiagram
    direction TB
    class BuiltManagedAgent~M, C = (), S = Standard~ {
        -agent Agent~M~
        -max_retries u32
        -context_manager OptionalContextManager
        +max_retries() u32
        +prompt(msg) String
        +chat(msg, &mut history) String
        +stream_prompt(msg) ManagedStream
        +stream_chat(msg, &mut history) ManagedStream
        +prompt_compact(msg) String
        +chat_compact(msg, &mut history) String
        +stream_prompt_compact(msg) ManagedStream
        +stream_chat_compact(msg, &mut history) ManagedStream
    }

    class ManagedExt {
        <<interface>>
        +managed() ManagedBuilder
    }

    class ManagedBuilder~M, CompState~ {
        -agent Agent~M~
        -max_retries u32
        -compaction CompState
        +max_retries(n) Self
        +with_compaction() ManagedBuilder
        +build() BuiltManagedAgent
    }

    class CompactionConfig~C~ {
        +model C
        +threshold usize
        +tokenizer fn
        +compaction_prompt fn
    }

    class Tool {
        <<interface>>
        +const NAME &'static str
        +description() String
        +parameters() Value
        +async call(args) Result~Output~
    }

    class ReadDocumentTool {
        +call(args) Result~String~
    }

    class WriteDocumentTool {
        +call(args) Result~String~
    }

    class GrepSearchTool {
        +call(args) Result~String~
    }

    class GlobSearchTool {
        +call(args) Result~String~
    }

    class ListDirectoryTool {
        +call(args) Result~String~
    }

    class CompactTool~M~ {
        -M model
        +call(args) Result~String~
    }

    class CompactError {
        <<enumeration>>
        +Model(String)
    }

    ManagedExt ..> ManagedBuilder : creates
    ManagedBuilder ..> BuiltManagedAgent : build()
    ManagedBuilder --> CompactionConfig : configures

    note for BuiltManagedAgent "OptionalContextManager = Option<Arc<dyn Any>>\nCaller owns history Vec<Message>"
    
    ReadDocumentTool ..|> Tool
    WriteDocumentTool ..|> Tool
    GrepSearchTool ..|> Tool
    GlobSearchTool ..|> Tool
    ListDirectoryTool ..|> Tool
    CompactTool ..|> Tool
    CompactTool ..> CompactError : throws
```

