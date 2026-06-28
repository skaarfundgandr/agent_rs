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
        +Option~PathBuf~ cwd
        +Option~String~ url
        +HashMap~String, String~ headers
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
    McpTransportSpec <|.. McpStdioTransportSpec
    McpTransportSpec <|.. McpStreamableHttpTransportSpec
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
        +connect() Result~McpRegistryRuntime~
        +tools() List~ToolDyn~
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
        +HashMap~String, String~ metadata
    }

    class ChunkingOptions {
        +usize chunk_words
        +usize chunk_overlap_words
    }

    class DocumentLoader {
        <<interface>>
        +load(path) Result~Document~
    }

    class PdfLoader {
        +load(path) Result~Document~
    }

    class TextLoader {
        +load(path) Result~Document~
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
        +open_or_create(db, index, dim, bit_width) Result~Self~
        +add_source(path, service) Result~usize~
        +add_source_dyn(path, embedder) Result~usize~
        +remove_source(name) Result~usize~
        +build(embedder) TurboVectorIndex
        +save(path) Result
        +chunk_count() Result~i64~
    }

    class EmbeddingVector {
        <<type alias>>
        Vec~f64~
    }

    class EmbeddingService~M~ {
        -model M
        +new(model) Self
        +ndims() usize
        +max_documents() usize
        +embed_text(text) EmbeddingVector
        +embed_texts(texts) List~EmbeddingVector~
        +embed_document(doc) List~EmbeddingVector~
    }

    class DocumentError {
        <<enumeration>>
        +Io(io::Error)
        +Pdf(String)
        +UnsupportedExtension(String)
        +SandboxEscape(String)
        +PermissionDenied(String)
        +Rag(String)
    }

    DocumentLoader <|.. PdfLoader
    DocumentLoader <|.. TextLoader
    DocumentLoader ..> Document : creates
    DocumentLoader ..> DocumentError : throws
    
    TextSplitter <|.. WordSplitter
    WordSplitter --> ChunkingOptions : configures
    TextSplitter ..> Chunk : creates

    RagPipeline --> EmbeddingService : uses
    RagPipeline --> TextSplitter : uses
    RagPipeline *--> Chunk : stores

    note for RagPipeline "SharedTurboIndex = Arc<RwLock<TurboIndex>>"
    note for EmbeddingVector "EmbeddingVector = Vec<f64>"
```

---

## 4. Agent Memory & Internal Tools

This diagram outlines the context-managed agent wrapper for conversation compaction and the set of sandboxed filesystem utilities.

```mermaid
classDiagram
    direction TB
    class BuiltManagedAgent~M, P, C~ {
        -agent Agent~M~
        -history SharedHistory
        -context_manager OptionalContextManager
        +history() Vec~Message~
        +prompt(msg) Response
        +chat(msg) Response
    }

    class ManagedExt {
        <<interface>>
        +managed() ManagedBuilder
    }

    class ManagedBuilder~M, P, CompState~ {
        -agent Agent~M~
        -initial_history Vec~Message~
        -compaction CompState
        +with_history(history) Self
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
        +definition(prompt) ToolDefinition
        +call(args) Result~Output~
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

    note for BuiltManagedAgent "SharedHistory = Arc<Mutex<Vec<Message>>>\nOptionalContextManager = Option<Arc<dyn Any>>"
    
    ReadDocumentTool ..|> Tool
    WriteDocumentTool ..|> Tool
    GrepSearchTool ..|> Tool
    GlobSearchTool ..|> Tool
    ListDirectoryTool ..|> Tool
    CompactTool ..|> Tool
    CompactTool ..> CompactError : throws
```

