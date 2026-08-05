# RAG Pipeline

A decoupled ingestion pipeline that transforms files into chunked, embedded vector indexes.

> **Pipeline reference:** See the [RAG processing flowchart](../diagrams/flowchart.md) for the end-to-end document ingestion flow, the [class diagram](../diagrams/class-diagram.md) for trait relationships (`DocumentLoader`, `TextSplitter`, `RagPipeline`), and the [sequence diagram](../diagrams/sequence-diagram.md) for how the pipeline is invoked at startup.

## Data Models

Located in `src/domain/rag.rs`:

```rust
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

pub struct Chunk {
    pub text: String,
    pub metadata: ChunkMetadata,
}

pub struct ChunkMetadata {
    pub source: String,      // source filename (e.g. "doc.txt")
    pub file_type: String,   // file type / extension (e.g. "txt", "pdf")
    pub chunk_index: usize,  // zero-based index within the source document
}

pub enum RagSourceType {
    File,
    Directory,
}

pub struct RagSource {
    pub path: PathBuf,
    pub source_type: RagSourceType,
}

pub struct ChunkingOptions {
    pub chunk_words: usize,         // default 220
    pub chunk_overlap_words: usize, // default 40
}
```

---

## `DocumentLoader`

Trait for reading source files into generic `Document` structs.

```rust
#[async_trait::async_trait]
pub trait DocumentLoader: Send + Sync {
    async fn load(&self, path: &Path) -> Result<Document>;
}
```

Implementers must be `Send + Sync` and provide an **async** `load`.

- **`PdfLoader`**: Extracts plain text from `.pdf` files.
- **`TextLoader`**: Reads plaintext content from any file it is asked to load — the `.txt`/`.md` restriction comes from `DEFAULT_EXTENSIONS`, not from the loader itself.

---

## `TextSplitter`

Trait for chunking documents into smaller searchable units.

```rust
pub trait TextSplitter {
    fn split(&self, document: &Document) -> Vec<Chunk>;
}
```

- **`WordSplitter`**: Splits document text by word boundaries using a sliding window.
  - Initialized via `WordSplitter::new(chunk_words: usize, chunk_overlap_words: usize)`.
  - Default configuration: 220 words per chunk, 40 words overlap.

---

## `RagPipeline`

Persistent, on-disk RAG pipeline backed by SQLite (chunk metadata) and turbovec (vector ANN index). Start here to build a RAG system.

> **Two on-disk files:** `rag_chunks` (SQLite, metadata + chunk text) and `.tvim` (turbovec, vector index). They must stay in sync; deleting both is the recovery procedure if the builder errors with "out of sync".

### Builder API (recommended)

Use `RagPipeline::builder()` for a fluent construction path:

```rust
let rag = RagPipeline::builder()
    .embedder(embedding_service)      // EmbeddingService<M>, auto-erased
    .store_at("./rag_data/")          // shorthand for db_path + index_path
    // OR: .db_path("..."), .index_path("...")
    .extensions(["txt", "md", "pdf"]) // file types to index (default: txt, md, pdf)
    .chunk_words(220)                 // words per chunk (default: 220)
    .chunk_overlap_words(40)          // overlap (default: 40)
    .bit_width(4)                     // turbovec quantization (default: 4)
    .sandbox(my_sandbox)              // optional, defaults to CWD
    .build()
    .await?;
// Returns BuiltRag { vector_index, indexer }
```

### `BuiltRag`

Output of the builder. Contains:
- **`vector_index: TurboVectorIndex`** — plug into `agent.dynamic_context(k, rag.vector_index)`. `top_n_ids`/`top_n` honor `VectorSearchRequest.threshold()` (minimum score filter; no filtering when unset). Derives `Clone` (all fields are `Arc`).
- **`indexer: RagIndexer`** — ingestion handle for managing sources.

### `RagIndexer`

Owns the pipeline, embedder, source registry, and sandbox. Methods:
- **`add(&path) -> Result<usize>`** — register + index a file or directory (sandbox-aware). No-op for already-registered sources (returns 0).
- **`remove(&path) -> Result<usize>`** — unregister + delete chunks (sandbox-aware). Chunks are matched by canonical path string.
- **`reindex(&path) -> Result<usize>`** — unconditional re-embed replacing existing chunks for an already-registered source.
- **`list() -> Vec<RagSource>`** — list registered sources.
- **`is_empty() -> bool`** — check if any sources are registered.
- **`chunk_count() -> Result<i64>`** — number of persisted chunks.
- **`pipeline() -> &Arc<RagPipeline>`** — access the underlying pipeline (staging API escape hatch).
- **`tool(policy) -> ManageRagTool`** — create a rig-compatible `ManageRagTool` delegate.
- **`search_tool() -> SearchRagTool`** — create a read-only `SearchRagTool` sharing the pipeline's live store and index.

**Source keying:** Chunks are stored keyed by the canonical path string (not the bare filename from loader metadata). The pipeline overrides `Document.metadata["source"]` with the canonical path on ingest. This prevents same-named files in different directories from clobbering each other's chunks.

### Pipeline Methods (via `rag.indexer.pipeline()`)

- **`add_source(path, &EmbeddingService<M>) -> Result<usize>`**
  High-level file ingestion: loads, chunks, embeds, and persists. Returns chunk count.
- **`add_source_dyn(path, &dyn ErasedEmbedder) -> Result<usize>`**
  Same as `add_source` but accepts a trait object.
- **`remove_source(source_name) -> Result<usize>`**
  Drops every chunk whose `source` (canonical path) matches. Returns number removed.
- **`save(&index_path) -> Result<()>`**
  Persists the turbovec index to disk.
- **`commit_pending(&service) -> Result<usize>`**
  Flushes staged (unpersisted) chunks into the turbovec index.
- **`chunk_count() -> Result<i64>`**
  Number of chunks currently persisted.
- **`store() -> &Arc<DocumentStore>`** / **`turbo() -> &Arc<RwLock<TurboIndex>>`**
  Accessors for advanced use.

**Persisted vs staged state:** Chunks inserted via `add_source` are persisted to SQLite immediately. The turbovec index is built in memory and must be flushed with `save()` or `commit_pending()` to survive restarts.

---

## `ErasedEmbedder` Trait

Object-safe trait for embedding without knowing the concrete model type. Implementors: `EmbeddingService<M>` (blanket impl) and custom wrappers.

```rust
pub trait ErasedEmbedder: WasmCompatSend + WasmCompatSync {
    fn embed_query<'a>(&'a self, text: &'a str) -> QueryFuture<'a>;
    fn embed_texts<'a>(&'a self, texts: Vec<String>) -> TextsFuture<'a>;
    fn ndims(&self) -> usize;
}
```

(`WasmCompatSend`/`WasmCompatSync` are rig-core marker traits equivalent to `Send`/`Sync` on native targets; on wasm32 with the `wasm` feature they become no-op markers.)

---

### Example Usage: Building a RAG Index

```rust,no_run
use std::path::Path;
use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::rag::RagPipeline;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize local fastembed embeddings
    let service = EmbeddingService::builder()
        .model(agent_rs::agent::embeddings::FastembedModel::AllMiniLML6V2)
        .build()?;

    // 2. Build persistent RAG pipeline
    let rag = RagPipeline::builder()
        .embedder(service)
        .store_at("rag_data/")
        .build()
        .await?;

    // 3. Ingest a PDF
    let chunks = rag.indexer.add(Path::new("orientation.pdf")).await?;
    println!("Indexed {chunks} chunks");

    // 4. Save to disk
    rag.indexer.pipeline().save(Path::new("rag_data/rag.tvim")).await?;

    // 5. Use vector_index with agents
    let _index = rag.vector_index;
    Ok(())
}
```
