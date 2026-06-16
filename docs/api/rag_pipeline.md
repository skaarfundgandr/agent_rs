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
    pub metadata: HashMap<String, String>,
}

pub enum RagSourceType {
    File,
    Directory,
}

pub struct RagSource {
    pub path: PathBuf,
    pub source_type: RagSourceType,
}
```

---

## `DocumentLoader`

Trait for reading source files into generic `Document` structs.

```rust
pub trait DocumentLoader {
    fn load(&self, path: &Path) -> Result<Document>;
}
```
* **`PdfLoader`**: Extracts plain text from `.pdf` files.
* **`TextLoader`**: Reads plaintext content from `.txt` and `.md` files.

---

## `TextSplitter`

Trait for chunking documents into smaller searchable units.

```rust
pub trait TextSplitter {
    fn split(&self, document: &Document) -> Vec<Chunk>;
}
```
* **`WordSplitter`**: Splits document text by word boundaries using a sliding window.
  - Initialized via `WordSplitter::new(chunk_words: usize, chunk_overlap_words: usize)`.

---

## `RagPipeline`

Persistent, on-disk RAG pipeline backed by SQLite (chunk metadata) and turbovec (vector ANN index). Start here to build a RAG system.

> **Two on-disk files:** `rag_chunks` (SQLite, metadata + chunk text) and `.tvim` (turbovec, vector index). They must stay in sync; deleting both is the recovery procedure if `open_or_create` errors with "out of sync".

### Methods
* **`open_or_create(db_path, index_path, dim, bit_width) -> Result<Self>`**
  Opens or creates the SQLite database and turbovec index. `dim` is the embedding dimension (must match your embedder); `bit_width` controls turbovec quantization (4 = 4-bit, good default).
* **`add_source(path, &EmbeddingService<M>) -> Result<usize>`**
  High-level file ingestion: loads, chunks, embeds, and persists. Returns chunk count. File type is selected by extension (`.pdf` → `PdfLoader`, else `TextLoader`).
* **`add_source_dyn(path, &dyn ErasedEmbedder) -> Result<usize>`**
  Same as `add_source` but accepts a trait object — use with `Arc<dyn ErasedEmbedder>`.
* **`remove_source(source_name) -> Result<usize>`**
  Drops every chunk whose `source` matches. Returns number removed.
* **`build(embedder: Arc<dyn ErasedEmbedder>) -> TurboVectorIndex`**
  Returns a rig-compatible `VectorStoreIndex` view sharing the same underlying state.
* **`save(&index_path) -> Result<()>`**
  Persists the turbovec index to disk.
* **`commit_pending(&service) -> Result<usize>`**
  Flushes staged (unpersisted) chunks into the turbovec index.
* **`chunk_count() -> Result<i64>`**
  Number of chunks currently persisted.
* **`store() -> &Arc<DocumentStore>`** / **`turbo() -> &Arc<RwLock<TurboIndex>>`**
  Accessors for advanced use.

**Persisted vs staged state:** Chunks inserted via `add_source` are persisted to SQLite immediately. The turbovec index is built in memory and must be flushed with `save()` or `commit_pending()` to survive restarts.

---

## `ErasedEmbedder` Trait

Object-safe trait for embedding without knowing the concrete model type. Implementors: `EmbeddingService<M>` (blanket impl) and custom wrappers.

```rust
pub trait ErasedEmbedder: Send + Sync {
    fn embed_query<'a>(&'a self, text: &'a str) -> QueryFuture<'a>;
    fn embed_texts<'a>(&'a self, texts: Vec<String>) -> TextsFuture<'a>;
    fn ndims(&self) -> usize;
}
```

### Example Usage: Building a RAG Index

```rust
use std::path::Path;
use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::rag::{DocumentLoader, PdfLoader, RagPipeline, WordSplitter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Initialize local fastembed embeddings
    let service = EmbeddingService::from_fastembed("Xenova/bge-small-en-v1.5".parse()?)?;
    let dim = service.ndims();

    // 2. Open or create persistent RAG pipeline
    let pipeline = RagPipeline::open_or_create(
        Path::new("rag_data/rag.db"),
        Path::new("rag_data/rag.tvim"),
        dim,
        4, // bit_width
    ).await?;

    // 3. Ingest a PDF
    let chunks = pipeline.add_source(Path::new("orientation.pdf"), &service).await?;
    println!("Indexed {chunks} chunks");

    // 4. Save to disk
    pipeline.save(Path::new("rag_data/rag.tvim")).await?;

    // 5. Build rig-compatible index for agents
    let index = pipeline.build(std::sync::Arc::new(service));
    Ok(())
}
```
