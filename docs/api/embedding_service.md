# Embedding Service

Wraps any Rig `EmbeddingModel` to provide structured document splitting, order-preserving batching, and error handling.

> **Pipeline reference:** See the [RAG flowchart](../diagrams/flowchart.md) for how embedding batching integrates with the ingestion pipeline, and the [class diagram](../diagrams/class-diagram.md) for `EmbeddingService`'s generic type constraints and relationships.

## `EmbeddingService<M>`

Generic over `M: EmbeddingModel`.

### Methods
- **`new(model: M) -> Self`**
  Wraps a concrete Rig embedding model. The wrapped model is stored privately — there is no accessor to retrieve it.
- **`ndims(&self) -> usize`**
  Returns the dimensions of the embedding vectors.
- **`max_documents(&self) -> usize`**
  Returns the maximum batch size accepted by the model provider in a single request.
- **`async embed_text(&self, text: impl AsRef<str>) -> Result<Embedding>`**
  Embeds a single string of text.
- **`async embed_texts<I, S>(&self, texts: I) -> Result<Vec<Embedding>>`**
  Embeds an iterator of text slices. Automatically batches the requests to respect `max_documents` while preserving original ordering. Returns an error if the collection is empty or if a batch returns a mismatched count.
- **`async embed_documents<T, I>(&self, documents: I) -> Result<Vec<(T, OneOrMany<Embedding>)>>`** where `T: Embed`, `I: IntoIterator<Item = T>`
  Extracts text fragments from each document implementing Rig's `Embed` trait and embeds them. There is no single-document variant — pass a one-element iterator to embed one document. Preserves the original document order and the order of embedded text fragments within each document, and batches requests to respect `max_documents`. Returns an error if any document produces no embeddable text, if the collection is empty, or if a batch returns a mismatched embedding count.
- **`builder() -> EmbeddingServiceBuilder`** *(requires `rag` feature)*
  Entry point for constructing a local fastembed-backed service. See [`EmbeddingServiceBuilder`](#embeddingservicebuilder-requires-rag-feature) below.

## `EmbeddingServiceBuilder` *(requires `rag` feature)*

Builds an `EmbeddingService<FastembedEmbeddingModel>` backed by a local `fastembed` model. Downloads the model from Hugging Face on first build (requires network or a pre-populated cache via `FASTEMBED_CACHE_DIR`).

```rust
use agent_rs::agent::embeddings::{EmbeddingService, FastembedModel};

let embedder = EmbeddingService::builder()
    .model(FastembedModel::BGESmallENV15)
    .cache_dir("./models")
    .show_progress(true)
    .build()?;
```

### Methods
- **`EmbeddingService::builder() -> EmbeddingServiceBuilder`**
  Creates a new builder with no model set.
- **`.model(model: FastembedModel) -> Self`**
  Sets the fastembed model variant. Required; `build()` errors without it.
- **`.cache_dir(dir: impl AsRef<Path>) -> Self`**
  Sets an explicit model cache directory via `TextInitOptions::with_cache_dir`. Does **not** mutate `FASTEMBED_CACHE_DIR` process-wide; fastembed still honors the env var as the default when no directory is set.
- **`.show_progress(show: bool) -> Self`**
  Toggles the model-download progress bar. Defaults to `false`.
- **`.execution_providers(providers: Vec<ExecutionProviderDispatch>) -> Self`**
  Sets an explicit, priority-ordered execution provider list for GPU acceleration. Supplied providers **replace** the feature-gated GPU auto-detect defaults; the CPU provider is always appended automatically as the final runtime fallback during `build()` — do **not** add it manually.
- **`.build() -> Result<EmbeddingService<FastembedEmbeddingModel>>`**
  Constructs the service. When no `.execution_providers()` is set, auto-adds the GPU providers enabled at compile time (`rag-cuda` → CUDA, `rag-directml` → DirectML, `rag-rocm` → ROCm). The CPU provider is appended unconditionally as the final fallback — whether or not `.execution_providers()` was set. Under `rag-load-dynamic`, all EP types are available at compile time, so the builder auto-registers CUDA + ROCm (Linux/macOS) or DirectML (Windows); it also loads the bundled ORT dylib (resolved at build time) if found, falling back to the system linker otherwise.

### Re-exports *(requires `rag` feature)*

When the `rag` feature is enabled, `agent_rs::agent::embeddings` re-exports:

- **`FastembedModel`** — alias for `fastembed::EmbeddingModel` (the enum of supported embedding model variants). Parse from a variant-name string: `"BGESmallENV15".parse::<FastembedModel>()?`.
- **`FastembedEmbeddingModel`** — the internal `EmbeddingModel` wrapper around a fastembed `TextEmbedding` instance. Its `EmbeddingModel::make()` factory is **unsupported** and returns an error; construct via `EmbeddingService::builder()`.
- **`EmbeddingServiceBuilder`** — the builder returned by `EmbeddingService::builder()`.
- **`ort`** — the `ort` crate re-exported for constructing `ExecutionProviderDispatch` values. EPs live at `agent_rs::agent::embeddings::ort::ep::*`.

