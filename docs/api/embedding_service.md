# Embedding Service

Wraps any Rig `EmbeddingModel` to provide structured document splitting, order-preserving batching, and error handling.

> **Pipeline reference:** See the [RAG flowchart](../diagrams/flowchart.md) for how embedding batching integrates with the ingestion pipeline, and the [class diagram](../diagrams/class-diagram.md) for `EmbeddingService`'s generic type constraints and relationships.

## `EmbeddingService<M>`

Generic over `M: EmbeddingModel`.

### Methods
- **`new(model: M) -> Self`**
  Wraps a concrete Rig embedding model.
- **`into_inner(self) -> M`**
  Consumes the service and returns the inner model.
- **`model(&self) -> &M`**
  Returns a reference to the wrapped Rig embedding model.
- **`ndims(&self) -> usize`**
  Returns the dimensions of the embedding vectors.
- **`max_documents(&self) -> usize`**
  Returns the maximum batch size accepted by the model provider in a single request.
- **`async embed_text(&self, text: impl AsRef<str>) -> Result<Embedding>`**
  Embeds a single string of text.
- **`async embed_texts<I, S>(&self, texts: I) -> Result<Vec<Embedding>>`**
  Embeds an iterator of text slices. Automatically batches the requests to respect `max_documents` while preserving original ordering. Returns an error if the collection is empty or if a batch returns a mismatched count.
- **`async embed_document<T: Embed>(&self, document: T) -> Result<(T, OneOrMany<Embedding>)>`**
  Extracts text fragments from a single document implementing Rig's `Embed` trait and embeds them.
- **`async embed_documents<T: Embed, I>(&self, documents: I) -> Result<Vec<(T, OneOrMany<Embedding>)>>`**
  Batches and embeds multiple `Embed` documents, maintaining original ordering. Returns an error if any document produces no embeddable text.
- **`from_fastembed(model: FastembedModel) -> Result<Self>`** *(requires `rag` feature)*
  Convenience constructor for a local `fastembed` model. Downloads the model from Hugging Face on first call (requires network or a pre-populated cache via `FASTEMBED_CACHE_DIR`).
- **`from_fastembed_with_cache_dir(model: FastembedModel, cache_dir: impl AsRef<Path>) -> Result<Self>`** *(requires `rag` feature)*
  Convenience constructor with an explicit cache directory. Unlike the old implementation, this does **not** mutate `FASTEMBED_CACHE_DIR` process-wide; it sets the cache directory via `TextInitOptions::with_cache_dir`. The env var is still honored by fastembed as the default when not set.
- **`from_fastembed_with_providers(model: FastembedModel, providers: Vec<ExecutionProviderDispatch>) -> Result<Self>`** *(requires `rag` feature)*
  Convenience constructor with opt-in execution providers for GPU acceleration. The provider list is priority-ordered; callers should append `CPUExecutionProvider::default().build()` as the final entry for runtime fallback.
- **`from_fastembed_with_providers_and_cache_dir(model: FastembedModel, providers: Vec<ExecutionProviderDispatch>, cache_dir: impl AsRef<Path>) -> Result<Self>`** *(requires `rag` feature)*
  Combines providers and cache directory in one constructor.

### Re-exports *(requires `rag` feature)*

When the `rag` feature is enabled, `agent_rs::agent::embeddings` re-exports:

- **`FastembedModel`** — alias for `fastembed::EmbeddingModel` (the enum of supported embedding model variants). Parse from a variant-name string: `"BGESmallENV15".parse::<FastembedModel>()?`.
- **`FastembedEmbeddingModel`** — the internal `EmbeddingModel` wrapper around a fastembed `TextEmbedding` instance. Its `EmbeddingModel::make()` factory is **unsupported** and returns an error; construct via `EmbeddingService::from_fastembed*`.
- **`ort`** — the `ort` crate re-exported for constructing `ExecutionProviderDispatch` values. EPs live at `agent_rs::agent::embeddings::ort::execution_providers::*`.

## Free Functions

### `service_from_client()`

```rust
pub fn service_from_client<C: EmbeddingsClient>(
    client: &C,
    model: impl Into<String>,
) -> EmbeddingService<C::EmbeddingModel>
```

Builds an `EmbeddingService` from any Rig embedding-capable client.

### `service_from_client_with_ndims()`

```rust
pub fn service_from_client_with_ndims<C: EmbeddingsClient>(
    client: &C,
    model: impl Into<String>,
    ndims: usize,
) -> EmbeddingService<C::EmbeddingModel>
```

Builds an `EmbeddingService` from a model name and explicit dimensions.
