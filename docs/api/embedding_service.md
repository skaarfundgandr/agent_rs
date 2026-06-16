# Embedding Service

Wraps any Rig `EmbeddingModel` to provide structured document splitting, order-preserving batching, and error handling.

> **Pipeline reference:** See the [RAG flowchart](../diagrams/flowchart.md) for how embedding batching integrates with the ingestion pipeline, and the [class diagram](../diagrams/class-diagram.md) for `EmbeddingService`'s generic type constraints and relationships.

## `EmbeddingService<M>`

Generic over `M: EmbeddingModel`.

### Methods
* **`new(model: M) -> Self`**
  Wraps a concrete Rig embedding model.
* **`ndims(&self) -> usize`**
  Returns the dimensions of the embedding vectors.
* **`max_documents(&self) -> usize`**
  Returns the maximum batch size accepted by the model provider in a single request.
* **`async embed_text(&self, text: impl AsRef<str>) -> Result<Embedding>`**
  Embeds a single string of text.
* **`async embed_texts<I, S>(&self, texts: I) -> Result<Vec<Embedding>>`**
  Embeds an iterator of text slices. Automatically batches the requests to respect `max_documents` while preserving original ordering.
* **`async embed_document<T: Embed>(&self, document: T) -> Result<(T, OneOrMany<Embedding>)>`**
  Extracts text fragments from a document implementing Rig's `Embed` trait and embeds them.
* **`async embed_documents<T: Embed, I>(&self, documents: I) -> Result<Vec<(T, OneOrMany<Embedding>)>>`**
  Batches and embeds multiple `Embed` documents, maintaining original ordering.
* **`from_fastembed(model: FastembedModel) -> Result<Self, FastembedError>`** *(requires `rag` feature)*
  Convenience constructor for a local `fastembed` model. Downloads the model from Hugging Face on first call (requires network or a pre-populated cache via `FASTEMBED_CACHE_DIR`). The `FASTEMBED_MODEL` env var selects the model at runtime (default `Xenova/bge-small-en-v1.5`).
