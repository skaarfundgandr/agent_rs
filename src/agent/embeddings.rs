#![cfg(feature = "rag")]

use std::cmp::max;

use anyhow::{Context, Result, bail};
use rig_core::embeddings::{Embedding, EmbeddingModel, embed::to_texts};
use rig_core::{Embed, OneOrMany};

use crate::rag::{ErasedEmbedder, QueryFuture, TextsFuture};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// A small, reusable embedding service for library consumers.
///
/// This type is intentionally generic over the Rig embedding model so it can be used with any
/// provider that implements `rig_core::embeddings::EmbeddingModel`.
///
/// Prefer constructing the underlying provider outside this module and then passing the embedding
/// model into [`EmbeddingService::new`].
#[derive(Clone, Debug)]
pub struct EmbeddingService<M> {
    model: M,
}

impl<M> EmbeddingService<M> {
    /// Create a new embedding service from a concrete Rig embedding model.
    ///
    /// # Arguments
    ///
    /// * `model` - The concrete Rig embedding model to wrap.
    ///
    /// # Returns
    ///
    /// Returns the initialized `EmbeddingService`.
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

impl<M> EmbeddingService<M>
where
    M: EmbeddingModel,
{
    /// Number of dimensions produced by the underlying embedding model.
    ///
    /// # Returns
    ///
    /// Returns the number of dimensions as a `usize`.
    pub fn ndims(&self) -> usize {
        self.model.ndims()
    }

    /// Maximum number of texts the provider accepts in a single request.
    ///
    /// # Returns
    ///
    /// Returns the maximum documents count as a `usize`.
    pub fn max_documents(&self) -> usize {
        max(1, M::MAX_DOCUMENTS)
    }

    /// Embed a single text value.
    ///
    /// # Arguments
    ///
    /// * `text` - The text slice or reference to embed.
    ///
    /// # Returns
    ///
    /// Returns the computed `Embedding` vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the model invocation fails.
    pub async fn embed_text(&self, text: impl AsRef<str>) -> Result<Embedding> {
        self.model
            .embed_text(text.as_ref())
            .await
            .map_err(Into::into)
    }

    /// Embed a list of plain text inputs.
    ///
    /// Inputs are batched to respect the provider's request limit while preserving order.
    ///
    /// # Arguments
    ///
    /// * `texts` - An iterator of text values to embed.
    ///
    /// # Returns
    ///
    /// Returns a vector of computed `Embedding` vectors in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding any batch fails, or if the provider returns an unexpected number of embeddings.
    pub async fn embed_texts<I, S>(&self, texts: I) -> Result<Vec<Embedding>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let texts: Vec<String> = texts.into_iter().map(Into::into).collect();

        if texts.is_empty() {
            bail!("cannot embed an empty text collection");
        }

        let batch_size = self.max_documents();
        let mut output = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let batch_texts: Vec<String> = batch.to_vec();
            let embeddings = self
                .model
                .embed_texts(batch_texts)
                .await
                .context("failed to embed text batch")?;

            if embeddings.len() != batch.len() {
                bail!(
                    "provider returned {} embeddings for {} texts",
                    embeddings.len(),
                    batch.len()
                );
            }

            output.extend(embeddings);
        }

        Ok(output)
    }

    /// Embed multiple documents implementing Rig's `Embed` trait.
    ///
    /// This preserves the original document order and the order of embedded text fragments within
    /// each document.
    ///
    /// # Arguments
    ///
    /// * `documents` - An iterator of document instances implementing `Embed`.
    ///
    /// # Returns
    ///
    /// Returns a vector of tuples containing each original document and its corresponding computed `Embedding` vector(s).
    ///
    /// # Errors
    ///
    /// Returns an error if any document produces no text, or if embedding the batch fails.
    pub async fn embed_documents<T, I>(
        &self,
        documents: I,
    ) -> Result<Vec<(T, OneOrMany<Embedding>)>>
    where
        T: Embed,
        I: IntoIterator<Item = T>,
    {
        let prepared: Vec<(T, Vec<String>)> = documents
            .into_iter()
            .map(|document| {
                let texts = to_texts(&document).map_err(anyhow::Error::new)?;

                if texts.is_empty() {
                    bail!("a document produced no embeddable text values");
                }

                Ok((document, texts))
            })
            .collect::<Result<_>>()?;

        if prepared.is_empty() {
            bail!("cannot embed an empty document collection");
        }

        let flattened: Vec<(usize, String)> = prepared
            .iter()
            .enumerate()
            .flat_map(|(doc_idx, (_, texts))| {
                texts.iter().cloned().map(move |text| (doc_idx, text))
            })
            .collect();

        let batch_size = self.max_documents();
        let mut grouped: Vec<Vec<Embedding>> = vec![Vec::new(); prepared.len()];

        for batch in flattened.chunks(batch_size) {
            let batch_texts: Vec<String> = batch.iter().map(|(_, text)| text.clone()).collect();
            let batch_embeddings = self
                .model
                .embed_texts(batch_texts)
                .await
                .context("failed to embed document batch")?;

            if batch_embeddings.len() != batch.len() {
                bail!(
                    "provider returned {} embeddings for {} document fragments",
                    batch_embeddings.len(),
                    batch.len()
                );
            }

            for ((doc_idx, _), embedding) in batch.iter().cloned().zip(batch_embeddings) {
                grouped[doc_idx].push(embedding);
            }
        }

        let mut result = Vec::with_capacity(prepared.len());
        for ((document, _), embeddings) in prepared.into_iter().zip(grouped) {
            let embeddings = OneOrMany::many(embeddings)
                .context("a document was expected to have at least one embedding")?;
            result.push((document, embeddings));
        }

        Ok(result)
    }
}

impl<M> ErasedEmbedder for EmbeddingService<M>
where
    M: EmbeddingModel + WasmCompatSend + WasmCompatSync + 'static,
{
    fn ndims(&self) -> usize {
        self.model.ndims()
    }

    fn embed_query<'a>(&'a self, text: &'a str) -> QueryFuture<'a> {
        Box::pin(async move {
            let e = self.embed_text(text).await?;
            Ok(e.vec.into_iter().map(|v| v as f32).collect())
        })
    }

    fn embed_texts<'a>(&'a self, texts: Vec<String>) -> TextsFuture<'a> {
        Box::pin(async move {
            let embeddings = self.embed_texts(texts).await?;
            Ok(embeddings
                .into_iter()
                .map(|e| e.vec.into_iter().map(|v| v as f32).collect())
                .collect())
        })
    }
}

impl EmbeddingService<rig_fastembed::EmbeddingModel> {
    /// Convenience constructor for a local `fastembed` model.
    ///
    /// Downloads the model from Hugging Face on first call (requires network
    /// or a pre-populated cache via `FASTEMBED_CACHE_DIR`).
    ///
    /// # Arguments
    ///
    /// * `model` - The fastembed model enum variant selecting which model to load.
    ///
    /// # Returns
    ///
    /// Returns the initialized `EmbeddingService` using the fastembed model.
    ///
    /// # Errors
    ///
    /// Returns a `FastembedError` if loading or downloading the model fails.
    pub fn from_fastembed(
        model: rig_fastembed::FastembedModel,
    ) -> Result<Self, rig_fastembed::FastembedError> {
        let client = rig_fastembed::Client::new();
        let model = client.embedding_model(&model)?;
        Ok(Self::new(model))
    }
}
