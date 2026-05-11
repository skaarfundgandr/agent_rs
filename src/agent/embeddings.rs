use std::cmp::max;

use anyhow::{Context, Result, bail};
use rig::embeddings::{Embedding, EmbeddingModel, embed::to_texts};
use rig::{Embed, OneOrMany, client::EmbeddingsClient};

/// A small, reusable embedding service for library consumers.
///
/// This type is intentionally generic over the Rig embedding model so it can be used with any
/// provider that implements `rig::embeddings::EmbeddingModel`.
///
/// Prefer constructing the underlying provider outside this module and then passing the embedding
/// model into [`EmbeddingService::new`].
pub struct EmbeddingService<M> {
    model: M,
}

impl<M> EmbeddingService<M> {
    /// Create a new embedding service from a concrete Rig embedding model.
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Consume the service and return the inner model.
    pub fn into_inner(self) -> M {
        self.model
    }

    /// Access the wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }
}

impl<M> EmbeddingService<M>
where
    M: EmbeddingModel,
{
    /// Number of dimensions produced by the underlying embedding model.
    pub fn ndims(&self) -> usize {
        self.model.ndims()
    }

    /// Maximum number of texts the provider accepts in a single request.
    pub fn max_documents(&self) -> usize {
        max(1, M::MAX_DOCUMENTS)
    }

    /// Embed a single text value.
    pub async fn embed_text(&self, text: impl AsRef<str>) -> Result<Embedding> {
        self.model
            .embed_text(text.as_ref())
            .await
            .map_err(Into::into)
    }

    /// Embed a list of plain text inputs.
    ///
    /// Inputs are batched to respect the provider's request limit while preserving order.
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

    /// Embed a single document implementing Rig's `Embed` trait.
    pub async fn embed_document<T>(&self, document: T) -> Result<(T, OneOrMany<Embedding>)>
    where
        T: Embed,
    {
        self.embed_documents(std::iter::once(document))
            .await?
            .into_iter()
            .next()
            .context("document embedding result was unexpectedly empty")
    }

    /// Embed multiple documents implementing Rig's `Embed` trait.
    ///
    /// This preserves the original document order and the order of embedded text fragments within
    /// each document.
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

/// Convenience helper that builds an [`EmbeddingService`] from any Rig embedding-capable client.
pub fn service_from_client<C>(
    client: &C,
    model: impl Into<String>,
) -> EmbeddingService<C::EmbeddingModel>
where
    C: EmbeddingsClient,
{
    EmbeddingService::new(client.embedding_model(model))
}

/// Convenience helper that builds an [`EmbeddingService`] from a model name and explicit dimensions.
pub fn service_from_client_with_ndims<C>(
    client: &C,
    model: impl Into<String>,
    ndims: usize,
) -> EmbeddingService<C::EmbeddingModel>
where
    C: EmbeddingsClient,
{
    EmbeddingService::new(client.embedding_model_with_ndims(model, ndims))
}
