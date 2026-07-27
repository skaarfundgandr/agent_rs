#![cfg(feature = "rag")]

use std::cmp::max;
use std::path::{Path, PathBuf};

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

pub use fastembed::EmbeddingModel as FastembedModel;
pub use ort;

use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct FastembedEmbeddingModel {
    model: fastembed::EmbeddingModel,
    ndims: usize,
    embedder: Option<Arc<Mutex<fastembed::TextEmbedding>>>,
    init_error: Option<String>,
}

// fastembed::TextEmbedding has no Debug impl; print static fields only.
impl std::fmt::Debug for FastembedEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastembedEmbeddingModel")
            .field("model", &self.model)
            .field("ndims", &self.ndims)
            .field("has_embedder", &self.embedder.is_some())
            .field("init_error", &self.init_error.as_deref())
            .finish()
    }
}

impl FastembedEmbeddingModel {
    fn build(
        model: fastembed::EmbeddingModel,
        options: fastembed::TextInitOptions,
    ) -> Result<Self> {
        let ndims = fastembed::TextEmbedding::get_model_info(&model)
            .context("failed to resolve fastembed model info")?
            .dim;
        let embedder = fastembed::TextEmbedding::try_new(options)
            .context("failed to initialize fastembed model")?;
        Ok(Self {
            model,
            ndims,
            embedder: Some(Arc::new(Mutex::new(embedder))),
            init_error: None,
        })
    }
}

impl EmbeddingModel for FastembedEmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_: &(), _: impl Into<String>, _: Option<usize>) -> Self {
        Self {
            model: fastembed::EmbeddingModel::default(),
            ndims: 0,
            embedder: None,
            init_error: Some(
                "`make` is not supported for fastembed models; construct via EmbeddingService::builder()"
                    .to_string(),
            ),
        }
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<Embedding>, rig_core::embeddings::EmbeddingError> {
        let Some(embedder) = &self.embedder else {
            return Err(rig_core::embeddings::EmbeddingError::ProviderError(
                self.init_error
                    .clone()
                    .unwrap_or_else(|| "fastembed model initialization failed".to_string()),
            ));
        };
        let documents: Vec<String> = documents.into_iter().collect();
        let mut guard = embedder.lock().map_err(|e| {
            rig_core::embeddings::EmbeddingError::ProviderError(format!(
                "embedding model lock poisoned: {e}"
            ))
        })?;
        let vectors = guard
            .embed(&documents, None)
            .map_err(|e| rig_core::embeddings::EmbeddingError::ProviderError(e.to_string()))?;
        if vectors.len() != documents.len() {
            return Err(rig_core::embeddings::EmbeddingError::ProviderError(
                format!(
                    "fastembed returned {} embeddings for {} documents",
                    vectors.len(),
                    documents.len()
                ),
            ));
        }
        Ok(documents
            .into_iter()
            .zip(vectors)
            .map(|(document, vec)| Embedding {
                document,
                vec: vec.into_iter().map(f64::from).collect(),
            })
            .collect())
    }
}

impl EmbeddingService<FastembedEmbeddingModel> {
    /// Starts building a local `fastembed`-backed embedding service.
    ///
    /// The returned [`EmbeddingServiceBuilder`] configures the model, cache
    /// directory, download progress reporting, and execution providers before
    /// initializing the service with [`EmbeddingServiceBuilder::build`].
    ///
    /// # Returns
    ///
    /// Returns a fresh builder with all fields unset and progress reporting
    /// disabled.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use agent_rs::agent::embeddings::{EmbeddingService, FastembedModel};
    ///
    /// let service = EmbeddingService::builder()
    ///     .model(FastembedModel::BGESmallENV15)
    ///     .show_progress(true)
    ///     .build()?;
    /// # anyhow::Ok(())
    /// ```
    pub fn builder() -> EmbeddingServiceBuilder {
        EmbeddingServiceBuilder {
            model: None,
            cache_dir: None,
            show_progress: false,
            providers: None,
        }
    }
}

/// Builder for a local [`fastembed`]-backed [`EmbeddingService`].
///
/// Downloads the selected model from Hugging Face on first
/// [`build`](EmbeddingServiceBuilder::build) (requires network or a
/// pre-populated cache). When no execution providers are supplied, the builder
/// appends any GPU provider enabled at compile time (`rag-cuda`,
/// `rag-directml`, `rag-rocm`) followed by the always-available CPU provider,
/// so GPU acceleration is opt-in via crate features while CPU remains the
/// runtime fallback.
///
/// This type is only available with the `rag` feature enabled.
///
/// # Examples
///
/// ```no_run
/// use agent_rs::agent::embeddings::{EmbeddingService, FastembedModel};
///
/// let service = EmbeddingService::builder()
///     .model(FastembedModel::BGESmallENV15)
///     .show_progress(true)
///     .build()?;
/// # anyhow::Ok(())
/// ```
pub struct EmbeddingServiceBuilder {
    model: Option<fastembed::EmbeddingModel>,
    cache_dir: Option<PathBuf>,
    show_progress: bool,
    providers: Option<Vec<fastembed::ExecutionProviderDispatch>>,
}

impl EmbeddingServiceBuilder {
    /// Selects which fastembed model to load.
    ///
    /// # Arguments
    ///
    /// * `model` - The fastembed model enum variant to initialize.
    ///
    /// # Returns
    ///
    /// Returns the builder for further configuration.
    pub fn model(mut self, model: fastembed::EmbeddingModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Pins the model cache to an explicit directory.
    ///
    /// Overrides the `FASTEMBED_CACHE_DIR` environment variable so the model is
    /// not re-downloaded when the working directory changes.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory used to store and load downloaded models.
    ///
    /// # Returns
    ///
    /// Returns the builder for further configuration.
    pub fn cache_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.cache_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Controls whether download progress is reported to stderr.
    ///
    /// # Arguments
    ///
    /// * `show` - Enables progress reporting when `true`.
    ///
    /// # Returns
    ///
    /// Returns the builder for further configuration.
    pub fn show_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }

    /// Sets a priority-ordered list of execution providers.
    ///
    /// `ort` tries each provider in sequence. When set, this list replaces the
    /// feature-gated GPU defaults; the CPU provider is always appended as the
    /// final fallback during [`build`](Self::build). Provider types are
    /// available at the re-exported ort path
    /// `agent_rs::agent::embeddings::ort::execution_providers::*`.
    ///
    /// # Arguments
    ///
    /// * `providers` - Priority-ordered execution providers to try.
    ///
    /// # Returns
    ///
    /// Returns the builder for further configuration.
    pub fn execution_providers(
        mut self,
        providers: Vec<fastembed::ExecutionProviderDispatch>,
    ) -> Self {
        self.providers = Some(providers);
        self
    }

    /// Initializes the embedding service from the configured options.
    ///
    /// When no execution providers were supplied, the GPU provider enabled at
    /// compile time (if any) is added first, and the CPU provider is always
    /// appended last as a runtime fallback.
    ///
    /// # Returns
    ///
    /// Returns the initialized [`EmbeddingService`] backed by the selected
    /// fastembed model.
    ///
    /// # Errors
    ///
    /// Returns an error if no model was set, or if loading or downloading the
    /// model fails.
    pub fn build(self) -> Result<EmbeddingService<FastembedEmbeddingModel>> {
        let model = self.model.context("model is required")?;

        let mut providers = self.providers.unwrap_or_default();

        if providers.is_empty() {
            #[cfg(feature = "rag-cuda")]
            {
                providers.push(ort::execution_providers::CUDAExecutionProvider::default().build());
            }
            #[cfg(feature = "rag-directml")]
            {
                providers
                    .push(ort::execution_providers::DirectMLExecutionProvider::default().build());
            }
            #[cfg(feature = "rag-rocm")]
            {
                providers.push(ort::execution_providers::ROCmExecutionProvider::default().build());
            }
        }

        providers.push(ort::execution_providers::CPUExecutionProvider::default().build());

        let mut opts = fastembed::TextInitOptions::new(model.clone())
            .with_show_download_progress(self.show_progress)
            .with_execution_providers(providers);

        if let Some(dir) = self.cache_dir {
            opts = opts.with_cache_dir(dir);
        }

        Ok(EmbeddingService::new(FastembedEmbeddingModel::build(
            model, opts,
        )?))
    }
}
