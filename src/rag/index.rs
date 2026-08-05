//! turbovec wrapper for ANN vector search.

use std::path::Path;
use turbovec::{AddError, ConstructError, IdMapIndex};

/// Wrapper over `turbovec::IdMapIndex` that owns the vector quantization +
/// search side. Persists to a single `.tvim` file via [`Self::save`] /
/// [`Self::load`].
pub struct TurboIndex {
    inner: IdMapIndex,
}

impl TurboIndex {
    /// Create a new empty index with eager dimensionality.
    ///
    /// `bit_width` must be one of `{2, 3, 4}`. Higher = more accurate, larger
    /// file. 4 is a good default for general-purpose embeddings.
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: IdMapIndex::new(dim, bit_width)?,
        })
    }

    /// Add `vectors` (flat row-major `ids.len() * dim` length) with their ids.
    pub fn add(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        self.inner.add_with_ids(vectors, ids)
    }

    /// Remove a vector by id. Returns `true` if the id was present.
    pub fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the `k` nearest neighbors of a single query vector.
    /// Returns flat `(scores, ids)` of length `k` each.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.inner.search(query, k)
    }

    /// Load a previously written `.tvim` index.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            inner: IdMapIndex::load(path)?,
        })
    }

    /// Persist the index to disk. Underlying call is `IdMapIndex::write`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        self.inner.write(path)
    }

    /// Number of vectors currently stored in the index.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Dimensionality of the vectors the index was created with.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Quantization bit width the index was created with (`2`, `3`, or `4`).
    pub fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
