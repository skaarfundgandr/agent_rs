# Migration Guide: v0.11.x → v0.12.0

## Overview

v0.12.0 removes the flat `from_fastembed*` constructors from `EmbeddingService`. The `EmbeddingService::builder()` API is now the sole way to construct a local fastembed-backed service. The builder also auto-detects GPU execution providers from the compile-time feature flags, so most GPU users no longer pass providers explicitly.

## Breaking: `from_fastembed*` Constructors Removed

All four flat constructors are removed:

- `from_fastembed(model)`
- `from_fastembed_with_cache_dir(model, cache_dir)`
- `from_fastembed_with_providers(model, providers)`
- `from_fastembed_with_providers_and_cache_dir(model, providers, cache_dir)`

**Before (v0.11.x):**
```rust
let embedder = EmbeddingService::from_fastembed("BGESmallENV15".parse()?)?;

let embedder = EmbeddingService::from_fastembed_with_providers_and_cache_dir(
    "BGESmallENV15".parse()?,
    vec![
        CUDAExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ],
    "./models",
)?;
```

**After (v0.12.0):**
```rust
let embedder = EmbeddingService::builder()
    .model(FastembedModel::BGESmallENV15)
    .build()?;

let embedder = EmbeddingService::builder()
    .model(FastembedModel::BGESmallENV15)
    .cache_dir("./models")
    .execution_providers(vec![
        CUDAExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ])
    .build()?;
```

EP types remain at the re-exported ort path:

```rust
use agent_rs::agent::embeddings::ort::ep::{
    CUDAExecutionProvider,
    CPUExecutionProvider,
};
```

## Behavioral: `show_progress` Defaults to `false`

The removed constructors hardcoded the model-download progress bar to `true`. The builder defaults it to `false`. Opt back in explicitly:

```rust
let embedder = EmbeddingService::builder()
    .model(FastembedModel::BGESmallENV15)
    .show_progress(true)
    .build()?;
```

## New: GPU Auto-Detect from Feature Flags

When no `.execution_providers()` is set, `build()` auto-adds the GPU providers enabled at compile time, followed by a CPU fallback:

| Feature | Auto-added provider |
|---|---|
| `rag-cuda` | CUDA |
| `rag-directml` | DirectML |
| `rag-rocm` | ROCm |

Calling `.execution_providers()` overrides auto-detect entirely; the supplied list is used as-is, so append `CPUExecutionProvider::default().build()` as the final entry for runtime fallback.

## New: `EmbeddingServiceBuilder` Re-export

`EmbeddingServiceBuilder` is re-exported from `agent_rs::agent`, alongside the existing `EmbeddingService` re-export.

## Verification

```bash
cargo test --all-features -- --include-ignored
cargo clippy --all-features --all-targets
cargo fmt --check
```

GPU compilation (e.g. `cargo check --features rag,rag-cuda`) and model downloads are **user-side verification** and are not part of automated execution. The default `cargo test --features rag` path runs CPU-only.
