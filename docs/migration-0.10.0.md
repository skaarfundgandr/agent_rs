# Migration Guide: v0.9.x → v0.10.0

## Overview

v0.10.0 replaces the `rig-fastembed 0.40.0` dependency with a direct `fastembed 5.17.3` dependency (unified to `ort =2.0.0-rc.12`). The default `rag` build is CPU-only; GPU acceleration is opt-in via four new features. `FASTEMBED_MODEL` env var and `.parse()` format changed from HuggingFace model codes to fastembed variant names.

## Dependency Changes

| Dependency | v0.9.x | v0.10.0 |
|---|---|---|
| `rig-fastembed` | 0.40.0 | removed |
| `fastembed` | — | 5.17.3 (new, default CPU-only) |
| `ort` | 2.0.0-rc.9 (via fastembed 4.9.1 inside rig-fastembed 0.40.0) | =2.0.0-rc.12 (new, shared dep) |
| `rag-cuda` | — | new, enables `ort/cuda` |
| `rag-directml` | — | new, enables `ort/directml` |
| `rag-rocm` | — | new, enables `ort/rocm` |
| `rag-load-dynamic` | — | new, enables `ort/load-dynamic` |

All other dependencies are unchanged.

## Breaking: Type Path Change

**`rig_fastembed::FastembedModel`** → **`agent_rs::agent::embeddings::FastembedModel`** (re-export of `fastembed::EmbeddingModel`).

`agent/mod.rs` still re-exports only `EmbeddingService`; the wrapper and re-exports live at `agent::embeddings::*`.

| Old path | New path |
|---|---|
| `rig_fastembed::FastembedModel` | `agent_rs::agent::embeddings::FastembedModel` |
| `rig_fastembed::FastembedModel::AllMiniLML6V2` | `agent_rs::agent::embeddings::FastembedModel::AllMiniLML6V2` (or `fastembed::EmbeddingModel::AllMiniLML6V2`) |

## Breaking: `FastembedError` Removed

`from_fastembed` and friends now return `anyhow::Result<Self>` instead of `Result<Self, FastembedError>`.

**Before (v0.9.x):**
```rust
use rig_fastembed::FastembedError;
let service = EmbeddingService::from_fastembed(model)?;
```

**After (v0.10.0):**
```rust
let service = EmbeddingService::from_fastembed(model)?; // anyhow::Result
```

## Breaking: `FASTEMBED_MODEL` Format

fastembed 5.x uses `FromStr` with variant Debug names (case-insensitive), not HuggingFace model codes. Existing `.env` or `.parse()` calls with `Xenova/...` strings will fail with "Unknown embedding model".

### Conversion Table

| Old code (v0.9.x) | New variant name (v0.10.0) |
|---|---|
| `Xenova/bge-small-en-v1.5` | `BGESmallENV15` |
| `Xenova/all-MiniLM-L6-v2` | `AllMiniLML6V2` |
| `Xenova/all-mpnet-base-v2` | `AllMpnetBaseV2` |
| `Xenova/multilingual-e5-large` | `MultilingualE5Large` |

This is a strict upstream `FromStr` change in fastembed 5.x; no lenient model-code fallback is provided.

## Breaking: `from_fastembed_with_cache_dir` No Longer Mutates Process Env

In v0.9.x, `from_fastembed_with_cache_dir` used `unsafe { std::env::set_var("FASTEMBED_CACHE_DIR", ...) }`. In v0.10.0, it sets the cache directory via `TextInitOptions::with_cache_dir(PathBuf)`. The `FASTEMBED_CACHE_DIR` env var is still honored by fastembed as the default when no explicit cache directory is provided.

## New: GPU Acceleration (Opt-In)

By default, `--features rag` builds CPU-only ONNX Runtime. Four new feature flags enable GPU backends:

| Feature | Hardware | ORT feature |
|---|---|---|
| `rag-cuda` | NVIDIA GPUs | `ort/cuda` |
| `rag-directml` | Windows GPUs | `ort/directml` |
| `rag-rocm` | AMD GPUs | `ort/rocm` |
| `rag-load-dynamic` | System-provided ORT dylib | `ort/load-dynamic` |

Two new constructors accept execution providers:

- `from_fastembed_with_providers(model, providers)`
- `from_fastembed_with_providers_and_cache_dir(model, providers, cache_dir)`

The provider list is priority-ordered. **Callers must append `CPUExecutionProvider::default().build()` as the final entry** for runtime fallback when no GPU is available.

EP types are available at the re-exported ort path:

```rust
use agent_rs::agent::embeddings::ort::ep::{
    CUDAExecutionProvider,
    CPUExecutionProvider,
};
```

### `rag-load-dynamic`

For consumers who want to avoid building ORT GPU binaries from source, set `ORT_DYLIB_PATH` to point at a system-provided or pre-installed ONNX Runtime shared library. The `rag-load-dynamic` feature enables `ort/load-dynamic`, which skips static linking and loads ORT at runtime from `ORT_DYLIB_PATH`.

## New: `ort` Re-export

`agent_rs::agent::embeddings::ort` re-exports the `ort` crate (pinned to `=2.0.0-rc.12`). Consumers construct `ExecutionProviderDispatch` values through this re-export without adding their own `ort` dependency.

### ort Unification

Downstream binaries pairing agent_rs with `transcribe-rs 0.3.11` now get a single `ort =2.0.0-rc.12` runtime — one GPU-enabled ONNX Runtime serves both embeddings and transcription.

## Verification

```bash
cargo test --all-features -- --include-ignored
cargo clippy --all-features --all-targets
cargo fmt --check
```

GPU compilation (e.g. `cargo check --features rag,rag-cuda`) and model downloads are **user-side verification** and are not part of automated execution. The default `cargo test --features rag` path runs CPU-only and matches v0.9.x behavior.
