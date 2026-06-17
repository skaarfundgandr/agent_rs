#![cfg(feature = "rag")]

use crate::agent::permission::{PermissionPolicy, PermissionResult};
use crate::domain::errors::DocumentError;
use crate::domain::rag::{RagSource, RagSourceType};
use crate::rag::{ErasedEmbedder, RagPipeline};
use crate::security::{SandboxConfig, validate_sandboxed_path};
use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A thread-safe registry that tracks RAG sources (files and directories).
///
/// The registry does **not** rebuild the vector index itself. It maintains the
/// authoritative list of registered paths so that consumers can read
/// [`sources()`](RagSourceRegistry::sources) and rebuild the pipeline when needed.
///
/// # Thread Safety
///
/// Intended to be wrapped in `Arc<Mutex<...>>` for shared ownership across tools.
/// The mutex is held only for the duration of each call, so contention is minimal.
#[derive(Debug, Clone)]
pub struct RagSourceRegistry {
    sources: Vec<RagSource>,
    supported_extensions: HashSet<String>,
}

impl RagSourceRegistry {
    /// Creates a new empty registry.
    ///
    /// # Arguments
    ///
    /// * `supported_extensions` - File extensions (without the dot) that the
    ///   consumer can load into the RAG pipeline. When adding a file, its
    ///   extension is validated against this set.
    pub fn new(supported_extensions: HashSet<String>) -> Self {
        Self {
            sources: Vec::new(),
            supported_extensions,
        }
    }

    /// Adds a source (file or directory) to the registry.
    ///
    /// The path is resolved relative to `sandbox` and canonicalized.
    /// Files are checked against the supported extension set, and duplicate
    /// paths are rejected.
    ///
    /// # Arguments
    ///
    /// * `path` - Relative or absolute path to the file or directory.
    /// * `sandbox` - Sandbox configuration within which the path is validated.
    ///
    /// # Returns
    ///
    /// A confirmation message including the source type and total count.
    ///
    /// # Errors
    ///
    /// * [`DocumentError::SandboxEscape`] if the path resolves outside all sandbox roots.
    /// * [`DocumentError::UnsupportedExtension`] if the file extension is not in the supported set.
    /// * [`DocumentError::Rag`] if the source is already indexed or the path does not exist.
    pub fn add_source(
        &mut self,
        path: &Path,
        sandbox: &SandboxConfig,
    ) -> Result<String, DocumentError> {
        let canonical = validate_sandboxed_path(sandbox, path)?;

        if self.sources.iter().any(|s| s.path == canonical) {
            return Err(DocumentError::Rag(format!(
                "Source already indexed: {}",
                path.display()
            )));
        }

        if canonical.is_file() {
            let ext = canonical.extension().and_then(|e| e.to_str()).unwrap_or("");

            if !self.supported_extensions.contains(ext) {
                return Err(DocumentError::UnsupportedExtension(ext.to_string()));
            }

            self.sources.push(RagSource {
                path: canonical,
                source_type: RagSourceType::File,
            });
        } else if canonical.is_dir() {
            self.sources.push(RagSource {
                path: canonical,
                source_type: RagSourceType::Directory,
            });
        } else {
            return Err(DocumentError::Rag(format!(
                "Path does not exist: {}",
                path.display()
            )));
        }

        Ok(format!(
            "Added source '{}' (total: {})",
            path.display(),
            self.sources.len()
        ))
    }

    /// Removes a source by its canonical path.
    ///
    /// # Arguments
    ///
    /// * `canonical_path` - The canonicalized file or directory path to remove.
    ///
    /// # Returns
    ///
    /// Returns a confirmation message as a `String` if successful.
    ///
    /// # Errors
    ///
    /// Returns [`DocumentError::Rag`] if the source is not found in the registry.
    pub fn remove_source(&mut self, canonical_path: &Path) -> Result<String, DocumentError> {
        let before = self.sources.len();
        self.sources.retain(|s| s.path != canonical_path);

        if self.sources.len() == before {
            return Err(DocumentError::Rag(format!(
                "Source not found: {}",
                canonical_path.display()
            )));
        }

        Ok(format!(
            "Removed source '{}' (total: {})",
            canonical_path.display(),
            self.sources.len()
        ))
    }

    /// Returns all registered sources.
    ///
    /// # Returns
    ///
    /// Returns a slice of all currently registered `RagSource`s.
    pub fn sources(&self) -> &[RagSource] {
        &self.sources
    }

    /// Returns a formatted list of all sources.
    ///
    /// # Returns
    ///
    /// Returns a formatted string listing all registered sources, or a message indicating none are registered.
    pub fn list_sources(&self) -> String {
        if self.sources.is_empty() {
            return "No sources registered.".to_string();
        }

        self.sources
            .iter()
            .map(|s| {
                let kind = match s.source_type {
                    RagSourceType::File => "FILE",
                    RagSourceType::Directory => "DIR",
                };
                format!("[{}] {}", kind, s.path.display())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns `true` if no sources are registered.
    ///
    /// # Returns
    ///
    /// Returns `true` if the registry has no sources, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

/// Arguments for the `manage_rag` tool.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ManageRagArgs {
    /// The action to perform: `"add"`, `"remove"`, or `"list"`.
    pub action: String,
    /// Path to the file or directory (relative to sandbox root).
    /// Required for `"add"` and `"remove"` actions.
    pub path: Option<String>,
}

/// A unified tool for managing RAG sources.
///
/// Supports three actions via a string enum argument:
/// - **add** — Register a file or directory as a RAG source. The path is
///   validated against the sandbox root and the file extension is checked
///   against the supported set. Duplicates are rejected.
/// - **remove** — Unregister a previously added source by path.
/// - **list** — Display all currently registered sources with their type.
///
/// After modifying the registry, the consumer should rebuild the RAG index
/// from the updated source list via [`RagSourceRegistry::sources()`].
#[derive(Clone)]
pub struct ManageRagTool {
    registry: Arc<Mutex<RagSourceRegistry>>,
    pipeline: Arc<RagPipeline>,
    embedder: Arc<dyn ErasedEmbedder>,
    sandbox: SandboxConfig,
    policy: PermissionPolicy,
}

impl ManageRagTool {
    /// Creates a new `ManageRagTool` backed by the given shared registry.
    ///
    /// # Arguments
    ///
    /// * `registry` - Shared registry that this tool reads from and writes to.
    /// * `sandbox` - Sandbox configuration within which all source paths are resolved
    ///   and validated. Paths that escape all sandbox roots are rejected.
    /// * `policy` - Permission policy evaluated before each tool invocation.
    pub fn new(
        registry: Arc<Mutex<RagSourceRegistry>>,
        pipeline: Arc<RagPipeline>,
        embedder: Arc<dyn ErasedEmbedder>,
        sandbox: SandboxConfig,
        policy: PermissionPolicy,
    ) -> Self {
        Self {
            registry,
            pipeline,
            embedder,
            sandbox,
            policy,
        }
    }
}

impl Tool for ManageRagTool {
    const NAME: &'static str = "manage_rag";

    type Error = DocumentError;
    type Args = ManageRagArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Manage RAG sources: add a file or directory, remove a source, or list all indexed sources. After add/remove, rebuild the RAG pipeline from the updated registry.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "list"],
                        "description": "Action to perform: 'add' registers a new source, 'remove' unregisters an existing source, 'list' shows all registered sources"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory (relative to sandbox root). Required for 'add' and 'remove' actions."
                    }
                },
                "required": ["action"]
            }),
        }
    }

    /// Dispatches the requested action against the shared registry.
    ///
    /// Acquires the registry mutex for the duration of each operation. The
    /// mutex is released before returning, so contention is minimal.
    ///
    /// # Errors
    ///
    /// * [`DocumentError::Rag`] for invalid actions, missing required arguments,
    ///   or registry-level errors (duplicate source, source not found).
    /// * [`DocumentError::PermissionDenied`] if the permission policy rejects
    ///   the invocation.
    /// * [`DocumentError::SandboxEscape`] if the path escapes all sandbox roots
    ///   (via `add_source`).
    /// * [`DocumentError::UnsupportedExtension`] if the file extension is not
    ///   in the supported set (via `add_source`).
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let description = match args.action.as_str() {
            "add" => format!(
                "Wants to add RAG source [{}]",
                args.path.as_deref().unwrap_or("")
            ),
            "remove" => format!(
                "Wants to remove RAG source [{}]",
                args.path.as_deref().unwrap_or("")
            ),
            "list" => "Wants to list all RAG sources".to_string(),
            other => {
                return Err(DocumentError::Rag(format!(
                    "Unknown action '{other}'. Valid actions: add, remove, list"
                )));
            }
        };

        match self.policy.evaluate(Self::NAME, &description).await {
            PermissionResult::Allow => {}
            PermissionResult::Deny { reason } => {
                return Err(DocumentError::PermissionDenied(format!(
                    "{description}: {reason}"
                )));
            }
            PermissionResult::DeferToUser => {
                return Err(DocumentError::PermissionDenied(format!(
                    "{description}: defer-to-user not yet supported"
                )));
            }
        }

        match args.action.as_str() {
            "add" => {
                let path_str = args.path.ok_or_else(|| {
                    DocumentError::Rag(
                        "The 'path' argument is required for the 'add' action".to_string(),
                    )
                })?;
                let path = Path::new(&path_str);

                let confirmation = {
                    let mut registry = self
                        .registry
                        .lock()
                        .map_err(|e| DocumentError::Rag(e.to_string()))?;
                    registry.add_source(path, &self.sandbox)?
                };

                let canonical = validate_sandboxed_path(&self.sandbox, path)?;
                let added = self
                    .pipeline
                    .add_source_dyn(&canonical, self.embedder.as_ref())
                    .await
                    .map_err(|e| DocumentError::Rag(format!("pipeline add failed: {e}")))?;

                Ok(format!("{confirmation} (indexed {added} chunks)"))
            }
            "remove" => {
                let path_str = args.path.ok_or_else(|| {
                    DocumentError::Rag(
                        "The 'path' argument is required for the 'remove' action".to_string(),
                    )
                })?;
                let path = Path::new(&path_str);
                let canonical = validate_sandboxed_path(&self.sandbox, path)?;

                let confirmation = {
                    let mut registry = self
                        .registry
                        .lock()
                        .map_err(|e| DocumentError::Rag(e.to_string()))?;
                    registry.remove_source(&canonical)?
                };

                let source_name = canonical
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                let removed = self
                    .pipeline
                    .remove_source(source_name)
                    .await
                    .map_err(|e| DocumentError::Rag(format!("pipeline remove failed: {e}")))?;

                Ok(format!("{confirmation} (removed {removed} chunks)"))
            }
            "list" => {
                let registry = self
                    .registry
                    .lock()
                    .map_err(|e| DocumentError::Rag(e.to_string()))?;
                Ok(registry.list_sources())
            }
            _ => unreachable!(),
        }
    }
}
