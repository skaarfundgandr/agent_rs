#![cfg(feature = "rag")]

use crate::agent::permission::PermissionPolicy;
use crate::domain::errors::DocumentError;
use crate::domain::rag::{RagSource, RagSourceType};
use crate::rag::RagPipeline;
use crate::security::SharedSandbox;
use anyhow::Result;
use rig_core::completion::ToolDefinition;
use rig_core::tool::Tool;
use serde_json::json;
use std::collections::HashSet;
use std::path::Path;

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

    /// Rebuild the registry from sources persisted in the SQLite store.
    ///
    /// Sources are read from the dedicated `rag_sources` table, which stores
    /// canonical paths and their types (`file` or `directory`). If the table
    /// is empty (e.g. an older database), the registry starts empty.
    pub async fn hydrate_from_store(
        pipeline: &RagPipeline,
        supported_extensions: HashSet<String>,
    ) -> Result<Self> {
        let sources = pipeline.list_registered_sources().await?;
        Ok(Self {
            sources,
            supported_extensions,
        })
    }

    /// Adds a source (file or directory) to the registry.
    ///
    /// The path is resolved relative to `sandbox` and canonicalized.
    /// Files are checked against the supported extension set. Duplicate
    /// paths are ignored (the registry is idempotent).
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
    /// * [`DocumentError::Rag`] if the path does not exist.
    pub fn add_source(
        &mut self,
        path: &Path,
        sandbox: &SharedSandbox,
    ) -> Result<String, DocumentError> {
        let canonical = sandbox.resolve_path_unchecked(path);

        if self.sources.iter().any(|s| s.path == canonical) {
            return Ok(format!(
                "Source already indexed: {} (total: {})",
                path.display(),
                self.sources.len()
            ));
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

/// A thin permission shell over [`RagIndexer`].
///
/// Supports three actions via a string enum argument:
/// - **add** — Register a file or directory as a RAG source. The path is
///   validated against the sandbox root and the file extension is checked
///   against the supported set. Duplicate adds are ignored.
/// - **remove** — Unregister a previously added source by path and delete
///   its chunks from the persisted store/index.
/// - **list** — Display all currently registered sources with their type.
///
/// Changes are persisted directly; the consumer does not need to rebuild the
/// index manually.
#[derive(Clone)]
pub struct ManageRagTool {
    pub(crate) indexer: crate::rag::RagIndexer,
    policy: PermissionPolicy,
}

impl ManageRagTool {
    /// Creates a new `ManageRagTool` backed by the given indexer.
    pub(crate) fn new(indexer: crate::rag::RagIndexer, policy: PermissionPolicy) -> Self {
        Self { indexer, policy }
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
            description: "Manage RAG sources: add a file or directory, remove a source, or list all indexed sources. Changes are persisted directly.".to_string(),
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

    /// Dispatches the requested action, delegating to [`RagIndexer`].
    ///
    /// # Errors
    ///
    /// * [`DocumentError::Rag`] for invalid actions, missing required arguments,
    ///   or indexer-level errors (duplicate source, source not found).
    /// * [`DocumentError::PermissionDenied`] if the permission policy rejects
    ///   the invocation.
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

        // In-sandbox paths are auto-allowed; the `list` action needs no gate
        // (it only reads the in-memory registry). Consult the gate only for
        // out-of-sandbox add/remove paths.
        let needs_gate = if args.action.as_str() == "list" {
            false
        } else {
            let p = args.path.as_deref().unwrap_or("");
            crate::security::validate_sandboxed_path_shared(self.indexer.sandbox(), Path::new(p))
                .is_err()
        };
        if needs_gate {
            self.indexer
                .sandbox()
                .check_permission(&self.policy, Self::NAME, &description)
                .await?;
        }

        match args.action.as_str() {
            "add" => {
                let path_str = args.path.ok_or_else(|| {
                    DocumentError::Rag(
                        "The 'path' argument is required for the 'add' action".to_string(),
                    )
                })?;
                let path = Path::new(&path_str);
                let added = self
                    .indexer
                    .add(path)
                    .await
                    .map_err(|e| DocumentError::Rag(e.to_string()))?;
                Ok(format!("indexed {added} chunks"))
            }
            "remove" => {
                let path_str = args.path.ok_or_else(|| {
                    DocumentError::Rag(
                        "The 'path' argument is required for the 'remove' action".to_string(),
                    )
                })?;
                let path = Path::new(&path_str);
                let removed = self
                    .indexer
                    .remove(path)
                    .await
                    .map_err(|e| DocumentError::Rag(e.to_string()))?;
                Ok(format!("removed {removed} chunks"))
            }
            "list" => {
                let sources = self.indexer.list();
                let output = if sources.is_empty() {
                    "No sources registered.".to_string()
                } else {
                    sources
                        .iter()
                        .map(|s| {
                            let kind = match s.source_type {
                                crate::domain::rag::RagSourceType::File => "FILE",
                                crate::domain::rag::RagSourceType::Directory => "DIR",
                            };
                            format!("[{}] {}", kind, s.path.display())
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(output)
            }
            _ => Err(DocumentError::Rag(format!(
                "Unknown action '{}'. Valid actions: add, remove, list",
                args.action
            ))),
        }
    }
}
