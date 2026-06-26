//! Directory walking used by `ingest::add_directory`.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Walk `dir` recursively and return the paths of every regular file whose
/// extension is in `extensions`. Hidden files and directories (those whose
/// name starts with `.`) are skipped. Symbolic links are not followed.
///
/// Returns the first walkdir error encountered wrapped with the directory
/// path for context, matching the previous inline behaviour.
pub(crate) fn walk_indexable(dir: &Path, extensions: &HashSet<String>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir).follow_links(false).into_iter() {
        let entry = entry.with_context(|| format!("error walking directory {}", dir.display()))?;

        if !entry.file_type().is_file() {
            continue;
        }

        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if !extensions.contains(ext) {
            continue;
        }

        files.push(entry.into_path());
    }
    Ok(files)
}
