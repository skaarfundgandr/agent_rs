use std::io::Write;
use std::path::Path;

use anyhow::Context;

use super::checkpoint::AgentCheckpoint;
use crate::agent::state::CURRENT_SCHEMA_VERSION;

/// Serialize `ckpt` to JSON and atomically write it to `path`.
///
/// Writes to a temporary file in the same directory, then renames it into
/// place so a crash mid-write cannot leave a truncated checkpoint file.
pub fn save_checkpoint(path: impl AsRef<Path>, ckpt: &AgentCheckpoint) -> anyhow::Result<()> {
    let path = path.as_ref();
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut tmp = tempfile::NamedTempFile::with_prefix_in(".checkpoint-", dir)
        .context("create temporary checkpoint file")?;
    let json = serde_json::to_string_pretty(ckpt).context("serialize checkpoint")?;
    tmp.write_all(json.as_bytes())
        .context("write checkpoint data")?;
    tmp.flush().context("flush checkpoint data")?;
    tmp.persist(path)
        .context("replace checkpoint file atomically")?;
    Ok(())
}

/// Read a JSON checkpoint from `path` and validate its schema version.
pub fn load_checkpoint(path: impl AsRef<Path>) -> anyhow::Result<AgentCheckpoint> {
    let json = std::fs::read_to_string(path).context("read checkpoint file")?;
    let ckpt: AgentCheckpoint = serde_json::from_str(&json).context("deserialize checkpoint")?;
    anyhow::ensure!(
        ckpt.metadata.schema_version == CURRENT_SCHEMA_VERSION,
        "unsupported schema_version {} (expected {})",
        ckpt.metadata.schema_version,
        CURRENT_SCHEMA_VERSION
    );
    Ok(ckpt)
}
