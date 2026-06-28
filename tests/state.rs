use std::collections::HashMap;

use rig_core::OneOrMany;
use rig_core::message::{Message, UserContent};

use agent_rs::agent::state::{
    AgentCheckpoint, CURRENT_SCHEMA_VERSION, CheckpointMetadata, load_checkpoint, save_checkpoint,
};

fn make_message(text: &str) -> Message {
    Message::User {
        content: OneOrMany::one(UserContent::text(text)),
    }
}

fn make_assistant_message(text: &str) -> Message {
    Message::assistant(text)
}

fn make_checkpoint(phase: &str, history: Vec<Message>, cycles: usize) -> AgentCheckpoint {
    AgentCheckpoint {
        history,
        compacted_context: None,
        phase: phase.to_string(),
        partial_results: HashMap::new(),
        metadata: CheckpointMetadata {
            created_at: "1234567890s".to_string(),
            cycles_completed: cycles,
            schema_version: CURRENT_SCHEMA_VERSION,
        },
    }
}

/// Compare two `Message` values for semantic equality.
///
/// rig-core's `Text.additional_params` uses `#[serde(flatten)]` which causes
/// `None` to deserialize as `Some(Object {})` after a JSON round-trip, making
/// direct `PartialEq` comparison unreliable on loaded data. This helper
/// serializes both sides to JSON and compares the output, which is the true
/// round-trip invariant.
fn messages_json_eq(a: &[Message], b: &[Message]) -> bool {
    let ja = serde_json::to_value(a).unwrap_or_default();
    let jb = serde_json::to_value(b).unwrap_or_default();
    ja == jb
}

#[test]
fn round_trip_basic() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let history = vec![
        make_message("hello"),
        make_assistant_message("hi there"),
        make_message("what's 2+2?"),
    ];
    let ckpt = make_checkpoint("research", history, 3);

    save_checkpoint(tmp.path(), &ckpt)?;
    let loaded = load_checkpoint(tmp.path())?;

    assert_eq!(loaded.phase, "research");
    assert_eq!(loaded.history.len(), 3);
    assert_eq!(loaded.metadata.cycles_completed, 3);
    assert_eq!(loaded.metadata.schema_version, CURRENT_SCHEMA_VERSION);
    // Compare via serialized JSON to avoid flatten-related PartialEq divergence
    assert!(messages_json_eq(&loaded.history, &ckpt.history));
    Ok(())
}

#[test]
fn schema_version_round_trip() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let ckpt = make_checkpoint("phase1", Vec::new(), 0);

    save_checkpoint(tmp.path(), &ckpt)?;
    let loaded = load_checkpoint(tmp.path())?;

    assert_eq!(loaded.metadata.schema_version, 1);
    Ok(())
}

#[test]
fn empty_checkpoint_round_trip() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let ckpt = AgentCheckpoint::default();

    save_checkpoint(tmp.path(), &ckpt)?;
    let loaded = load_checkpoint(tmp.path())?;

    assert!(loaded.history.is_empty());
    assert_eq!(loaded.phase, "");
    assert!(loaded.partial_results.is_empty());
    assert_eq!(loaded.metadata.schema_version, CURRENT_SCHEMA_VERSION);
    Ok(())
}

#[test]
fn large_history_round_trip() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let history: Vec<Message> = (0..1000)
        .map(|i| make_message(&format!("message {i}")))
        .collect();
    let ckpt = make_checkpoint("long-run", history, 50);

    save_checkpoint(tmp.path(), &ckpt)?;
    let loaded = load_checkpoint(tmp.path())?;

    assert_eq!(loaded.history.len(), 1000);
    assert!(messages_json_eq(&loaded.history, &ckpt.history));
    Ok(())
}

#[test]
#[allow(clippy::unwrap_used)]
fn load_rejects_unsupported_schema_version() -> anyhow::Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let bad_json = serde_json::json!({
        "history": [],
        "compacted_context": null,
        "phase": "test",
        "partial_results": {},
        "metadata": {
            "created_at": "0s",
            "cycles_completed": 0,
            "schema_version": 99
        }
    });
    std::fs::write(tmp.path(), serde_json::to_string(&bad_json)?)?;

    let result = load_checkpoint(tmp.path());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("unsupported schema_version"),
        "expected schema_version complaint, got: {err_msg}"
    );
    Ok(())
}
