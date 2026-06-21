#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs_lib::domain::errors::DocumentError;
use agent_rs_lib::security::{
    SandboxConfig, find_containing_root, relative_display_path, validate_sandboxed_path,
};
use std::fs;
use std::path::Path;

#[test]
fn test_single_root() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();
    assert_eq!(config.len(), 1);
    assert_eq!(config.primary(), tmp.path());
}

#[test]
fn test_multi_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();
    assert_eq!(config.len(), 2);
    assert_eq!(config.primary(), tmp1.path());
}

#[test]
fn test_validate_within_root() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();
    fs::write(tmp.path().join("test.txt"), "hello").unwrap();

    let result = validate_sandboxed_path(&config, Path::new("test.txt")).unwrap();
    assert!(result.starts_with(&config.canonical_roots()[0]));
}

#[test]
fn test_validate_escape_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();

    let err = validate_sandboxed_path(&config, Path::new("../escaped.txt")).unwrap_err();
    assert!(matches!(err, DocumentError::SandboxEscape(_)));
}

#[test]
fn test_validate_multi_root_first_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

    fs::write(tmp1.path().join("file.txt"), "primary").unwrap();
    fs::write(tmp2.path().join("other.txt"), "secondary").unwrap();

    // File in primary resolves to primary
    let result = validate_sandboxed_path(&config, Path::new("file.txt")).unwrap();
    assert!(result.to_string_lossy().contains("file.txt"));

    // File in secondary resolves to secondary
    let result2 = validate_sandboxed_path(&config, Path::new("other.txt")).unwrap();
    assert!(result2.to_string_lossy().contains("other.txt"));
}

#[test]
fn test_validate_multi_root_escape_rejected() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

    let err = validate_sandboxed_path(&config, Path::new("../../etc/passwd")).unwrap_err();
    assert!(matches!(err, DocumentError::SandboxEscape(_)));
}

#[test]
fn test_find_containing_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

    // Create the file so canonicalize() works on Windows
    fs::write(tmp2.path().join("file.txt"), "content").unwrap();
    let path = tmp2.path().join("file.txt");
    let found = find_containing_root(&config, &path).unwrap();
    assert_eq!(found, tmp2.path());
}

#[test]
fn test_relative_display_path() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();
    let file = tmp.path().join("src/main.rs");

    let display = relative_display_path(&config, &file);
    assert_eq!(display, "src/main.rs");
}

#[test]
fn test_relative_display_path_multi_root() {
    let tmp1 = tempfile::tempdir().unwrap();
    let tmp2 = tempfile::tempdir().unwrap();
    let config =
        SandboxConfig::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();

    let file = tmp2.path().join("docs/readme.md");
    let display = relative_display_path(&config, &file);
    assert_eq!(display, "docs/readme.md");
}

#[test]
fn test_validate_nonexistent_path_under_primary() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();

    let result = validate_sandboxed_path(&config, Path::new("newdir/newfile.txt")).unwrap();
    assert!(
        result.to_string_lossy().contains("newdir/newfile.txt")
            || result.to_string_lossy().contains("newdir\\newfile.txt")
    );
}

#[test]
fn test_empty_roots_returns_error() {
    let err = SandboxConfig::new(vec![]).unwrap_err();
    assert!(matches!(err, DocumentError::Rag(_)));
    assert!(
        err.to_string()
            .contains("SandboxConfig requires at least one root")
    );
}

#[test]
fn test_sandbox_config_clone() {
    let tmp = tempfile::tempdir().unwrap();
    let config = SandboxConfig::single(tmp.path()).unwrap();
    let cloned = config.clone();
    assert_eq!(config.len(), cloned.len());
}
