#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use agent_rs_lib::domain::errors::DocumentError;
use agent_rs_lib::security::{
    SandboxConfig, SharedSandbox, find_containing_root, relative_display_path,
    validate_sandboxed_path,
};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

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

#[tokio::test]
async fn test_symlink_glob_rejects_targets_outside_sandbox() {
    use agent_rs_lib::agent::permission::PermissionPolicy;
    use agent_rs_lib::agent::tools::glob::{GlobSearchArgs, GlobSearchTool};
    use rig_core::tool::Tool;

    let sandbox_dir = tempfile::tempdir().unwrap();
    let outside_dir = tempfile::tempdir().unwrap();

    fs::write(outside_dir.path().join("secret.txt"), "classified").unwrap();
    fs::write(sandbox_dir.path().join("safe.txt"), "public").unwrap();

    let symlink_path = sandbox_dir.path().join("link_outside");
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside_dir.path().join("secret.txt"), &symlink_path).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(outside_dir.path().join("secret.txt"), &symlink_path)
        .unwrap();

    let sandbox = Arc::new(SharedSandbox::from(
        SandboxConfig::single(sandbox_dir.path()).unwrap(),
    ));
    let tool = GlobSearchTool::new(Arc::clone(&sandbox), PermissionPolicy::AllowAll);

    let result = tool
        .call(GlobSearchArgs {
            pattern: "**/*".to_string(),
            directory: None,
        })
        .await
        .unwrap();

    assert!(
        !result.contains("secret.txt"),
        "Glob should not return files outside sandbox via symlink, got: {result}"
    );
    assert!(
        result.contains("safe.txt"),
        "Glob should return files inside sandbox, got: {result}"
    );
}

#[tokio::test]
async fn test_symlink_grep_rejects_targets_outside_sandbox() {
    use agent_rs_lib::agent::permission::PermissionPolicy;
    use agent_rs_lib::agent::tools::search::{GrepSearchArgs, GrepSearchTool};
    use rig_core::tool::Tool;

    let sandbox_dir = tempfile::tempdir().unwrap();
    let outside_dir = tempfile::tempdir().unwrap();

    fs::write(outside_dir.path().join("secret.txt"), "classified content").unwrap();
    fs::write(sandbox_dir.path().join("safe.txt"), "public content").unwrap();

    let symlink_path = sandbox_dir.path().join("link_outside");
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside_dir.path().join("secret.txt"), &symlink_path).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(outside_dir.path().join("secret.txt"), &symlink_path)
        .unwrap();

    let sandbox = Arc::new(SharedSandbox::from(
        SandboxConfig::single(sandbox_dir.path()).unwrap(),
    ));
    let tool = GrepSearchTool::new(
        Arc::clone(&sandbox),
        HashSet::from(["txt".to_string()]),
        PermissionPolicy::AllowAll,
    );

    let result = tool
        .call(GrepSearchArgs {
            query: "content".to_string(),
            path: None,
            case_sensitive: None,
        })
        .await
        .unwrap();

    let canonical_outside = outside_dir.path().canonicalize().unwrap();
    for line in result.lines() {
        if let Some(path_part) = line.split(':').next() {
            let path = std::path::Path::new(path_part);
            if path.is_absolute() {
                assert!(
                    !path.starts_with(&canonical_outside),
                    "Grep returned path outside sandbox: {path_part}"
                );
            }
        }
    }
}
