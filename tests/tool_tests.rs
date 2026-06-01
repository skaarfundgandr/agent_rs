use agent_rs_lib::agent::permission::PermissionPolicy;
use agent_rs_lib::agent::tools::directory::{ListDirectoryArgs, ListDirectoryTool};
use agent_rs_lib::agent::tools::document::{
    ReadDocumentArgs, ReadDocumentTool, WriteDocumentArgs, WriteDocumentTool,
};
use agent_rs_lib::agent::tools::glob::{GlobSearchArgs, GlobSearchTool};
use agent_rs_lib::agent::tools::search::{GrepSearchArgs, GrepSearchTool};
use agent_rs_lib::security::SandboxConfig;
use rig::tool::Tool;
use std::collections::HashSet;
use std::fs;

#[tokio::test]
async fn test_read_txt() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ReadDocumentTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        HashSet::from(["txt", "md", "pdf"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    let file_path = temp_dir.path().join("test_read.txt");
    fs::write(&file_path, "Hello World").unwrap();

    let args = ReadDocumentArgs {
        path: "test_read.txt".to_string(),
    };
    let result = tool.call(args).await.unwrap();

    assert_eq!(result, "Hello World");
}

#[tokio::test]
#[ignore]
async fn test_read_pdf() {
    let tool = ReadDocumentTool::new(
        SandboxConfig::single("./").unwrap(),
        HashSet::from(["txt", "md", "pdf"].map(String::from)),
        PermissionPolicy::AllowAll,
    );
    let path = "Stellaron Architecture Overview.pdf";

    let args = ReadDocumentArgs {
        path: path.to_string(),
    };
    let result = tool.call(args).await.unwrap();

    assert!(!result.is_empty());
    assert!(result.contains("Stellaron"));
}

#[tokio::test]
async fn test_write_document() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = WriteDocumentTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        HashSet::from(["txt", "md"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    // Test overwrite
    let args = WriteDocumentArgs {
        path: "test_write.txt".to_string(),
        content: "First line\n".to_string(),
        append: None,
    };
    tool.call(args).await.unwrap();

    // Test append
    let args = WriteDocumentArgs {
        path: "test_write.txt".to_string(),
        content: "Second line".to_string(),
        append: Some(true),
    };
    tool.call(args).await.unwrap();

    let content = fs::read_to_string(temp_dir.path().join("test_write.txt")).unwrap();
    assert_eq!(content, "First line\nSecond line");
}

#[tokio::test]
async fn test_sandbox_escape_read() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ReadDocumentTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        HashSet::from(["txt", "md", "pdf"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    let args = ReadDocumentArgs {
        path: "../escaped.txt".to_string(),
    };
    let err = tool
        .call(args)
        .await
        .expect_err("should reject path traversal");

    assert!(
        matches!(
            err,
            agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
        ),
        "Expected SandboxEscape error, got {:?}",
        err
    );
}

#[tokio::test]
async fn test_sandbox_escape_write() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = WriteDocumentTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        HashSet::from(["txt", "md"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    let args = WriteDocumentArgs {
        path: "../../escaped_write.txt".to_string(),
        content: "malicious".to_string(),
        append: None,
    };
    let err = tool
        .call(args)
        .await
        .expect_err("should reject path traversal");

    assert!(
        matches!(
            err,
            agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
        ),
        "Expected SandboxEscape error, got {:?}",
        err
    );
}

#[tokio::test]
async fn test_list_directory() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Create some structure
    fs::create_dir(temp_dir.path().join("sub_dir")).unwrap();
    fs::write(temp_dir.path().join("file.txt"), "hello").unwrap();
    fs::write(temp_dir.path().join("another.md"), "world").unwrap();

    let tool = ListDirectoryTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        PermissionPolicy::AllowAll,
    );

    let args = ListDirectoryArgs { path: None };
    let result = tool.call(args).await.unwrap();

    assert!(result.contains("[DIR]  sub_dir"));
    assert!(result.contains("[FILE] another.md (5 bytes)"));
    assert!(result.contains("[FILE] file.txt (5 bytes)"));
}

#[tokio::test]
async fn test_list_directory_sandbox_escape() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ListDirectoryTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        PermissionPolicy::AllowAll,
    );

    let args = ListDirectoryArgs {
        path: Some("../".to_string()),
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}

#[tokio::test]
async fn test_grep_search() {
    let temp_dir = tempfile::tempdir().unwrap();

    let sub_dir = temp_dir.path().join("sub");
    fs::create_dir(&sub_dir).unwrap();

    fs::write(
        temp_dir.path().join("file1.txt"),
        "Hello World\nRust is awesome\nhello world",
    )
    .unwrap();
    fs::write(sub_dir.join("file2.md"), "Another world here").unwrap();

    let allowed = HashSet::from(["txt", "md"].map(String::from));
    let tool = GrepSearchTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        allowed,
        PermissionPolicy::AllowAll,
    );

    // Test case insensitive search
    let args = GrepSearchArgs {
        query: "world".to_string(),
        path: None,
        case_sensitive: None,
    };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("file1.txt:1: Hello World"));
    assert!(result.contains("file1.txt:3: hello world"));
    assert!(result.contains("file2.md:1: Another world here"));

    // Test case sensitive search
    let args_sensitive = GrepSearchArgs {
        query: "Hello".to_string(),
        path: None,
        case_sensitive: Some(true),
    };
    let result_sensitive = tool.call(args_sensitive).await.unwrap();
    assert!(result_sensitive.contains("file1.txt:1: Hello World"));
    assert!(!result_sensitive.contains("file1.txt:3: hello world"));
}

#[tokio::test]
async fn test_grep_search_sandbox_escape() {
    let temp_dir = tempfile::tempdir().unwrap();
    let allowed = HashSet::from(["txt", "md"].map(String::from));
    let tool = GrepSearchTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        allowed,
        PermissionPolicy::AllowAll,
    );

    let args = GrepSearchArgs {
        query: "test".to_string(),
        path: Some("../../".to_string()),
        case_sensitive: None,
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}

#[tokio::test]
async fn test_glob_search() {
    let temp_dir = tempfile::tempdir().unwrap();

    let sub_dir = temp_dir.path().join("src");
    fs::create_dir(&sub_dir).unwrap();

    fs::write(temp_dir.path().join("file1.txt"), "hello").unwrap();
    fs::write(sub_dir.join("file2.rs"), "fn main() {}").unwrap();
    fs::write(sub_dir.join("file3.txt"), "world").unwrap();

    let tool = GlobSearchTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        PermissionPolicy::AllowAll,
    );

    // Match files recursively
    let args = GlobSearchArgs {
        pattern: "**/*.txt".to_string(),
        directory: None,
    };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("file1.txt"));
    assert!(result.contains("src/file3.txt"));
    assert!(!result.contains("file2.rs"));

    // Match specific folder
    let args2 = GlobSearchArgs {
        pattern: "src/*.rs".to_string(),
        directory: None,
    };
    let result2 = tool.call(args2).await.unwrap();
    assert!(result2.contains("src/file2.rs"));
    assert!(!result2.contains("file1.txt"));

    // Match specific folder via directory argument
    let args3 = GlobSearchArgs {
        pattern: "*.rs".to_string(),
        directory: Some("src".to_string()),
    };
    let result3 = tool.call(args3).await.unwrap();
    assert!(result3.contains("file2.rs"));
    assert!(!result3.contains("file1.txt"));
}

#[tokio::test]
async fn test_glob_search_sandbox_escape() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = GlobSearchTool::new(
        SandboxConfig::single(temp_dir.path()).unwrap(),
        PermissionPolicy::AllowAll,
    );

    let args = GlobSearchArgs {
        pattern: "../**/*".to_string(),
        directory: None,
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}

// ============================================================================
// Multi-root sandbox tests (integration-level, not covered by unit tests)
// ============================================================================

#[tokio::test]
async fn test_multi_root_read_from_secondary() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    // File lives in secondary root
    fs::write(secondary.path().join("shared.txt"), "from secondary").unwrap();

    let tool = ReadDocumentTool::new(
        sandbox,
        HashSet::from(["txt"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    let args = ReadDocumentArgs {
        path: "shared.txt".to_string(),
    };
    let result = tool.call(args).await.unwrap();
    assert_eq!(result, "from secondary");
}

#[tokio::test]
async fn test_multi_root_write_to_primary() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let tool = WriteDocumentTool::new(
        sandbox,
        HashSet::from(["txt"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    // New file written to primary root (default)
    let args = WriteDocumentArgs {
        path: "new_file.txt".to_string(),
        content: "written to primary".to_string(),
        append: None,
    };
    tool.call(args).await.unwrap();

    let content = fs::read_to_string(primary.path().join("new_file.txt")).unwrap();
    assert_eq!(content, "written to primary");

    // Verify not in secondary
    assert!(!secondary.path().join("new_file.txt").exists());
}

#[tokio::test]
async fn test_multi_root_escape_rejected() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let tool = ReadDocumentTool::new(
        sandbox,
        HashSet::from(["txt"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    // Try to escape both roots
    let args = ReadDocumentArgs {
        path: "../../etc/passwd".to_string(),
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}

#[tokio::test]
async fn test_multi_root_list_directory() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    fs::write(primary.path().join("primary.txt"), "p").unwrap();
    fs::write(secondary.path().join("secondary.txt"), "s").unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let tool = ListDirectoryTool::new(sandbox, PermissionPolicy::AllowAll);

    // List primary
    let args = ListDirectoryArgs { path: None };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("primary.txt"));
    assert!(!result.contains("secondary.txt"));

    // List secondary by path
    let args2 = ListDirectoryArgs {
        path: Some(secondary.path().to_string_lossy().to_string()),
    };
    let result2 = tool.call(args2).await.unwrap();
    assert!(result2.contains("secondary.txt"));
    assert!(!result2.contains("primary.txt"));
}

#[tokio::test]
async fn test_multi_root_grep() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    fs::write(primary.path().join("main.txt"), "Hello from primary").unwrap();
    fs::write(secondary.path().join("docs.txt"), "Hello from secondary").unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let allowed = HashSet::from(["txt"].map(String::from));
    let tool = GrepSearchTool::new(sandbox, allowed, PermissionPolicy::AllowAll);

    let args = GrepSearchArgs {
        query: "Hello".to_string(),
        path: None,
        case_sensitive: None,
    };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("Hello from primary"));
    // Grep searches only from the resolved path (primary by default),
    // not across all roots. This is intentional — path-based resolution
    // means the user chooses which root to search.
}

#[tokio::test]
async fn test_multi_root_grep_from_secondary() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    fs::write(primary.path().join("main.txt"), "Hello from primary").unwrap();
    fs::write(secondary.path().join("docs.txt"), "Hello from secondary").unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let allowed = HashSet::from(["txt"].map(String::from));
    let tool = GrepSearchTool::new(sandbox, allowed, PermissionPolicy::AllowAll);

    // Search explicitly in secondary root
    let args = GrepSearchArgs {
        query: "Hello".to_string(),
        path: Some(secondary.path().to_string_lossy().to_string()),
        case_sensitive: None,
    };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("Hello from secondary"));
    assert!(!result.contains("Hello from primary"));
}

#[tokio::test]
async fn test_multi_root_glob_across_roots() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    fs::write(primary.path().join("a.txt"), "hello").unwrap();
    fs::write(secondary.path().join("b.txt"), "world").unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let tool = GlobSearchTool::new(sandbox, PermissionPolicy::AllowAll);

    let args = GlobSearchArgs {
        pattern: "**/*.txt".to_string(),
        directory: None,
    };
    let result = tool.call(args).await.unwrap();
    // Should find files from both roots
    assert!(result.contains("a.txt"));
    assert!(result.contains("b.txt"));
}

#[tokio::test]
async fn test_multi_root_glob_escape_rejected() {
    let primary = tempfile::tempdir().unwrap();
    let secondary = tempfile::tempdir().unwrap();

    let sandbox = SandboxConfig::new(vec![
        primary.path().to_path_buf(),
        secondary.path().to_path_buf(),
    ])
    .unwrap();

    let tool = GlobSearchTool::new(sandbox, PermissionPolicy::AllowAll);

    let args = GlobSearchArgs {
        pattern: "../**/*".to_string(),
        directory: None,
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}

#[tokio::test]
async fn test_sandbox_config_try_from() {
    let tmp = tempfile::tempdir().unwrap();
    let sandbox = SandboxConfig::try_from(tmp.path()).unwrap();
    assert_eq!(sandbox.len(), 1);

    let tool = ReadDocumentTool::new(
        sandbox,
        HashSet::from(["txt"].map(String::from)),
        PermissionPolicy::AllowAll,
    );

    fs::write(tmp.path().join("test.txt"), "content").unwrap();
    let args = ReadDocumentArgs {
        path: "test.txt".to_string(),
    };
    let result = tool.call(args).await.unwrap();
    assert_eq!(result, "content");
}

#[tokio::test]
async fn test_sandbox_config_default() {
    let sandbox = SandboxConfig::default();
    assert_eq!(sandbox.len(), 1);
    assert_eq!(sandbox.primary(), std::path::Path::new("."));
}
