use agent_rs_lib::agent::permission::PermissionPolicy;
use agent_rs_lib::agent::tools::directory::{ListDirectoryArgs, ListDirectoryTool};
use agent_rs_lib::agent::tools::document::{
    ReadDocumentArgs, ReadDocumentTool, WriteDocumentArgs, WriteDocumentTool,
};
use agent_rs_lib::agent::tools::glob::{GlobSearchArgs, GlobSearchTool};
use agent_rs_lib::agent::tools::search::{GrepSearchArgs, GrepSearchTool};
use rig::tool::Tool;
use std::collections::HashSet;
use std::fs;

#[tokio::test]
async fn test_read_txt() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ReadDocumentTool::new(
        temp_dir.path(),
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
        "./",
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
        temp_dir.path(),
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
        temp_dir.path(),
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
        temp_dir.path(),
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

    let tool = ListDirectoryTool::new(temp_dir.path(), PermissionPolicy::AllowAll);

    let args = ListDirectoryArgs { path: None };
    let result = tool.call(args).await.unwrap();

    assert!(result.contains("[DIR]  sub_dir"));
    assert!(result.contains("[FILE] another.md (5 bytes)"));
    assert!(result.contains("[FILE] file.txt (5 bytes)"));
}

#[tokio::test]
async fn test_list_directory_sandbox_escape() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = ListDirectoryTool::new(temp_dir.path(), PermissionPolicy::AllowAll);

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
    let tool = GrepSearchTool::new(temp_dir.path(), allowed, PermissionPolicy::AllowAll);

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
    let tool = GrepSearchTool::new(temp_dir.path(), allowed, PermissionPolicy::AllowAll);

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

    let tool = GlobSearchTool::new(temp_dir.path(), PermissionPolicy::AllowAll);

    // Match files recursively
    let args = GlobSearchArgs {
        pattern: "**/*.txt".to_string(),
    };
    let result = tool.call(args).await.unwrap();
    assert!(result.contains("file1.txt"));
    assert!(result.contains("src/file3.txt"));
    assert!(!result.contains("file2.rs"));

    // Match specific folder
    let args2 = GlobSearchArgs {
        pattern: "src/*.rs".to_string(),
    };
    let result2 = tool.call(args2).await.unwrap();
    assert!(result2.contains("src/file2.rs"));
    assert!(!result2.contains("file1.txt"));
}

#[tokio::test]
async fn test_glob_search_sandbox_escape() {
    let temp_dir = tempfile::tempdir().unwrap();
    let tool = GlobSearchTool::new(temp_dir.path(), PermissionPolicy::AllowAll);

    let args = GlobSearchArgs {
        pattern: "../**/*".to_string(),
    };
    let err = tool.call(args).await.expect_err("should reject escape");

    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
}
