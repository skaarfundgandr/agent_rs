use agent_rs_lib::security::SandboxConfig;
use std::fs;
use std::path::Path;

use agent_rs_lib::security::validate_sandboxed_path;

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
    assert!(matches!(
        err,
        agent_rs_lib::domain::errors::DocumentError::SandboxEscape(_)
    ));
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
