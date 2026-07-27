#[cfg(feature = "rag-load-dynamic")]
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");

    #[cfg(feature = "rag-load-dynamic")]
    {
        if let Some(path) = find_ort_dylib() {
            println!("cargo:rustc-env=ORT_DYLIB_BUNDLED_PATH={}", path.display());
        }
    }
}

#[cfg(feature = "rag-load-dynamic")]
fn find_ort_dylib() -> Option<PathBuf> {
    let lib_name = if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };

    if let Ok(dir) = std::env::var("ORT_LIB_LOCATION") {
        let candidate = PathBuf::from(&dir).join(lib_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let dfbin = PathBuf::from(home)
        .join(".cache")
        .join("ort.pyke.io")
        .join("dfbin");
    for triple in std::fs::read_dir(&dfbin).ok()?.flatten() {
        let Ok(hashes) = std::fs::read_dir(triple.path()) else {
            continue;
        };
        for hash_dir in hashes.flatten() {
            let candidate = hash_dir
                .path()
                .join("onnxruntime")
                .join("lib")
                .join(lib_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}
