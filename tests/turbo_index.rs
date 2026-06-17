#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs_lib::rag::TurboIndex;
use tempfile::tempdir;

#[test]
fn turbo_index_add_search_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("idx.tvim");

    let mut idx = TurboIndex::new(8, 4).expect("new");
    let vecs: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    idx.add(&vecs, &[10, 20, 30]).expect("add");
    assert_eq!(idx.len(), 3);
    assert_eq!(idx.dim(), 8);

    let (_scores, ids) = idx.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], 10);

    idx.save(&path).expect("save");
    let loaded = TurboIndex::load(&path).expect("load");
    assert_eq!(loaded.len(), 3);
    let (_, ids2) = loaded.search(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
    assert_eq!(ids2[0], 20);
}
