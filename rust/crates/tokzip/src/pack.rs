//! Priors packing for the module build (`build.rs` compiles this file together with the codec;
//! the library only unpacks, in `Models::from_priors`).
//!
//! Trained priors leave most literal-tree nodes untrained (contexts the training documents never
//! reached), and an untrained node's whole subtree is untrained too, so each 256-node literal
//! tree is stored as a depth-first walk: one flag bit per visited node — 1: the node's value
//! follows in the value stream and its children are walked; 0: the node and its subtree stay at
//! `PRIORS_DEFAULT` and are skipped. Every other node is stored verbatim.
//!
//! Packed layout: the two class tables (512 bytes), the values of the nodes before the literal
//! trees, then per literal tree its flag bits (packed LSB-first, padded to a byte) followed by
//! its values.

use crate::lz::{LIT, PRIORS_DEFAULT, PRIORS_SIZE};

/// Packs a raw serialized model (`PRIORS_SIZE` bytes).
pub fn pack_priors(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len(), PRIORS_SIZE, "priors size mismatch");
    let mut out = raw[..512 + LIT].to_vec();
    for tree in raw[512 + LIT..].as_chunks::<256>().0 {
        let mut flags = Vec::new();
        let mut values = Vec::new();
        walk(tree, 1, &mut flags, &mut values);
        out.extend(flags.chunks(8).map(|bits| {
            bits.iter()
                .enumerate()
                .fold(0u8, |byte, (i, &bit)| byte | (bit << i))
        }));
        out.extend_from_slice(&values);
    }
    out
}

fn walk(tree: &[u8], node: usize, flags: &mut Vec<u8>, values: &mut Vec<u8>) {
    if subtree_is_default(tree, node) {
        flags.push(0);
        return;
    }
    flags.push(1);
    values.push(tree[node]);
    if node < 128 {
        walk(tree, 2 * node, flags, values);
        walk(tree, 2 * node + 1, flags, values);
    }
}

fn subtree_is_default(tree: &[u8], node: usize) -> bool {
    node >= 256
        || (tree[node] == PRIORS_DEFAULT
            && subtree_is_default(tree, 2 * node)
            && subtree_is_default(tree, 2 * node + 1))
}
