//! Asset packing for the module build (`build.rs` compiles this file together with the codec;
//! the library only unpacks, in `lz::Models::from_packed` and `lang::primed`).
//!
//! A language's raw priors (`priors/<language>.bin`, `PRIORS_SIZE` bytes) split into its own
//! part — the nodes before `LIT`, stored verbatim — and its group's literal part, identical for
//! every language of the group: the two class tables, then every 256-node tree from `LIT` on
//! (the plain literal trees, then the two matched-literal trees) as a depth-first walk with one
//! flag bit per visited node — 1: the node's value follows in the value stream and its children
//! are walked; 0: the node and its subtree stay at `PRIORS_DEFAULT` and are skipped. Trained
//! priors leave most literal nodes at the default (contexts the training never reached, and
//! nodes pruned for saving too little), and a default node's subtree is default too.
//!
//! A dictionary suffix is embedded coded by the codec itself: its length as a varint, then the
//! range-coded body of the suffix coded as one segment with the language's models and no
//! dictionary.

use crate::lz::{encode_doc, Models, Primed, Segment, LIT, PRIORS_DEFAULT, PRIORS_SIZE};
use crate::varint::push_varint;

/// The language's own model nodes of a raw serialized model.
pub fn language_part(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len(), PRIORS_SIZE, "priors size mismatch");
    raw[512..512 + LIT].to_vec()
}

/// The group's literal part of a raw serialized model, packed.
pub fn literal_part(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len(), PRIORS_SIZE, "priors size mismatch");
    let mut out = raw[..512].to_vec();
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

/// A dictionary suffix coded with the models of `raw` and no dictionary.
pub fn pack_dictionary(suffix: &[u8], raw: &[u8]) -> Vec<u8> {
    let primed = Primed::new(Vec::new(), Models::from_raw_priors(raw));
    let lookup = |_: u8| &primed;
    let segments = [Segment {
        end: suffix.len(),
        lang: 0,
    }];
    let mut out = Vec::new();
    push_varint(&mut out, suffix.len() as u64);
    out.extend_from_slice(&encode_doc(&lookup, suffix, &segments));
    out
}
