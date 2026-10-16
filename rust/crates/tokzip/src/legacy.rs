use crate::lang;
use crate::lz::{decode_doc, LiteralModels, Models, Primed, Segment};
use crate::varint::read_varint;
use std::sync::{Arc, OnceLock};

struct PackedAsset {
    bytes: &'static [u8],
    checksum: u32,
}

struct Assets {
    dictionary: PackedAsset,
    priors: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/v1-assets.rs"));

pub const LANGUAGE_COUNT: usize = ASSETS.len();
static PRIMED: [OnceLock<Primed>; LANGUAGE_COUNT] = [const { OnceLock::new() }; LANGUAGE_COUNT];
static LITERALS: [OnceLock<Arc<LiteralModels>>; GROUP_PRIORS.len()] =
    [const { OnceLock::new() }; GROUP_PRIORS.len()];

pub fn primed(language: u8) -> &'static Primed {
    PRIMED[usize::from(language)].get_or_init(|| {
        let group = usize::from(GROUP_IDS[usize::from(language)]);
        let literals = LITERALS[group].get_or_init(|| {
            let base = GROUP_BASES[group] as u8;
            let models = lang::language_models(base);
            let raw = restore(
                &GROUP_PRIORS[group],
                &models,
                lang::packed_literal_priors(base),
            );
            LiteralModels::from_packed(&raw)
        });
        let current = lang::primed(language);
        let assets = &ASSETS[usize::from(language)];
        let dictionary = restore(&assets.dictionary, &current.models, &current.bytes);
        Primed::new(dictionary, literals.with_language(assets.priors))
    })
}

fn restore(packed: &PackedAsset, models: &Models, prefix: &[u8]) -> Vec<u8> {
    let (len, body) = read_varint(packed.bytes).expect("v1 asset length");
    let dictionary = Primed::new(prefix.to_vec(), models.clone());
    let segments = [Segment {
        end: len as usize,
        lang: 0,
    }];
    let mut raw = Vec::new();
    decode_doc(&|_| &dictionary, body, len as usize, &segments, &mut raw)
        .expect("v1 asset decodes");
    assert_eq!(crc32fast::hash(&raw), packed.checksum, "v1 asset checksum");
    raw
}
