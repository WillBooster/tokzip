//! Embeds the trained assets (`src/pack.rs`): every `priors/<language>.bin` (the language's
//! own model nodes) and `priors/<group>.bin` (the group's packed literal priors) as committed —
//! flat priors in their place while a model layout change leaves them stale — every
//! `dict/<language>.bin` and `dict/<group>.bin` coded by the codec itself
//! (`$OUT_DIR/<language>.dict`, `$OUT_DIR/<group>.dict`), and the detection table of every
//! language's trained dictionary (its group's shared part and its suffix, `$OUT_DIR/grams`).
//! `lang.rs` includes
//! them through the generated `$OUT_DIR/assets.rs`. The codec sources are compiled into this
//! script by path, so the packed form always matches the code that unpacks it.

#[path = "src/error.rs"]
mod error;
#[allow(dead_code)]
#[path = "src/grams.rs"]
mod grams;
#[allow(dead_code)]
#[path = "src/languages.rs"]
mod languages;
#[allow(dead_code)]
#[path = "src/lz.rs"]
mod lz;
#[path = "src/pack.rs"]
mod pack;
#[allow(dead_code)]
#[path = "src/rc.rs"]
mod rc;
#[allow(dead_code)]
#[path = "src/varint.rs"]
mod varint;

pub use error::DecodeError;
use languages::{Group, LANGUAGES};
use std::path::{Path, PathBuf};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    for file in [
        "src/error.rs",
        "src/grams.rs",
        "src/languages.rs",
        "src/lz.rs",
        "src/pack.rs",
        "src/rc.rs",
        "src/varint.rs",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    let read = |path: &Path| {
        println!("cargo:rerun-if-changed={}", path.display());
        std::fs::read(path)
    };
    // Group parts exist only for groups with a shared dictionary budget; everything else is
    // required, and a missing language dictionary would otherwise ship silently as none.
    let read_optional = |path: &Path| read(path).unwrap_or_default();
    let read_required =
        |path: &Path| read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // A model layout change leaves the committed priors stale until the trainer (which needs
    // this build) rewrites them: flat priors stand in meanwhile.
    let flat_raw = pack::flat_raw();
    let flat_language = pack::language_part(&flat_raw);
    let group_literals: Vec<Vec<u8>> = Group::ALL
        .iter()
        .map(|group| {
            let path = root.join("priors").join(format!("{}.bin", group.name()));
            let part = read_optional(&path);
            if lz::Models::try_from_packed(&flat_language, &part).is_some() {
                part
            } else {
                println!(
                    "cargo:warning={} does not match the model layout; embedding flat priors until retrained",
                    path.display()
                );
                pack::literal_part(&flat_raw)
            }
        })
        .collect();
    let language_parts: Vec<Vec<u8>> = LANGUAGES
        .iter()
        .map(|(name, _)| {
            let path = root.join("priors").join(format!("{name}.bin"));
            let part = read_optional(&path);
            if part.len() == lz::LIT {
                part
            } else {
                println!(
                    "cargo:warning={} has {} bytes, expected {}; embedding flat priors until retrained",
                    path.display(),
                    part.len(),
                    lz::LIT
                );
                flat_language.clone()
            }
        })
        .collect();
    let models: Vec<lz::Models> = LANGUAGES
        .iter()
        .zip(&language_parts)
        .map(|((_, group), part)| lz::Models::from_packed(part, &group_literals[*group as usize]))
        .collect();
    let mut shared_parts: Vec<Vec<u8>> = Vec::new();
    let mut group_priors = String::from("[\n");
    let mut group_dicts = String::from("[\n");
    for group in Group::ALL {
        let literal_path = out.join(format!("{}.lit", group.name()));
        std::fs::write(&literal_path, &group_literals[group as usize]).expect("write group priors");
        group_priors.push_str(&format!("    include_bytes!({literal_path:?}),\n"));
        let part = read_optional(&root.join("dict").join(format!("{}.bin", group.name())));
        // Coded with the models of the group's first language (`lang.rs` decodes it so).
        let packed = LANGUAGES
            .iter()
            .position(|(_, g)| *g == group)
            .filter(|_| !part.is_empty())
            .map_or_else(Vec::new, |first| {
                pack::pack_dictionary(&part, &models[first])
            });
        let dict_path = out.join(format!("{}.dict", group.name()));
        std::fs::write(&dict_path, packed).expect("write packed shared dictionary");
        group_dicts.push_str(&format!("    include_bytes!({dict_path:?}),\n"));
        shared_parts.push(part);
    }
    group_priors.push(']');
    group_dicts.push(']');
    let mut trained_dictionaries: Vec<Vec<u8>> = Vec::new();
    let mut assets = String::from("[\n");
    for (i, (name, group)) in LANGUAGES.iter().enumerate() {
        let suffix = read_required(&root.join("dict").join(format!("{name}.bin")));
        let priors_path = out.join(format!("{name}.priors"));
        std::fs::write(&priors_path, &language_parts[i]).expect("write packed priors");
        let dict_path = out.join(format!("{name}.dict"));
        std::fs::write(&dict_path, pack::pack_dictionary(&suffix, &models[i]))
            .expect("write packed dictionary");
        let mut trained = shared_parts[*group as usize].clone();
        trained.extend_from_slice(&suffix);
        trained_dictionaries.push(trained);
        assets.push_str(&format!(
            "    Assets {{ packed_suffix: include_bytes!({dict_path:?}), priors: include_bytes!({priors_path:?}) }},\n"
        ));
    }
    assets.push(']');
    let grams_path = out.join("grams");
    std::fs::write(&grams_path, grams::gram_table(&trained_dictionaries)).expect("write grams");
    std::fs::write(
        out.join("assets.rs"),
        format!(
            "static ASSETS: [Assets; {}] = {assets};\nstatic GROUP_PRIORS: [&[u8]; {}] = {group_priors};\nstatic GROUP_DICTS: [&[u8]; {}] = {group_dicts};\nstatic GRAM_TABLE: &[u8] = include_bytes!({grams_path:?});\n",
            LANGUAGES.len(),
            Group::ALL.len(),
            Group::ALL.len()
        ),
    )
    .expect("write assets.rs");
}
