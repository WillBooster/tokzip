//! Packs the trained assets the module embeds (`src/pack.rs`): every `priors/<language>.bin`
//! into the language's own nodes (`$OUT_DIR/<language>.priors`) and, once per model group, the
//! literal part with its untrained subtrees skipped (`$OUT_DIR/<group>.lit`); every
//! `dict/<language>.bin` coded by the codec itself (`$OUT_DIR/<language>.dict`). `lang.rs`
//! includes them through the generated `$OUT_DIR/assets.rs`. The codec sources are compiled
//! into this script by path, so the packed form always matches the code that unpacks it.

#[path = "src/error.rs"]
mod error;
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
        "src/languages.rs",
        "src/lz.rs",
        "src/pack.rs",
        "src/rc.rs",
        "src/varint.rs",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    // Every language's raw priors first (flat when stale), then the group literal parts —
    // the first language's of each group — so every language is packed against exactly the
    // models `lang::primed` rebuilds: its own nodes plus its group's literal part.
    let mut raws: Vec<Vec<u8>> = Vec::new();
    for (name, _) in LANGUAGES {
        let priors_path = root.join("priors").join(format!("{name}.bin"));
        println!("cargo:rerun-if-changed={}", priors_path.display());
        let mut raw = std::fs::read(&priors_path).expect("priors");
        // A model layout change leaves the committed priors at the old size until the trainer
        // (which needs this build) rewrites them: embed flat priors meanwhile.
        if raw.len() != lz::PRIORS_SIZE {
            println!(
                "cargo:warning={} has {} bytes, expected {}; embedding flat priors until retrained",
                priors_path.display(),
                raw.len(),
                lz::PRIORS_SIZE
            );
            raw = vec![0; 512];
            raw.resize(lz::PRIORS_SIZE, lz::PRIORS_DEFAULT);
        }
        raws.push(raw);
    }
    let mut group_first: Vec<Option<usize>> = vec![None; Group::ALL.len()];
    for (i, (name, group)) in LANGUAGES.iter().enumerate() {
        match group_first[*group as usize] {
            None => group_first[*group as usize] = Some(i),
            Some(first) => {
                // The trainer gives every language of a group the same literal part; priors
                // from before a group change differ until retrained, and the first language's
                // part is embedded meanwhile — for every language of the group, so the packed
                // dictionaries stay decodable with the embedded models.
                if pack::literal_part(&raws[i]) != pack::literal_part(&raws[first]) {
                    println!(
                        "cargo:warning={name} and {} are both {group:?} but have different literal priors; embedding {}'s until retrained",
                        LANGUAGES[first].0,
                        LANGUAGES[first].0
                    );
                    let (own, shared) = raws.split_at_mut(i.max(first));
                    let (source, target) = if first < i {
                        (&own[first], &mut shared[0])
                    } else {
                        (&shared[0], &mut own[i])
                    };
                    target[..512].copy_from_slice(&source[..512]);
                    target[512 + lz::LIT..].copy_from_slice(&source[512 + lz::LIT..]);
                }
            }
        }
    }
    let mut assets = String::from("[\n");
    for (i, (name, _)) in LANGUAGES.iter().enumerate() {
        let dict_path = root.join("dict").join(format!("{name}.bin"));
        println!("cargo:rerun-if-changed={}", dict_path.display());
        let suffix = std::fs::read(&dict_path).expect("dict");
        std::fs::write(
            out.join(format!("{name}.priors")),
            pack::language_part(&raws[i]),
        )
        .expect("write packed priors");
        std::fs::write(
            out.join(format!("{name}.dict")),
            pack::pack_dictionary(&suffix, &raws[i]),
        )
        .expect("write packed dictionary");
        assets.push_str(&format!(
            "    Assets {{ packed_suffix: include_bytes!({:?}), priors: include_bytes!({:?}) }},\n",
            out.join(format!("{name}.dict")),
            out.join(format!("{name}.priors"))
        ));
    }
    assets.push(']');
    let mut groups = String::from("[\n");
    for group in Group::ALL {
        let part =
            group_first[group as usize].map_or_else(Vec::new, |i| pack::literal_part(&raws[i]));
        let path = out.join(format!("{}.lit", group.name()));
        std::fs::write(&path, part).expect("write packed literal priors");
        groups.push_str(&format!("    include_bytes!({path:?}),\n"));
    }
    groups.push(']');
    std::fs::write(
        out.join("assets.rs"),
        format!(
            "static ASSETS: [Assets; {}] = {assets};\nstatic GROUP_PRIORS: [&[u8]; {}] = {groups};\n",
            LANGUAGES.len(),
            Group::ALL.len()
        ),
    )
    .expect("write assets.rs");
}
