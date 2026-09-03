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
    let mut assets = String::from("[\n");
    let mut literal_parts: Vec<Option<(String, Vec<u8>)>> = vec![None; Group::ALL.len()];
    for (name, group) in LANGUAGES {
        let priors_path = root.join("priors").join(format!("{name}.bin"));
        let dict_path = root.join("dict").join(format!("{name}.bin"));
        println!("cargo:rerun-if-changed={}", priors_path.display());
        println!("cargo:rerun-if-changed={}", dict_path.display());
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
        let suffix = std::fs::read(&dict_path).expect("dict");
        std::fs::write(out.join(format!("{name}.priors")), pack::language_part(&raw))
            .expect("write packed priors");
        std::fs::write(
            out.join(format!("{name}.dict")),
            pack::pack_dictionary(&suffix, &raw),
        )
        .expect("write packed dictionary");
        let literal = pack::literal_part(&raw);
        match &literal_parts[group as usize] {
            // The trainer gives every language of a group the same literal part; priors from
            // before a group change differ until retrained, and the first language's part is
            // embedded meanwhile.
            Some((first, part)) => {
                if *part != literal {
                    println!(
                        "cargo:warning={name} and {first} are both {group:?} but have different literal priors; embedding {first}'s until retrained"
                    );
                }
            }
            None => literal_parts[group as usize] = Some((name.to_string(), literal)),
        }
        assets.push_str(&format!(
            "    Assets {{ packed_suffix: include_bytes!({:?}), priors: include_bytes!({:?}) }},\n",
            out.join(format!("{name}.dict")),
            out.join(format!("{name}.priors"))
        ));
    }
    assets.push_str("]");
    let mut groups = String::from("[\n");
    for group in Group::ALL {
        let part = literal_parts[group as usize]
            .take()
            .map_or_else(Vec::new, |(_, part)| part);
        let path = out.join(format!("{}.lit", group.name()));
        std::fs::write(&path, part).expect("write packed literal priors");
        groups.push_str(&format!("    include_bytes!({path:?}),\n"));
    }
    groups.push_str("]");
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
