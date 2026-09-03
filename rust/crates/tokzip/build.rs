//! Packs the trained priors the module embeds: every `priors/<language>.bin` with its untrained
//! literal subtrees skipped (`src/pack.rs`), into `$OUT_DIR/<language>.priors`, which `lang.rs`
//! includes. The codec sources are compiled into this script by path, so the packed form always
//! matches the model layout that unpacks it.

#[path = "src/error.rs"]
mod error;
#[allow(dead_code)]
#[path = "src/lz.rs"]
mod lz;
#[path = "src/pack.rs"]
mod pack;
#[allow(dead_code)]
#[path = "src/rc.rs"]
mod rc;

pub use error::DecodeError;
use std::path::{Path, PathBuf};

fn main() {
    let priors_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../priors");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    for file in ["src/error.rs", "src/lz.rs", "src/pack.rs", "src/rc.rs"] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rerun-if-changed={}", priors_dir.display());
    for entry in std::fs::read_dir(&priors_dir).expect("priors/") {
        let path = entry.expect("priors entry").path();
        if path.extension().is_none_or(|ext| ext != "bin") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let mut raw = std::fs::read(&path).expect("priors");
        // A model layout change leaves the committed priors at the old size until the trainer
        // (which needs this build) rewrites them: embed flat priors meanwhile.
        if raw.len() != lz::PRIORS_SIZE {
            println!(
                "cargo:warning={} has {} bytes, expected {}; embedding flat priors until retrained",
                path.display(),
                raw.len(),
                lz::PRIORS_SIZE
            );
            raw = vec![0; 512];
            raw.resize(lz::PRIORS_SIZE, lz::PRIORS_DEFAULT);
        }
        let name = path
            .file_stem()
            .expect("file stem")
            .to_str()
            .expect("utf-8 name");
        std::fs::write(out.join(format!("{name}.priors")), pack::pack_priors(&raw))
            .expect("write packed priors");
    }
}
