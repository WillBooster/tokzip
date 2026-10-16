use super::{languages::LANGUAGES, lz, pack, read_required};
use std::path::Path;

struct Snapshot {
    groups: Vec<Vec<u8>>,
    languages: Vec<Language>,
}

struct Language {
    name: String,
    group: u8,
    dictionary: Vec<u8>,
    priors: Vec<u8>,
}

pub fn emit(
    root: &Path,
    out: &Path,
    models: &[lz::Models],
    dictionaries: &[Vec<u8>],
    group_literals: &[Vec<u8>],
) {
    let snapshot = load(root);
    let bases: Vec<usize> = (0..snapshot.groups.len())
        .map(|g| {
            snapshot
                .languages
                .iter()
                .position(|l| usize::from(l.group) == g)
                .expect("v1 group member")
        })
        .collect();
    let group_assets = emit_groups(out, &snapshot, &bases, models, group_literals);
    let language_assets = emit_languages(out, &snapshot, models, dictionaries);
    let groups: Vec<u8> = snapshot.languages.iter().map(|l| l.group).collect();
    std::fs::write(out.join("v1-assets.rs"), format!(
        "static ASSETS: [Assets; {}] = {language_assets};\nstatic GROUP_PRIORS: [PackedAsset; {}] = {group_assets};\nstatic GROUP_IDS: [u8; {}] = {groups:?};\nstatic GROUP_BASES: [usize; {}] = {bases:?};\n",
        snapshot.languages.len(), snapshot.groups.len(), groups.len(), bases.len()
    )).expect("write v1 assets");
}

fn load(root: &Path) -> Snapshot {
    let manifest: serde_json::Value =
        serde_json::from_slice(&read_required(&root.join("dict/v1/manifest.json")))
            .expect("v1 manifest");
    for (file, checksum) in manifest["crc32"].as_object().expect("v1 checksums") {
        let expected = u32::from_str_radix(checksum.as_str().expect("v1 checksum string"), 16)
            .expect("v1 checksum");
        assert_eq!(
            crc32fast::hash(&read_required(&root.join(file))),
            expected,
            "frozen v1 asset changed: {file}"
        );
    }
    let groups: Vec<Vec<u8>> = manifest["groups"]
        .as_array()
        .expect("v1 groups")
        .iter()
        .map(|name| {
            let name = name.as_str().expect("v1 group name");
            let priors = read_required(&root.join(format!("priors/v1/{name}.bin")));
            assert!(
                lz::LiteralModels::try_from_packed(&priors).is_some(),
                "v1 literal model layout requires a retained decoder"
            );
            priors
        })
        .collect();
    let languages = manifest["languages"]
        .as_array()
        .expect("v1 languages")
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let name = entry["name"]
                .as_str()
                .expect("v1 language name")
                .to_string();
            assert_eq!(
                LANGUAGES[i].0, name,
                "released language ids must stay stable"
            );
            let group = entry["group"].as_u64().expect("v1 language group") as u8;
            let group_name = manifest["groups"][usize::from(group)]
                .as_str()
                .expect("v1 group");
            let mut dictionary = read_required(&root.join("dict/v1/wrapper.bin"));
            if group_name == "code" {
                dictionary.extend(read_required(&root.join("dict/v1/code.bin")));
            }
            dictionary.extend(read_required(&root.join(format!("dict/v1/{name}.bin"))));
            let priors = read_required(&root.join(format!("priors/v1/{name}.bin")));
            assert_eq!(
                priors.len(),
                lz::LIT,
                "v1 language model layout requires a retained decoder"
            );
            Language {
                name,
                group,
                dictionary,
                priors,
            }
        })
        .collect();
    Snapshot { groups, languages }
}

fn emit_groups(
    out: &Path,
    snapshot: &Snapshot,
    bases: &[usize],
    models: &[lz::Models],
    group_literals: &[Vec<u8>],
) -> String {
    let mut assets = String::from("[\n");
    for (group, raw) in snapshot.groups.iter().enumerate() {
        let base = bases[group];
        let current_group = LANGUAGES[base].1 as usize;
        let packed = pack::pack_dictionary(raw, &models[base], &group_literals[current_group]);
        let path = out.join(format!("v1-group-{group}.priors"));
        std::fs::write(&path, packed).expect("write v1 group priors");
        assets.push_str(&format!(
            "    PackedAsset {{ bytes: include_bytes!({path:?}), checksum: {} }},\n",
            crc32fast::hash(raw)
        ));
    }
    assets.push(']');
    assets
}

fn emit_languages(
    out: &Path,
    snapshot: &Snapshot,
    models: &[lz::Models],
    dictionaries: &[Vec<u8>],
) -> String {
    let mut assets = String::from("[\n");
    for (i, language) in snapshot.languages.iter().enumerate() {
        let name = &language.name;
        let dictionary = out.join(format!("v1-{name}.dict"));
        std::fs::write(
            &dictionary,
            pack::pack_dictionary(&language.dictionary, &models[i], &dictionaries[i]),
        )
        .expect("write v1 dictionary");
        let priors = out.join(format!("v1-{name}.priors"));
        std::fs::write(&priors, &language.priors).expect("write v1 language priors");
        assets.push_str(&format!("    Assets {{ dictionary: PackedAsset {{ bytes: include_bytes!({dictionary:?}), checksum: {} }}, priors: include_bytes!({priors:?}) }},\n", crc32fast::hash(&language.dictionary)));
    }
    assets.push(']');
    assets
}
