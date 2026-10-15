//! The embedded languages: ids, names, and model groups. Compiled into both the library and
//! its build script (which packs the assets by this table), so it holds nothing else.

/// Languages whose literal statistics are pooled: the group shares one set of literal class
/// tables and literal-tree priors (the bulk of every model), trained on all of its documents.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// Latin-script prose.
    Prose,
    Japanese,
    Chinese,
    Code,
}

// The budgets and names drive the trainer and the build script; the library reads the table.
#[allow(dead_code)]
impl Group {
    pub const ALL: [Group; 4] = [Group::Prose, Group::Japanese, Group::Chinese, Group::Code];

    pub fn name(self) -> &'static str {
        match self {
            Group::Prose => "prose",
            Group::Japanese => "japanese",
            Group::Chinese => "chinese",
            Group::Code => "code",
        }
    }

    /// Budget of the dictionary part every language of the group shares (`dict/<group>.bin`,
    /// between the wrapper and the language's own suffix): programming languages have most of
    /// their frequent fragments — English words, Markdown, license headers, C-family syntax —
    /// in common, so one shared part replaces most of what each code dictionary repeated.
    pub fn shared_budget(self) -> usize {
        match self {
            Group::Code => 64 * 1024,
            _ => 0,
        }
    }

    /// Dictionary suffix budget (beyond the group's shared part). Ratio keeps improving with
    /// budget, but every language ships in the wasm module, so the budget is bounded by the
    /// module size target: a 256 KB prose dictionary codes the corpus 0.2 pp smaller but a
    /// production mix 0.2 pp larger (denser detection tables misdetect more) for +225 KB of
    /// module; with the shared part, 48 KB per code language costs 0.1 pp on the corpus against
    /// 64 KB and saves 65 KB of module, and 32 KB costs 0.2 pp.
    pub fn dictionary_budget(self) -> usize {
        match self {
            Group::Code => 48 * 1024,
            _ => 128 * 1024,
        }
    }
}

/// Language ids (the index) are frame-format identity, as are the dictionaries and priors
/// they name: changing any of them is a new format version.
pub const LANGUAGES: [(&str, Group); 21] = [
    ("text", Group::Prose),
    ("en-US", Group::Prose),
    ("ja-JP", Group::Japanese),
    ("html", Group::Code),
    ("css", Group::Code),
    ("javascript", Group::Code),
    ("typescript", Group::Code),
    ("c", Group::Code),
    ("cpp", Group::Code),
    ("csharp", Group::Code),
    ("dart", Group::Code),
    ("haskell", Group::Code),
    ("java", Group::Code),
    ("jsp", Group::Code),
    ("php", Group::Code),
    ("python", Group::Code),
    ("ruby", Group::Code),
    ("rust", Group::Code),
    ("zig", Group::Code),
    ("zh-CN", Group::Chinese),
    ("zh-TW", Group::Chinese),
];
pub const LANGUAGE_COUNT: usize = LANGUAGES.len();
