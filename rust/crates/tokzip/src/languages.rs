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

    /// Dictionary suffix budget. Ratio keeps improving with budget, but every language ships
    /// in the wasm module, so the budget is bounded by the module size target: halving a prose
    /// dictionary costs about 1 pp on its documents, halving a code dictionary about 0.4 pp,
    /// and prompts and answers are mostly prose.
    pub fn dictionary_budget(self) -> usize {
        match self {
            Group::Code => 64 * 1024,
            _ => 128 * 1024,
        }
    }
}

/// Language ids (the index) are frame-format identity, as are the dictionaries and priors
/// they name.
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
