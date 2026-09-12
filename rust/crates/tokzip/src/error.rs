//! The decoder's error type, shared with the asset packer (`build.rs` compiles this file too).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    ChecksumMismatch,
    Corrupt,
    /// The declared content length exceeds the caller's limit, or the output cannot be allocated.
    TooLarge,
}

impl DecodeError {
    pub fn code(self) -> u32 {
        match self {
            Self::Truncated => 1,
            Self::BadMagic => 2,
            Self::UnsupportedVersion => 3,
            Self::ChecksumMismatch => 4,
            Self::Corrupt => 5,
            Self::TooLarge => 6,
        }
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "frame truncated"),
            Self::BadMagic => write!(f, "bad magic byte"),
            Self::UnsupportedVersion => write!(f, "unsupported format version"),
            Self::ChecksumMismatch => write!(f, "content checksum mismatch"),
            Self::Corrupt => write!(f, "corrupt compressed body"),
            Self::TooLarge => write!(f, "content too large for the length limit or memory"),
        }
    }
}

impl std::error::Error for DecodeError {}
