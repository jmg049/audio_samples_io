//! AIFF / AIFF-C format support (read + write + streaming).
//!
//! See [`aiff_file`] for the format details. [`streaming`] provides an
//! incremental reader for large files, real-time playback, or non-file
//! `Read + Seek` sources; a streaming writer is not yet implemented — use
//! [`crate::read`]/[`crate::write`] or the [`AiffFile`] type directly for that.

pub mod aiff_file;
pub mod extended;
pub mod streaming;

pub use aiff_file::{AiffFile, AiffFileInfo, write_aiff};
pub use extended::{decode_extended, encode_extended};
pub use streaming::StreamedAiffFile;
