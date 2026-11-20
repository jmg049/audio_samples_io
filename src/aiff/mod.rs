//! AIFF and AIFF-C files.
//!
//! [`AiffFile`] reads a whole file and [`write_aiff`] writes one; [`aiff_file`]
//! describes the format and the encodings read. [`StreamedAiffFile`] reads a file, or
//! any `Read + Seek` source, incrementally. There is no streaming AIFF writer: write
//! whole files with [`write`](crate::write) or [`write_aiff`], or stream to WAV or FLAC.

pub mod aiff_file;
pub mod extended;
pub mod streaming;

pub use aiff_file::{AiffFile, AiffFileInfo, write_aiff};
pub use extended::{decode_extended, encode_extended};
pub use streaming::StreamedAiffFile;
