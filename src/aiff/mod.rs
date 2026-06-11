//! AIFF / AIFF-C format support (read + write).
//!
//! See [`aiff_file`] for the format details. Streaming readers/writers are not
//! yet implemented for AIFF; use [`crate::read`]/[`crate::write`] or the
//! [`AiffFile`] type directly.

pub mod aiff_file;
pub mod extended;

pub use aiff_file::{AiffFile, AiffFileInfo, write_aiff};
pub use extended::{decode_extended, encode_extended};
