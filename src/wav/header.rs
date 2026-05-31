//! Standalone WAV header construction and size pre-calculation.
//!
//! These helpers let callers build a correct finite-length WAV header up front — without
//! holding the sample data — and compute the exact final file size from a frame count. Useful
//! for multipart/streaming uploads and for anticipating output size before writing.
//!
//! The bytes produced here are byte-for-byte identical to the header [`crate::wav::wav_file`]
//! writes, so a precomputed size always matches the file that is ultimately written.

use std::io::Write;

use crate::error::AudioIOResult;
use crate::types::ValidatedSampleType;
use crate::wav::FormatCode;

/// Standard WAVE extensible sub-format GUID tail (`...-0000-0010-8000-00AA00389B71`).
const SUBFORMAT_GUID_TAIL: [u8; 12] = [0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71];

/// True if the given format requires a `WAVE_FORMAT_EXTENSIBLE` fmt chunk.
///
/// Matches the write path: extensible is used for more than two channels, or for sample
/// formats whose container size is non-standard for a base fmt chunk (24-bit, 64-bit float).
pub const fn needs_extensible(channels: u16, sample_type: ValidatedSampleType) -> bool {
    channels > 2 || matches!(sample_type, ValidatedSampleType::I24 | ValidatedSampleType::F64)
}

/// Length in bytes of the fmt chunk *body* (16 base, or 40 extensible).
const fn fmt_body_len(channels: u16, sample_type: ValidatedSampleType) -> usize {
    if needs_extensible(channels, sample_type) {
        40
    } else {
        16
    }
}

/// Byte length of the complete WAV header that precedes the audio data
/// (RIFF/WAVE + fmt chunk + the 8-byte `data` chunk header).
pub const fn wav_header_len(channels: u16, sample_type: ValidatedSampleType) -> usize {
    // RIFF id+size+WAVE (12) + fmt header (8) + fmt body + data header (8)
    12 + 8 + fmt_body_len(channels, sample_type) + 8
}

/// Byte length of the raw audio payload for `total_frames` frames (excludes word-alignment pad).
pub const fn wav_data_len(channels: u16, sample_type: ValidatedSampleType, total_frames: usize) -> usize {
    total_frames * channels as usize * sample_type.bytes_per_sample().get()
}

/// Exact total size of the finished WAV file in bytes, including header and the word-alignment
/// pad byte appended after an odd-length data chunk.
pub const fn wav_file_len(channels: u16, sample_type: ValidatedSampleType, total_frames: usize) -> usize {
    let data = wav_data_len(channels, sample_type, total_frames);
    let padded = data + (data & 1);
    wav_header_len(channels, sample_type) + padded
}

/// Windows speaker-position channel mask for a given channel count (mono → FRONT_CENTER).
const fn channel_mask(channels: u16) -> u32 {
    match channels {
        1 => 0x4,
        2 => 0x3,
        3 => 0x7,
        4 => 0x33,
        5 => 0x37,
        6 => 0x3F,
        7 => 0x13F,
        8 => 0x63F,
        _ => {
            if channels < 32 {
                (1u32 << channels) - 1
            } else {
                0xFFFFFFFF
            }
        },
    }
}

const fn format_code(sample_type: ValidatedSampleType) -> FormatCode {
    match sample_type {
        ValidatedSampleType::F32 | ValidatedSampleType::F64 => FormatCode::IeeeFloat,
        _ => FormatCode::Pcm,
    }
}

fn write_fmt_chunk<W: Write>(
    w: &mut W,
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
) -> AudioIOResult<()> {
    let bits = sample_type.bits_per_sample().get() as u16;
    let bytes = sample_type.bytes_per_sample().get() as u16;
    let block_align = channels * bytes;
    let byte_rate = sample_rate * block_align as u32;
    let fc = format_code(sample_type);

    if needs_extensible(channels, sample_type) {
        w.write_all(b"fmt ")?;
        w.write_all(&40u32.to_le_bytes())?;
        w.write_all(&FormatCode::Extensible.as_u16().to_le_bytes())?;
        w.write_all(&channels.to_le_bytes())?;
        w.write_all(&sample_rate.to_le_bytes())?;
        w.write_all(&byte_rate.to_le_bytes())?;
        w.write_all(&block_align.to_le_bytes())?;
        w.write_all(&bits.to_le_bytes())?;
        w.write_all(&22u16.to_le_bytes())?; // cbSize
        w.write_all(&bits.to_le_bytes())?; // wValidBitsPerSample
        w.write_all(&channel_mask(channels).to_le_bytes())?;
        // SubFormat GUID: Data1 = format code as u32 LE, then the standard tail.
        w.write_all(&u32::from(fc.as_u16()).to_le_bytes())?;
        w.write_all(&SUBFORMAT_GUID_TAIL)?;
    } else {
        w.write_all(b"fmt ")?;
        w.write_all(&16u32.to_le_bytes())?;
        w.write_all(&fc.as_u16().to_le_bytes())?;
        w.write_all(&channels.to_le_bytes())?;
        w.write_all(&sample_rate.to_le_bytes())?;
        w.write_all(&byte_rate.to_le_bytes())?;
        w.write_all(&block_align.to_le_bytes())?;
        w.write_all(&bits.to_le_bytes())?;
    }
    Ok(())
}

/// Build a complete, finite WAV header (RIFF/WAVE + fmt + `data` header) for a known frame count.
///
/// The returned bytes are exactly what precedes the audio payload in the finished file: write
/// these, then write `total_frames` frames of interleaved little-endian samples, then (if the
/// data length is odd) a single pad byte. The RIFF and `data` size fields are already final, so
/// no seeking/backpatching is required — ideal for `!Seek` sinks and multipart uploads.
pub fn build_wav_header(
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
    total_frames: usize,
) -> AudioIOResult<Vec<u8>> {
    let data_size = wav_data_len(channels, sample_type, total_frames);
    let padded = data_size + (data_size & 1);
    let fmt_total = 8 + fmt_body_len(channels, sample_type);
    let file_size = 4 + fmt_total + 8 + padded;

    let mut header = Vec::with_capacity(wav_header_len(channels, sample_type));
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(file_size as u32).to_le_bytes());
    header.extend_from_slice(b"WAVE");
    write_fmt_chunk(&mut header, channels, sample_rate, sample_type)?;
    header.extend_from_slice(b"data");
    header.extend_from_slice(&(data_size as u32).to_le_bytes());
    Ok(header)
}

/// Build a WAV header for a stream of *unknown* length.
///
/// The RIFF and `data` size fields are set to `0xFFFFFFFF`, the convention streaming encoders
/// (ffmpeg, etc.) use when the final length can't be known up front. The resulting file is
/// non-standard but widely readable — including by this crate's own reader, which clamps such
/// size fields to the bytes actually present. Prefer [`build_wav_header`] whenever the frame
/// count *is* known.
pub fn build_wav_header_infinite(
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
) -> AudioIOResult<Vec<u8>> {
    let mut header = Vec::with_capacity(wav_header_len(channels, sample_type));
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    write_fmt_chunk(&mut header, channels, sample_rate, sample_type)?;
    header.extend_from_slice(b"data");
    header.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_len_matches_built_header() {
        for (ch, st) in [
            (1u16, ValidatedSampleType::I16),
            (2, ValidatedSampleType::F32),
            (1, ValidatedSampleType::I24), // extensible
            (6, ValidatedSampleType::I16), // extensible (>2 ch)
        ] {
            let h = build_wav_header(ch, 44_100, st, 100).expect("build header");
            assert_eq!(h.len(), wav_header_len(ch, st), "ch={ch} st={st:?}");
        }
    }

    #[test]
    fn file_len_accounts_for_pad_byte() {
        // Mono 8-bit, odd frame count → odd data size → +1 pad byte → even file size.
        let st = ValidatedSampleType::U8;
        let total = wav_file_len(1, st, 3);
        assert_eq!(total % 2, 0);
        assert_eq!(total, wav_header_len(1, st) + 4); // 3 data bytes + 1 pad
    }

    #[test]
    fn header_declares_expected_sizes() {
        let h = build_wav_header(2, 48_000, ValidatedSampleType::I16, 10).expect("build header");
        // data size = 10 frames * 2 ch * 2 bytes = 40
        let data_size = u32::from_le_bytes([h[h.len() - 4], h[h.len() - 3], h[h.len() - 2], h[h.len() - 1]]);
        assert_eq!(data_size, 40);
        let riff_size = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
        assert_eq!(riff_size as usize, wav_file_len(2, ValidatedSampleType::I16, 10) - 8);
    }
}
