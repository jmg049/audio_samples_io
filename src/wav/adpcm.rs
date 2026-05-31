//! ADPCM decode to 16-bit linear PCM.
//!
//! Supports the two ADPCM variants commonly found in `.wav` files:
//! * **Microsoft ADPCM** (`WAVE_FORMAT_ADPCM`, `0x0002`) — classic Windows/game-era audio.
//! * **IMA / DVI ADPCM** (`WAVE_FORMAT_DVI_ADPCM`, `0x0011`).
//!
//! Both are block-based: the `data` chunk is a sequence of fixed-size blocks (`nBlockAlign`
//! bytes), each carrying a small predictor state header followed by 4-bit nibbles. Decoding
//! expands every nibble to a signed 16-bit linear sample.
//!
//! Only whole blocks are decoded; a ragged trailing partial block is dropped, matching the
//! way the rest of the reader tolerates truncated/streaming files.

use crate::{
    error::{AudioIOError, AudioIOResult},
    wav::fmt::FmtChunk,
};

/// Microsoft ADPCM delta adaptation table (indexed by nibble).
const MS_ADAPT: [i32; 16] = [
    230, 230, 230, 230, 307, 409, 512, 614, 768, 614, 512, 409, 307, 230, 230, 230,
];

/// IMA ADPCM step-index adjustment table (indexed by nibble).
const IMA_INDEX: [i32; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

/// IMA ADPCM quantiser step-size table (89 entries).
const IMA_STEP: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107,
    118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
    6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767,
];

#[inline]
fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Read `wSamplesPerBlock` from the fmt extension (bytes 18..20). Returns `None` if absent.
fn samples_per_block(fmt: &FmtChunk<'_>) -> Option<usize> {
    let b = fmt.as_bytes();
    if b.len() >= 20 {
        Some(u16::from_le_bytes([b[18], b[19]]) as usize)
    } else {
        None
    }
}

/// Number of interleaved 16-bit samples that [`decode`] will produce for `data_len` data bytes.
///
/// Counts whole blocks only (a ragged trailing block is dropped), so this matches the length of
/// the vector returned by [`decode`] exactly.
pub fn decoded_sample_count(fmt: &FmtChunk<'_>, data_len: usize) -> usize {
    let channels = fmt.channels() as usize;
    let block_align = fmt.block_align() as usize;
    let spb = samples_per_block(fmt).unwrap_or(0);
    if channels == 0 || block_align == 0 {
        return 0;
    }
    (data_len / block_align) * spb * channels
}

/// Decode an ADPCM `data` chunk to interleaved 16-bit linear PCM.
///
/// Dispatches on the fmt chunk's format code; returns an error for non-ADPCM formats or
/// malformed headers (missing `wSamplesPerBlock`, zero block size, etc.).
pub fn decode(fmt: &FmtChunk<'_>, data: &[u8]) -> AudioIOResult<Vec<i16>> {
    use crate::wav::FormatCode;
    match fmt.format_code() {
        FormatCode::MsAdpcm => decode_ms(fmt, data),
        FormatCode::ImaAdpcm => decode_ima(fmt, data),
        other => Err(AudioIOError::unsupported_format(format!(
            "not an ADPCM format: {other}"
        ))),
    }
}

fn decode_ms(fmt: &FmtChunk<'_>, data: &[u8]) -> AudioIOResult<Vec<i16>> {
    let channels = fmt.channels() as usize;
    let block_align = fmt.block_align() as usize;
    let spb = samples_per_block(fmt)
        .ok_or_else(|| AudioIOError::unsupported_format("MS ADPCM fmt chunk missing wSamplesPerBlock"))?;
    let ext = fmt.as_bytes();

    // Extension layout: [18..20] wSamplesPerBlock, [20..22] wNumCoef, [22..] aCoef pairs.
    if ext.len() < 22 {
        return Err(AudioIOError::unsupported_format(
            "MS ADPCM fmt chunk missing coefficient table",
        ));
    }
    let num_coef = u16::from_le_bytes([ext[20], ext[21]]) as usize;
    if ext.len() < 22 + num_coef * 4 {
        return Err(AudioIOError::unsupported_format("MS ADPCM coefficient table truncated"));
    }
    let coefs: Vec<(i32, i32)> = (0..num_coef)
        .map(|i| {
            let o = 22 + i * 4;
            (
                i16::from_le_bytes([ext[o], ext[o + 1]]) as i32,
                i16::from_le_bytes([ext[o + 2], ext[o + 3]]) as i32,
            )
        })
        .collect();
    if coefs.is_empty() || block_align == 0 || channels == 0 {
        return Err(AudioIOError::unsupported_format("invalid MS ADPCM parameters"));
    }

    let header_len = 7 * channels; // per channel: 1 predictor index + 2 delta + 2 samp1 + 2 samp2
    let mut out = Vec::with_capacity(decoded_sample_count(fmt, data.len()));

    for block in data.chunks_exact(block_align) {
        if block.len() < header_len {
            break;
        }
        let mut predictor = vec![0usize; channels];
        let mut delta = vec![0i32; channels];
        let mut samp1 = vec![0i32; channels];
        let mut samp2 = vec![0i32; channels];

        let mut pos = 0;
        for p in predictor.iter_mut() {
            // Clamp a bogus predictor index into the available coefficient set.
            *p = (block[pos] as usize).min(coefs.len() - 1);
            pos += 1;
        }
        for d in delta.iter_mut() {
            *d = i16::from_le_bytes([block[pos], block[pos + 1]]) as i32;
            pos += 2;
        }
        for s in samp1.iter_mut() {
            *s = i16::from_le_bytes([block[pos], block[pos + 1]]) as i32;
            pos += 2;
        }
        for s in samp2.iter_mut() {
            *s = i16::from_le_bytes([block[pos], block[pos + 1]]) as i32;
            pos += 2;
        }

        // The block header's two stored samples are the first two output samples (oldest first).
        out.extend(samp2.iter().map(|&s| clamp_i16(s)));
        out.extend(samp1.iter().map(|&s| clamp_i16(s)));

        // Remaining samples come from the nibble stream: high nibble first within each byte,
        // channels round-robin (stereo: high = left, low = right).
        let nibbles_total = spb.saturating_sub(2) * channels;
        let mut count = 0usize;
        'block: for &byte in &block[pos..] {
            for nib in [byte >> 4, byte & 0x0F] {
                if count >= nibbles_total {
                    break 'block;
                }
                let ch = count % channels;
                let (c1, c2) = coefs[predictor[ch]];
                let signed = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                let predict = (samp1[ch] * c1 + samp2[ch] * c2) / 256;
                let new = clamp_i16(predict + signed * delta[ch]) as i32;
                out.push(new as i16);

                let mut new_delta = (MS_ADAPT[nib as usize] * delta[ch]) / 256;
                if new_delta < 16 {
                    new_delta = 16;
                }
                delta[ch] = new_delta;
                samp2[ch] = samp1[ch];
                samp1[ch] = new;
                count += 1;
            }
        }
    }
    Ok(out)
}

fn decode_ima(fmt: &FmtChunk<'_>, data: &[u8]) -> AudioIOResult<Vec<i16>> {
    let channels = fmt.channels() as usize;
    let block_align = fmt.block_align() as usize;
    let spb = samples_per_block(fmt)
        .ok_or_else(|| AudioIOError::unsupported_format("IMA ADPCM fmt chunk missing wSamplesPerBlock"))?;
    if channels == 0 || block_align == 0 {
        return Err(AudioIOError::unsupported_format("invalid IMA ADPCM parameters"));
    }

    let header_len = 4 * channels; // per channel: 2-byte initial sample + 1-byte index + 1 reserved
    let mut out = Vec::with_capacity(decoded_sample_count(fmt, data.len()));

    for block in data.chunks_exact(block_align) {
        if block.len() < header_len {
            break;
        }
        let mut predictor = vec![0i32; channels];
        let mut index = vec![0i32; channels];

        let mut pos = 0;
        for ch in 0..channels {
            predictor[ch] = i16::from_le_bytes([block[pos], block[pos + 1]]) as i32;
            index[ch] = (block[pos + 2] as i32).clamp(0, 88);
            pos += 4;
        }

        // Decode each channel into its own buffer, then interleave. Data is laid out in 4-byte
        // (8-nibble) words, one channel per word, round-robin; nibbles are low-then-high.
        let mut chan: Vec<Vec<i16>> = (0..channels)
            .map(|ch| {
                let mut v = Vec::with_capacity(spb);
                v.push(clamp_i16(predictor[ch])); // the stored predictor is the first sample
                v
            })
            .collect();

        let words = &block[header_len..];
        let mut word_off = 0;
        let mut ch = 0;
        while word_off + 4 <= words.len() {
            for &byte in &words[word_off..word_off + 4] {
                for nib in [byte & 0x0F, byte >> 4] {
                    let step = IMA_STEP[index[ch] as usize];
                    let mut diff = step >> 3;
                    if nib & 4 != 0 {
                        diff += step;
                    }
                    if nib & 2 != 0 {
                        diff += step >> 1;
                    }
                    if nib & 1 != 0 {
                        diff += step >> 2;
                    }
                    if nib & 8 != 0 {
                        predictor[ch] -= diff;
                    } else {
                        predictor[ch] += diff;
                    }
                    predictor[ch] = predictor[ch].clamp(i16::MIN as i32, i16::MAX as i32);
                    index[ch] = (index[ch] + IMA_INDEX[nib as usize]).clamp(0, 88);
                    chan[ch].push(predictor[ch] as i16);
                }
            }
            word_off += 4;
            ch = (ch + 1) % channels;
        }

        // Interleave, capped at the declared samples-per-block (and the shortest channel).
        let frames = chan.iter().map(|c| c.len()).min().unwrap_or(0).min(spb);
        for i in 0..frames {
            for c in chan.iter() {
                out.push(c[i]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal IMA ADPCM fmt chunk (20 bytes: base + cbSize + wSamplesPerBlock).
    fn ima_fmt(channels: u16, block_align: u16, samples_per_block: u16) -> Vec<u8> {
        let mut v = vec![0u8; 20];
        v[0..2].copy_from_slice(&0x0011u16.to_le_bytes());
        v[2..4].copy_from_slice(&channels.to_le_bytes());
        v[4..8].copy_from_slice(&8000u32.to_le_bytes());
        v[12..14].copy_from_slice(&block_align.to_le_bytes());
        v[14..16].copy_from_slice(&4u16.to_le_bytes()); // bits per sample
        v[16..18].copy_from_slice(&2u16.to_le_bytes()); // cbSize
        v[18..20].copy_from_slice(&samples_per_block.to_le_bytes());
        v
    }

    /// Build a minimal MS ADPCM fmt chunk with a single coefficient pair.
    fn ms_fmt(channels: u16, block_align: u16, samples_per_block: u16, coef: (i16, i16)) -> Vec<u8> {
        let mut v = vec![0u8; 26];
        v[0..2].copy_from_slice(&0x0002u16.to_le_bytes());
        v[2..4].copy_from_slice(&channels.to_le_bytes());
        v[4..8].copy_from_slice(&8000u32.to_le_bytes());
        v[12..14].copy_from_slice(&block_align.to_le_bytes());
        v[14..16].copy_from_slice(&4u16.to_le_bytes());
        v[16..18].copy_from_slice(&32u16.to_le_bytes()); // cbSize
        v[18..20].copy_from_slice(&samples_per_block.to_le_bytes());
        v[20..22].copy_from_slice(&1u16.to_le_bytes()); // wNumCoef
        v[22..24].copy_from_slice(&coef.0.to_le_bytes());
        v[24..26].copy_from_slice(&coef.1.to_le_bytes());
        v
    }

    #[test]
    fn ima_mono_block_decodes_hand_traced_values() {
        let fmt_bytes = ima_fmt(1, 8, 9);
        let fmt = FmtChunk::from_bytes(&fmt_bytes).expect("valid test fmt chunk");
        // header: predictor=100, index=0, reserved=0; then one 4-byte word with nibbles [4,0,0,...].
        let block = [0x64u8, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00];
        let decoded = decode(&fmt, &block).expect("decode test block");
        assert_eq!(decoded, vec![100, 107, 108, 109, 109, 109, 109, 109, 109]);
        assert_eq!(decoded.len(), decoded_sample_count(&fmt, block.len()));
    }

    #[test]
    fn ms_mono_block_decodes_hand_traced_values() {
        // coef (256, 0) makes the predictor simply track the previous sample.
        let fmt_bytes = ms_fmt(1, 8, 4, (256, 0));
        let fmt = FmtChunk::from_bytes(&fmt_bytes).expect("valid test fmt chunk");
        // header: predictor idx=0, delta=16, samp1=0, samp2=0; then nibble byte 0x10 -> [1, 0].
        let block = [0x00u8, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10];
        let decoded = decode(&fmt, &block).expect("decode test block");
        assert_eq!(decoded, vec![0, 0, 16, 16]);
        assert_eq!(decoded.len(), decoded_sample_count(&fmt, block.len()));
    }

    #[test]
    fn non_adpcm_format_is_rejected() {
        let mut fmt_bytes = vec![0u8; 16];
        fmt_bytes[0..2].copy_from_slice(&0x0001u16.to_le_bytes()); // PCM
        let fmt = FmtChunk::from_bytes(&fmt_bytes).expect("valid test fmt chunk");
        assert!(decode(&fmt, &[0u8; 8]).is_err());
    }
}
