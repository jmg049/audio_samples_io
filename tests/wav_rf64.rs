//! RF64/BW64 (64-bit RIFF) read support.
//!
//! Files are hand-crafted so the fixtures stay tiny: the point is the header
//! plumbing (form id, ds64 sizes, 0xFFFFFFFF placeholders), not large payloads.

#![cfg(feature = "wav")]

use std::io::Write;

use audio_samples_io::types::FileType;
use audio_samples_io::{info, open_streamed, peek_native_type, read};

/// Build an RF64/BW64 file holding mono 16-bit PCM `samples`.
///
/// The 32-bit RIFF and data size fields carry the 0xFFFFFFFF placeholder; the
/// true sizes live in ds64, exactly as EBU Tech 3306 prescribes. When
/// `trailing_chunk` is set, a junk chunk follows the data payload — a reader
/// that ignores ds64 and clamps to EOF would wrongly count those bytes as audio.
fn rf64_bytes(form: &[u8; 4], samples: &[i16], sample_rate: u32, trailing_chunk: bool) -> Vec<u8> {
    let data_size = samples.len() * 2;
    let trailing = if trailing_chunk { 8 + 4 } else { 0 };
    // Everything after the 8-byte RIFF header:
    // WAVE (4) + ds64 hdr+body (8+28) + fmt hdr+body (8+16) + data hdr (8) + payload + trailing
    let riff_size = 4 + 36 + 24 + 8 + data_size + trailing;

    let mut f = Vec::new();
    f.extend_from_slice(form);
    f.extend_from_slice(&u32::MAX.to_le_bytes()); // 32-bit size placeholder
    f.extend_from_slice(b"WAVE");

    // ds64
    f.extend_from_slice(b"ds64");
    f.extend_from_slice(&28u32.to_le_bytes());
    f.extend_from_slice(&(riff_size as u64).to_le_bytes());
    f.extend_from_slice(&(data_size as u64).to_le_bytes());
    f.extend_from_slice(&(samples.len() as u64).to_le_bytes());
    f.extend_from_slice(&0u32.to_le_bytes()); // empty chunk-size table

    // fmt (mono 16-bit PCM)
    let block_align = 2u16;
    f.extend_from_slice(b"fmt ");
    f.extend_from_slice(&16u32.to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes()); // PCM
    f.extend_from_slice(&1u16.to_le_bytes()); // channels
    f.extend_from_slice(&sample_rate.to_le_bytes());
    f.extend_from_slice(&(sample_rate * block_align as u32).to_le_bytes());
    f.extend_from_slice(&block_align.to_le_bytes());
    f.extend_from_slice(&16u16.to_le_bytes());

    // data with placeholder size
    f.extend_from_slice(b"data");
    f.extend_from_slice(&u32::MAX.to_le_bytes());
    for s in samples {
        f.extend_from_slice(&s.to_le_bytes());
    }

    if trailing_chunk {
        f.extend_from_slice(b"junk");
        f.extend_from_slice(&4u32.to_le_bytes());
        f.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    f
}

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut file = std::fs::File::create(&path).expect("create fixture");
    file.write_all(bytes).expect("write fixture");
    path
}

fn test_samples() -> Vec<i16> {
    (0..480).map(|i| (i * 37 % 1000) as i16 - 500).collect()
}

#[test]
fn rf64_full_read_round_trip() {
    let samples = test_samples();
    let path = write_tmp("aio_rf64_read.wav", &rf64_bytes(b"RF64", &samples, 48_000, false));

    assert_eq!(FileType::detect(&path), FileType::WAV);
    assert_eq!(
        peek_native_type(&path).expect("peek"),
        audio_samples_io::ValidatedSampleType::I16
    );

    let audio = read::<_, i16>(&path).expect("read RF64");
    assert_eq!(audio.sample_rate().get(), 48_000);
    assert_eq!(audio.samples_per_channel().get(), samples.len());
    let back: Vec<i16> = audio.to_interleaved_vec().into_iter().collect();
    assert_eq!(back, samples);

    std::fs::remove_file(&path).ok();
}

#[test]
fn bw64_form_id_is_accepted() {
    let samples = test_samples();
    let path = write_tmp("aio_bw64_read.wav", &rf64_bytes(b"BW64", &samples, 44_100, false));

    let audio = read::<_, i16>(&path).expect("read BW64");
    assert_eq!(audio.samples_per_channel().get(), samples.len());

    std::fs::remove_file(&path).ok();
}

#[test]
fn rf64_data_size_comes_from_ds64_not_eof_clamp() {
    // A junk chunk after the data payload: counting bytes to EOF would inflate
    // the sample count by 6 (junk header 8 + body 4 = 12 bytes = 6 i16 samples).
    let samples = test_samples();
    let path = write_tmp("aio_rf64_trailing.wav", &rf64_bytes(b"RF64", &samples, 48_000, true));

    let base = info(&path).expect("info");
    assert_eq!(
        base.total_samples,
        samples.len(),
        "ds64 data size must win over EOF clamping"
    );

    let audio = read::<_, i16>(&path).expect("read");
    assert_eq!(audio.samples_per_channel().get(), samples.len());

    std::fs::remove_file(&path).ok();
}

#[test]
fn rf64_streamed_reader_reports_and_reads_frames() {
    use audio_samples::{AudioSamples, nzu, sample_rate};

    let samples = test_samples();
    let path = write_tmp("aio_rf64_streamed.wav", &rf64_bytes(b"RF64", &samples, 48_000, true));

    let mut streamed = open_streamed(&path).expect("open_streamed");
    assert_eq!(streamed.total_frames(), samples.len());
    assert_eq!(streamed.num_channels(), 1);

    let mut buffer = AudioSamples::<i16>::zeros_mono(nzu!(480), sample_rate!(48000));
    let frames = streamed.read_frames_into(&mut buffer, nzu!(480)).expect("read frames");
    assert_eq!(frames, samples.len());

    std::fs::remove_file(&path).ok();
}

#[test]
fn rf64_without_ds64_is_rejected() {
    let samples = test_samples();
    let mut bytes = rf64_bytes(b"RF64", &samples, 48_000, false);
    // Corrupt the ds64 id so the mandatory chunk is "missing".
    bytes[12..16].copy_from_slice(b"nope");
    let path = write_tmp("aio_rf64_no_ds64.wav", &bytes);

    assert!(read::<_, i16>(&path).is_err(), "RF64 without ds64 must be rejected");

    std::fs::remove_file(&path).ok();
}
