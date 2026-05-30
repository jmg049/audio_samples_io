//! Robustness tests for awkward real-world WAV files plus the write-side metadata,
//! header-builder, and non-seekable streaming APIs.
//!
//! Each test pins behaviour for a quirk that strict readers tend to reject — streaming size
//! placeholders, ragged data chunks, companded/ADPCM encodings — and round-trips for the
//! newer write APIs.
#![cfg(feature = "wav")]

use audio_samples_io::traits::{AudioFile, AudioFileRead};
use audio_samples_io::{OpenOptions, WavFile};

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "asio_wav_{name}_{}_{:?}.wav",
        std::process::id(),
        std::thread::current().id()
    ))
}

/// Build a 16-byte base `fmt ` chunk with internally consistent derived fields.
fn fmt_chunk(format: u16, channels: u16, sample_rate: u32, bits: u16) -> Vec<u8> {
    let bytes_per = bits / 8;
    let block_align = channels * bytes_per;
    let byte_rate = sample_rate * block_align as u32;
    let mut v = Vec::new();
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&format.to_le_bytes());
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&block_align.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v
}

/// Assemble a complete RIFF/WAVE file from a fmt chunk and data bytes, allowing the RIFF
/// size and `data` size fields to be overridden to simulate streaming/malformed producers.
/// `trailing` is appended verbatim after the (padded) data chunk to exercise post-`data` chunks.
fn assemble(
    fmt: &[u8],
    data: &[u8],
    riff_size_override: Option<u32>,
    data_size_override: Option<u32>,
    trailing: &[u8],
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(fmt);
    body.extend_from_slice(b"data");
    let dsize = data_size_override.unwrap_or(data.len() as u32);
    body.extend_from_slice(&dsize.to_le_bytes());
    body.extend_from_slice(data);
    if data.len() % 2 == 1 {
        body.push(0); // word-align the data chunk
    }
    body.extend_from_slice(trailing);

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    let rsize = riff_size_override.unwrap_or(body.len() as u32);
    out.extend_from_slice(&rsize.to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Wrap raw fmt-chunk-data bytes (the bytes after the 8-byte chunk header) into a `fmt ` chunk.
fn fmt_chunk_raw(fmt_data: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&(fmt_data.len() as u32).to_le_bytes());
    v.extend_from_slice(fmt_data);
    if fmt_data.len() % 2 == 1 {
        v.push(0);
    }
    v
}

fn read_i16(bytes: &[u8], name: &str) -> Vec<i16> {
    let path = temp_path(name);
    std::fs::write(&path, bytes).expect("write temp wav");
    let wav = <WavFile as AudioFile>::open_with_options(&path, OpenOptions::default())
        .unwrap_or_else(|e| panic!("open {name}: {e}"));
    let audio = <WavFile as AudioFileRead>::read::<i16>(&wav)
        .unwrap_or_else(|e| panic!("read {name}: {e}"));
    let out = audio.to_interleaved_vec().into_vec();
    std::fs::remove_file(&path).ok();
    out
}

fn i16_data(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

#[test]
fn streaming_max_size_fields_are_clamped() {
    // ffmpeg/live-capture writes 0xFFFFFFFF when the final length is unknown.
    let fmt = fmt_chunk(0x0001, 1, 44_100, 16);
    let data = i16_data(&[2, -3, 5, -7]);
    let bytes = assemble(&fmt, &data, Some(0xFFFF_FFFF), Some(0xFFFF_FFFF), &[]);
    assert_eq!(read_i16(&bytes, "streaming_max"), vec![2, -3, 5, -7]);
}

#[test]
fn oversized_data_size_is_clamped() {
    // Declared data size larger than the bytes present: clamp to what's actually there.
    let fmt = fmt_chunk(0x0001, 1, 44_100, 16);
    let data = i16_data(&[10, 20, 30]);
    let bytes = assemble(&fmt, &data, None, Some(1_000_000), &[]);
    assert_eq!(read_i16(&bytes, "oversized_data"), vec![10, 20, 30]);
}

#[test]
fn ragged_mono_data_truncates_to_whole_samples() {
    // Mono 16-bit, declared data size has a stray trailing byte (7 = 3 samples + 1).
    let fmt = fmt_chunk(0x0001, 1, 44_100, 16);
    let mut data = i16_data(&[100, 200, 300]);
    data.push(0x7F); // ragged trailing byte
    let bytes = assemble(&fmt, &data, None, Some(7), &[]);
    // Whole samples decoded, ragged tail dropped.
    assert_eq!(read_i16(&bytes, "ragged_mono"), vec![100, 200, 300]);
}

#[test]
fn ragged_stereo_data_truncates_to_whole_frames() {
    // Stereo 16-bit (frame = 4 bytes); 6 data bytes = 1 whole frame + a ragged 2-byte tail.
    let fmt = fmt_chunk(0x0001, 2, 44_100, 16);
    let data = i16_data(&[111, 222, 333]); // 6 bytes: L=111,R=222 then ragged 333
    let bytes = assemble(&fmt, &data, None, None, &[]);
    // Only the whole frame survives.
    assert_eq!(read_i16(&bytes, "ragged_stereo"), vec![111, 222]);
}

#[test]
fn mulaw_decodes_to_linear_pcm() {
    // mu-law (0x0007), 8-bit mono. Bytes chosen for known G.711 expansions.
    let fmt = fmt_chunk(0x0007, 1, 8_000, 8);
    let data = [0xFFu8, 0x00, 0x80, 0x7F];
    let bytes = assemble(&fmt, &data, None, None, &[]);
    // Companded telephony audio expands to 16-bit linear PCM.
    assert_eq!(read_i16(&bytes, "mulaw"), vec![0, -32124, 32124, 0]);
}

#[test]
fn alaw_decodes_to_linear_pcm() {
    // a-law (0x0006), 8-bit mono.
    let fmt = fmt_chunk(0x0006, 1, 8_000, 8);
    let data = [0xD5u8, 0x55, 0x2A, 0xAA];
    let bytes = assemble(&fmt, &data, None, None, &[]);
    assert_eq!(read_i16(&bytes, "alaw"), vec![8, -8, -32256, 32256]);
}

#[test]
fn mulaw_metadata_reports_decoded_i16() {
    use audio_samples::SampleType;
    use audio_samples_io::traits::AudioFileMetadata;
    let fmt = fmt_chunk(0x0007, 1, 8_000, 8);
    let data = [0xFFu8, 0x00, 0x80, 0x7F];
    let bytes = assemble(&fmt, &data, None, None, &[]);
    let path = temp_path("mulaw_meta");
    std::fs::write(&path, &bytes).unwrap();
    let wav = <WavFile as AudioFile>::open_with_options(&path, OpenOptions::default()).unwrap();
    let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
    // Decoded output is 16-bit linear; sample count equals the number of companded bytes.
    assert_eq!(info.sample_type, SampleType::I16);
    assert_eq!(info.total_samples, 4);
    std::fs::remove_file(&path).ok();
}

#[test]
fn precomputed_size_and_header_match_written_file() {
    use audio_samples::{AudioSamples, nzu, sample_rate};
    use audio_samples_io::wav::{build_wav_header, wav_file_len};
    use audio_samples_io::{ValidatedSampleType, types::FileType, write_with};

    let sr = sample_rate!(44_100);

    // (label, written bytes, channels, ValidatedSampleType, frames)
    let mut cases: Vec<(&str, Vec<u8>, u16, ValidatedSampleType, usize)> = Vec::new();

    let mono_i16 = AudioSamples::<i16>::zeros_mono(nzu!(101), sr);
    let mut b = Vec::new();
    write_with(std::io::Cursor::new(&mut b), &mono_i16, FileType::WAV).unwrap();
    cases.push(("mono_i16", b, 1, ValidatedSampleType::I16, 101));

    let stereo_f32 = AudioSamples::<f32>::zeros_multi(audio_samples::channels!(2), nzu!(64), sr);
    let mut b = Vec::new();
    write_with(std::io::Cursor::new(&mut b), &stereo_f32, FileType::WAV).unwrap();
    cases.push(("stereo_f32", b, 2, ValidatedSampleType::F32, 64));

    // f64 mono → extensible fmt chunk path.
    let mono_f64 = AudioSamples::<f64>::zeros_mono(nzu!(50), sr);
    let mut b = Vec::new();
    write_with(std::io::Cursor::new(&mut b), &mono_f64, FileType::WAV).unwrap();
    cases.push(("mono_f64_ext", b, 1, ValidatedSampleType::F64, 50));

    for (label, written, ch, st, frames) in cases {
        assert_eq!(
            written.len(),
            wav_file_len(ch, st, frames),
            "{label}: precomputed file size mismatch"
        );
        let header = build_wav_header(ch, 44_100, st, frames).unwrap();
        assert_eq!(
            &written[..header.len()],
            &header[..],
            "{label}: precomputed header bytes differ from writer output"
        );
    }
}

#[test]
fn metadata_round_trips() {
    use audio_samples::{AudioSamples, nzu, sample_rate};
    use audio_samples_io::traits::AudioFile;
    use audio_samples_io::wav::chunks::ChunkID;
    use audio_samples_io::wav::list_info::InfoMetadata;
    use audio_samples_io::wav::{CuePoint, WavMetadata};
    use audio_samples_io::write_with_metadata;

    let path = temp_path("metadata_roundtrip");
    let audio = AudioSamples::<i16>::zeros_mono(nzu!(64), sample_rate!(44_100));

    let meta = WavMetadata {
        info: Some(InfoMetadata {
            title: Some("My Track".to_string()),
            artist: Some("Some Artist".to_string()),
            genre: Some("Electronic".to_string()),
            ..Default::default()
        }),
        cue_points: vec![
            CuePoint {
                id: 1,
                position: 0,
                data_chunk_id: ChunkID::new(b"data"),
                chunk_start: 0,
                block_start: 0,
                sample_offset: 22_050,
            },
            CuePoint {
                id: 2,
                position: 0,
                data_chunk_id: ChunkID::new(b"data"),
                chunk_start: 0,
                block_start: 0,
                sample_offset: 44_100,
            },
        ],
    };

    write_with_metadata(&path, &audio, &meta).expect("write with metadata");

    let wav = <WavFile as AudioFile>::open_with_options(&path, OpenOptions::default())
        .expect("reopen tagged file");

    // INFO tags survive the round-trip.
    let list = wav.list().expect("list parse").expect("LIST chunk present");
    let info = list.info_metadata().expect("is INFO").expect("parses");
    assert_eq!(info.title.as_deref(), Some("My Track"));
    assert_eq!(info.artist.as_deref(), Some("Some Artist"));
    assert_eq!(info.genre.as_deref(), Some("Electronic"));

    // Cue points survive the round-trip.
    let cue = wav.cue().expect("cue parse").expect("cue chunk present");
    let points = cue.cue_points().expect("cue points");
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].sample_offset, 22_050);
    assert_eq!(points[1].sample_offset, 44_100);

    // Audio still reads back correctly alongside the metadata.
    let read = <WavFile as AudioFileRead>::read::<i16>(&wav).expect("read audio");
    assert_eq!(read.samples_per_channel().get(), 64);

    std::fs::remove_file(&path).ok();
}

#[test]
fn ima_adpcm_mono_decodes_via_public_api() {
    // IMA ADPCM (0x0011), mono, block_align=8, 9 samples/block.
    let mut fmt_data = vec![0u8; 20];
    fmt_data[0..2].copy_from_slice(&0x0011u16.to_le_bytes());
    fmt_data[2..4].copy_from_slice(&1u16.to_le_bytes()); // channels
    fmt_data[4..8].copy_from_slice(&8000u32.to_le_bytes());
    fmt_data[12..14].copy_from_slice(&8u16.to_le_bytes()); // block_align
    fmt_data[14..16].copy_from_slice(&4u16.to_le_bytes()); // bits
    fmt_data[16..18].copy_from_slice(&2u16.to_le_bytes()); // cbSize
    fmt_data[18..20].copy_from_slice(&9u16.to_le_bytes()); // samples per block
    let fmt = fmt_chunk_raw(&fmt_data);

    // predictor=100, index=0, reserved=0; word with nibbles [4,0,0,...].
    let data = [0x64u8, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00];
    let bytes = assemble(&fmt, &data, None, None, &[]);
    // Block-compressed ADPCM is decoded to linear PCM.
    assert_eq!(
        read_i16(&bytes, "ima_adpcm"),
        vec![100, 107, 108, 109, 109, 109, 109, 109, 109]
    );
}

#[test]
fn trailing_chunk_after_data_is_ignored() {
    let fmt = fmt_chunk(0x0001, 1, 44_100, 16);
    let data = i16_data(&[1, 2, 3, 4]);
    // A LIST chunk after data — must be ignored, audio read unaffected.
    let mut trailing = Vec::new();
    trailing.extend_from_slice(b"LIST");
    trailing.extend_from_slice(&4u32.to_le_bytes());
    trailing.extend_from_slice(b"INFO");
    let bytes = assemble(&fmt, &data, None, None, &trailing);
    assert_eq!(read_i16(&bytes, "trailing"), vec![1, 2, 3, 4]);
}
