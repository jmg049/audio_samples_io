//! Integration tests for the streaming FLAC writer (`create_streamed_flac` /
//! `StreamedFlacWriter`).
//!
//! These mirror the structure of `tests/wav_write.rs`: round-trip through the public API
//! and check stream structure (sample rate, channels, sample count). FLAC content fidelity
//! for non-trivial audio has known decoder caveats, so — like the existing FLAC roundtrip
//! tests — these assert structure, not exact sample values.

#![cfg(feature = "flac")]

use std::io::Cursor;
use std::time::Duration;

use audio_samples::{AudioSamples, channels, nzu, sample_rate, sine_wave};
use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
use audio_samples_io::{create_streamed_flac, create_streamed_flac_writer, info, read};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

/// A single chunk written and finalized round-trips with the right structure.
#[test]
fn stream_write_single_chunk_roundtrips() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.1), sr, 0.5);
    let expected = audio.samples_per_channel().get();

    let path = tmp("aio_flac_stream_single.flac");
    {
        let mut writer = create_streamed_flac::<_, i16>(&path, 1, 44100).expect("create streaming FLAC writer");
        let n = writer.write_frames(&audio).expect("write_frames");
        assert_eq!(n, expected, "write_frames should report frames written");
        writer.finalize().expect("finalize");
        assert!(writer.is_finalized());
        assert_eq!(writer.frames_written(), expected);
    }

    let back = read::<_, i16>(&path).expect("read back streamed FLAC");
    assert_eq!(back.sample_rate(), sr);
    assert_eq!(back.num_channels().get(), 1);
    assert_eq!(back.samples_per_channel().get(), expected);

    std::fs::remove_file(&path).ok();
}

/// Many chunks spanning multiple full blocks plus a trailing partial block accumulate
/// correctly across `write_frames` calls.
#[test]
fn stream_write_multi_chunk_accumulates() {
    let sr = sample_rate!(44100);
    // Default level uses a 1152-sample block, so these chunks exercise several full
    // blocks (4000) and a final partial block (4000 + 4000 + 2000 = 10000 samples).
    let chunk_a = sine_wave::<i16>(330.0, Duration::from_secs_f64(4000.0 / 44100.0), sr, 0.4);
    let chunk_b = sine_wave::<i16>(440.0, Duration::from_secs_f64(4000.0 / 44100.0), sr, 0.4);
    let chunk_c = sine_wave::<i16>(550.0, Duration::from_secs_f64(2000.0 / 44100.0), sr, 0.4);
    let expected =
        chunk_a.samples_per_channel().get() + chunk_b.samples_per_channel().get() + chunk_c.samples_per_channel().get();

    let path = tmp("aio_flac_stream_multi.flac");
    {
        let mut writer = create_streamed_flac::<_, i16>(&path, 1, 44100).expect("create writer");
        writer.write_frames(&chunk_a).expect("write a");
        writer.write_frames(&chunk_b).expect("write b");
        writer.write_frames(&chunk_c).expect("write c");
        writer.finalize().expect("finalize");
        assert_eq!(writer.frames_written(), expected);
    }

    let meta = info(&path).expect("info on streamed FLAC");
    assert_eq!(meta.channels, 1);
    assert_eq!(meta.sample_rate.get(), 44100);

    let back = read::<_, i16>(&path).expect("read back");
    assert_eq!(back.samples_per_channel().get(), expected);

    std::fs::remove_file(&path).ok();
}

/// Stereo silence round-trips through the streaming writer.
#[test]
fn stream_write_stereo_roundtrips() {
    let sr = sample_rate!(48000);
    let audio = AudioSamples::<i16>::zeros_multi(channels!(2), nzu!(5000), sr);

    let path = tmp("aio_flac_stream_stereo.flac");
    {
        let mut writer = create_streamed_flac::<_, i16>(&path, 2, 48000).expect("create writer");
        writer.write_frames(&audio).expect("write stereo");
        writer.finalize().expect("finalize");
    }

    let meta = info(&path).expect("info");
    assert_eq!(meta.channels, 2);
    assert_eq!(meta.sample_rate.get(), 48000);

    let back = read::<_, i16>(&path).expect("read back stereo");
    assert_eq!(back.num_channels().get(), 2);
    assert_eq!(back.samples_per_channel().get(), 5000);

    std::fs::remove_file(&path).ok();
}

/// The 24-bit path (f32 input) round-trips structurally.
#[test]
fn stream_write_f32_24bit_roundtrips() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);
    let expected = audio.samples_per_channel().get();

    let path = tmp("aio_flac_stream_f32.flac");
    {
        let mut writer = create_streamed_flac::<_, f32>(&path, 1, 44100).expect("create writer");
        writer.write_frames(&audio).expect("write f32");
        writer.finalize().expect("finalize");
    }

    let back = read::<_, f32>(&path).expect("read back f32");
    assert_eq!(back.sample_rate(), sr);
    assert_eq!(back.samples_per_channel().get(), expected);

    std::fs::remove_file(&path).ok();
}

/// Streaming to an in-memory `Cursor` produces a file the decoder accepts.
#[test]
fn stream_write_in_memory_cursor() {
    let sr = sample_rate!(22050);
    let audio = sine_wave::<i16>(880.0, Duration::from_secs_f64(0.05), sr, 0.6);
    let expected = audio.samples_per_channel().get();

    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = create_streamed_flac_writer::<_, i16>(cursor, 1, 22050).expect("create cursor writer");
        writer.write_frames(&audio).expect("write");
        writer.finalize().expect("finalize");
    }

    assert_eq!(&buffer[0..4], b"fLaC", "stream should start with the FLAC marker");

    let path = tmp("aio_flac_stream_cursor.flac");
    std::fs::write(&path, &buffer).expect("write buffer to temp file");
    let back = read::<_, i16>(&path).expect("read back from buffer");
    assert_eq!(back.sample_rate(), sr);
    assert_eq!(back.samples_per_channel().get(), expected);
    std::fs::remove_file(&path).ok();
}

/// A channel-count mismatch between the writer config and the supplied frames errors.
#[test]
fn stream_write_channel_mismatch_errors() {
    let sr = sample_rate!(44100);
    let mono = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.02), sr, 0.5);

    let path = tmp("aio_flac_stream_mismatch.flac");
    let mut writer = create_streamed_flac::<_, i16>(&path, 2, 44100).expect("create stereo writer");
    let result = writer.write_frames(&mono);
    assert!(
        result.is_err(),
        "writing mono frames to a 2-channel writer should error"
    );
    // Finalize so the file is valid/closed, then clean up.
    writer.finalize().ok();
    drop(writer);
    std::fs::remove_file(&path).ok();
}

/// `finalize()` is idempotent.
#[test]
fn stream_write_finalize_idempotent() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.02), sr, 0.5);

    let path = tmp("aio_flac_stream_idem.flac");
    let mut writer = create_streamed_flac::<_, i16>(&path, 1, 44100).expect("create writer");
    writer.write_frames(&audio).expect("write");
    writer.finalize().expect("first finalize");
    writer.finalize().expect("second finalize should be a no-op");
    assert!(writer.is_finalized());
    drop(writer);
    std::fs::remove_file(&path).ok();
}

/// The format-agnostic [`create_streamed`] dispatches on the file extension and round-trips
/// for both WAV and FLAC through one code path. (Requires the `wav` feature too.)
#[cfg(feature = "wav")]
#[test]
fn unified_create_streamed_dispatches_by_extension() {
    use audio_samples_io::create_streamed;

    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);
    let expected = audio.samples_per_channel().get();

    for ext in ["wav", "flac"] {
        let path = tmp(&format!("aio_unified_stream.{ext}"));
        {
            let mut writer = create_streamed::<_, i16>(&path, 1, 44100).expect("create_streamed");
            writer.write_frames(&audio).expect("write");
            writer.finalize().expect("finalize");
        }
        let back = read::<_, i16>(&path).expect("read back");
        assert_eq!(back.sample_rate(), sr, "sample rate for .{ext}");
        assert_eq!(back.samples_per_channel().get(), expected, "sample count for .{ext}");
        std::fs::remove_file(&path).ok();
    }
}

/// [`create_streamed_with`] writes either format to an in-memory cursor given an explicit
/// [`FileType`]. (Requires the `wav` feature too.)
#[cfg(feature = "wav")]
#[test]
fn unified_create_streamed_with_explicit_format() {
    use std::io::Cursor;

    use audio_samples_io::create_streamed_with;
    use audio_samples_io::types::FileType;

    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.03), sr, 0.5);
    let expected = audio.samples_per_channel().get();

    for (format, ext, magic) in [
        (FileType::WAV, "wav", &b"RIFF"[..]),
        (FileType::FLAC, "flac", &b"fLaC"[..]),
    ] {
        let mut buf = Vec::new();
        {
            let cursor = Cursor::new(&mut buf);
            let mut writer = create_streamed_with::<_, i16>(cursor, 1, 44100, format).expect("create_with");
            writer.write_frames(&audio).expect("write");
            writer.finalize().expect("finalize");
        }
        assert_eq!(&buf[0..4], magic, "magic bytes for {format:?}");

        let path = tmp(&format!("aio_unified_with.{ext}"));
        std::fs::write(&path, &buf).expect("write temp");
        let back = read::<_, i16>(&path).expect("read back");
        assert_eq!(back.samples_per_channel().get(), expected, "{format:?}");
        std::fs::remove_file(&path).ok();
    }
}
