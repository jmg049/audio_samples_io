//! Content-based format detection: the read paths must dispatch on the bytes
//! actually in the file, not on what the extension claims.

#![cfg(all(feature = "wav", feature = "flac"))]

use std::time::Duration;

use audio_samples::{sample_rate, sine_wave};
use audio_samples_io::{info, read, types::FileType, write};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn read_dispatches_on_contents_not_extension() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);

    // WAV bytes behind a .flac extension.
    let wav_path = tmp("aio_sniff_real_wav.wav");
    let lying_path = tmp("aio_sniff_lying_ext.flac");
    write(&wav_path, &audio).expect("write wav");
    std::fs::rename(&wav_path, &lying_path).expect("rename");

    let detected = FileType::detect(&lying_path);
    assert_eq!(detected, FileType::WAV, "magic bytes win over the extension");

    let back = read::<_, i16>(&lying_path).expect("read WAV bytes despite .flac extension");
    assert_eq!(back.samples_per_channel(), audio.samples_per_channel());

    let base = info(&lying_path).expect("info on mismatched extension");
    assert_eq!(base.file_type, FileType::WAV);

    std::fs::remove_file(&lying_path).ok();
}

#[test]
fn read_works_without_any_extension() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);

    let flac_path = tmp("aio_sniff_noext_src.flac");
    let bare_path = tmp("aio_sniff_noext");
    write(&flac_path, &audio).expect("write flac");
    std::fs::rename(&flac_path, &bare_path).expect("rename");

    assert_eq!(FileType::detect(&bare_path), FileType::FLAC);
    let back = read::<_, i16>(&bare_path).expect("read FLAC without extension");
    assert_eq!(back.samples_per_channel(), audio.samples_per_channel());

    std::fs::remove_file(&bare_path).ok();
}

#[test]
fn unreadable_path_falls_back_to_extension() {
    // Nothing to sniff: dispatch must behave exactly as extension-only detection,
    // so the WAV reader is selected and reports the underlying io error.
    let missing = tmp("aio_sniff_does_not_exist.wav");
    std::fs::remove_file(&missing).ok();
    let err = read::<_, i16>(&missing).expect_err("missing file should not read");
    assert!(
        matches!(err, audio_samples_io::AudioIOError::Io(_)),
        "expected Io error, got: {err:?}"
    );
}
