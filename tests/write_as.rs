//! `write_as` and `write_flac_file_as` give the file that converting first and then writing gives.

#![cfg(all(feature = "wav", feature = "flac"))]

use std::num::NonZeroUsize;

use audio_samples::{AudioSamples, AudioTypeConversion, I24, sample_rate};
use audio_samples_io::{CompressionLevel, WriteOptions, write_as, write_flac_file_as, write_with_options};
use ndarray::{Array1, Array2};

fn stereo() -> AudioSamples<'static, f64> {
    let data = Array2::from_shape_fn((2, 30_000), |(c, i)| ((i * (c + 3)) as f64 * 0.013).sin() * 0.7);
    AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap()
}

fn mono() -> AudioSamples<'static, f32> {
    let data: Array1<f32> = (0..30_001).map(|i| (i as f32 * 0.011).sin() * 0.9).collect();
    AudioSamples::new_mono(data, sample_rate!(22050)).unwrap()
}

macro_rules! same_file_as_converting_first {
    ($U:ty, $ext:literal, $audio:expr) => {{
        let dir = tempfile::tempdir().unwrap();
        let direct = dir.path().join(concat!("direct.", $ext));
        let converted = dir.path().join(concat!("converted.", $ext));
        write_as::<$U, _, _>(&direct, $audio, WriteOptions::default()).unwrap();
        write_with_options(&converted, &$audio.to_format::<$U>(), WriteOptions::default()).unwrap();
        assert_eq!(std::fs::read(&direct).unwrap(), std::fs::read(&converted).unwrap());
    }};
}

#[test]
fn wav_matches_converting_first() {
    let (s, m) = (stereo(), mono());
    same_file_as_converting_first!(i16, "wav", &s);
    same_file_as_converting_first!(I24, "wav", &s);
    same_file_as_converting_first!(i32, "wav", &s);
    same_file_as_converting_first!(f32, "wav", &s);
    same_file_as_converting_first!(u8, "wav", &s);
    same_file_as_converting_first!(f64, "wav", &s);
    same_file_as_converting_first!(i16, "wav", &m);
    same_file_as_converting_first!(f32, "wav", &m);
}

#[test]
fn flac_matches_converting_first() {
    let (s, m) = (stereo(), mono());
    same_file_as_converting_first!(i16, "flac", &s);
    same_file_as_converting_first!(I24, "flac", &s);
    same_file_as_converting_first!(i16, "flac", &m);
    same_file_as_converting_first!(I24, "flac", &m);
}

#[test]
fn flac_without_md5_differs_only_in_the_signature() {
    let dir = tempfile::tempdir().unwrap();
    let (with, without) = (dir.path().join("with.flac"), dir.path().join("without.flac"));
    let audio = stereo();
    let level = CompressionLevel::default();
    write_flac_file_as::<i16, _, _>(&with, &audio, WriteOptions::default(), level, true).unwrap();
    write_flac_file_as::<i16, _, _>(&without, &audio, WriteOptions::default(), level, false).unwrap();
    let (with, without) = (std::fs::read(with).unwrap(), std::fs::read(without).unwrap());
    const SIGNATURE: std::ops::Range<usize> = 26..42;
    assert_ne!(with[SIGNATURE], [0; 16]);
    assert_eq!(without[SIGNATURE], [0; 16]);
    assert_eq!(with[..SIGNATURE.start], without[..SIGNATURE.start]);
    assert_eq!(with[SIGNATURE.end..], without[SIGNATURE.end..]);
}

/// The thread limit belongs to the process, and the tests of this file run at the same time.
static THREAD_LIMIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn thread_limit_replaces_the_default() {
    let _guard = THREAD_LIMIT.lock().unwrap();
    audio_samples_io::set_thread_limit(NonZeroUsize::new(3));
    assert_eq!(audio_samples_io::thread_limit(), 3);
    audio_samples_io::set_thread_limit(None);
    assert!(audio_samples_io::thread_limit() >= 1);
}

/// Large enough for the mapped write on tmpfs, which has its own conversion path.
#[cfg(target_os = "linux")]
#[test]
fn write_as_on_tmpfs_matches_converting_first() {
    let Ok(dir) = tempfile::tempdir_in("/dev/shm") else {
        return;
    };
    let _guard = THREAD_LIMIT.lock().unwrap();
    audio_samples_io::set_thread_limit(NonZeroUsize::new(4));
    let data = Array2::from_shape_fn((2, 12_000_000), |(c, i)| (((i + c) % 997) as f32 / 997.0) - 0.5);
    let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
    let direct = dir.path().join("direct.wav");
    let converted = dir.path().join("converted.wav");
    write_as::<i16, _, _>(&direct, &audio, WriteOptions::default()).unwrap();
    write_with_options(&converted, &audio.to_format::<i16>(), WriteOptions::default()).unwrap();
    audio_samples_io::set_thread_limit(None);
    assert!(std::fs::read(&direct).unwrap() == std::fs::read(&converted).unwrap());
}
