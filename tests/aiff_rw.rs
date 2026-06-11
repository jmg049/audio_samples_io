//! AIFF/AIFF-C read + write round-trips.

#![cfg(feature = "aiff")]

use std::io::Write;
use std::time::Duration;

use audio_samples::{AudioSamples, I24, sample_rate, sine_wave, traits::StandardSample};
use audio_samples_io::types::FileType;
use audio_samples_io::{info, peek_native_type, read, write};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

fn roundtrip<T>(name: &str)
where
    T: StandardSample + std::fmt::Debug + PartialEq + 'static,
{
    let sr = sample_rate!(44100);
    let audio: AudioSamples<'static, T> = sine_wave::<T>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);

    let path = tmp(name);
    write(&path, &audio).expect("write aiff");

    let back = read::<_, T>(&path).expect("read aiff");
    assert_eq!(back.sample_rate(), sr);
    assert_eq!(back.samples_per_channel(), audio.samples_per_channel());

    let original: Vec<T> = audio.to_interleaved_vec().into_iter().collect();
    let decoded: Vec<T> = back.to_interleaved_vec().into_iter().collect();
    assert_eq!(original, decoded, "lossless native round-trip");

    std::fs::remove_file(&path).ok();
}

#[test]
fn aiff_round_trips_every_sample_type() {
    roundtrip::<u8>("aio_aiff_rt_u8.aiff");
    roundtrip::<i16>("aio_aiff_rt_i16.aiff");
    roundtrip::<I24>("aio_aiff_rt_i24.aiff");
    roundtrip::<i32>("aio_aiff_rt_i32.aiff");
    roundtrip::<f32>("aio_aiff_rt_f32.aiff");
    roundtrip::<f64>("aio_aiff_rt_f64.aiff");
}

#[test]
fn aiff_reports_metadata() {
    let sr = sample_rate!(48000);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.1), sr, 0.5);
    let path = tmp("aio_aiff_info.aiff");
    write(&path, &audio).expect("write");

    assert_eq!(FileType::detect(&path), FileType::AIFF);
    assert_eq!(
        peek_native_type(&path).expect("peek"),
        audio_samples_io::ValidatedSampleType::I16
    );

    let base = info(&path).expect("info");
    assert_eq!(base.file_type, FileType::AIFF);
    assert_eq!(base.sample_rate.get(), 48_000);
    assert_eq!(base.channels, 1);
    assert_eq!(base.bits_per_sample, 16);
    assert_eq!(base.total_samples, audio.samples_per_channel().get());

    std::fs::remove_file(&path).ok();
}

#[test]
fn aiff_cross_type_read_converts() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<i16>(440.0, Duration::from_secs_f64(0.05), sr, 0.5);
    let path = tmp("aio_aiff_cross.aiff");
    write(&path, &audio).expect("write");

    let as_f32 = read::<_, f32>(&path).expect("read as f32");
    assert_eq!(as_f32.samples_per_channel(), audio.samples_per_channel());
    let max = as_f32
        .to_interleaved_vec()
        .into_iter()
        .fold(0.0f32, |acc, s| acc.max(s.abs()));
    assert!((0.4..=0.6).contains(&max), "expected ~0.5 amplitude, got {max}");

    std::fs::remove_file(&path).ok();
}

#[test]
fn aiff_stereo_preserves_channel_order() {
    use std::num::NonZeroU32;

    use ndarray::Array2;

    let frames = 256usize;
    let mut planar = Vec::with_capacity(frames * 2);
    planar.extend((0..frames).map(|i| (i as i16) * 10)); // left ramp
    planar.extend((0..frames).map(|i| -(i as i16) * 10)); // right ramp, negated
    let data = Array2::from_shape_vec((2, frames), planar).expect("shape");
    let audio = AudioSamples::new_multi_channel(data, NonZeroU32::new(44_100).expect("nz")).expect("audio");

    let path = tmp("aio_aiff_stereo.aiff");
    write(&path, &audio).expect("write");

    let back = read::<_, i16>(&path).expect("read");
    assert_eq!(back.num_channels().get(), 2);
    let iv: Vec<i16> = back.to_interleaved_vec().into_iter().collect();
    assert_eq!(iv[0], 0);
    assert_eq!(iv[1], 0);
    assert_eq!(iv[2], 10); // L frame 1
    assert_eq!(iv[3], -10); // R frame 1

    std::fs::remove_file(&path).ok();
}

/// Hand-crafted AIFF-C with `sowt` (little-endian) sound data, as written by
/// Apple tools: same COMM layout plus compression fields, LE payload.
#[test]
fn aifc_sowt_little_endian_reads_correctly() {
    let samples: Vec<i16> = (0..64).map(|i| i * 100).collect();
    let channels = 1u16;
    let data_size = samples.len() * 2;

    let mut f = Vec::new();
    f.extend_from_slice(b"FORM");
    let comm_len = 18 + 4 + 2; // base + compression code + empty pstring
    let form_size = 4 + (8 + 4) + (8 + comm_len) + (8 + 8 + data_size);
    f.extend_from_slice(&(form_size as u32).to_be_bytes());
    f.extend_from_slice(b"AIFC");

    f.extend_from_slice(b"FVER");
    f.extend_from_slice(&4u32.to_be_bytes());
    f.extend_from_slice(&0xA280_5140u32.to_be_bytes());

    f.extend_from_slice(b"COMM");
    f.extend_from_slice(&(comm_len as u32).to_be_bytes());
    f.extend_from_slice(&channels.to_be_bytes());
    f.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    f.extend_from_slice(&16u16.to_be_bytes());
    f.extend_from_slice(&audio_samples_io::aiff::encode_extended(22_050.0));
    f.extend_from_slice(b"sowt");
    f.extend_from_slice(&[0u8, 0u8]);

    f.extend_from_slice(b"SSND");
    f.extend_from_slice(&((8 + data_size) as u32).to_be_bytes());
    f.extend_from_slice(&0u32.to_be_bytes());
    f.extend_from_slice(&0u32.to_be_bytes());
    for s in &samples {
        f.extend_from_slice(&s.to_le_bytes()); // sowt: little-endian
    }

    let path = tmp("aio_aifc_sowt.aiff");
    std::fs::File::create(&path)
        .and_then(|mut file| file.write_all(&f))
        .expect("write fixture");

    let back = read::<_, i16>(&path).expect("read sowt");
    assert_eq!(back.sample_rate().get(), 22_050);
    let decoded: Vec<i16> = back.to_interleaved_vec().into_iter().collect();
    assert_eq!(decoded, samples);

    std::fs::remove_file(&path).ok();
}

#[test]
fn aiff_float_write_produces_aifc() {
    let sr = sample_rate!(44100);
    let audio = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.02), sr, 0.5);
    let path = tmp("aio_aiff_fl32.aiff");
    write(&path, &audio).expect("write");

    let bytes = std::fs::read(&path).expect("read raw");
    assert_eq!(&bytes[8..12], b"AIFC", "float audio must use the AIFF-C form");
    assert_eq!(&bytes[12..16], b"FVER", "AIFF-C requires the FVER chunk first");

    std::fs::remove_file(&path).ok();
}
