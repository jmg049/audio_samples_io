#![cfg(feature = "flac")]

use std::{
    fs, hint::black_box, io::Cursor, num::NonZeroU32, path::PathBuf, sync::Arc, time::Duration,
};

use audio_samples::{
    AudioSample, AudioSamples, ConvertTo, chirp, cosine_wave, sawtooth_wave, sine_wave,
    square_wave,
    traits::{ConvertFrom, StandardSample},
};
use symphonia::core::{
    codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};
use audio_samples_io::{
    flac::{CompressionLevel, FlacFile, write_flac},
    traits::{AudioFile, AudioFileRead},
    types::OpenOptions,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array2;

const SAMPLE_RATES: &[u32] = &[44_100, 96_000];
const CHANNEL_OPTIONS: &[usize] = &[1, 2, 6];
const ASSET_DIR: &str = "target/bench_assets_flac";
const SIGNAL_DURATION_MS: u64 = 250;

#[derive(Clone)]
struct ReadScenario {
    path: PathBuf,
    bytes: u64,
}

fn bench_flac_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_read");
    configure_group(&mut group);

    for &sample_rate in SAMPLE_RATES {
        for &channels in CHANNEL_OPTIONS {
            let label = case_label(sample_rate, channels);
            let scenario = prepare_read_scenario::<i16>(sample_rate, channels);

            bench_audio_samples_io_read::<i16>(&mut group, &scenario, &label);
            bench_audio_samples_io_read::<i32>(&mut group, &scenario, &label);
            bench_audio_samples_io_read::<f32>(&mut group, &scenario, &label);
            bench_symphonia_read(&mut group, &scenario, &label);
        }
    }

    group.finish();
}

fn bench_flac_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("flac_write");
    configure_group(&mut group);

    for &sample_rate in SAMPLE_RATES {
        for &channels in CHANNEL_OPTIONS {
            let label = case_label(sample_rate, channels);

            // i16
            let audio_i16 = Arc::new(generate_audio::<i16>(sample_rate, channels));
            let payload_i16 = data_payload_bytes(audio_i16.as_ref());
            bench_audio_samples_io_write(&mut group, Arc::clone(&audio_i16), payload_i16, &label);

            // i32 (encoded as 24-bit)
            let audio_i32 = Arc::new(generate_audio::<i32>(sample_rate, channels));
            let payload_i32 = data_payload_bytes(audio_i32.as_ref());
            bench_audio_samples_io_write(&mut group, Arc::clone(&audio_i32), payload_i32, &label);
        }
    }

    group.finish();
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(8));
}

fn prepare_read_scenario<T>(sample_rate: u32, channels: usize) -> ReadScenario
where
    T: StandardSample + 'static,
    f64: ConvertTo<T> + ConvertFrom<T>,
{
    let audio = generate_audio::<T>(sample_rate, channels);
    let path = flac_asset_path::<T>(sample_rate, channels);
    let file = fs::File::create(&path).expect("create flac asset");
    write_flac(file, &audio, CompressionLevel::DEFAULT).expect("write flac asset");
    let bytes = fs::metadata(&path).expect("asset metadata").len();
    ReadScenario { path, bytes }
}

fn bench_audio_samples_io_read<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scenario: &ReadScenario,
    case_label: &str,
) where
    T: StandardSample + 'static,
{
    let bench_id = BenchmarkId::new(format!("audio-{}", T::LABEL), case_label.to_string());
    let path = scenario.path.clone();

    group.throughput(Throughput::Bytes(scenario.bytes));
    group.bench_function(bench_id, move |b| {
        b.iter_batched(
            || FlacFile::open_with_options(&path, OpenOptions::default()).expect("open flac"),
            |flac_file| {
                let samples = flac_file.read::<T>().expect("read flac");
                black_box(samples);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_symphonia_read(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scenario: &ReadScenario,
    case_label: &str,
) {
    let bench_id = BenchmarkId::new("symphonia-i32", case_label.to_string());
    let path = scenario.path.clone();

    group.throughput(Throughput::Bytes(scenario.bytes));
    group.bench_function(bench_id, move |b| {
        b.iter_batched(
            || std::fs::File::open(&path).expect("open flac"),
            |file| {
                let mss = MediaSourceStream::new(Box::new(file), Default::default());
                let mut hint = Hint::new();
                hint.with_extension("flac");
                let mut format = symphonia::default::get_probe()
                    .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
                    .expect("probe flac")
                    .format;
                let track_id = format.default_track().expect("track").id;
                let mut decoder = symphonia::default::get_codecs()
                    .make(
                        &format.default_track().expect("track").codec_params,
                        &DecoderOptions::default(),
                    )
                    .expect("make decoder");
                loop {
                    let packet = match format.next_packet() {
                        Ok(p) if p.track_id() == track_id => p,
                        Ok(_) => continue,
                        Err(_) => break,
                    };
                    black_box(decoder.decode(&packet).expect("decode"));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_audio_samples_io_write<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    audio: Arc<AudioSamples<'static, T>>,
    payload_bytes: u64,
    case_label: &str,
) where
    T: StandardSample + 'static,
{
    let bench_id = BenchmarkId::new(format!("audio-{}", T::LABEL), case_label.to_string());
    let capacity = buffer_capacity(payload_bytes);

    group.throughput(Throughput::Bytes(payload_bytes));
    group.bench_function(bench_id, move |b| {
        let samples = Arc::clone(&audio);
        b.iter_batched(
            || Cursor::new(Vec::with_capacity(capacity)),
            move |writer| {
                write_flac(writer, samples.as_ref(), CompressionLevel::DEFAULT)
                    .expect("write flac");
            },
            BatchSize::SmallInput,
        );
    });
}

fn generate_audio<T>(sample_rate: u32, channels: usize) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
    f64: ConvertTo<T> + ConvertFrom<T>,
{
    let duration = Duration::from_millis(SIGNAL_DURATION_MS);
    let mut planar = Vec::new();
    let mut frames_per_channel = None;

    for channel_idx in 0..channels {
        let mono = channel_signal::<T>(channel_idx, duration, sample_rate);
        let arr = mono.into_array1().expect("generated tone should be mono");
        let len = arr.len();
        if let Some(expected) = frames_per_channel {
            assert_eq!(len, expected, "generator length mismatch");
        } else {
            frames_per_channel = Some(len);
            planar.reserve(len * channels);
        }
        planar.extend(arr.into_iter());
    }

    let frames = frames_per_channel.expect("at least one channel");
    let data = Array2::from_shape_vec((channels, frames), planar)
        .expect("channel stacking should succeed");
    AudioSamples::new_multi_channel(
        data,
        NonZeroU32::new(sample_rate).expect("sample rate must be non-zero"),
    )
    .unwrap()
}

fn channel_signal<T>(
    channel_idx: usize,
    duration: Duration,
    sample_rate: u32,
) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
    f64: ConvertTo<T> + ConvertFrom<T>,
{
    let base_freq = 110.0 + 55.0 * channel_idx as f64;
    let amplitude = 0.35 + 0.1 * (channel_idx % 4) as f64;
    let sample_rate = NonZeroU32::new(sample_rate).expect("sample rate must be non-zero");

    match channel_idx % 5 {
        0 => sine_wave::<T>(base_freq, duration, sample_rate, amplitude),
        1 => cosine_wave::<T>(base_freq * 1.5, duration, sample_rate, amplitude * 0.9),
        2 => square_wave::<T>(base_freq * 0.75, duration, sample_rate, amplitude * 0.8),
        3 => sawtooth_wave::<T>(base_freq * 1.2, duration, sample_rate, amplitude * 0.7),
        _ => chirp::<T>(
            base_freq,
            base_freq * 3.0,
            duration,
            sample_rate,
            amplitude * 0.85,
        ),
    }
}

fn flac_asset_path<T: AudioSample>(sample_rate: u32, channels: usize) -> PathBuf {
    let mut dir = flac_assets_dir();
    dir.push(format!("{}_{}hz_{}ch.flac", T::LABEL, sample_rate, channels));
    dir
}

fn flac_assets_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ASSET_DIR);
    fs::create_dir_all(&dir).expect("Failed to create asset directory");
    dir
}

fn case_label(sample_rate: u32, channels: usize) -> String {
    format!("{sample_rate}hz_{channels}ch")
}

fn data_payload_bytes<T>(audio: &AudioSamples<T>) -> u64
where
    T: StandardSample,
{
    let frames = audio.samples_per_channel().get() as u64;
    let channels = audio.num_channels().get() as u64;
    frames * channels * bytes_per_sample::<T>()
}

fn bytes_per_sample<T>() -> u64
where
    T: StandardSample,
{
    T::BYTES as u64
}

fn buffer_capacity(payload: u64) -> usize {
    payload as usize + 1024
}

criterion_group!(
    name = flac_benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(8))
        .configure_from_args();
    targets = bench_flac_read, bench_flac_write
);
criterion_main!(flac_benches);
