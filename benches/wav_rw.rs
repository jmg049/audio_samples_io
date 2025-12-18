use std::{
    any::TypeId, fs, hint::black_box, io::Cursor, num::NonZeroU32, path::PathBuf, sync::Arc,
    time::Duration,
};

use audio_io::{
    self,
    traits::{AudioFile, AudioFileRead},
    types::OpenOptions,
    wav::wav_file::WavFile,
};
use audio_samples::{
    AudioSample, AudioSamples, ConvertTo, I24, chirp, cosine_wave, sawtooth_wave, sine_wave,
    square_wave,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hound::{Sample, SampleFormat, WavReader, WavSpec};
use ndarray::Array2;

const SAMPLE_RATES: &[u32] = &[44_100, 96_000];
const CHANNEL_OPTIONS: &[usize] = &[1, 2, 6];
const ASSET_DIR: &str = "target/bench_assets";
const SIGNAL_DURATION_MS: u64 = 250;

#[derive(Clone)]
struct ReadScenario {
    path: PathBuf,
    bytes: u64,
}

fn bench_wav_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_read");
    configure_group(&mut group);

    for &sample_rate in SAMPLE_RATES {
        for &channels in CHANNEL_OPTIONS {
            bench_read_case_with_hound::<i16>(&mut group, sample_rate, channels);
            bench_read_case_audio_only::<I24>(&mut group, sample_rate, channels);
            bench_read_case_with_hound::<i32>(&mut group, sample_rate, channels);
            bench_read_case_with_hound::<f32>(&mut group, sample_rate, channels);
            bench_read_case_audio_only::<f64>(&mut group, sample_rate, channels);
        }
    }

    group.finish();
}

fn bench_wav_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("wav_write");
    configure_group(&mut group);

    for &sample_rate in SAMPLE_RATES {
        for &channels in CHANNEL_OPTIONS {
            bench_write_case_with_hound::<i16>(&mut group, sample_rate, channels);
            bench_write_case_audio_only::<I24>(&mut group, sample_rate, channels);
            bench_write_case_with_hound::<i32>(&mut group, sample_rate, channels);
            bench_write_case_with_hound::<f32>(&mut group, sample_rate, channels);
            bench_write_case_audio_only::<f64>(&mut group, sample_rate, channels);
        }
    }

    group.finish();
}

fn bench_read_case_with_hound<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    sample_rate: u32,
    channels: usize,
) where
    T: AudioSample + Sample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let scenario = prepare_read_scenario::<T>(sample_rate, channels);
    let label = case_label(sample_rate, channels);

    bench_audio_io_read::<T>(group, &scenario, &label);
    if channels <= 2 {
        bench_hound_read_impl::<T>(group, &scenario, &label);
    }
}

fn bench_read_case_audio_only<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    sample_rate: u32,
    channels: usize,
) where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let scenario = prepare_read_scenario::<T>(sample_rate, channels);
    let label = case_label(sample_rate, channels);
    bench_audio_io_read::<T>(group, &scenario, &label);
}

fn bench_write_case_with_hound<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    sample_rate: u32,
    channels: usize,
) where
    T: AudioSample + Sample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let audio = Arc::new(generate_audio::<T>(sample_rate, channels));
    let payload_bytes = data_payload_bytes(audio.as_ref());
    let label = case_label(sample_rate, channels);

    bench_audio_io_write(group, Arc::clone(&audio), payload_bytes, &label);
    if channels <= 2 {
        let interleaved = Arc::new(audio.to_interleaved_vec());
        bench_hound_write_impl::<T>(
            group,
            interleaved,
            payload_bytes,
            sample_rate,
            channels,
            &label,
        );
    }
}

fn bench_write_case_audio_only<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    sample_rate: u32,
    channels: usize,
) where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let audio = Arc::new(generate_audio::<T>(sample_rate, channels));
    let payload_bytes = data_payload_bytes(audio.as_ref());
    let label = case_label(sample_rate, channels);
    bench_audio_io_write(group, audio, payload_bytes, &label);
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(8));
}

fn prepare_read_scenario<T>(sample_rate: u32, channels: usize) -> ReadScenario
where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let audio = generate_audio::<T>(sample_rate, channels);
    let path = asset_path::<T>(sample_rate, channels);
    audio_io::write(&path, &audio).expect("failed to create wav asset");
    let bytes = fs::metadata(&path).expect("asset metadata").len();
    ReadScenario { path, bytes }
}

fn bench_audio_io_read<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scenario: &ReadScenario,
    case_label: &str,
) where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let bench_id = BenchmarkId::new(format!("audio-{}", T::LABEL), case_label.to_string());
    let path = scenario.path.clone();

    group.throughput(Throughput::Bytes(scenario.bytes));
    group.bench_function(bench_id, move |b| {
        b.iter_batched(
            || WavFile::open_with_options(&path, OpenOptions::default()).expect("open wav"),
            |wav_file| {
                let samples = wav_file.read::<T>().expect("read wav");
                black_box(samples);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_hound_read_impl<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scenario: &ReadScenario,
    case_label: &str,
) where
    T: AudioSample + Sample,
{
    let bench_id = BenchmarkId::new(format!("hound-{}", T::LABEL), case_label.to_string());
    let path = scenario.path.clone();

    group.throughput(Throughput::Bytes(scenario.bytes));
    group.bench_function(bench_id, move |b| {
        b.iter_batched(
            || WavReader::open(&path).expect("open wav"),
            |mut reader| {
                let samples: Vec<T> = reader
                    .samples::<T>()
                    .map(|result| result.expect("hound read"))
                    .collect();
                black_box(samples);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_audio_io_write<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    audio: Arc<AudioSamples<'static, T>>,
    payload_bytes: u64,
    case_label: &str,
) where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let bench_id = BenchmarkId::new(format!("audio-{}", T::LABEL), case_label.to_string());
    let capacity = buffer_capacity(payload_bytes);

    group.throughput(Throughput::Bytes(payload_bytes));
    group.bench_function(bench_id, move |b| {
        let samples = Arc::clone(&audio);
        b.iter_batched(
            || Cursor::new(Vec::with_capacity(capacity)),
            move |writer| {
                audio_io::write_with(writer, samples.as_ref(), audio_io::types::FileType::WAV)
                    .expect("write wav");
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_hound_write_impl<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    samples: Arc<Vec<T>>,
    payload_bytes: u64,
    sample_rate: u32,
    channels: usize,
    case_label: &str,
) where
    T: AudioSample + Sample + 'static,
{
    let bench_id = BenchmarkId::new(format!("hound-{}", T::LABEL), case_label.to_string());
    let capacity = buffer_capacity(payload_bytes);
    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: T::BITS as u16,
        sample_format: sample_format_for::<T>(),
    };

    group.throughput(Throughput::Bytes(payload_bytes));
    group.bench_function(bench_id, move |b| {
        let signal = Arc::clone(&samples);
        b.iter_batched(
            || Cursor::new(Vec::with_capacity(capacity)),
            move |writer| {
                let mut wav_writer = hound::WavWriter::new(writer, spec).expect("hound writer");
                for sample in signal.iter() {
                    wav_writer.write_sample(*sample).expect("hound write");
                }
                wav_writer.finalize().expect("finalize");
            },
            BatchSize::SmallInput,
        );
    });
}

fn sample_format_for<T: 'static>() -> SampleFormat {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        SampleFormat::Float
    } else {
        SampleFormat::Int
    }
}

fn generate_audio<T>(sample_rate: u32, channels: usize) -> AudioSamples<'static, T>
where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
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
}

fn channel_signal<T>(
    channel_idx: usize,
    duration: Duration,
    sample_rate: u32,
) -> AudioSamples<'static, T>
where
    T: AudioSample + 'static,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let base_freq = 110.0 + 55.0 * channel_idx as f64;
    let amplitude = 0.35 + 0.1 * (channel_idx % 4) as f64;
    match channel_idx % 5 {
        0 => sine_wave::<T, f64>(base_freq, duration, sample_rate, amplitude),
        1 => cosine_wave::<T, f64>(base_freq * 1.5, duration, sample_rate, amplitude * 0.9),
        2 => square_wave::<T, f64>(base_freq * 0.75, duration, sample_rate, amplitude * 0.8),
        3 => sawtooth_wave::<T, f64>(base_freq * 1.2, duration, sample_rate, amplitude * 0.7),
        _ => chirp::<T, f64>(
            base_freq,
            base_freq * 3.0,
            duration,
            sample_rate,
            amplitude * 0.85,
        ),
    }
}

fn asset_path<T: AudioSample>(sample_rate: u32, channels: usize) -> PathBuf {
    let mut dir = assets_dir();
    dir.push(format!("{}_{}hz_{}ch.wav", T::LABEL, sample_rate, channels));
    dir
}

fn assets_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(ASSET_DIR);
    fs::create_dir_all(&dir).expect("Failed to create asset directory");
    dir
}

fn case_label(sample_rate: u32, channels: usize) -> String {
    format!("{}hz_{}ch", sample_rate, channels)
}

fn data_payload_bytes<T: AudioSample>(audio: &AudioSamples<T>) -> u64 {
    let frames = audio.samples_per_channel() as u64;
    let channels = audio.num_channels() as u64;
    frames * channels * bytes_per_sample::<T>()
}

fn bytes_per_sample<T: AudioSample>() -> u64 {
    T::BYTES as u64
}

fn buffer_capacity(payload: u64) -> usize {
    payload as usize + 1024
}

criterion_group!(
    name = wav_benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(8))
        .configure_from_args();
    targets =  bench_wav_read, bench_wav_write
);
criterion_main!(wav_benches);
