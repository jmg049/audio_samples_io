//! `StreamedAiffFile` incremental read tests.

#![cfg(feature = "aiff")]

use std::io::{BufReader, Cursor};
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::{AudioSamples, nzu, sample_rate, sine_wave};
use audio_samples_io::aiff::StreamedAiffFile;
use audio_samples_io::write;

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn streamed_aiff_matches_full_decode() {
    let sr = sample_rate!(44100);
    let audio: AudioSamples<'static, f32> = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.25), sr, 0.5);
    let path = tmp("aio_aiff_stream_full.aiff");
    write(&path, &audio).expect("write aiff");

    let mut streamed = StreamedAiffFile::new(BufReader::new(std::fs::File::open(&path).expect("open"))).expect("parse");
    assert_eq!(streamed.sample_rate(), 44100);
    assert_eq!(streamed.num_channels(), 1);
    assert_eq!(streamed.total_frames(), audio.samples_per_channel().get());

    let stream_sr = std::num::NonZeroU32::new(streamed.sample_rate()).unwrap();
    let mut buffer = AudioSamples::<f32>::zeros_mono(nzu!(512), stream_sr);
    let mut collected = Vec::new();
    loop {
        let n = streamed.read_frames_into(&mut buffer, nzu!(512)).expect("read chunk");
        if n == 0 {
            break;
        }
        let mono = buffer.as_mono().expect("mono");
        collected.extend(mono.iter().take(n).copied());
    }

    let expected: Vec<f32> = audio.to_interleaved_vec().into_iter().collect();
    assert_eq!(collected, expected, "chunked read must match a full decode");
    assert_eq!(streamed.remaining_frames(), 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn streamed_aiff_partial_final_chunk() {
    let sr = sample_rate!(44100);
    // Deliberately not a multiple of the read chunk size (512), so the last
    // `read_frames_into` call returns a short read.
    let frames = 1000usize;
    let samples: Vec<f32> = (0..frames).map(|i| (i as f32 / frames as f32) - 0.5).collect();
    let audio = AudioSamples::<f32>::new_mono(ndarray::Array1::from_vec(samples), sr).unwrap();
    let path = tmp("aio_aiff_stream_partial.aiff");
    write(&path, &audio).expect("write aiff");

    let mut streamed = StreamedAiffFile::new(BufReader::new(std::fs::File::open(&path).expect("open"))).expect("parse");
    assert_eq!(streamed.total_frames(), frames);

    let mut buffer = AudioSamples::<f32>::zeros_mono(nzu!(512), sr);
    let first = streamed.read_frames_into(&mut buffer, nzu!(512)).expect("first chunk");
    assert_eq!(first, 512);

    let second = streamed.read_frames_into(&mut buffer, nzu!(512)).expect("second chunk");
    assert_eq!(second, frames - 512, "final chunk must report the true short count");

    let third = streamed.read_frames_into(&mut buffer, nzu!(512)).expect("exhausted");
    assert_eq!(third, 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn streamed_aiff_seek_and_reset() {
    let sr = sample_rate!(44100);
    let frames = 2048usize;
    let samples: Vec<f32> = (0..frames).map(|i| i as f32).collect();
    let audio = AudioSamples::<f32>::new_mono(ndarray::Array1::from_vec(samples), sr).unwrap();
    let path = tmp("aio_aiff_stream_seek.aiff");
    write(&path, &audio).expect("write aiff");

    let mut streamed = StreamedAiffFile::new(BufReader::new(std::fs::File::open(&path).expect("open"))).expect("parse");

    streamed.seek_to_frame(1000).expect("seek");
    assert_eq!(streamed.current_frame(), 1000);

    let mut buffer = AudioSamples::<f32>::zeros_mono(nzu!(4), sr);
    streamed
        .read_frames_into(&mut buffer, nzu!(4))
        .expect("read after seek");
    let mono = buffer.as_mono().expect("mono");
    assert_eq!(mono[0], 1000.0);

    streamed.reset().expect("reset");
    assert_eq!(streamed.current_frame(), 0);
    streamed
        .read_frames_into(&mut buffer, nzu!(4))
        .expect("read after reset");
    let mono = buffer.as_mono().expect("mono");
    assert_eq!(mono[0], 0.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn streamed_aiff_stereo_channel_order() {
    use std::num::NonZeroU32;

    use ndarray::Array2;

    let frames = 300usize;
    let mut planar = Vec::with_capacity(frames * 2);
    planar.extend((0..frames).map(|i| i as f32));
    planar.extend((0..frames).map(|i| -(i as f32)));
    let data = Array2::from_shape_vec((2, frames), planar).expect("shape");
    let audio = AudioSamples::new_multi_channel(data, NonZeroU32::new(44_100).unwrap()).unwrap();

    let path = tmp("aio_aiff_stream_stereo.aiff");
    write(&path, &audio).expect("write");

    let mut streamed = StreamedAiffFile::new(BufReader::new(std::fs::File::open(&path).expect("open"))).expect("parse");
    assert_eq!(streamed.num_channels(), 2);

    let mut buffer = AudioSamples::<f32>::zeros_multi(
        NonZeroU32::new(2).unwrap(),
        NonZeroUsize::new(frames).unwrap(),
        sample_rate!(44100),
    );
    let n = streamed
        .read_frames_into(&mut buffer, NonZeroUsize::new(frames).unwrap())
        .expect("read all");
    assert_eq!(n, frames);

    let multi = buffer.as_multi_channel().expect("multi");
    assert_eq!(multi[[0, 1]], 1.0);
    assert_eq!(multi[[1, 1]], -1.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn streamed_aiff_reads_from_in_memory_cursor() {
    let sr = sample_rate!(22050);
    let audio: AudioSamples<'static, i16> = sine_wave::<i16>(220.0, Duration::from_secs_f64(0.05), sr, 0.5);

    let mut bytes = Vec::new();
    audio_samples_io::aiff::write_aiff(&mut bytes, &audio).expect("write to memory");

    let mut streamed = StreamedAiffFile::new(Cursor::new(bytes)).expect("parse from cursor");
    assert_eq!(streamed.sample_rate(), 22050);

    let mut buffer = AudioSamples::<i16>::zeros_mono(nzu!(256), sr);
    let mut total = 0usize;
    while streamed.read_frames_into(&mut buffer, nzu!(256)).expect("read") > 0 {
        total += 1;
        if total > 1000 {
            panic!("read loop did not terminate");
        }
    }
    assert_eq!(streamed.remaining_frames(), 0);
}
