//! Read buffers in every layout: a one-channel buffer from `zeros_multi` is valid for a mono
//! file in every reader, keeps its layout, and a buffer whose channel count or sample rate
//! differs from the file's is an error that leaves the buffer and the stream unchanged.

#![cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]

use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use audio_samples::{AudioSamples, channels, nzu, sample_rate, sine_wave};
use audio_samples_io::traits::{AudioFile, AudioFileRead, AudioStreamRead, AudioStreamReader};

const RATE: u32 = 22_050;
const FRAMES: usize = 5_000;
const BLOCK: NonZeroUsize = nzu!(1024);

fn dir() -> tempfile::TempDir {
    tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).expect("temp dir")
}

/// A mono tone written to `name` in `dir`, and the file read back whole (the reference).
fn mono_file(dir: &Path, name: &str) -> (PathBuf, Vec<f32>) {
    let tone = sine_wave::<f32>(
        440.0,
        Duration::from_secs_f64(FRAMES as f64 / f64::from(RATE)),
        sample_rate!(22050),
        0.5,
    );
    let path = dir.join(name);
    audio_samples_io::write(&path, &tone).expect("write");
    let whole: AudioSamples<'static, f32> = audio_samples_io::read(&path).expect("read");
    assert_eq!(whole.num_channels().get(), 1);
    (path, whole.to_interleaved_vec().into_vec())
}

/// Reads `stream` to the end through a one-channel `zeros_multi` buffer, the documented
/// streaming pattern, and returns the samples.
fn stream_through_multi<S: AudioStreamRead>(stream: &mut S) -> Vec<f32> {
    let channels = NonZeroU32::new(u32::from(AudioStreamReader::num_channels(stream))).unwrap();
    let rate = NonZeroU32::new(AudioStreamReader::sample_rate(stream)).unwrap();
    let mut buffer = AudioSamples::<f32>::zeros_multi(channels, BLOCK, rate);
    let mut out = Vec::new();
    while stream.remaining_frames() > 0 {
        let n = stream.read_frames_into(&mut buffer, BLOCK).expect("read block");
        assert!(n > 0);
        assert!(buffer.is_multi_channel(), "the buffer keeps its layout");
        assert_eq!(buffer.num_channels().get(), 1);
        out.extend(buffer.to_interleaved_vec().iter().take(n).copied());
    }
    out
}

/// The stream rejects buffers with another channel count or sample rate without moving.
fn rejects_mismatched_buffers<S: AudioStreamRead>(stream: &mut S) {
    let before = stream.current_frame();
    let mut stereo = AudioSamples::<f32>::zeros_multi(channels!(2), BLOCK, sample_rate!(22050));
    assert!(stream.read_frames_into(&mut stereo, BLOCK).is_err());
    let mut other_rate = AudioSamples::<f32>::zeros_mono(BLOCK, sample_rate!(44100));
    assert!(stream.read_frames_into(&mut other_rate, BLOCK).is_err());
    assert_eq!(stream.current_frame(), before);
    assert!(stereo.to_interleaved_vec().iter().all(|&x| x == 0.0));
}

#[cfg(feature = "wav")]
#[test]
fn wav_streams_mono_into_one_channel_multi_buffer() {
    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.wav");

    let mut seekable = audio_samples_io::open_streamed(&path).expect("open");
    rejects_mismatched_buffers(&mut seekable);
    assert_eq!(stream_through_multi(&mut seekable), reference);

    let file = std::io::BufReader::new(std::fs::File::open(&path).expect("open"));
    let mut forward = audio_samples_io::StreamedWavFile::new_forward_only(file).expect("parse");
    rejects_mismatched_buffers(&mut forward);
    assert_eq!(stream_through_multi(&mut forward), reference);
}

#[cfg(feature = "flac")]
#[test]
fn flac_streams_mono_into_one_channel_multi_buffer() {
    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.flac");

    let mut stream = audio_samples_io::open_streamed_flac(&path).expect("open");
    rejects_mismatched_buffers(&mut stream);
    assert_eq!(stream_through_multi(&mut stream), reference);
}

#[cfg(feature = "aiff")]
#[test]
fn aiff_streams_mono_into_one_channel_multi_buffer() {
    use audio_samples_io::aiff::StreamedAiffFile;

    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.aiff");

    let file = std::io::BufReader::new(std::fs::File::open(&path).expect("open"));
    let mut seekable = StreamedAiffFile::new(file).expect("parse");
    rejects_mismatched_buffers(&mut seekable);
    assert_eq!(stream_through_multi(&mut seekable), reference);

    let file = std::io::BufReader::new(std::fs::File::open(&path).expect("open"));
    let mut forward = StreamedAiffFile::new_forward_only(file).expect("parse");
    rejects_mismatched_buffers(&mut forward);
    assert_eq!(stream_through_multi(&mut forward), reference);
}

/// Whole-file `read_into` fills a one-channel `zeros_multi` buffer and rejects one with
/// another channel count or sample rate.
fn whole_file_read_into<'a, F>(file: &'a F, reference: &[f32])
where
    F: AudioFileRead<'a>,
{
    let frames = NonZeroUsize::new(reference.len()).unwrap();
    let mut buffer = AudioSamples::<f32>::zeros_multi(channels!(1), frames, sample_rate!(22050));
    file.read_into(&mut buffer).expect("read_into");
    assert!(buffer.is_multi_channel());
    assert_eq!(buffer.to_interleaved_vec().as_slice(), reference);

    let half = NonZeroUsize::new(reference.len() / 2).unwrap();
    let mut stereo = AudioSamples::<f32>::zeros_multi(channels!(2), half, sample_rate!(22050));
    assert!(file.read_into(&mut stereo).is_err());
    let mut other_rate = AudioSamples::<f32>::zeros_mono(frames, sample_rate!(44100));
    assert!(file.read_into(&mut other_rate).is_err());
}

#[cfg(feature = "wav")]
#[test]
fn wav_read_into_one_channel_multi_buffer() {
    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.wav");
    let file = audio_samples_io::WavFile::open(&path).expect("open");
    whole_file_read_into(&file, &reference);
}

#[cfg(feature = "flac")]
#[test]
fn flac_read_into_one_channel_multi_buffer() {
    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.flac");
    let file = audio_samples_io::flac::FlacFile::open(&path).expect("open");
    whole_file_read_into(&file, &reference);
}

#[cfg(feature = "aiff")]
#[test]
fn aiff_read_into_one_channel_multi_buffer() {
    let dir = dir();
    let (path, reference) = mono_file(dir.path(), "mono.aiff");
    let file = audio_samples_io::AiffFile::open(&path).expect("open");
    whole_file_read_into(&file, &reference);
}

/// A stereo file read into a mono buffer was laid out as one long channel; it is an error.
#[cfg(feature = "wav")]
#[test]
fn stereo_file_into_mono_buffer_is_an_error() {
    let dir = dir();
    let path = dir.path().join("stereo.wav");
    let stereo = AudioSamples::<f32>::zeros_multi(channels!(2), nzu!(2048), sample_rate!(22050));
    audio_samples_io::write(&path, &stereo).expect("write");
    let mut stream = audio_samples_io::open_streamed(&path).expect("open");
    let mut mono = AudioSamples::<f32>::zeros_mono(BLOCK, sample_rate!(22050));
    assert!(stream.read_frames_into(&mut mono, BLOCK).is_err());
    assert_eq!(mono.samples_per_channel(), BLOCK);
}
