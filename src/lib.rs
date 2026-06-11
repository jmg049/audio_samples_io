pub mod error;
pub mod traits;
pub mod types;

#[cfg(feature = "wav")]
pub mod wav;

#[cfg(feature = "wav")]
pub use crate::wav::{
    StreamedWavFile, StreamedWavWriter, build_wav_header, wav_data_len, wav_file::WavFile, wav_file_len, wav_header_len,
};

#[cfg(feature = "flac")]
pub mod flac;

#[cfg(feature = "flac")]
pub use crate::flac::{CompressionLevel, StreamedFlacFile, StreamedFlacWriter};

#[cfg(any(feature = "wav", feature = "flac"))]
pub mod streaming;
#[cfg(any(feature = "wav", feature = "flac"))]
pub use crate::streaming::StreamedAudioWriter;

#[cfg(feature = "numpy")]
pub mod python;

#[cfg(feature = "resampling")]
use std::num::NonZeroU32;
use std::{
    any::TypeId,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};

#[cfg(feature = "resampling")]
pub use audio_samples::operations::ResamplingQuality;
#[cfg(feature = "resampling")]
pub use audio_samples::operations::resample;
use audio_samples::{AudioSamples, traits::StandardSample};

#[cfg(feature = "numpy")]
pub use crate::python::read_pyarray;
#[cfg(all(feature = "numpy", target_endian = "little"))]
pub use crate::python::{NativeAudioArray, read_pyarray_native};
pub use crate::{
    error::{AudioIOError, AudioIOResult},
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioStreamReader},
    types::{BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType, WriteOptions},
};

pub(crate) const MAX_WAV_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GB limit
pub(crate) const MAX_MMAP_SIZE: u64 = 512 * 1024 * 1024; // 512MB for memory mapping

/// Convenience trait for types that implement both Read and Seek
pub trait ReadSeek: Read + Seek {}

impl<RS> ReadSeek for RS where RS: Read + Seek {}

pub trait WriteSeek: Write + Seek {}

impl<WS> WriteSeek for WS where WS: Write + Seek {}

// Public API

/// Peek at the native sample type of an audio file with minimal I/O.
///
/// Uses a small buffered read (one syscall for the first 64 KB covers any WAV/FLAC header)
/// instead of mmapping the entire file.  This is significantly cheaper than [`info`] when
/// only the sample type is needed, such as when auto-detecting the read target type.
pub fn peek_native_type<P: AsRef<Path>>(fp: P) -> AudioIOResult<ValidatedSampleType> {
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to peek WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                use crate::wav::wav_file::parse_wav_header_streaming;
                let file = File::open(path)?;
                let mut reader = BufReader::with_capacity(65536, file);
                let (info, _) = parse_wav_header_streaming(&mut reader)?;
                ValidatedSampleType::try_from(info.sample_type).map_err(|_| {
                    AudioIOError::unsupported_format(format!("Unsupported native sample type: {:?}", info.sample_type))
                })
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to peek FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                use crate::flac::FlacFile;
                use crate::traits::{AudioFile, AudioFileMetadata};
                let flac_file = FlacFile::open_with_options(path, OpenOptions::default())?;
                let info = flac_file.base_info()?;
                ValidatedSampleType::try_from(info.sample_type).map_err(|_| {
                    AudioIOError::unsupported_format(format!("Unsupported native sample type: {:?}", info.sample_type))
                })
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "peek_native_type does not support: {other:?}"
        ))),
    }
}

/// Get basic audio information from a file
///
/// Automatically detects the file format and extracts metadata.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn info<P: AsRef<Path>>(fp: P) -> AudioIOResult<BaseAudioInfo> {
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to read WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                let wav_file = WavFile::open_metadata(path)?;
                wav_file.base_info()
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to read FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                use crate::flac::FlacFile;
                use crate::traits::AudioFileMetadata;
                let flac_file = FlacFile::open_metadata(path)?;
                flac_file.base_info()
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {other:?}"
        ))),
    }
}

/// Read audio samples from a file
///
/// Returns owned AudioSamples containing all audio data from the file.
/// Automatically detects file format and handles sample type conversion.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn read<P, T>(fp: P) -> AudioIOResult<AudioSamples<'static, T>>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to read WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                let wav_file = WavFile::open_with_options(path, OpenOptions::default())?;
                let samples = wav_file.read::<T>()?;
                Ok(samples.into_owned())
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to read FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                use crate::flac::FlacFile;
                use crate::traits::{AudioFile, AudioFileRead};
                let flac_file = FlacFile::open_with_options(path, OpenOptions::default())?;
                let samples = flac_file.read::<T>()?;
                Ok(samples.into_owned())
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {other:?}"
        ))),
    }
}

#[cfg(feature = "resampling")]
pub fn read_and_resample<P, T>(
    fp: P,
    target_sr: NonZeroU32,
    quality: Option<ResamplingQuality>,
) -> AudioIOResult<AudioSamples<'static, T>>
where
    P: AsRef<Path>,
    T: StandardSample,
{
    let signal = read(fp)?;
    resample::<T>(&signal, target_sr, quality.unwrap_or(ResamplingQuality::Fast)).map_err(AudioIOError::AudioSamples)
}

/// Open a WAV file for streaming reads.
///
/// Unlike `read()` which loads the entire file, this opens the file for incremental reading,
/// parsing only the header initially. Returns a concrete [`StreamedWavFile`] for full API access.
///
/// For format-agnostic streaming use [`open_streamed_dyn`], which returns a trait object.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::open_streamed;
/// use audio_samples_io::traits::AudioFileMetadata;
/// use audio_samples::{AudioSamples, nzu};
/// use std::num::NonZeroU32;
///
/// let mut streamed = open_streamed("large_file.wav")?;
/// let channels = NonZeroU32::new(streamed.num_channels() as u32).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("channels must be non-zero".to_string()))?;
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("sample_rate must be non-zero".to_string()))?;
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, nzu!(1024), sample_rate);
///
/// while streamed.remaining_frames() > 0 {
///     let frames = streamed.read_frames_into(&mut buffer, nzu!(1024))?;
///     // Process frames...
/// }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn open_streamed<P>(fp: P) -> AudioIOResult<StreamedWavFile<BufReader<File>>>
where
    P: AsRef<Path>,
{
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            StreamedWavFile::new_with_path(reader, path.to_path_buf())
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format for streaming: {other:?}"
        ))),
    }
}

/// Open any `Read + Seek` source for streaming WAV reads.
///
/// This allows streaming from any source implementing `Read + Seek`,
/// such as network streams with range request support, in-memory cursors,
/// or custom I/O implementations.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::open_streamed_reader;
/// use std::io::Cursor;
///
/// let wav_bytes: Vec<u8> = load_from_network();
/// let cursor = Cursor::new(wav_bytes);
/// let mut streamed = open_streamed_reader(cursor)?;
/// # fn load_from_network() -> Vec<u8> { vec![] }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn open_streamed_reader<R>(reader: R) -> AudioIOResult<wav::StreamedWavFile<R>>
where
    R: ReadSeek,
{
    wav::StreamedWavFile::new(reader)
}

/// Open a FLAC file for streaming reads.
///
/// Parses only the metadata header on construction and decodes frames on demand.
/// Returns a concrete [`StreamedFlacFile`] for full API access.
///
/// For format-agnostic streaming use [`open_streamed_dyn`], which returns a trait object.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::open_streamed_flac;
/// use audio_samples_io::traits::{AudioFileMetadata, AudioStreamRead, AudioStreamReader};
/// use audio_samples::{AudioSamples, nzu};
/// use std::num::NonZeroU32;
///
/// let mut streamed = open_streamed_flac("large_file.flac")?;
/// let channels = NonZeroU32::new(streamed.num_channels() as u32).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("channels must be non-zero".to_string()))?;
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).ok_or_else(|| audio_samples_io::error::AudioIOError::UnsupportedFormat("sample_rate must be non-zero".to_string()))?;
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, nzu!(1024), sample_rate);
///
/// while streamed.remaining_frames() > 0 {
///     let frames = streamed.read_frames_into(&mut buffer, nzu!(1024))?;
///     // Process frames...
/// }
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "flac")]
pub fn open_streamed_flac<P>(fp: P) -> AudioIOResult<StreamedFlacFile<BufReader<File>>>
where
    P: AsRef<Path>,
{
    let path = fp.as_ref();
    match FileType::detect(path) {
        FileType::FLAC => {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            StreamedFlacFile::new_with_path(reader, path.to_path_buf())
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format for FLAC streaming: {other:?}"
        ))),
    }
}

/// Open any `Read + Seek` source for streaming FLAC reads.
///
/// This allows streaming from any source implementing `Read + Seek`,
/// such as in-memory cursors or custom I/O implementations.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::open_streamed_flac_reader;
/// use std::io::Cursor;
///
/// let flac_bytes: Vec<u8> = std::fs::read("audio.flac").unwrap();
/// let cursor = Cursor::new(flac_bytes);
/// let mut streamed = open_streamed_flac_reader(cursor)?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "flac")]
pub fn open_streamed_flac_reader<R>(reader: R) -> AudioIOResult<flac::StreamedFlacFile<R>>
where
    R: ReadSeek,
{
    flac::StreamedFlacFile::new(reader)
}

/// Open an audio file for streaming reads, returning a trait object.
///
/// This function returns a `Box<dyn AudioStreamReader>` which provides
/// format-agnostic streaming access. Use this when you need to work with
/// multiple formats through a unified interface.
///
/// For format-specific access with full functionality (including generic
/// `read_frames_into<T>()`), use [`open_streamed()`] or [`open_streamed_reader()`]
/// which return concrete types.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::open_streamed_dyn;
/// use audio_samples_io::traits::AudioStreamReader;
///
/// fn process_any_stream(mut stream: Box<dyn AudioStreamReader>) -> Result<(), Box<dyn std::error::Error>> {
///     println!("channels={} sample_rate={} total_frames={}",
///         stream.num_channels(), stream.sample_rate(), stream.total_frames());
///     stream.seek_to_frame(1000)?;
///     Ok(())
/// }
///
/// let stream = open_streamed_dyn("audio.wav")?;
/// process_any_stream(stream);
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
pub fn open_streamed_dyn<P>(fp: P) -> AudioIOResult<Box<dyn AudioStreamReader>>
where
    P: AsRef<Path>,
{
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled for WAV streaming",
            ));

            #[cfg(feature = "wav")]
            {
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                let streamed = StreamedWavFile::new_with_path(reader, path.to_path_buf())?;
                Ok(Box::new(streamed))
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled for FLAC streaming",
            ));

            #[cfg(feature = "flac")]
            {
                let file = File::open(path)?;
                let reader = BufReader::new(file);
                let streamed = StreamedFlacFile::new_with_path(reader, path.to_path_buf())?;
                Ok(Box::new(streamed))
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format for streaming: {other:?}"
        ))),
    }
}

/// Create a streaming writer to a file path, choosing WAV or FLAC from the extension.
///
/// Returns a format-agnostic [`StreamedAudioWriter`]; for the concrete per-format writer
/// use [`create_streamed_writer`] (WAV) or [`create_streamed_flac`] (FLAC).
///
/// The sample type is inferred from the generic parameter `T`. Use the turbofish
/// syntax when the type cannot be inferred: `create_streamed::<f32>(...)`.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::create_streamed;
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, channels, nzu, sample_rate};
///
/// let mut writer = create_streamed::<_, f32>("output.wav", 2, 44100)?;
///
/// let sr = sample_rate!(44100);
/// let chunk = AudioSamples::<f32>::zeros_multi(channels!(2), nzu!(1024), sr);
/// writer.write_frames(&chunk)?;
/// writer.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(any(feature = "wav", feature = "flac"))]
pub fn create_streamed<P, T>(
    fp: P,
    channels: u16,
    sample_rate: u32,
) -> AudioIOResult<StreamedAudioWriter<BufWriter<File>>>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();
    let format = FileType::from_path(path);
    let file = File::create(path)?;
    // 256 KiB: buffers ~8 typical streaming chunks (4096-frame stereo f32 = 32 KiB)
    // before issuing a write syscall, reducing syscall count ~8× vs the 8 KiB default.
    let writer = BufWriter::with_capacity(256 * 1024, file);
    create_streamed_with::<_, T>(writer, channels, sample_rate, format)
}

/// Create a streaming writer to a file path with explicit [`WriteOptions`].
///
/// Identical to [`create_streamed`] (format chosen by extension) but lets you control the
/// write-buffer size.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::{create_streamed_with_options, WriteOptions};
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, channels, nzu, sample_rate};
///
/// // 1 MiB buffer for large streaming chunks.
/// let opts = WriteOptions { write_buf_capacity: 1024 * 1024 };
/// let mut writer = create_streamed_with_options::<_, f32>("output.wav", 2, 44100, opts)?;
///
/// let sr = sample_rate!(44100);
/// let chunk = AudioSamples::<f32>::zeros_multi(channels!(2), nzu!(8192), sr);
/// writer.write_frames(&chunk)?;
/// writer.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(any(feature = "wav", feature = "flac"))]
pub fn create_streamed_with_options<P, T>(
    fp: P,
    channels: u16,
    sample_rate: u32,
    opts: WriteOptions,
) -> AudioIOResult<StreamedAudioWriter<BufWriter<File>>>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();
    let format = FileType::from_path(path);
    let file = File::create(path)?;
    let writer = BufWriter::with_capacity(opts.write_buf_capacity, file);
    create_streamed_with::<_, T>(writer, channels, sample_rate, format)
}

/// Create a streaming writer to any [`WriteSeek`] destination with an explicit format.
///
/// The format-agnostic, bring-your-own-writer counterpart of [`write_with`]: returns a
/// [`StreamedAudioWriter`] that encodes as WAV or FLAC according to `format`. The output
/// sample type / bit depth is derived from `T`.
///
/// ```no_run
/// use audio_samples_io::{create_streamed_with, types::FileType};
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, nzu, sample_rate};
/// use std::io::Cursor;
///
/// let mut buf = Vec::new();
/// let mut writer = create_streamed_with::<_, i16>(Cursor::new(&mut buf), 1, 44100, FileType::FLAC)?;
/// writer.write_frames(&AudioSamples::<i16>::zeros_mono(nzu!(1024), sample_rate!(44100)))?;
/// writer.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(any(feature = "wav", feature = "flac"))]
pub fn create_streamed_with<W, T>(
    writer: W,
    channels: u16,
    sample_rate: u32,
    format: FileType,
) -> AudioIOResult<StreamedAudioWriter<W>>
where
    W: WriteSeek,
    T: StandardSample + 'static,
{
    match format {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            {
                let _ = (writer, channels, sample_rate);
                Err(AudioIOError::missing_feature(
                    "'wav' feature must be enabled for WAV streaming writes",
                ))
            }
            #[cfg(feature = "wav")]
            {
                Ok(StreamedAudioWriter::Wav(wav_writer_for_type::<T, W>(
                    writer,
                    channels,
                    sample_rate,
                )?))
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            {
                let _ = (writer, channels, sample_rate);
                Err(AudioIOError::missing_feature(
                    "'flac' feature must be enabled for FLAC streaming writes",
                ))
            }
            #[cfg(feature = "flac")]
            {
                Ok(StreamedAudioWriter::Flac(flac_writer_for_type::<T, W>(
                    writer,
                    channels,
                    sample_rate,
                )?))
            }
        },
        other => Err(AudioIOError::unsupported_format(format!(
            "Unsupported output format for streaming write: {other:?}"
        ))),
    }
}

/// Create a streaming FLAC writer to a file path.
///
/// FLAC is a block-based codec, so this writer buffers each block and encodes it
/// incrementally as frames are written, using the same encoder as the bulk [`write`]
/// path. The STREAMINFO header's total sample count is back-patched on
/// [`finalize`](crate::traits::AudioStreamWriter::finalize), which is why a seekable
/// destination is required. The output bit depth is derived from `T` (16-bit for `i16`,
/// 24-bit otherwise), matching [`write`].
///
/// FLAC's concrete return type can't be folded into [`create_streamed`] (which is
/// WAV-only); this mirrors the [`open_streamed`]/[`open_streamed_flac`] split.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::create_streamed_flac;
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, channels, nzu, sample_rate};
///
/// let mut writer = create_streamed_flac::<_, i16>("output.flac", 2, 44100)?;
///
/// let chunk = AudioSamples::<i16>::zeros_multi(channels!(2), nzu!(4096), sample_rate!(44100));
/// writer.write_frames(&chunk)?;
/// writer.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "flac")]
pub fn create_streamed_flac<P, T>(
    fp: P,
    channels: u16,
    sample_rate: u32,
) -> AudioIOResult<StreamedFlacWriter<BufWriter<File>>>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();
    match FileType::from_path(path) {
        FileType::FLAC => {
            let file = File::create(path)?;
            let writer = BufWriter::with_capacity(256 * 1024, file);
            flac_writer_for_type::<T, _>(writer, channels, sample_rate)
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported output format for streaming FLAC write: {other:?}"
        ))),
    }
}

/// Create a streaming FLAC writer to any [`WriteSeek`] destination (e.g. an in-memory
/// cursor). See [`create_streamed_flac`] for behaviour and bit-depth rules.
#[cfg(feature = "flac")]
pub fn create_streamed_flac_writer<W, T>(
    writer: W,
    channels: u16,
    sample_rate: u32,
) -> AudioIOResult<StreamedFlacWriter<W>>
where
    W: WriteSeek,
    T: StandardSample + 'static,
{
    flac_writer_for_type::<T, W>(writer, channels, sample_rate)
}

/// Construct a [`StreamedFlacWriter`] for sample type `T` at the default compression level.
///
/// Shared by [`create_streamed_flac`] and [`create_streamed_flac_writer`].
#[cfg(feature = "flac")]
fn flac_writer_for_type<T, W>(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<StreamedFlacWriter<W>>
where
    T: StandardSample + 'static,
    W: WriteSeek,
{
    let sample_type = validated_sample_type_of::<T>()?;
    StreamedFlacWriter::new(writer, channels, sample_rate, sample_type, CompressionLevel::default())
}

/// Create a streaming WAV writer to any `WriteSeek` destination.
///
/// The sample type is inferred from the generic parameter `T`. Allows streaming to
/// in-memory buffers, network streams, or any custom `WriteSeek` implementation.
#[cfg(feature = "wav")]
pub fn create_streamed_writer<W, T>(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<StreamedWavWriter<W>>
where
    W: WriteSeek,
    T: StandardSample + 'static,
{
    wav_writer_for_type::<T, W>(writer, channels, sample_rate)
}

/// Dispatch to the appropriate `StreamedWavWriter` constructor based on `T`.
///
/// Shared by `create_streamed` and `create_streamed_writer` to avoid duplication.
#[cfg(feature = "wav")]
fn wav_writer_for_type<T, W>(writer: W, channels: u16, sample_rate: u32) -> AudioIOResult<StreamedWavWriter<W>>
where
    T: StandardSample + 'static,
    W: WriteSeek,
{
    use audio_samples::I24;
    let type_id = TypeId::of::<T>();
    match type_id {
        id if id == TypeId::of::<u8>() || id == TypeId::of::<i16>() => {
            StreamedWavWriter::new_i16(writer, channels, sample_rate)
        },
        id if id == TypeId::of::<I24>() => StreamedWavWriter::new_i24(writer, channels, sample_rate),
        id if id == TypeId::of::<i32>() => StreamedWavWriter::new_i32(writer, channels, sample_rate),
        id if id == TypeId::of::<f32>() => StreamedWavWriter::new_f32(writer, channels, sample_rate),
        id if id == TypeId::of::<f64>() => StreamedWavWriter::new_f64(writer, channels, sample_rate),
        _ => Err(AudioIOError::unsupported_format(format!(
            "No WAV encoding for sample type (TypeId: {type_id:?})"
        ))),
    }
}

/// Map a Rust sample type to its [`ValidatedSampleType`].
#[cfg(any(feature = "wav", feature = "flac"))]
fn validated_sample_type_of<T>() -> AudioIOResult<ValidatedSampleType>
where
    T: StandardSample + 'static,
{
    use audio_samples::I24;
    let id = TypeId::of::<T>();
    if id == TypeId::of::<u8>() {
        Ok(ValidatedSampleType::U8)
    } else if id == TypeId::of::<i16>() {
        Ok(ValidatedSampleType::I16)
    } else if id == TypeId::of::<I24>() {
        Ok(ValidatedSampleType::I24)
    } else if id == TypeId::of::<i32>() {
        Ok(ValidatedSampleType::I32)
    } else if id == TypeId::of::<f32>() {
        Ok(ValidatedSampleType::F32)
    } else if id == TypeId::of::<f64>() {
        Ok(ValidatedSampleType::F64)
    } else {
        Err(AudioIOError::unsupported_format(format!(
            "No WAV encoding for sample type (TypeId: {id:?})"
        )))
    }
}

/// Create a non-seekable streaming WAV writer ([`WavSink`](crate::wav::WavSink)) over any
/// `Write` destination — stdout, a pipe, a socket, etc.
///
/// Because a `!Seek` sink cannot backpatch size fields, the header is written with final sizes
/// up front. Pass `total_frames = Some(n)` when the frame count is known (recommended; produces
/// a fully standard file and verifies the count on `finalize`), or `None` for an open-ended
/// stream (uses the `0xFFFFFFFF` streaming-size convention).
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::create_streamed_sink;
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, nzu, sample_rate};
///
/// let stdout = std::io::stdout();
/// let mut sink = create_streamed_sink::<_, i16>(stdout.lock(), 1, 44100, Some(1024))?;
/// let audio = AudioSamples::<f32>::zeros_mono(nzu!(1024), sample_rate!(44100));
/// sink.write_frames(&audio)?;
/// sink.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn create_streamed_sink<W, T>(
    writer: W,
    channels: u16,
    sample_rate: u32,
    total_frames: Option<usize>,
) -> AudioIOResult<wav::WavSink<W>>
where
    W: Write,
    T: StandardSample + 'static,
{
    let sample_type = validated_sample_type_of::<T>()?;
    wav::WavSink::new(writer, channels, sample_rate, sample_type, total_frames)
}

/// Open an audio file for reading/writing operations
///
/// Returns a trait object that can be used for low-level file operations.
/// For simple use cases, prefer the `read()` and `info()` convenience functions.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn open<P>(fp: P) -> AudioIOResult<Box<dyn AudioFile>>
where
    P: AsRef<Path>,
{
    let path = fp.as_ref();

    match FileType::detect(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to open WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                let wav_file = WavFile::open_with_options(path, OpenOptions::default())?;
                Ok(Box::new(wav_file))
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to open FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                use crate::flac::FlacFile;
                use crate::traits::AudioFile;
                let flac_file = FlacFile::open_with_options(path, OpenOptions::default())?;
                Ok(Box::new(flac_file))
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {other:?}"
        ))),
    }
}

pub fn write<P, T>(fp: P, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    write_with_options(fp, audio, WriteOptions::default())
}

/// Write audio samples to a file with explicit [`WriteOptions`].
///
/// Identical to [`write`] but lets you control the write-buffer size:
///
/// ```no_run
/// use audio_samples_io::{write_with_options, WriteOptions};
/// use audio_samples::{AudioSamples, sine_wave, sample_rate};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_secs(60), sample_rate!(44100), 0.5);
///
/// // 16 MiB buffer for a 60-second file (~21 MiB stereo f32).
/// write_with_options("long.wav", &audio, WriteOptions { write_buf_capacity: 16 * 1024 * 1024 })?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
pub fn write_with_options<P, T>(fp: P, audio: &AudioSamples<T>, opts: WriteOptions) -> AudioIOResult<()>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();

    match FileType::from_path(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to write WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                let file = std::fs::File::create(path)?;
                crate::wav::wav_file::write_wav(file, audio, opts)
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to write FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                let file = std::fs::File::create(path)?;
                let buf_writer = std::io::BufWriter::with_capacity(opts.write_buf_capacity, file);
                crate::flac::write_flac(buf_writer, audio, CompressionLevel::default())
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported format: {other:?}"
        ))),
    }
}

/// Write a WAV file with trailing metadata chunks (LIST/INFO tags, cue points).
///
/// Like [`write`], but also serialises the given [`WavMetadata`](crate::wav::WavMetadata) after
/// the audio data — letting you persist tags/markers that a plain read→write round-trip would
/// drop. WAV only.
///
/// ```no_run
/// use audio_samples_io::write_with_metadata;
/// use audio_samples_io::wav::WavMetadata;
/// use audio_samples_io::wav::list_info::InfoMetadata;
/// use audio_samples::{AudioSamples, sine_wave, sample_rate};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.5);
/// let mut meta = WavMetadata::default();
/// meta.info = Some(InfoMetadata { title: Some("My Track".into()), ..Default::default() });
/// write_with_metadata("tagged.wav", &audio, &meta)?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn write_with_metadata<P, T>(
    fp: P,
    audio: &AudioSamples<T>,
    metadata: &crate::wav::WavMetadata,
) -> AudioIOResult<()>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let file = std::fs::File::create(fp)?;
    crate::wav::wav_file::write_wav_with_metadata(file, audio, WriteOptions::default(), metadata)
}

/// Write a WAV file with trailing metadata to any `Write` destination (e.g. an in-memory buffer).
#[cfg(feature = "wav")]
pub fn write_with_metadata_to<T, W>(
    writer: W,
    audio: &AudioSamples<T>,
    metadata: &crate::wav::WavMetadata,
) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    crate::wav::wav_file::write_wav_with_metadata(writer, audio, WriteOptions::default(), metadata)
}

/// Write audio data to any `Write` destination with explicit format specification.
///
/// This function allows writing audio data to any destination implementing `Write`,
/// such as in-memory buffers, network streams, or custom I/O implementations.
/// Unlike `write()` which determines format from file extension, this function
/// requires an explicit format parameter.
///
/// # Arguments
///
/// * `writer` - The destination implementing `Write`
/// * `audio` - The audio samples to write
/// * `format` - The target audio format (WAV, FLAC, etc.)
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::{write_with, types::FileType};
/// use audio_samples::{AudioSamples, sine_wave, sample_rate};
/// use std::io::Cursor;
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.5);
/// let mut buffer = Vec::new();
/// let cursor = Cursor::new(&mut buffer);
///
/// write_with(cursor, &audio, FileType::WAV)?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
pub fn write_with<T, W>(writer: W, audio: &AudioSamples<T>, format: FileType) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    write_with_writer_options(writer, audio, format, WriteOptions::default())
}

/// Write audio data to any `Write` destination with explicit format and [`WriteOptions`].
///
/// Identical to [`write_with`] but lets you control the write-buffer size.
pub fn write_with_writer_options<T, W>(
    writer: W,
    audio: &AudioSamples<T>,
    format: FileType,
    opts: WriteOptions,
) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    match format {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to write WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                crate::wav::wav_file::write_wav(writer, audio, opts)
            }
        },
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to write FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                crate::flac::write_flac(writer, audio, CompressionLevel::default())
            }
        },
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported format for write_with: {other:?}"
        ))),
    }
}

#[cfg(all(test, feature = "wav"))]
mod lib_tests {
    use std::time::Duration;

    use audio_samples::sample_rate;

    use super::*;

    #[test]
    fn test_info_function() {
        let info_result = info("resources/test.wav");
        assert!(info_result.is_ok(), "Failed to get info from test WAV file");

        let audio_info = info_result.expect("Expected successful info retrieval");
        assert_eq!(audio_info.file_type, FileType::WAV);
        assert!(audio_info.sample_rate.get() > 0, "Sample rate should be positive");
        assert!(audio_info.channels > 0, "Channel count should be positive");
        println!("Audio info: {audio_info:#}");
    }

    #[test]
    fn test_read_function() {
        let audio_result = read::<_, f32>("resources/test.wav");
        assert!(audio_result.is_ok(), "Failed to read test WAV file");

        let audio_samples = audio_result.expect("Expected successful audio read");
        println!(
            "Read {} samples at {} Hz",
            audio_samples.len(),
            audio_samples.sample_rate()
        );
    }

    #[test]
    fn test_open_function() {
        let file_result = open("resources/test.wav");
        assert!(file_result.is_ok(), "Failed to open test WAV file");
    }

    #[test]
    fn test_write_function() {
        use std::fs;

        use audio_samples::sine_wave;

        // Generate test audio
        let sample_rate = sample_rate!(44100);
        let sine_samples = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.1), sample_rate, 0.5);

        // Test writing WAV file
        let output_path = std::env::temp_dir().join("test_lib_write.wav");
        write(&output_path, &sine_samples).expect("Failed to write WAV file");

        // Verify file exists and can be read back
        assert!(fs::metadata(&output_path).is_ok(), "Output file should exist");

        let read_back = read::<_, f32>(&output_path).expect("Failed to read back WAV file");
        assert_eq!(read_back.sample_rate(), sample_rate);
        assert_eq!(read_back.total_samples(), sine_samples.total_samples());
        assert_eq!(read_back.num_channels(), sine_samples.num_channels());

        // Verify the actual audio data matches (approximately for floating point)
        let original_bytes = sine_samples.bytes().expect("bytes should be available");
        let read_bytes = read_back.bytes().expect("bytes should be available");
        assert_eq!(
            original_bytes.as_slice().len(),
            read_bytes.as_slice().len(),
            "Audio data size should match"
        );

        // For f32, check that values are very close (allowing for minor precision differences)
        let original_samples: &[f32] = bytemuck::cast_slice(original_bytes.as_slice());
        let read_samples: &[f32] = bytemuck::cast_slice(read_bytes.as_slice());

        for (i, (orig, read)) in original_samples.iter().zip(read_samples.iter()).enumerate() {
            let diff = (orig - read).abs();
            assert!(
                diff < 1e-6,
                "Sample {i} differs too much: {orig} vs {read} (diff: {diff})"
            );
        }

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_write_with_function() {
        use std::io::Cursor;

        use audio_samples::sine_wave;

        // Generate test audio
        let sample_rate = sample_rate!(22050);
        let sine_samples = sine_wave::<i16>(880.0, Duration::from_secs_f64(0.05), sample_rate, 0.8);

        // Write to in-memory buffer
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        write_with(cursor, &sine_samples, FileType::WAV).expect("Failed to write with cursor");

        // Verify buffer has WAV data
        assert!(buffer.len() > 44, "Buffer should contain WAV header and data");
        assert_eq!(&buffer[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(&buffer[8..12], b"WAVE", "Should contain WAVE identifier");

        // Verify we can read the WAV data back from the buffer
        let temp_file = std::env::temp_dir().join("test_write_with_buffer.wav");
        std::fs::write(&temp_file, &buffer).expect("Failed to write buffer to temp file");

        let read_back = read::<_, i16>(&temp_file).expect("Failed to read back WAV from buffer");
        assert_eq!(read_back.sample_rate(), sample_rate);
        assert_eq!(read_back.total_samples(), sine_samples.total_samples());
        assert_eq!(read_back.num_channels(), sine_samples.num_channels());

        // Verify the actual audio data matches exactly for i16
        let original_bytes = sine_samples.bytes().expect("bytes should be available");
        let read_bytes = read_back.bytes().expect("bytes should be available");
        assert_eq!(
            original_bytes.as_slice(),
            read_bytes.as_slice(),
            "Audio data should match exactly for i16"
        );

        // Clean up
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_write_with_format_parameter() {
        use std::io::Cursor;

        use audio_samples::sine_wave;

        // Generate test audio
        let sample_rate = sample_rate!(44100);
        let sine_samples = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.01), sample_rate, 0.5);

        // Test with explicit WAV format
        let mut wav_buffer = Vec::new();
        let wav_cursor = Cursor::new(&mut wav_buffer);
        write_with(wav_cursor, &sine_samples, FileType::WAV).expect("Failed to write WAV format");

        // Verify WAV buffer has data and proper WAV header
        assert!(wav_buffer.len() > 44, "WAV buffer should contain header and data");
        assert_eq!(&wav_buffer[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(&wav_buffer[8..12], b"WAVE", "Should contain WAVE identifier");

        // Test error for unsupported format
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        let result = write_with(cursor, &sine_samples, FileType::MP3);
        assert!(result.is_err(), "Should return error for unsupported format");

        let error_msg = format!("{}", result.expect_err("Expected error"));
        assert!(
            error_msg.contains("Unsupported format"),
            "Error should mention unsupported format"
        );
    }

    #[test]
    fn test_write_different_formats() {
        use std::fs;

        use audio_samples::{AudioTypeConversion, sine_wave};

        let sample_rate = sample_rate!(48000);
        let sine_base = sine_wave::<f32>(1000.0, Duration::from_secs_f64(0.02), sample_rate, 0.3);

        // Test different sample types
        let test_cases = vec![
            ("i16", std::env::temp_dir().join("test_format_i16.wav")),
            ("f32", std::env::temp_dir().join("test_format_f32.wav")),
        ];

        for (format_name, output_path) in test_cases {
            match format_name {
                "i16" => {
                    let samples_i16 = sine_base.to_format::<i16>();
                    write(&output_path, &samples_i16).expect("Failed to write i16 WAV");

                    // Verify the written file can be read back as i16
                    let read_back = read::<_, i16>(&output_path).expect("Failed to read back i16 WAV");
                    assert_eq!(read_back.sample_rate(), sample_rate, "Sample rate mismatch for i16");
                    assert_eq!(
                        read_back.total_samples(),
                        samples_i16.total_samples(),
                        "Sample count mismatch for i16"
                    );

                    // Verify WAV file properties using info function
                    let wav_info = info(&output_path).expect("Failed to get WAV info for i16");
                    assert_eq!(wav_info.bits_per_sample, 16, "Bits per sample should be 16 for i16");
                    assert_eq!(
                        wav_info.sample_type,
                        audio_samples::SampleType::I16,
                        "Sample type should be I16"
                    );
                },
                "f32" => {
                    write(&output_path, &sine_base).expect("Failed to write f32 WAV");

                    // Verify the written file can be read back as f32
                    let read_back = read::<_, f32>(&output_path).expect("Failed to read back f32 WAV");
                    assert_eq!(read_back.sample_rate(), sample_rate, "Sample rate mismatch for f32");
                    assert_eq!(
                        read_back.total_samples(),
                        sine_base.total_samples(),
                        "Sample count mismatch for f32"
                    );

                    // Verify WAV file properties using info function
                    let wav_info = info(&output_path).expect("Failed to get WAV info for f32");
                    assert_eq!(wav_info.bits_per_sample, 32, "Bits per sample should be 32 for f32");
                    assert_eq!(
                        wav_info.sample_type,
                        audio_samples::SampleType::F32,
                        "Sample type should be F32"
                    );
                },
                _ => unreachable!("Unknown format"),
            }

            // Verify file was created and is a valid WAV
            assert!(
                fs::metadata(&output_path).is_ok(),
                "File should exist for {format_name}"
            );

            // Clean up
            fs::remove_file(&output_path).ok();
        }
    }

    #[test]
    fn test_unsupported_format_error() {
        use audio_samples::sine_wave;

        let sample_rate = sample_rate!(44100);
        let sine_samples = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.01), sample_rate, 0.1);

        // Try to write to unsupported format
        let result = write(std::env::temp_dir().join("test.mp3"), &sine_samples);
        assert!(result.is_err(), "Should fail for unsupported format");

        let error_msg = format!("{}", result.expect_err("Expected error"));
        assert!(
            error_msg.contains("Unsupported"),
            "Error should mention unsupported format"
        );
    }
}
