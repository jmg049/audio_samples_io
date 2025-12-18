// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![allow(clippy::result_large_err)] // Allow large error types for comprehensive error handling
#![allow(clippy::missing_const_for_fn)] // Functions may need mutations in the future
#![allow(clippy::collapsible_if)] // Sometimes clearer to have separate conditions
#![allow(clippy::missing_panics_doc)] // Panics are converted to proper errors where needed
#![allow(clippy::needless_borrows_for_generic_args)] // Sometimes clearer with explicit borrows
#![allow(clippy::if_same_then_else)] // Similar blocks may diverge in the future
#![allow(clippy::unnecessary_cast)] // Explicit casts for clarity
#![allow(clippy::identity_op)] // Explicit operations for clarity

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`
#![warn(clippy::unwrap_used)] // Avoids using `unwrap()`
#![warn(clippy::panic)] // Avoids using `panic!` in production code

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)] // Allow functions with many parameters (very few and far between)

pub mod error;
pub mod traits;
pub mod types;

#[cfg(feature = "wav")]
pub mod wav;

#[cfg(feature = "wav")]
pub use crate::wav::{StreamedWavFile, StreamedWavWriter, wav_file::WavFile};

#[cfg(feature = "flac")]
pub mod flac;

#[cfg(feature = "flac")]
pub use crate::flac::CompressionLevel;

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::Path,
};

use audio_samples::{AudioSample, AudioSamples, ConvertTo, I24};

pub use crate::{
    error::{AudioIOError, AudioIOResult},
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioStreamReader},
    types::{BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType},
};


pub(crate) const MAX_WAV_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GB limit
pub(crate) const MAX_MMAP_SIZE: u64 = 512 * 1024 * 1024; // 512MB for memory mapping

/// Convenience trait for types that implement both Read and Seek
pub trait ReadSeek: Read + Seek {}

impl<RS: Read + Seek> ReadSeek for RS where RS: Read + Seek {}

// Public API

/// Get basic audio information from a file
///
/// Automatically detects the file format and extracts metadata.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn info<P: AsRef<Path>>(fp: P) -> AudioIOResult<BaseAudioInfo> {
    let path = fp.as_ref();

    match FileType::from_path(path) {
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
        }
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to read FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                // TODO: Implement FLAC info extraction when FLAC support is added
                Err(crate::error::AudioIOError::unsupported_format(
                    "FLAC info extraction not yet implemented",
                ))
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {:?}",
            other
        ))),
    }
}

/// Read audio samples from a file
///
/// Returns owned AudioSamples containing all audio data from the file.
/// Automatically detects file format and handles sample type conversion.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn read<P: AsRef<Path>, T: AudioSample>(fp: P) -> AudioIOResult<AudioSamples<'static, T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let path = fp.as_ref();

    match FileType::from_path(path) {
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
        }
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to read FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                // TODO: Implement FLAC reading when FLAC support is added
                Err(crate::error::AudioIOError::unsupported_format(
                    "FLAC reading not yet implemented",
                ))
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {:?}",
            other
        ))),
    }
}

pub fn read_with<'a, R: ReadSeek, T: AudioSample>(
    _reader: R,
) -> AudioIOResult<AudioSamples<'a, T>> {
    todo!()
}

/// Open a WAV file for streaming reads.
///
/// Unlike `read()` which loads the entire file (lazily if using the mmap backing), this opens the file for
/// incremental reading, parsing only the header initially.
///
/// # Example
///
/// ```no_run
/// use audio_io::open_streamed;
/// use audio_io::traits::AudioFileMetadata;
/// use audio_samples::AudioSamples;
/// use std::num::NonZeroU32;
///
/// let mut streamed = open_streamed("large_file.wav")?;
/// let channels = streamed.num_channels() as usize;
/// let sample_rate = NonZeroU32::new(streamed.sample_rate()).unwrap();
/// let mut buffer = AudioSamples::<f32>::zeros_multi(channels, 1024, sample_rate);
///
/// while streamed.remaining_frames() > 0 {
///     let frames = streamed.read_frames_into(&mut buffer, 1024)?;
///     // Process frames...
/// }
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn open_streamed<P: AsRef<Path>>(fp: P) -> AudioIOResult<StreamedWavFile<BufReader<File>>> {
    let path = fp.as_ref();

    match FileType::from_path(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled for WAV streaming",
            ));

            #[cfg(feature = "wav")]
            {
                let file = File::open(path)?;
                let reader = std::io::BufReader::new(file);
                StreamedWavFile::new_with_path(reader, path.to_path_buf())
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format for streaming: {:?}",
            other
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
/// use audio_io::open_streamed_reader;
/// use std::io::Cursor;
///
/// let wav_bytes: Vec<u8> = load_from_network();
/// let cursor = Cursor::new(wav_bytes);
/// let mut streamed = open_streamed_reader(cursor)?;
/// # fn load_from_network() -> Vec<u8> { vec![] }
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn open_streamed_reader<R: ReadSeek>(reader: R) -> AudioIOResult<wav::StreamedWavFile<R>> {
    wav::StreamedWavFile::new(reader)
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
/// use audio_io::open_streamed_dyn;
/// use audio_io::traits::AudioStreamReader;
///
/// fn process_any_stream(mut stream: Box<dyn AudioStreamReader>) {
///     println!("Total frames: {}", stream.total_frames());
///     stream.seek_to_frame(1000).unwrap();
/// }
///
/// let stream = open_streamed_dyn("audio.wav")?;
/// process_any_stream(stream);
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
pub fn open_streamed_dyn<P: AsRef<Path>>(fp: P) -> AudioIOResult<Box<dyn AudioStreamReader>> {
    let path = fp.as_ref();

    match FileType::from_path(path) {
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
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format for streaming: {:?}",
            other
        ))),
    }
}

/// Create a streaming WAV writer to a file path.
///
/// This function creates a new WAV file for incremental writing. Audio data
/// is written as it's provided, without buffering the entire file in memory.
///
/// # Arguments
///
/// * `fp` - Output file path
/// * `channels` - Number of audio channels
/// * `sample_rate` - Sample rate in Hz
/// * `sample_type` - Target sample type for encoding
///
/// # Example
///
/// ```no_run
/// use audio_io::{create_streamed, types::ValidatedSampleType};
/// use audio_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::AudioSamples;
/// use std::num::NonZeroU32;
///
/// let mut writer = create_streamed(
///     "output.wav",
///     2,  // stereo
///     44100,
///     ValidatedSampleType::F32,
/// )?;
///
/// // Write audio in chunks
/// let sample_rate = NonZeroU32::new(44100).unwrap();
/// let chunk = AudioSamples::<f32>::zeros_multi(2, 1024, sample_rate);
/// writer.write_frames(&chunk)?;
///
/// // Always finalize to update headers
/// writer.finalize()?;
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn create_streamed<P: AsRef<Path>>(
    fp: P,
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
) -> AudioIOResult<StreamedWavWriter<BufWriter<File>>> {
    let path = fp.as_ref();

    match FileType::from_path(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled for WAV writing",
            ));

            #[cfg(feature = "wav")]
            {
                let file = File::create(path)?;
                let writer = BufWriter::new(file);
                match sample_type {
                    ValidatedSampleType::I16 => {
                        StreamedWavWriter::new_i16(writer, channels, sample_rate)
                    }
                    ValidatedSampleType::I24 => {
                        StreamedWavWriter::new_i24(writer, channels, sample_rate)
                    }
                    ValidatedSampleType::I32 => {
                        StreamedWavWriter::new_i32(writer, channels, sample_rate)
                    }
                    ValidatedSampleType::F32 => {
                        StreamedWavWriter::new_f32(writer, channels, sample_rate)
                    }
                    ValidatedSampleType::F64 => {
                        StreamedWavWriter::new_f64(writer, channels, sample_rate)
                    }
                }
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported output format for streaming write: {:?}",
            other
        ))),
    }
}

/// Create a streaming WAV writer to any `Write + Seek` destination.
///
/// This allows streaming to any destination implementing `Write + Seek`,
/// such as in-memory buffers, network streams, or custom I/O implementations.
///
/// # Example
///
/// ```no_run
/// use audio_io::{create_streamed_writer, types::ValidatedSampleType};
/// use audio_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::AudioSamples;
/// use std::io::Cursor;
/// use std::num::NonZeroU32;
///
/// let mut buffer = Vec::new();
/// let cursor = Cursor::new(&mut buffer);
///
/// let mut writer = create_streamed_writer(
///     cursor,
///     1,  // mono
///     22050,
///     ValidatedSampleType::I16,
/// )?;
///
/// let sample_rate = NonZeroU32::new(22050).unwrap();
/// let audio = AudioSamples::<f32>::zeros_mono(1024, sample_rate);
/// writer.write_frames(&audio)?;
/// writer.finalize()?;
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
#[cfg(feature = "wav")]
pub fn create_streamed_writer<W: Write + Seek>(
    writer: W,
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
) -> AudioIOResult<StreamedWavWriter<W>> {
    match sample_type {
        ValidatedSampleType::I16 => StreamedWavWriter::new_i16(writer, channels, sample_rate),
        ValidatedSampleType::I24 => StreamedWavWriter::new_i24(writer, channels, sample_rate),
        ValidatedSampleType::I32 => StreamedWavWriter::new_i32(writer, channels, sample_rate),
        ValidatedSampleType::F32 => StreamedWavWriter::new_f32(writer, channels, sample_rate),
        ValidatedSampleType::F64 => StreamedWavWriter::new_f64(writer, channels, sample_rate),
    }
}

/// Open an audio file for reading/writing operations
///
/// Returns a trait object that can be used for low-level file operations.
/// For simple use cases, prefer the `read()` and `info()` convenience functions.
/// Currently supports WAV and FLAC formats (with appropriate features enabled).
pub fn open<P: AsRef<Path>>(fp: P) -> AudioIOResult<Box<dyn AudioFile>> {
    let path = fp.as_ref();

    match FileType::from_path(path) {
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
        }
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to open FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                // TODO: Implement FLAC opening when FLAC support is added
                Err(crate::error::AudioIOError::unsupported_format(
                    "FLAC opening not yet implemented",
                ))
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported file format: {:?}",
            other
        ))),
    }
}

pub fn write<P: AsRef<Path>, T: AudioSample>(fp: P, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let path = fp.as_ref();

    // Determine format from file extension
    match FileType::from_path(path) {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to write WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                let file = std::fs::File::create(path)?;
                let buf_writer = std::io::BufWriter::new(file);
                crate::wav::wav_file::write_wav(buf_writer, audio)
            }
        }
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to write FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                let file = std::fs::File::create(path)?;
                let buf_writer = std::io::BufWriter::new(file);
                crate::flac::write_flac(buf_writer, audio, CompressionLevel::default())
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported  format: {:?}",
            other
        ))),
    }
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
/// use audio_io::{write_with, types::FileType};
/// use audio_samples::{AudioSamples, sine_wave};
/// use std::io::Cursor;
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44100, 0.5);
/// let mut buffer = Vec::new();
/// let cursor = Cursor::new(&mut buffer);
///
/// write_with(cursor, &audio, FileType::WAV)?;
/// # Ok::<(), audio_io::error::AudioIOError>(())
/// ```
pub fn write_with<T: AudioSample, W: Write>(
    writer: W,
    audio: &AudioSamples<T>,
    format: FileType,
) -> AudioIOResult<()>
where
    i16: ConvertTo<T>,
    audio_samples::I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    match format {
        FileType::WAV => {
            #[cfg(not(feature = "wav"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'wav' feature must be enabled to write WAV files",
            ));

            #[cfg(feature = "wav")]
            {
                crate::wav::wav_file::write_wav(writer, audio)
            }
        }
        FileType::FLAC => {
            #[cfg(not(feature = "flac"))]
            return Err(crate::error::AudioIOError::missing_feature(
                "'flac' feature must be enabled to write FLAC files",
            ));

            #[cfg(feature = "flac")]
            {
                crate::flac::write_flac(writer, audio, CompressionLevel::default())
            }
        }
        other => Err(crate::error::AudioIOError::unsupported_format(format!(
            "Unsupported format for write_with: {:?}",
            other
        ))),
    }
}

#[cfg(all(test, feature = "wav"))]
mod lib_tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_info_function() {
        let info_result = info("resources/test.wav");
        assert!(info_result.is_ok(), "Failed to get info from test WAV file");

        let audio_info = info_result.unwrap();
        assert_eq!(audio_info.file_type, FileType::WAV);
        assert!(audio_info.sample_rate > 0, "Sample rate should be positive");
        assert!(audio_info.channels > 0, "Channel count should be positive");
        println!("Audio info: {:#}", audio_info);
    }

    #[test]
    fn test_read_function() {
        let audio_result = read::<_, f32>("resources/test.wav");
        assert!(audio_result.is_ok(), "Failed to read test WAV file");

        let audio_samples = audio_result.unwrap();
        assert!(audio_samples.len() > 0, "Audio samples should not be empty");
        assert!(
            audio_samples.sample_rate().get() > 0,
            "Sample rate should be positive"
        );
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

        let audio_file = file_result.unwrap();
        assert!(audio_file.len() > 0, "File should not be empty");
    }

    #[test]
    fn test_write_function() {
        use audio_samples::sine_wave;
        use std::fs;

        // Generate test audio
        let sample_rate = 44100;
        let sine_samples =
            sine_wave::<f32, f32>(440.0, Duration::from_secs_f64(0.1), sample_rate, 0.5);

        // Test writing WAV file
        let output_path = std::env::temp_dir().join("test_lib_write.wav");
        write(&output_path, &sine_samples).expect("Failed to write WAV file");

        // Verify file exists and can be read back
        assert!(
            fs::metadata(&output_path).is_ok(),
            "Output file should exist"
        );

        let read_back = read::<_, f32>(&output_path).expect("Failed to read back WAV file");
        assert_eq!(read_back.sample_rate().get(), sample_rate);
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
                "Sample {} differs too much: {} vs {} (diff: {})",
                i,
                orig,
                read,
                diff
            );
        }

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_write_with_function() {
        use audio_samples::sine_wave;
        use std::io::Cursor;

        // Generate test audio
        let sample_rate = 22050;
        let sine_samples =
            sine_wave::<i16, f32>(880.0, Duration::from_secs_f64(0.05), sample_rate, 0.8);

        // Write to in-memory buffer
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        write_with(cursor, &sine_samples, FileType::WAV).expect("Failed to write with cursor");

        // Verify buffer has WAV data
        assert!(
            buffer.len() > 44,
            "Buffer should contain WAV header and data"
        );
        assert_eq!(&buffer[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(&buffer[8..12], b"WAVE", "Should contain WAVE identifier");

        // Verify we can read the WAV data back from the buffer
        let temp_file = std::env::temp_dir().join("test_write_with_buffer.wav");
        std::fs::write(&temp_file, &buffer).expect("Failed to write buffer to temp file");

        let read_back = read::<_, i16>(&temp_file).expect("Failed to read back WAV from buffer");
        assert_eq!(read_back.sample_rate().get(), sample_rate);
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
        use audio_samples::sine_wave;
        use std::io::Cursor;

        // Generate test audio
        let sample_rate = 44100;
        let sine_samples =
            sine_wave::<f32, f32>(440.0, Duration::from_secs_f64(0.01), sample_rate, 0.5);

        // Test with explicit WAV format
        let mut wav_buffer = Vec::new();
        let wav_cursor = Cursor::new(&mut wav_buffer);
        write_with(wav_cursor, &sine_samples, FileType::WAV).expect("Failed to write WAV format");

        // Verify WAV buffer has data and proper WAV header
        assert!(
            wav_buffer.len() > 44,
            "WAV buffer should contain header and data"
        );
        assert_eq!(&wav_buffer[0..4], b"RIFF", "Should start with RIFF header");
        assert_eq!(
            &wav_buffer[8..12],
            b"WAVE",
            "Should contain WAVE identifier"
        );

        // Test error for unsupported format
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        let result = write_with(cursor, &sine_samples, FileType::MP3);
        assert!(
            result.is_err(),
            "Should return error for unsupported format"
        );

        let error_msg = format!("{}", result.unwrap_err());
        assert!(
            error_msg.contains("Unsupported format"),
            "Error should mention unsupported format"
        );
    }

    #[test]
    fn test_write_different_formats() {
        use audio_samples::{AudioTypeConversion, sine_wave};
        use std::fs;

        let sample_rate = 48000;
        let sine_base =
            sine_wave::<f32, f32>(1000.0, Duration::from_secs_f64(0.02), sample_rate, 0.3);

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
                    let read_back =
                        read::<_, i16>(&output_path).expect("Failed to read back i16 WAV");
                    assert_eq!(
                        read_back.sample_rate().get(),
                        sample_rate,
                        "Sample rate mismatch for i16"
                    );
                    assert_eq!(
                        read_back.total_samples(),
                        samples_i16.total_samples(),
                        "Sample count mismatch for i16"
                    );

                    // Verify WAV file properties using info function
                    let wav_info = info(&output_path).expect("Failed to get WAV info for i16");
                    assert_eq!(
                        wav_info.bits_per_sample, 16,
                        "Bits per sample should be 16 for i16"
                    );
                    assert_eq!(
                        wav_info.sample_type,
                        audio_samples::SampleType::I16,
                        "Sample type should be I16"
                    );
                }
                "f32" => {
                    write(&output_path, &sine_base).expect("Failed to write f32 WAV");

                    // Verify the written file can be read back as f32
                    let read_back =
                        read::<_, f32>(&output_path).expect("Failed to read back f32 WAV");
                    assert_eq!(
                        read_back.sample_rate().get(),
                        sample_rate,
                        "Sample rate mismatch for f32"
                    );
                    assert_eq!(
                        read_back.total_samples(),
                        sine_base.total_samples(),
                        "Sample count mismatch for f32"
                    );

                    // Verify WAV file properties using info function
                    let wav_info = info(&output_path).expect("Failed to get WAV info for f32");
                    assert_eq!(
                        wav_info.bits_per_sample, 32,
                        "Bits per sample should be 32 for f32"
                    );
                    assert_eq!(
                        wav_info.sample_type,
                        audio_samples::SampleType::F32,
                        "Sample type should be F32"
                    );
                }
                _ => panic!("Unknown format"),
            }

            // Verify file was created and is a valid WAV
            assert!(
                fs::metadata(&output_path).is_ok(),
                "File should exist for {}",
                format_name
            );

            // Clean up
            fs::remove_file(&output_path).ok();
        }
    }

    #[test]
    fn test_unsupported_format_error() {
        use audio_samples::sine_wave;

        let sample_rate = 44100;
        let sine_samples =
            sine_wave::<f32, f32>(440.0, Duration::from_secs_f64(0.01), sample_rate, 0.1);

        // Try to write to unsupported format
        let result = write(std::env::temp_dir().join("test.mp3"), &sine_samples);
        assert!(result.is_err(), "Should fail for unsupported format");

        let error_msg = format!("{}", result.unwrap_err());
        assert!(
            error_msg.contains("Unsupported"),
            "Error should mention unsupported format"
        );
    }
}
