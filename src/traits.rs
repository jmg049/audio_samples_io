use core::fmt::{Debug, Display};
use std::{path::Path, time::Duration};

use crate::{
    error::{AudioIOError, AudioIOResult},
    types::{AudioInfo, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType},
};
use audio_samples::{AudioSample, AudioSamples, ConvertTo, I24, SampleType};

/// Marker trait for audio info structs
pub trait AudioInfoMarker {}

pub trait SpecificAudioInfo: Debug + Display + Send + Sync {
    /// Get the specific audio info as a trait object allocated on the heap
    fn specific_info(&self) -> impl AudioInfoMarker;
}

pub trait AudioFileMetadata {
    // CONSTRUCTORS (metadata-only)
    /// Open file for metadata operations only (no sample type needed)
    fn open_metadata<P: AsRef<Path>>(path: P) -> AudioIOResult<Self>
    where
        Self: Sized;

    // FILE INFO (no sample type needed)
    /// Get basic audio information from file headers
    fn base_info(&self) -> AudioIOResult<BaseAudioInfo>;
    /// Get format-specific audio information
    fn specific_info(&self) -> impl AudioInfoMarker;
    /// Get complete audio information (base + specific)
    fn info(&self) -> AudioIOResult<AudioInfo<impl AudioInfoMarker>> {
        let fp = self.file_path().to_path_buf();
        let base_info = self.base_info()?;
        let specific_info = self.specific_info();
        Ok(AudioInfo {
            fp,
            base_info,
            specific_info,
        })
    }
    /// Get the file format type
    fn file_type(&self) -> FileType;
    /// Get the file path
    fn file_path(&self) -> &Path;

    // SIZE INFO (no sample type needed)
    /// Get total number of samples in the file
    fn total_samples(&self) -> usize;
    /// Get duration of the audio
    fn duration(&self) -> AudioIOResult<Duration>;
    /// Get samples per channel
    fn samples_per_channel(&self) -> AudioIOResult<usize> {
        let base = self.base_info()?;
        if base.channels > 0 {
            Ok(self.total_samples() / base.channels as usize)
        } else {
            Err(AudioIOError::corrupted_data_simple(
                "Invalid channel count in audio file",
                base.channels.to_string(),
            ))
        }
    }

    /// Get the sample data type of the audio file
    /// SampleType should be validated during file opening
    fn sample_type(&self) -> ValidatedSampleType;

    /// Get the number of channels in the audio file
    fn num_channels(&self) -> u16;
}

pub trait AudioFile {
    fn open<P: AsRef<Path>>(fp: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        AudioFile::open_with_options(fp, OpenOptions::default())
    }

    fn open_with_options<P: AsRef<Path>>(fp: P, options: OpenOptions) -> AudioIOResult<Self>
    where
        Self: Sized;

    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for reading audio samples from a file
///
/// ## Why not just implement std::io::Read?
///
/// The standard library's `Read` trait is designed for reading raw bytes, which while close to the use case of audio_samples_io,
/// does not directly map to reading audio samples of various types and formats, hence the need for a more specific implementation
///
/// ## Generic Constraints
///
/// The trait does not impose any specific type constraints at the declaration level and instead uses generic constraints on individual methods.
/// This allows for greater flexibility in implementing the trait and avoids also allows the WavFile struct to not have an associated generic type parameter,
/// which complicates usage in terms of forcing the user to specify a type parameter when opening a file even if they don't intend to read samples immediately.
pub trait AudioFileRead<'a> {
    fn read<T>(&'a self) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;
}

pub trait AudioFileWrite: AudioFile {
    /// Write audio samples to the file
    fn write<P: AsRef<Path>, T: AudioSample>(&mut self, out_fp: P) -> AudioIOResult<()>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;
}

pub trait SupportsSampleTypes<const N: usize>: AudioFile {
    const SUPPORTED_SAMPLE_TYPES: [SampleType; N];

    fn supports_sample_type(sample_type: SampleType) -> bool {
        Self::SUPPORTED_SAMPLE_TYPES.contains(&sample_type)
    }
}

// ============================================================================
// STREAMING TRAITS
// ============================================================================

/// Base trait for streaming audio readers (object-safe).
///
/// This trait provides non-generic methods for position tracking and seeking,
/// enabling use as a trait object (`Box<dyn AudioStreamReader>`).
///
/// For reading actual audio data, see [`AudioStreamRead`] which extends this trait
/// with generic methods for type-safe sample conversion.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::traits::AudioStreamReader;
///
/// fn process_stream(stream: &mut dyn AudioStreamReader) {
///     println!("Position: {}/{}", stream.current_frame(), stream.total_frames());
///     stream.seek_to_frame(1000).unwrap();
///     stream.reset().unwrap();
/// }
/// ```
pub trait AudioStreamReader {
    /// Get the current frame position (0-indexed).
    fn current_frame(&self) -> usize;

    /// Get the number of remaining frames from current position.
    fn remaining_frames(&self) -> usize;

    /// Get the total number of frames in the stream.
    fn total_frames(&self) -> usize;

    /// Get the sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Get the number of bytes per frame (block align).
    fn bytes_per_frame(&self) -> usize;

    /// Seek to a specific frame position.
    ///
    /// # Arguments
    ///
    /// * `frame` - The frame index to seek to (0-indexed)
    ///
    /// # Errors
    ///
    /// Returns an error if the frame is beyond the end of the stream or seek fails.
    fn seek_to_frame(&mut self, frame: usize) -> AudioIOResult<()>;

    /// Reset to the beginning of the audio data.
    fn reset(&mut self) -> AudioIOResult<()>;
}

/// Trait for streaming audio reads with type conversion.
///
/// This trait extends [`AudioStreamReader`] with generic methods for reading
/// audio samples with automatic type conversion. Due to the generic type parameter,
/// this trait is NOT object-safe.
///
/// Types implementing this trait should also implement [`AudioFileMetadata`]
/// to provide access to audio format information.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::traits::{AudioStreamRead, AudioStreamReader, AudioFileMetadata};
/// use audio_samples::AudioSamples;
/// use std::num::NonZeroU32;
///
/// fn read_all<S: AudioStreamRead + AudioFileMetadata>(stream: &mut S) {
///     let channels = stream.num_channels() as usize;
///     let sample_rate = NonZeroU32::new(stream.sample_rate()).unwrap();
///     let mut buffer = AudioSamples::<f32>::zeros_multi(channels, 1024, sample_rate);
///     
///     while stream.remaining_frames() > 0 {
///         let frames = stream.read_frames_into(&mut buffer, 1024).unwrap();
///         // Process frames...
///     }
/// }
/// ```
pub trait AudioStreamRead: AudioStreamReader + AudioFileMetadata {
    /// Read frames into a pre-allocated `AudioSamples` buffer.
    ///
    /// This is the primary zero-allocation read method. After initial buffer
    /// allocation, repeated calls reuse the same memory.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Pre-allocated `AudioSamples` to fill with frame data
    /// * `frame_count` - Maximum number of frames to read
    ///
    /// # Returns
    ///
    /// The actual number of frames read (may be less at end of stream).
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or data is corrupted.
    fn read_frames_into<T>(
        &mut self,
        buffer: &mut AudioSamples<'_, T>,
        frame_count: usize,
    ) -> AudioIOResult<usize>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;
}

// ============================================================================
// STREAMING WRITE TRAITS
// ============================================================================

/// Base trait for streaming audio writers (object-safe).
///
/// This trait provides non-generic methods for streaming write operations,
/// enabling use as a trait object (`Box<dyn AudioStreamWriter>`).
///
/// For writing actual audio data, see [`AudioStreamWrite`] which extends this
/// trait with generic methods for type-safe sample encoding.
///
/// # Finalization
///
/// Audio formats often require updating headers with final size information.
/// Call [`finalize()`](AudioStreamWriter::finalize) when done writing to ensure
/// the output is valid. The internal `finalized` flag prevents double-finalization.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::traits::AudioStreamWriter;
///
/// fn finish_stream(writer: &mut dyn AudioStreamWriter) -> Result<(), audio_samples_io::error::AudioIOError> {
///     writer.flush()?;
///     writer.finalize()?;
///     Ok(())
/// }
/// ```
pub trait AudioStreamWriter {
    /// Flush any buffered data to the underlying writer.
    ///
    /// This should be called periodically to ensure data is written,
    /// especially for long-running streaming operations.
    fn flush(&mut self) -> AudioIOResult<()>;

    /// Finalize the audio stream, updating headers with final size information.
    ///
    /// This method should be called exactly once when writing is complete.
    /// It updates format-specific headers (e.g., WAV RIFF/data chunk sizes)
    /// and flushes all remaining data.
    ///
    /// # Idempotency
    ///
    /// Implementations should track finalization state internally. Calling
    /// `finalize()` multiple times should succeed without re-writing headers.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking to update headers fails, or if the
    /// underlying writer cannot be flushed.
    fn finalize(&mut self) -> AudioIOResult<()>;

    /// Check if the stream has been finalized.
    fn is_finalized(&self) -> bool;

    /// Get the number of frames written so far.
    fn frames_written(&self) -> usize;

    /// Get the sample rate this writer was configured with.
    fn sample_rate(&self) -> u32;

    /// Get the number of channels this writer was configured with.
    fn num_channels(&self) -> u16;
}

/// Trait for streaming audio writes with type conversion.
///
/// This trait extends [`AudioStreamWriter`] with generic methods for writing
/// audio samples. Due to the generic type parameter, this trait is NOT object-safe.
///
/// # Sample Type
///
/// Streaming writers are typically configured with a target sample type at
/// construction (e.g., `StreamedWavWriter::new_i16()`) for predictable output.
/// The `write_frames()` method converts from the input sample type to the
/// writer's configured output type.
///
/// # Example
///
/// ```no_run
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::AudioSamples;
///
/// fn write_audio<W: AudioStreamWrite>(
///     writer: &mut W,
///     audio: &AudioSamples<f32>,
/// ) -> Result<(), audio_samples_io::error::AudioIOError> {
///     writer.write_frames(audio)?;
///     writer.finalize()?;
///     Ok(())
/// }
/// ```
pub trait AudioStreamWrite: AudioStreamWriter {
    /// Write audio frames to the stream.
    ///
    /// Samples are converted from the input type `T` to the writer's configured
    /// output sample type. For multi-channel audio, the input should match the
    /// writer's channel configuration.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples to write
    ///
    /// # Returns
    ///
    /// The number of frames written.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The channel count doesn't match the writer's configuration
    /// - The underlying writer fails
    /// - The stream has already been finalized
    fn write_frames<T>(&mut self, samples: &AudioSamples<'_, T>) -> AudioIOResult<usize>
    where
        T: AudioSample + 'static,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>;
}
