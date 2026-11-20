//! One streaming writer type for WAV and FLAC.
//!
//! [`StreamedAudioWriter`] writes a file frame block by frame block, in WAV or FLAC, so
//! code that streams audio out does not depend on the format. [`create_streamed`]
//! chooses the format from the file extension and
//! [`create_streamed_with`](crate::create_streamed_with) takes it explicitly. Each call to
//! [`write_frames`](AudioStreamWrite::write_frames) encodes one block, and
//! [`finalize`](AudioStreamWriter::finalize) completes the header.
//!
//! The type is an enum over the concrete writers rather than a trait object, because
//! `write_frames` is generic over the sample type, which makes [`AudioStreamWrite`] not
//! object-safe.
//!
//! [`create_streamed`]: crate::create_streamed

use audio_samples::{AudioSamples, traits::StandardSample};

use crate::{
    WriteSeek,
    error::AudioIOResult,
    traits::{AudioStreamWrite, AudioStreamWriter},
};

/// A streaming audio writer that is agnostic to the underlying file format.
///
/// Returned by [`create_streamed`](crate::create_streamed) and
/// [`create_streamed_with`](crate::create_streamed_with). It implements
/// [`AudioStreamWriter`] and [`AudioStreamWrite`], forwarding every call to the wrapped
/// WAV or FLAC writer, so you can write format-independent code:
///
/// ```no_run
/// use audio_samples_io::create_streamed;
/// use audio_samples_io::traits::{AudioStreamWrite, AudioStreamWriter};
/// use audio_samples::{AudioSamples, channels, nzu, sample_rate};
///
/// // Same code path whether the extension is .wav or .flac.
/// let mut writer = create_streamed::<_, i16>("out.flac", 2, 44100)?;
/// let chunk = AudioSamples::<i16>::zeros_multi(channels!(2), nzu!(4096), sample_rate!(44100));
/// writer.write_frames(&chunk)?;
/// writer.finalize()?;
/// # Ok::<(), audio_samples_io::error::AudioIOError>(())
/// ```
#[derive(Debug)]
pub enum StreamedAudioWriter<W>
where
    W: WriteSeek,
{
    /// A streaming WAV writer.
    #[cfg(feature = "wav")]
    Wav(crate::wav::StreamedWavWriter<W>),
    /// A streaming FLAC writer.
    #[cfg(feature = "flac")]
    Flac(crate::flac::StreamedFlacWriter<W>),
}

impl<W> AudioStreamWriter for StreamedAudioWriter<W>
where
    W: WriteSeek,
{
    fn flush(&mut self) -> AudioIOResult<()> {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => w.flush(),
            #[cfg(feature = "flac")]
            Self::Flac(w) => w.flush(),
        }
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => w.finalize(),
            #[cfg(feature = "flac")]
            Self::Flac(w) => w.finalize(),
        }
    }

    fn is_finalized(&self) -> bool {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => w.is_finalized(),
            #[cfg(feature = "flac")]
            Self::Flac(w) => w.is_finalized(),
        }
    }

    fn frames_written(&self) -> usize {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => w.frames_written(),
            #[cfg(feature = "flac")]
            Self::Flac(w) => w.frames_written(),
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => AudioStreamWriter::sample_rate(w),
            #[cfg(feature = "flac")]
            Self::Flac(w) => AudioStreamWriter::sample_rate(w),
        }
    }

    fn num_channels(&self) -> u16 {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => AudioStreamWriter::num_channels(w),
            #[cfg(feature = "flac")]
            Self::Flac(w) => AudioStreamWriter::num_channels(w),
        }
    }
}

impl<W> AudioStreamWrite for StreamedAudioWriter<W>
where
    W: WriteSeek,
{
    fn write_frames<T>(&mut self, samples: &AudioSamples<'_, T>) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        match self {
            #[cfg(feature = "wav")]
            Self::Wav(w) => w.write_frames(samples),
            #[cfg(feature = "flac")]
            Self::Flac(w) => w.write_frames(samples),
        }
    }
}
