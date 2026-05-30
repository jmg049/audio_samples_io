//! Format-agnostic streaming writer.
//!
//! [`AudioStreamWrite::write_frames`] is generic over the sample type `T`, which makes the
//! trait **not object-safe** — so `Box<dyn AudioStreamWrite>` is impossible (the same reason
//! the streaming *reader* trait object [`open_streamed_dyn`](crate::open_streamed_dyn) can
//! only expose the non-generic surface).
//!
//! To provide a single type that works across formats while keeping the generic
//! `write_frames`, this module wraps each format's concrete writer in an enum and forwards
//! the trait methods. Dispatch is a cheap `match` (static dispatch, no vtable), so there is
//! no per-call cost beyond the branch.

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
