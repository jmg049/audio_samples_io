//! Non-seekable streaming WAV writer (`WavSink`).
//!
//! Unlike [`StreamedWavWriter`](crate::wav::StreamedWavWriter), which backpatches the RIFF and
//! `data` size fields on `finalize()` and therefore needs `Write + Seek`, `WavSink` writes a
//! *final* header up front and never seeks. This makes it suitable for `!Seek` sinks such as
//! stdout, pipes, and network sockets.
//!
//! Two modes:
//! * **Known length** — pass `Some(total_frames)`; the header carries exact sizes and the sink
//!   verifies on `finalize()` that exactly that many frames were written.
//! * **Unknown length** — pass `None`; the header uses the `0xFFFFFFFF` streaming convention.
//!   The resulting file is non-standard but widely readable (this crate's reader clamps such
//!   sizes to the bytes actually present).

use audio_samples::{AudioSamples, StandardSample};

use crate::{
    error::{AudioIOError, AudioIOResult},
    traits::{AudioStreamWrite, AudioStreamWriter},
    types::ValidatedSampleType,
    wav::{
        header::{build_wav_header, build_wav_header_infinite},
        streaming_writer::write_frames_converted,
    },
};
use std::io::Write;

/// A streaming WAV writer for `!Seek` destinations. See the [module docs](self).
#[derive(Debug)]
pub struct WavSink<W>
where
    W: Write,
{
    writer: W,
    channels: u16,
    sample_rate: u32,
    sample_type: ValidatedSampleType,
    /// Declared total frame count when the length is known up front; `None` for unknown length.
    declared_frames: Option<usize>,
    frames_written: usize,
    data_bytes_written: u64,
    finalized: bool,
}

impl<W> WavSink<W>
where
    W: Write,
{
    /// Create a non-seekable WAV sink.
    ///
    /// Pass `total_frames = Some(n)` when the final frame count is known (the header gets exact
    /// sizes); pass `None` for an open-ended stream (the header uses `0xFFFFFFFF` sizes).
    pub fn new(
        mut writer: W,
        channels: u16,
        sample_rate: u32,
        sample_type: ValidatedSampleType,
        total_frames: Option<usize>,
    ) -> AudioIOResult<Self> {
        if channels == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "Invalid channel count",
                "Channel count must be at least 1",
            ));
        }

        let header = match total_frames {
            Some(frames) => build_wav_header(channels, sample_rate, sample_type, frames)?,
            None => build_wav_header_infinite(channels, sample_rate, sample_type)?,
        };
        writer.write_all(&header)?;

        Ok(WavSink {
            writer,
            channels,
            sample_rate,
            sample_type,
            declared_frames: total_frames,
            frames_written: 0,
            data_bytes_written: 0,
            finalized: false,
        })
    }

    /// Sample type this sink encodes to.
    pub const fn target_sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }
}

impl<W> AudioStreamWrite for WavSink<W>
where
    W: Write,
{
    fn write_frames<T>(&mut self, samples: &AudioSamples<'_, T>) -> AudioIOResult<usize>
    where
        T: StandardSample + 'static,
    {
        if self.finalized {
            return Err(AudioIOError::corrupted_data_simple(
                "Cannot write to finalized stream",
                "Call write_frames before finalize()",
            ));
        }

        let input_channels = samples.num_channels();
        if input_channels.get() != self.channels as u32 {
            return Err(AudioIOError::corrupted_data_simple(
                "Channel count mismatch",
                format!(
                    "Sink configured for {} channels, got {} channels",
                    self.channels, input_channels
                ),
            ));
        }

        let frames_per_channel = samples.samples_per_channel().get();

        // With a known length, refuse to overrun the header's declared data size.
        if let Some(declared) = self.declared_frames {
            if self.frames_written + frames_per_channel > declared {
                return Err(AudioIOError::corrupted_data_simple(
                    "Too many frames for declared length",
                    format!(
                        "Sink declared {declared} frames; writing {} more would exceed it (already {})",
                        frames_per_channel, self.frames_written
                    ),
                ));
            }
        }

        let interleaved = samples.data.as_interleaved_vec();
        let bytes_written = write_frames_converted::<T, W>(
            &mut self.writer,
            &interleaved,
            self.sample_type,
            self.channels,
        )?;

        self.frames_written += frames_per_channel;
        self.data_bytes_written += bytes_written as u64;
        Ok(frames_per_channel)
    }
}

impl<W> AudioStreamWriter for WavSink<W>
where
    W: Write,
{
    fn flush(&mut self) -> AudioIOResult<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn finalize(&mut self) -> AudioIOResult<()> {
        if self.finalized {
            return Ok(());
        }

        // For a known length, the header already declared the exact sizes; writing a different
        // number of frames would leave those sizes wrong, so reject the mismatch.
        if let Some(declared) = self.declared_frames {
            if self.frames_written != declared {
                return Err(AudioIOError::corrupted_data_simple(
                    "Frame count does not match declared length",
                    format!(
                        "Sink declared {declared} frames but {} were written",
                        self.frames_written
                    ),
                ));
            }
        }

        // Word-align the data chunk. The header's size fields already account for this pad byte
        // (both the exact and the 0xFFFFFFFF header), so we only need to emit it.
        if self.data_bytes_written % 2 == 1 {
            self.writer.write_all(&[0])?;
        }

        self.writer.flush()?;
        self.finalized = true;
        Ok(())
    }

    fn is_finalized(&self) -> bool {
        self.finalized
    }

    fn frames_written(&self) -> usize {
        self.frames_written
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn num_channels(&self) -> u16 {
        self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::{WavFile, wav_file_len};
    use crate::{
        OpenOptions,
        traits::{AudioFile, AudioFileRead},
    };
    use audio_samples::{AudioSamples, nzu, sample_rate};

    #[test]
    fn known_length_sink_produces_exact_standard_file() {
        // A plain Vec is Write but not Seek — exactly the target use case.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut sink = WavSink::new(&mut buf, 1, 44_100, ValidatedSampleType::I16, Some(256))
                .expect("create sink");
            let audio = AudioSamples::<f32>::zeros_mono(nzu!(256), sample_rate!(44_100));
            sink.write_frames(&audio).expect("write");
            sink.finalize().expect("finalize");
        }
        // Exact size, and a valid file our own reader accepts.
        assert_eq!(buf.len(), wav_file_len(1, ValidatedSampleType::I16, 256));
        assert_eq!(&buf[0..4], b"RIFF");

        let path = std::env::temp_dir().join(format!("wavsink_{}.wav", std::process::id()));
        std::fs::write(&path, &buf).expect("write temp file");
        let wav = <WavFile as AudioFile>::open_with_options(&path, OpenOptions::default())
            .expect("open temp wav");
        let read = <WavFile as AudioFileRead>::read::<i16>(&wav).expect("read wav");
        assert_eq!(read.samples_per_channel().get(), 256);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn unknown_length_sink_is_readable_after_clamping() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut sink =
                WavSink::new(&mut buf, 1, 8_000, ValidatedSampleType::I16, None).expect("create");
            let audio = AudioSamples::<f32>::zeros_mono(nzu!(100), sample_rate!(8_000));
            sink.write_frames(&audio).expect("write");
            sink.finalize().expect("finalize");
        }
        // 0xFFFFFFFF size fields, but our reader clamps and reads the real 100 frames.
        let path = std::env::temp_dir().join(format!("wavsink_inf_{}.wav", std::process::id()));
        std::fs::write(&path, &buf).expect("write temp file");
        let wav = <WavFile as AudioFile>::open_with_options(&path, OpenOptions::default())
            .expect("open temp wav");
        let read = <WavFile as AudioFileRead>::read::<i16>(&wav).expect("read wav");
        assert_eq!(read.samples_per_channel().get(), 100);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn declared_length_overrun_is_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        let mut sink =
            WavSink::new(&mut buf, 1, 44_100, ValidatedSampleType::I16, Some(10)).expect("create");
        let audio = AudioSamples::<f32>::zeros_mono(nzu!(20), sample_rate!(44_100));
        assert!(sink.write_frames(&audio).is_err());
    }
}
