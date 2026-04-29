//! FLAC decoded audio data handling.
//!
//! This module provides `DecodedAudio`, analogous to WAV's `DataChunk`,
//! which handles conversion from FLAC's internal i32 representation
//! to any target `AudioSample` type via the `ConvertTo` traits.

use std::num::NonZeroU32;

use audio_samples::{AudioSamples, I24, traits::StandardSample};
use ndarray::{Array1, Array2};

use crate::error::{AudioIOError, AudioIOResult};

/// Decoded FLAC audio data.
///
/// FLAC always decodes to i32 samples internally (per the specification).
/// This struct wraps the decoded data and provides type-safe conversion
/// to any target `AudioSample` type, mirroring WAV's `DataChunk::read_samples`.
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Samples per channel (planar format)
    channels: Vec<Vec<i32>>,
    /// Original bits per sample from FLAC stream
    bits_per_sample: u8,
    /// Sample rate in Hz
    sample_rate: u32,
}

impl DecodedAudio {
    /// Create new decoded audio from channel data.
    pub const fn new(channels: Vec<Vec<i32>>, bits_per_sample: u8, sample_rate: u32) -> Self {
        DecodedAudio {
            channels,
            bits_per_sample,
            sample_rate,
        }
    }

    /// Number of channels.
    pub const fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    /// Total samples across all channels.
    pub fn total_samples(&self) -> usize {
        self.num_channels() * self.samples_per_channel()
    }

    /// Bits per sample of the source data.
    pub const fn bits_per_sample(&self) -> u8 {
        self.bits_per_sample
    }

    /// Sample rate in Hz.
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Read samples converting from FLAC's internal format to target type T,
    /// returning AudioSamples (the standard output format).
    ///
    /// This is the main method for getting decoded audio as AudioSamples.
    /// The returned AudioSamples owns its data, so it can be coerced to any lifetime.
    pub fn read_samples<'a, T>(&self, sample_rate: NonZeroU32) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: StandardSample + 'static,
    {
        let num_channels = self.num_channels();
        let samples_per_channel = self.samples_per_channel();

        if num_channels == 0 || samples_per_channel == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "Empty audio data",
                "No channels or samples",
            ));
        }

        if num_channels == 1 {
            // Mono: single allocation, direct conversion into Array1's buffer.
            let data = Array1::from_shape_fn(samples_per_channel, |i| {
                self.convert_one_sample::<T>(self.channels[0][i])
            });
            AudioSamples::new_mono(data, sample_rate).map_err(Into::into)
        } else {
            // Multi-channel: one flat allocation, fill channel-by-channel.
            // Avoids the N intermediate Vec<T> allocations from the old path.
            let mut flat = Vec::with_capacity(num_channels * samples_per_channel);
            for ch in &self.channels {
                for &s in ch {
                    flat.push(self.convert_one_sample::<T>(s));
                }
            }
            let arr =
                Array2::from_shape_vec((num_channels, samples_per_channel), flat).map_err(|e| {
                    AudioIOError::corrupted_data_simple("Array shape error", e.to_string())
                })?;
            AudioSamples::new_multi_channel(arr, sample_rate).map_err(Into::into)
        }
    }

    /// Convert a single i32 FLAC sample to target type T.
    #[inline(always)]
    fn convert_one_sample<T>(&self, s: i32) -> T
    where
        T: StandardSample + 'static,
    {
        use audio_samples::I24;
        match self.bits_per_sample {
            1..=8 => {
                let shift = 16 - self.bits_per_sample;
                T::convert_from((s << shift) as i16)
            }
            9..=16 => T::convert_from(s as i16),
            17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
            _ => T::convert_from(s),
        }
    }

    /// Read samples in planar format as a flat Vec<T>.
    /// Channels are concatenated: [ch0_samples..., ch1_samples..., ...]
    pub fn read_samples_planar<T>(&self) -> AudioIOResult<Vec<T>>
    where
        T: StandardSample + 'static,
    {
        let mut result = Vec::with_capacity(self.total_samples());

        for channel in &self.channels {
            let converted = self.convert_channel_samples::<T>(channel)?;
            result.extend(converted);
        }

        Ok(result)
    }

    /// Read samples for a single channel.
    pub fn read_channel_samples<T>(&self, channel: usize) -> AudioIOResult<Vec<T>>
    where
        T: StandardSample + 'static,
    {
        let samples = self.channels.get(channel).ok_or_else(|| {
            AudioIOError::corrupted_data_simple(
                "Channel index out of bounds",
                format!(
                    "Requested channel {}, have {}",
                    channel,
                    self.num_channels()
                ),
            )
        })?;

        self.convert_channel_samples::<T>(samples)
    }

    /// Read all samples in interleaved format.
    pub fn read_samples_interleaved<T>(&self) -> AudioIOResult<Vec<T>>
    where
        T: StandardSample + 'static,
    {
        let num_channels = self.num_channels();
        let samples_per_channel = self.samples_per_channel();

        if num_channels == 0 || samples_per_channel == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(num_channels * samples_per_channel);

        // Convert each channel first
        let converted_channels: Vec<Vec<T>> = self
            .channels
            .iter()
            .map(|ch| self.convert_channel_samples::<T>(ch))
            .collect::<AudioIOResult<_>>()?;

        // Interleave
        for i in 0..samples_per_channel {
            for ch in &converted_channels {
                result.push(ch[i]);
            }
        }

        Ok(result)
    }

    /// Convert a channel's i32 samples to target type T.
    ///
    /// FLAC stores samples at their native bit depth, sign-extended in i32.
    /// We convert based on the original bit depth to maintain proper scaling.
    fn convert_channel_samples<T>(&self, samples: &[i32]) -> AudioIOResult<Vec<T>>
    where
        T: StandardSample + 'static,
    {
        match self.bits_per_sample {
            1..=8 => {
                // 8-bit or less: scale to 16-bit range then convert
                let shift = 16 - self.bits_per_sample;
                Ok(samples
                    .iter()
                    .map(|&s| {
                        let scaled = (s << shift) as i16;
                        T::convert_from(scaled)
                    })
                    .collect())
            }
            9..=16 => {
                // 9-16 bit: treat as i16
                Ok(samples.iter().map(|&s| T::convert_from(s as i16)).collect())
            }
            17..=24 => {
                // 17-24 bit: convert via I24
                Ok(samples
                    .iter()
                    .map(|&s| T::convert_from(I24::wrapping_from_i32(s)))
                    .collect())
            }
            25..=32 => {
                // 25-32 bit: use full i32
                Ok(samples.iter().map(|&s| T::convert_from(s)).collect())
            }
            _ => Err(AudioIOError::corrupted_data_simple(
                "Invalid bits per sample",
                format!("{} bits", self.bits_per_sample),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoded_audio_basic() {
        let channels = vec![vec![1000i32, 2000, 3000], vec![-1000i32, -2000, -3000]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.total_samples(), 6);
        assert_eq!(audio.bits_per_sample(), 16);
        assert_eq!(audio.sample_rate(), 44100);
    }

    #[test]
    fn test_read_samples_planar_i16() {
        let channels = vec![vec![1000i32, 2000], vec![-1000i32, -2000]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let samples: Vec<i16> = audio.read_samples_planar().unwrap();
        assert_eq!(samples, vec![1000i16, 2000, -1000, -2000]);
    }

    #[test]
    fn test_read_samples_interleaved() {
        let channels = vec![vec![100i32, 200], vec![300i32, 400]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let samples: Vec<i16> = audio.read_samples_interleaved().unwrap();
        // Interleaved: [ch0[0], ch1[0], ch0[1], ch1[1]]
        assert_eq!(samples, vec![100i16, 300, 200, 400]);
    }

    #[test]
    fn test_read_channel_samples() {
        let channels = vec![vec![100i32, 200], vec![300i32, 400]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let ch0: Vec<i16> = audio.read_channel_samples(0).unwrap();
        let ch1: Vec<i16> = audio.read_channel_samples(1).unwrap();

        assert_eq!(ch0, vec![100i16, 200]);
        assert_eq!(ch1, vec![300i16, 400]);
    }

    #[test]
    fn test_24bit_conversion() {
        // 24-bit sample at full scale
        let channels = vec![vec![0x7FFFFFi32, -0x800000i32]];
        let audio = DecodedAudio::new(channels, 24, 48000);

        let samples: Vec<I24> = audio.read_samples_planar().unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn test_read_samples_to_audio_samples() {
        let channels = vec![vec![100i32, 200], vec![300i32, 400]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let sample_rate = NonZeroU32::new(44100).unwrap();
        let samples: AudioSamples<'static, i16> = audio.read_samples(sample_rate).unwrap();
        assert_eq!(samples.num_channels().get(), 2);
        assert_eq!(samples.samples_per_channel().get(), 2);
        assert_eq!(samples.sample_rate(), sample_rate);
    }

    // =========================================================================
    // Additional data.rs tests
    // =========================================================================

    #[test]
    fn test_read_samples_mono() {
        let channels = vec![vec![1000i32, 2000, 3000, 4000]];
        let audio = DecodedAudio::new(channels, 16, 48000);

        let sample_rate = NonZeroU32::new(48000).unwrap();
        let samples: AudioSamples<'static, i16> = audio.read_samples(sample_rate).unwrap();

        assert_eq!(samples.num_channels().get(), 1, "mono");
        assert_eq!(samples.samples_per_channel().get(), 4, "4 samples/ch");
        assert_eq!(samples.sample_rate(), sample_rate);
        assert_eq!(samples.total_samples().get(), 4);
    }

    #[test]
    fn test_read_samples_multi_channel_shape() {
        let n = 8;
        let channels: Vec<Vec<i32>> = (0..6).map(|ch| vec![(ch as i32) * 100; n]).collect();
        let audio = DecodedAudio::new(channels, 24, 96000);

        let sample_rate = NonZeroU32::new(96000).unwrap();
        let samples: AudioSamples<'static, I24> = audio.read_samples(sample_rate).unwrap();

        assert_eq!(samples.num_channels().get(), 6, "6 channels");
        assert_eq!(samples.samples_per_channel().get(), n, "n samples/ch");
        assert_eq!(samples.total_samples().get(), 6 * n, "total samples");
    }

    #[test]
    fn test_empty_audio_returns_error() {
        let audio = DecodedAudio::new(vec![], 16, 44100);
        let sample_rate = NonZeroU32::new(44100).unwrap();
        let result: Result<AudioSamples<'static, i16>, _> = audio.read_samples(sample_rate);
        assert!(result.is_err(), "empty channels should return error");
    }

    #[test]
    fn test_empty_samples_per_channel_returns_error() {
        let audio = DecodedAudio::new(vec![vec![]], 16, 44100);
        let sample_rate = NonZeroU32::new(44100).unwrap();
        let result: Result<AudioSamples<'static, i16>, _> = audio.read_samples(sample_rate);
        assert!(result.is_err(), "zero samples_per_channel should return error");
    }

    #[test]
    fn test_16bit_conversion_preserves_values() {
        // At 16-bit, raw i32 values should come back as i16 unchanged
        let samples_i32 = vec![0i32, 100, -100, 16383, -16384, 32767, -32768];
        let channels = vec![samples_i32.clone()];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let result: Vec<i16> = audio.read_samples_planar().unwrap();
        let expected: Vec<i16> = samples_i32.iter().map(|&s| s as i16).collect();
        assert_eq!(result, expected, "16-bit conversion should preserve values");
    }

    #[test]
    fn test_read_samples_as_f32_normalises() {
        // At 16-bit, i16::MAX should map to approximately 1.0 in f32
        let channels = vec![vec![32767i32, -32768, 0]];
        let audio = DecodedAudio::new(channels, 16, 44100);
        let sample_rate = NonZeroU32::new(44100).unwrap();

        let samples: AudioSamples<'static, f32> = audio.read_samples(sample_rate).unwrap();
        let iv = samples.to_interleaved_vec();
        assert!(iv[0].abs() > 0.9, "max i16 should map to near 1.0 in f32: {}", iv[0]);
        assert!(iv[1] < -0.9, "min i16 should map to near -1.0 in f32: {}", iv[1]);
        assert!(iv[2].abs() < 1e-6, "zero should map to zero in f32: {}", iv[2]);
    }

    #[test]
    fn test_read_samples_as_f64_normalises() {
        let channels = vec![vec![32767i32, -32768, 0]];
        let audio = DecodedAudio::new(channels, 16, 44100);
        let sample_rate = NonZeroU32::new(44100).unwrap();

        let samples: AudioSamples<'static, f64> = audio.read_samples(sample_rate).unwrap();
        let iv = samples.to_interleaved_vec();
        assert!(iv[0].abs() > 0.9, "max i16 → near 1.0 in f64");
        assert!(iv[1] < -0.9, "min i16 → near -1.0 in f64");
        assert!(iv[2].abs() < 1e-12, "zero → 0.0 in f64");
    }

    #[test]
    fn test_total_samples_correct() {
        let channels = vec![vec![0i32; 100]; 3];
        let audio = DecodedAudio::new(channels, 16, 44100);
        assert_eq!(audio.total_samples(), 300, "3 channels × 100 samples = 300");
    }

    #[test]
    fn test_read_channel_samples_oob() {
        let channels = vec![vec![1i32, 2], vec![3i32, 4]];
        let audio = DecodedAudio::new(channels, 16, 44100);

        let result: Result<Vec<i16>, _> = audio.read_channel_samples(5);
        assert!(result.is_err(), "out-of-bounds channel index should fail");
    }

    #[test]
    fn test_8bit_conversion() {
        // 8-bit sample: should be scaled up to 16-bit range
        let channels = vec![vec![127i32, -128]]; // max/min 8-bit
        let audio = DecodedAudio::new(channels, 8, 44100);

        let samples: Vec<i16> = audio.read_samples_planar().unwrap();
        assert_eq!(samples.len(), 2);
        // Should be scaled to 16-bit range: 127 << 8 = 32512
        assert!(samples[0] > 0, "positive 8-bit value should scale positive");
    }
}
