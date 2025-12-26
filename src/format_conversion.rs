//! Module which provides conversion utilities for audio formats.
//!
//! For example, converting between WAV and FLAC formats
//!
//! # Example
//!
//! ```no_run
//! use audio_samples_io::format_conversion::convert_to_format;
//! use audio_samples_io::wav::WavFile;
//!
//! let wav = WavFile::open("input.wav")?;
//! let flac = convert_to_format(&wav)?;
//! let flac_eile = wav.convert_to_format()?;
//!
//! flac.write("output.flac")?;
//!

use crate::{error::AudioIOResult, flac::FlacFile, traits::AudioFile, wav::WavFile};

pub struct Mp3<'a> {
    data: &'a [u8],
}

pub struct Aac<'a> {
    data: &'a [u8],
}

pub struct Ogg<'a> {
    data: &'a [u8],
}

pub trait ConvertFormat<'a>
where
    Self: AudioFile,
{
    fn convert_to_wav(self) -> AudioIOResult<WavFile<'a>>;
    fn convert_to_flac(self) -> AudioIOResult<FlacFile<'a>>;
    fn convert_to_mp3(self) -> AudioIOResult<Mp3<'a>>;
    fn convert_to_aac(self) -> AudioIOResult<Aac<'a>>;
    fn convert_to_ogg(self) -> AudioIOResult<Ogg<'a>>;
}

impl<'a> ConvertFormat<'a> for WavFile<'a> {
    fn convert_to_wav(self) -> AudioIOResult<WavFile<'a>> {
        Ok(self)
    }

    fn convert_to_flac(self) -> AudioIOResult<FlacFile<'a>> {
        todo!()
    }

    fn convert_to_mp3(self) -> AudioIOResult<Mp3<'a>> {
        todo!()
    }

    fn convert_to_aac(self) -> AudioIOResult<Aac<'a>> {
        todo!()
    }

    fn convert_to_ogg(self) -> AudioIOResult<Ogg<'a>> {
        todo!()
    }
}

impl<'a> ConvertFormat<'a> for FlacFile<'a> {
    fn convert_to_wav(self) -> AudioIOResult<WavFile<'a>> {
        todo!()
    }

    fn convert_to_flac(self) -> AudioIOResult<FlacFile<'a>> {
        Ok(self)
    }

    fn convert_to_mp3(self) -> AudioIOResult<Mp3<'a>> {
        todo!()
    }

    fn convert_to_aac(self) -> AudioIOResult<Aac<'a>> {
        todo!()
    }

    fn convert_to_ogg(self) -> AudioIOResult<Ogg<'a>> {
        todo!()
    }
}
