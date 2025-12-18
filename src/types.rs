use core::fmt::{Debug, Display, Formatter, Result as FmtResult};
use core::str::FromStr;
use std::borrow::{Borrow, Cow};
use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::Duration;

use audio_samples::SampleType;
use memmap2::Mmap;

use crate::error::AudioIOError;
use crate::traits::AudioInfoMarker;

/// Meant to reflect the SampleType enum but only allow valid sample types for audio files, everthing but unknown
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatedSampleType {
    I16,
    I24,
    I32,
    F32,
    F64,
}

impl ValidatedSampleType {
    pub const fn bits_per_sample(&self) -> usize {
        match self {
            ValidatedSampleType::I16 => 16,
            ValidatedSampleType::I24 => 24,
            ValidatedSampleType::I32 | ValidatedSampleType::F32 => 32,
            ValidatedSampleType::F64 => 64,
        }
    }

    pub const fn bytes_per_sample(&self) -> usize {
        self.bits_per_sample() / 8
    }
}

impl TryFrom<SampleType> for ValidatedSampleType {
    type Error = AudioIOError;

    fn try_from(value: SampleType) -> Result<Self, Self::Error> {
        match value {
            SampleType::I16 => Ok(ValidatedSampleType::I16),
            SampleType::I24 => Ok(ValidatedSampleType::I24),
            SampleType::I32 => Ok(ValidatedSampleType::I32),
            SampleType::F32 => Ok(ValidatedSampleType::F32),
            SampleType::F64 => Ok(ValidatedSampleType::F64),
            _ => Err(AudioIOError::corrupted_data_simple(
                "Unsupported sample type for audio file",
                format!("{:?}", value),
            )),
        }
    }
}

impl From<ValidatedSampleType> for SampleType {
    fn from(val: ValidatedSampleType) -> Self {
        match val {
            ValidatedSampleType::I16 => SampleType::I16,
            ValidatedSampleType::I24 => SampleType::I24,
            ValidatedSampleType::I32 => SampleType::I32,
            ValidatedSampleType::F32 => SampleType::F32,
            ValidatedSampleType::F64 => SampleType::F64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioInfo<I: AudioInfoMarker> {
    pub fp: PathBuf,
    pub base_info: BaseAudioInfo,
    pub specific_info: I,
}

/// Audio file container format types
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FileType {
    /// WAV container
    #[default]
    WAV,
    /// MP3 container
    MP3,
    /// OGG container
    OGG,
    /// FLAC container
    FLAC,
    /// AIFF container
    AIFF,
    /// Unknown or unsupported container
    Unknown,
}

impl FileType {
    /// Canonical lowercase file extension
    pub const fn as_str(self) -> &'static str {
        match self {
            FileType::WAV => "wav",
            FileType::MP3 => "mp3",
            FileType::OGG => "ogg",
            FileType::FLAC => "flac",
            FileType::AIFF => "aiff",
            FileType::Unknown => "unknown",
        }
    }

    /// Human-readable descriptive name
    pub const fn description(self) -> &'static str {
        match self {
            FileType::WAV => "Waveform Audio File Format",
            FileType::MP3 => "MPEG-1 Audio Layer III",
            FileType::OGG => "Ogg Vorbis Container",
            FileType::FLAC => "Free Lossless Audio Codec",
            FileType::AIFF => "Audio Interchange File Format",
            FileType::Unknown => "Unknown or unsupported audio container",
        }
    }

    /// True if this container is compressed
    pub const fn is_compressed(self) -> bool {
        matches!(self, FileType::MP3 | FileType::OGG | FileType::FLAC)
    }

    /// True if this container is lossless
    pub const fn is_lossless(self) -> bool {
        matches!(self, FileType::WAV | FileType::FLAC | FileType::AIFF)
    }

    /// Detect file type from path extension without allocating
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) else {
            return FileType::Unknown;
        };

        ext.parse().unwrap_or(FileType::Unknown)
    }
}

impl Display for FileType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if f.alternate() {
            // "{:#}" → human-readable container name
            write!(f, "{}", self.description())
        } else {
            // "{}" → canonical extension
            write!(f, "{}", self.as_str())
        }
    }
}

impl FromStr for FileType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "wav" | "WAV" => Ok(FileType::WAV),
            "mp3" | "MP3" => Ok(FileType::MP3),
            "ogg" | "OGG" => Ok(FileType::OGG),
            "flac" | "FLAC" => Ok(FileType::FLAC),
            "aiff" | "aif" | "AIFF" | "AIF" => Ok(FileType::AIFF),
            _ => Err(()),
        }
    }
}

/// Audio file information structure
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BaseAudioInfo {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Bits per sample (8, 16, 24, 32)
    pub bits_per_sample: u16,
    /// Bytes per sample
    pub bytes_per_sample: u16,

    /// Byte rate (bytes per second)
    pub byte_rate: u32,
    /// Block align (bytes per sample frame)
    pub block_align: u16,

    /// Total number of samples
    pub total_samples: usize,
    /// Duration in seconds
    pub duration: Duration,
    /// Audio file format
    pub file_type: FileType,
    /// Sample type
    pub sample_type: SampleType,
}

impl BaseAudioInfo {
    pub const fn new(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        bytes_per_sample: u16,
        byte_rate: u32,
        block_align: u16,
        total_samples: usize,
        duration: Duration,
        file_type: FileType,
        sample_type: SampleType,
    ) -> Self {
        BaseAudioInfo {
            sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            total_samples,
            duration,
            file_type,
            sample_type,
        }
    }
}

impl Display for BaseAudioInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        //
        // -------- COMPACT MODE --------
        //
        if !f.alternate() {
            return write!(
                f,
                "{} {} | {} Hz, {} ch, {}-bit, {:.2} s",
                self.file_type,
                self.sample_type,
                self.sample_rate,
                self.channels,
                self.bits_per_sample,
                self.duration.as_secs_f32()
            );
        }

        //
        // -------- PRETTY MODE --------
        //

        // ============ COLOURED VERSION ============
        #[cfg(feature = "colored")]
        {
            // Helper functions (NOT closures with impl Trait)
            fn label(s: &str) -> ColoredString {
                s.bold().bright_blue()
            }

            fn value<T: ToString>(v: T) -> ColoredString {
                v.to_string().bright_green()
            }

            writeln!(f, "{}", "Audio Info".bold().underline())?;

            writeln!(f, "├─ {}: {}", label("File Type"), value(self.format))?;
            writeln!(
                f,
                "├─ {}: {}",
                label("Sample Type"),
                value(format!("{}", self.sample_type))
            )?;
            writeln!(
                f,
                "├─ {}: {}",
                label("Sample Rate"),
                value(format!("{} Hz", self.sample_rate))
            )?;
            writeln!(f, "├─ {}: {}", label("Channels"), value(self.channels))?;
            writeln!(
                f,
                "├─ {}: {}",
                label("Bits per Sample"),
                value(format!("{}-bit", self.bits_per_sample))
            )?;
            writeln!(
                f,
                "├─ {}: {}",
                label("Bytes per Sample"),
                value(format!("{} bytes", self.bytes_per_sample))
            )?;
            writeln!(
                f,
                "├─ {}: {}",
                label("Total Samples"),
                value(self.n_samples)
            )?;
            writeln!(
                f,
                "└─ {}: {}",
                label("Duration"),
                value(format!("{:.2} s", self.duration.as_secs_f32()))
            )
        }

        // ============ NON-COLOURED VERSION ============
        #[cfg(not(feature = "colored"))]
        {
            writeln!(f, "Audio Info:")?;
            writeln!(f, "├─ File Type: {}", self.file_type)?;
            writeln!(f, "├─ Sample Type: {}", self.sample_type)?;
            writeln!(f, "├─ Sample Rate: {} Hz", self.sample_rate)?;
            writeln!(f, "├─ Channels: {}", self.channels)?;
            writeln!(f, "├─ Bits per Sample: {}-bit", self.bits_per_sample)?;
            writeln!(f, "├─ Bytes per Sample: {} bytes", self.bytes_per_sample)?;
            writeln!(f, "├─ Total Samples: {}", self.total_samples)?;
            writeln!(f, "└─ Duration: {:.2} s", self.duration.as_secs_f32())
        }
    }
}

/// Unified view over audio byte storage
#[non_exhaustive]
pub enum AudioDataSource<'a> {
    /// Owned heap-allocated byte buffer
    Owned(Vec<u8>),

    /// Memory-mapped file (zero-copy, OS-backed)
    MemoryMapped(Mmap),

    /// Borrowed byte slice
    Borrowed(&'a [u8]),
}

impl<'a> AudioDataSource<'a> {
    /// Returns the audio data as a contiguous byte slice
    #[inline]
    pub fn as_bytes(&'a self) -> &'a [u8] {
        match self {
            AudioDataSource::Owned(data) => data.as_slice(),
            AudioDataSource::MemoryMapped(mmap) => mmap.as_ref(),
            AudioDataSource::Borrowed(slice) => slice,
        }
    }

    /// Returns the length of the buffer in bytes
    #[inline]
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns true if the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a borrowed view if possible, otherwise allocates
    pub fn to_cow(&'a self) -> Cow<'a, [u8]> {
        match self {
            AudioDataSource::Borrowed(slice) => Cow::Borrowed(slice),
            AudioDataSource::Owned(vec) => Cow::Borrowed(vec.as_slice()),
            AudioDataSource::MemoryMapped(mmap) => Cow::Borrowed(mmap.as_ref()),
        }
    }

    /// Forces this source into an owned buffer
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            AudioDataSource::Owned(data) => data,
            AudioDataSource::Borrowed(slice) => slice.to_vec(),
            AudioDataSource::MemoryMapped(mmap) => mmap.as_ref().to_vec(),
        }
    }

    /// Create a memory-mapped source from a file
    pub fn from_file(file: &File) -> io::Result<Self> {
        let mmap = unsafe { Mmap::map(file)? };
        Ok(AudioDataSource::MemoryMapped(mmap))
    }
}

impl<'a> Deref for AudioDataSource<'a> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl<'a> From<Vec<u8>> for AudioDataSource<'a> {
    fn from(value: Vec<u8>) -> Self {
        AudioDataSource::Owned(value)
    }
}

impl<'a> From<&'a [u8]> for AudioDataSource<'a> {
    fn from(value: &'a [u8]) -> Self {
        AudioDataSource::Borrowed(value)
    }
}

impl<'a> From<Mmap> for AudioDataSource<'a> {
    fn from(value: Mmap) -> Self {
        AudioDataSource::MemoryMapped(value)
    }
}

impl<'a> From<AudioDataSource<'a>> for Vec<u8> {
    fn from(value: AudioDataSource<'a>) -> Self {
        value.into_owned()
    }
}

impl<'a> AsRef<[u8]> for AudioDataSource<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> Borrow<[u8]> for AudioDataSource<'a> {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> IntoIterator for &'a AudioDataSource<'a> {
    type Item = u8;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, u8>>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_bytes().iter().copied()
    }
}

impl<'a> PartialEq<[u8]> for AudioDataSource<'a> {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }
}

impl<'a> PartialEq<AudioDataSource<'a>> for [u8] {
    fn eq(&self, other: &AudioDataSource<'a>) -> bool {
        self == other.as_bytes()
    }
}

impl<'a> From<Cow<'a, [u8]>> for AudioDataSource<'a> {
    fn from(value: Cow<'a, [u8]>) -> Self {
        match value {
            Cow::Borrowed(slice) => AudioDataSource::Borrowed(slice),
            Cow::Owned(vec) => AudioDataSource::Owned(vec),
        }
    }
}

impl<'a> Debug for AudioDataSource<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            AudioDataSource::Owned(data) => f
                .debug_struct("AudioDataSource::Owned")
                .field("len", &data.len())
                .finish(),
            AudioDataSource::MemoryMapped(mmap) => f
                .debug_struct("AudioDataSource::MemoryMapped")
                .field("len", &mmap.len())
                .finish(),
            AudioDataSource::Borrowed(slice) => f
                .debug_struct("AudioDataSource::Borrowed")
                .field("len", &slice.len())
                .finish(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    pub use_memory_map: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            use_memory_map: true,
        }
    }
}

#[allow(dead_code)]
const fn _assert_send_sync()
where
    AudioDataSource<'static>: Send + Sync,
{
}
