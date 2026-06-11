use core::fmt::{Debug, Display, Formatter, Result as FmtResult};
use core::str::FromStr;
use std::borrow::{Borrow, Cow};
use std::fs::File;
use std::io;
use std::num::{NonZeroU32, NonZeroUsize};
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
    U8,
    I16,
    I24,
    I32,
    F32,
    F64,
}

impl ValidatedSampleType {
    pub const fn bits_per_sample(&self) -> NonZeroUsize {
        match self {
            ValidatedSampleType::U8 => audio_samples::nzu!(8),
            ValidatedSampleType::I16 => audio_samples::nzu!(16),
            ValidatedSampleType::I24 => audio_samples::nzu!(24),
            ValidatedSampleType::I32 | ValidatedSampleType::F32 => audio_samples::nzu!(32),
            ValidatedSampleType::F64 => audio_samples::nzu!(64),
        }
    }

    pub const fn bytes_per_sample(&self) -> NonZeroUsize {
        self.bits_per_sample().div_ceil(audio_samples::nzu!(8))
    }
}

impl TryFrom<SampleType> for ValidatedSampleType {
    type Error = AudioIOError;

    fn try_from(value: SampleType) -> Result<Self, Self::Error> {
        match value {
            SampleType::U8 => Ok(ValidatedSampleType::U8),
            SampleType::I16 => Ok(ValidatedSampleType::I16),
            SampleType::I24 => Ok(ValidatedSampleType::I24),
            SampleType::I32 => Ok(ValidatedSampleType::I32),
            SampleType::F32 => Ok(ValidatedSampleType::F32),
            SampleType::F64 => Ok(ValidatedSampleType::F64),
            _ => Err(AudioIOError::unsupported_format(format!(
                "Unsupported sample type: {value:?}"
            ))),
        }
    }
}

impl From<ValidatedSampleType> for SampleType {
    fn from(val: ValidatedSampleType) -> Self {
        match val {
            ValidatedSampleType::U8 => SampleType::U8,
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

    /// Number of leading bytes [`FileType::from_magic_bytes`] needs to classify
    /// any supported container.
    pub const MAGIC_LEN: usize = 12;

    /// Detect file type from the leading bytes of the file contents.
    ///
    /// Recognises the magic numbers of every supported container:
    ///
    /// | Signature | Detected type |
    /// |---|---|
    /// | `RIFF....WAVE`, `RF64....WAVE`, `BW64....WAVE` | [`FileType::WAV`] |
    /// | `fLaC` | [`FileType::FLAC`] |
    /// | `FORM....AIFF`, `FORM....AIFC` | [`FileType::AIFF`] |
    /// | `OggS` | [`FileType::OGG`] |
    /// | `ID3` tag or an MPEG frame sync (`0xFF 0xEx/0xFx`) | [`FileType::MP3`] |
    ///
    /// Returns [`FileType::Unknown`] when `header` is too short or matches no
    /// known signature. Pass at least [`FileType::MAGIC_LEN`] bytes when available.
    pub const fn from_magic_bytes(header: &[u8]) -> Self {
        match header {
            // 32-bit RIFF and the RF64/BW64 64-bit variants share the WAVE form type.
            [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'A', b'V', b'E', ..]
            | [b'R', b'F', b'6', b'4', _, _, _, _, b'W', b'A', b'V', b'E', ..]
            | [b'B', b'W', b'6', b'4', _, _, _, _, b'W', b'A', b'V', b'E', ..] => FileType::WAV,
            [b'f', b'L', b'a', b'C', ..] => FileType::FLAC,
            [b'F', b'O', b'R', b'M', _, _, _, _, b'A', b'I', b'F', b'F', ..]
            | [b'F', b'O', b'R', b'M', _, _, _, _, b'A', b'I', b'F', b'C', ..] => FileType::AIFF,
            [b'O', b'g', b'g', b'S', ..] => FileType::OGG,
            [b'I', b'D', b'3', ..] => FileType::MP3,
            // Bare MPEG audio frame sync: 11 set bits across the first two bytes.
            [0xFF, b1, ..] if *b1 & 0xE0 == 0xE0 => FileType::MP3,
            _ => FileType::Unknown,
        }
    }

    /// Detect file type from contents first, falling back to the extension.
    ///
    /// Reads up to [`FileType::MAGIC_LEN`] bytes from `path` and matches them with
    /// [`FileType::from_magic_bytes`]. When the file cannot be opened (e.g. it does
    /// not exist yet) or its leading bytes match no known signature, the result of
    /// [`FileType::from_path`] is returned instead, so extension-only behaviour is
    /// preserved for paths that cannot be sniffed.
    pub fn detect<P: AsRef<Path>>(path: P) -> Self {
        use std::io::Read;

        let path = path.as_ref();
        let mut header = [0u8; Self::MAGIC_LEN];
        let sniffed = std::fs::File::open(path)
            .and_then(|mut f| {
                let mut filled = 0;
                // A regular file returns everything in one read, but loop for
                // correctness on short reads from special files.
                loop {
                    match f.read(&mut header[filled..])? {
                        0 => break,
                        n => filled += n,
                    }
                    if filled == header.len() {
                        break;
                    }
                }
                Ok(Self::from_magic_bytes(&header[..filled]))
            })
            .unwrap_or(FileType::Unknown);

        match sniffed {
            FileType::Unknown => Self::from_path(path),
            known => known,
        }
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
    pub sample_rate: NonZeroU32,
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
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        sample_rate: NonZeroU32,
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

        // -------- PRETTY MODE --------
        //

        // ============ COLOURED VERSION ============
        #[cfg(feature = "colored")]
        {
            use colored::{ColoredString, Colorize};
            // Helper functions (NOT closures with impl Trait)
            fn label(s: &str) -> ColoredString {
                s.bold().bright_blue()
            }

            fn value<T: ToString>(v: T) -> ColoredString {
                v.to_string().bright_green()
            }

            writeln!(f, "{}", "Audio Info".bold().underline())?;

            writeln!(f, "├─ {}: {}", label("File Type"), value(self.file_type))?;
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
            writeln!(f, "├─ {}: {}", label("Total Samples"), value(self.total_samples))?;
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
        OpenOptions { use_memory_map: true }
    }
}

/// Default write-buffer capacity: 4 MiB.
///
/// Large enough to hold a full 10-second stereo file (≤ 3.5 MiB) in one shot,
/// flushing the kernel page-cache in a single `write()` syscall.
pub const DEFAULT_WRITE_BUF_CAPACITY: usize = 4 * 1024 * 1024;

/// Options controlling how audio data is written.
///
/// # Example
///
/// ```
/// use audio_samples_io::types::WriteOptions;
///
/// // Default: 4 MiB write buffer.
/// let default_opts = WriteOptions::default();
///
/// // Smaller buffer for memory-constrained environments.
/// let small_opts = WriteOptions { write_buf_capacity: 256 * 1024 };
///
/// // Larger buffer for writing many multi-minute files sequentially.
/// let large_opts = WriteOptions { write_buf_capacity: 16 * 1024 * 1024 };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WriteOptions {
    /// Size of the internal write buffer in bytes.
    ///
    /// A larger buffer reduces the number of `write()` syscalls at the cost of a larger
    /// upfront allocation.  The default (`DEFAULT_WRITE_BUF_CAPACITY`, 4 MiB) fully buffers
    /// most short audio files before issuing any I/O, and cuts syscall count significantly
    /// for longer ones.  Reduce this in memory-constrained environments; increase it when
    /// writing many large files sequentially.
    pub write_buf_capacity: usize,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            write_buf_capacity: DEFAULT_WRITE_BUF_CAPACITY,
        }
    }
}

#[allow(dead_code)]
const fn _assert_send_sync()
where
    AudioDataSource<'static>: Send + Sync,
{
}

#[cfg(test)]
mod tests {
    use super::FileType;

    #[test]
    fn magic_bytes_classify_known_signatures() {
        assert_eq!(FileType::from_magic_bytes(b"RIFF\x24\x08\x00\x00WAVE"), FileType::WAV);
        assert_eq!(FileType::from_magic_bytes(b"RF64\xFF\xFF\xFF\xFFWAVE"), FileType::WAV);
        assert_eq!(FileType::from_magic_bytes(b"BW64\xFF\xFF\xFF\xFFWAVE"), FileType::WAV);
        assert_eq!(FileType::from_magic_bytes(b"fLaC\x00\x00\x00\x22"), FileType::FLAC);
        assert_eq!(FileType::from_magic_bytes(b"FORM\x00\x00\x10\x00AIFF"), FileType::AIFF);
        assert_eq!(FileType::from_magic_bytes(b"FORM\x00\x00\x10\x00AIFC"), FileType::AIFF);
        assert_eq!(
            FileType::from_magic_bytes(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00"),
            FileType::OGG
        );
        assert_eq!(
            FileType::from_magic_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00"),
            FileType::MP3
        );
        assert_eq!(FileType::from_magic_bytes(&[0xFF, 0xFB, 0x90, 0x00]), FileType::MP3);
        assert_eq!(FileType::from_magic_bytes(&[0xFF, 0xE3, 0x18, 0xC4]), FileType::MP3);
    }

    #[test]
    fn magic_bytes_reject_non_signatures() {
        assert_eq!(FileType::from_magic_bytes(b""), FileType::Unknown);
        assert_eq!(FileType::from_magic_bytes(b"RIFF"), FileType::Unknown); // too short for form type
        assert_eq!(
            FileType::from_magic_bytes(b"RIFF\x00\x00\x00\x00AVI "),
            FileType::Unknown
        ); // RIFF but not WAVE
        assert_eq!(
            FileType::from_magic_bytes(b"FORM\x00\x00\x00\x008SVX"),
            FileType::Unknown
        ); // IFF but not AIFF
        assert_eq!(FileType::from_magic_bytes(&[0xFF, 0x7F, 0x00, 0x00]), FileType::Unknown); // bad MPEG sync
        assert_eq!(FileType::from_magic_bytes(b"random bytes"), FileType::Unknown);
    }

    #[test]
    fn detect_falls_back_to_extension_for_missing_file() {
        assert_eq!(FileType::detect("/nonexistent/path/audio.wav"), FileType::WAV);
        assert_eq!(FileType::detect("/nonexistent/path/audio.flac"), FileType::FLAC);
        assert_eq!(FileType::detect("/nonexistent/path/audio.xyz"), FileType::Unknown);
    }
}
