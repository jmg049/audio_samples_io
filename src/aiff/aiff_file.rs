//! AIFF/AIFF-C file reading and writing.
//!
//! AIFF is the big-endian IFF cousin of WAV: a `FORM` container holding a
//! `COMM` chunk (format parameters, with the sample rate as an 80-bit extended
//! float) and an `SSND` chunk (interleaved big-endian PCM). AIFF-C adds a
//! compression type to `COMM` and a mandatory `FVER` chunk; in practice the
//! "compression" codes that matter are byte-order and float markers, all of
//! which decode here:
//!
//! | Code | Meaning |
//! |---|---|
//! | `NONE`, `twos`, `in24`, `in32` | big-endian integer PCM |
//! | `sowt` | little-endian 16-bit PCM (Apple, "twos" reversed) |
//! | `fl32`/`FL32` | IEEE f32 big-endian |
//! | `fl64`/`FL64` | IEEE f64 big-endian |
//!
//! 8-bit AIFF samples are *signed* (unlike WAV's unsigned u8 convention); the
//! reader and writer flip the convention so `u8` behaves identically across
//! formats.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration;

use audio_samples::{AudioSamples, I24, traits::StandardSample};

use crate::aiff::extended::{decode_extended, encode_extended};
use crate::error::{AudioIOError, AudioIOResult, ErrorPosition};
use crate::traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioInfoMarker};
use crate::types::{AudioDataSource, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType};

/// How the SSND payload is encoded, derived from form type + compression code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoundEncoding {
    /// Big-endian two's-complement integers (plain AIFF, `NONE`, `twos`, `in24`, `in32`).
    PcmBigEndian,
    /// Little-endian 16-bit integers (`sowt`).
    PcmLittleEndian,
    /// Big-endian IEEE f32 (`fl32`).
    Float32,
    /// Big-endian IEEE f64 (`fl64`).
    Float64,
}

/// AIFF-specific metadata returned by [`AudioFileMetadata::specific_info`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiffFileInfo {
    /// `AIFF` or `AIFC`.
    pub form_type: [u8; 4],
    /// AIFF-C compression code (`NONE` for plain AIFF).
    pub compression: [u8; 4],
    /// Chunk ids found in the file, in order.
    pub available_chunks: Vec<[u8; 4]>,
}

impl AudioInfoMarker for AiffFileInfo {}

/// An AIFF/AIFF-C file, fully loaded or memory-mapped.
#[derive(Debug)]
pub struct AiffFile<'a> {
    data_source: AudioDataSource<'a>,
    file_path: PathBuf,
    form_type: [u8; 4],
    compression: [u8; 4],
    available_chunks: Vec<[u8; 4]>,
    channels: u16,
    num_frames: usize,
    bits_per_sample: u16,
    sample_rate: NonZeroU32,
    sample_type: ValidatedSampleType,
    encoding: SoundEncoding,
    /// Byte range of the audio payload inside the file (after SSND offset).
    sound_range: std::ops::Range<usize>,
}

fn be_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(bytes[at..at + 4].try_into().expect("4-byte slice"))
}

fn be_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_be_bytes(bytes[at..at + 2].try_into().expect("2-byte slice"))
}

impl<'a> AiffFile<'a> {
    fn parse(data_source: AudioDataSource<'a>, file_path: PathBuf) -> AudioIOResult<Self> {
        let bytes = data_source.as_bytes();

        if bytes.len() < 12 {
            return Err(AudioIOError::corrupted_data(
                "File too small to be a valid AIFF file",
                format!("File size: {}", bytes.len()),
                ErrorPosition::new(0).with_description("start of file"),
            ));
        }
        if &bytes[0..4] != b"FORM" {
            return Err(AudioIOError::corrupted_data(
                "Data does not start with FORM header",
                format!("Found: {:?}", &bytes[0..4]),
                ErrorPosition::new(0).with_description("FORM header at file start"),
            ));
        }
        let form_type: [u8; 4] = bytes[8..12].try_into().expect("4-byte slice");
        let is_aifc = &form_type == b"AIFC";
        if &form_type != b"AIFF" && !is_aifc {
            return Err(AudioIOError::corrupted_data(
                "FORM type is not AIFF or AIFC",
                format!("Found: {:?}", form_type),
                ErrorPosition::new(8).with_description("FORM type"),
            ));
        }

        let mut available_chunks = Vec::new();
        let mut comm: Option<(u16, usize, u16, f64, [u8; 4])> = None;
        let mut sound_range: Option<std::ops::Range<usize>> = None;

        let mut offset = 12;
        while offset + 8 <= bytes.len() {
            let id: [u8; 4] = bytes[offset..offset + 4].try_into().expect("4-byte slice");
            let declared = be_u32(bytes, offset + 4) as usize;
            let body_start = offset + 8;
            // Tolerate over-declared sizes on the final chunk, mirroring the WAV reader.
            let size = declared.min(bytes.len() - body_start);
            available_chunks.push(id);

            match &id {
                b"COMM" => {
                    if size < 18 {
                        return Err(AudioIOError::corrupted_data(
                            "COMM chunk too small",
                            format!("Expected at least 18 bytes, found {size}"),
                            ErrorPosition::new(body_start).with_description("COMM chunk body"),
                        ));
                    }
                    let channels = be_u16(bytes, body_start);
                    let num_frames = be_u32(bytes, body_start + 2) as usize;
                    let bits = be_u16(bytes, body_start + 6);
                    let rate_bytes: [u8; 10] = bytes[body_start + 8..body_start + 18]
                        .try_into()
                        .expect("10-byte slice");
                    let rate = decode_extended(&rate_bytes);
                    let compression: [u8; 4] = if is_aifc && size >= 22 {
                        bytes[body_start + 18..body_start + 22]
                            .try_into()
                            .expect("4-byte slice")
                    } else {
                        *b"NONE"
                    };
                    comm = Some((channels, num_frames, bits, rate, compression));
                },
                b"SSND" => {
                    if size < 8 {
                        return Err(AudioIOError::corrupted_data(
                            "SSND chunk too small",
                            format!("Expected at least 8 bytes, found {size}"),
                            ErrorPosition::new(body_start).with_description("SSND chunk body"),
                        ));
                    }
                    // SSND body: offset (u32) + blockSize (u32) + sound data.
                    let data_offset = be_u32(bytes, body_start) as usize;
                    let sound_start = body_start + 8 + data_offset;
                    let sound_end = (body_start + size).min(bytes.len());
                    if sound_start > sound_end {
                        return Err(AudioIOError::corrupted_data(
                            "SSND offset runs past the chunk",
                            format!("offset {data_offset} with chunk size {size}"),
                            ErrorPosition::new(body_start).with_description("SSND offset field"),
                        ));
                    }
                    sound_range = Some(sound_start..sound_end);
                },
                _ => {},
            }

            let padded = size + (size & 1); // IFF chunks are word-aligned
            offset = body_start + padded;
        }

        let (channels, num_frames, bits_per_sample, rate, compression) = comm.ok_or_else(|| {
            AudioIOError::corrupted_data(
                "COMM chunk not found in AIFF file",
                format!("Found chunks: {available_chunks:?}"),
                ErrorPosition::new(12).with_description("chunk data section"),
            )
        })?;
        let mut sound_range = sound_range.ok_or_else(|| {
            AudioIOError::corrupted_data(
                "SSND chunk not found in AIFF file",
                format!("Found chunks: {available_chunks:?}"),
                ErrorPosition::new(12).with_description("chunk data section"),
            )
        })?;

        if channels == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "Invalid channel count",
                "COMM declares zero channels",
            ));
        }
        let sample_rate = NonZeroU32::new(rate.round() as u32).ok_or_else(|| {
            AudioIOError::corrupted_data_simple("Invalid sample rate", format!("COMM declares {rate} Hz"))
        })?;

        let (encoding, sample_type) = match &compression {
            b"NONE" | b"twos" | b"in24" | b"in32" => (
                SoundEncoding::PcmBigEndian,
                match bits_per_sample {
                    1..=8 => ValidatedSampleType::U8,
                    9..=16 => ValidatedSampleType::I16,
                    17..=24 => ValidatedSampleType::I24,
                    25..=32 => ValidatedSampleType::I32,
                    other => {
                        return Err(AudioIOError::unsupported_format(format!(
                            "Unsupported AIFF bit depth: {other}"
                        )));
                    },
                },
            ),
            b"sowt" => (SoundEncoding::PcmLittleEndian, ValidatedSampleType::I16),
            b"fl32" | b"FL32" => (SoundEncoding::Float32, ValidatedSampleType::F32),
            b"fl64" | b"FL64" => (SoundEncoding::Float64, ValidatedSampleType::F64),
            other => {
                return Err(AudioIOError::unsupported_format(format!(
                    "Unsupported AIFF-C compression type: {:?}",
                    String::from_utf8_lossy(other)
                )));
            },
        };

        // Clamp the frame count to the bytes actually present, mirroring the
        // WAV reader's tolerance of over-declared headers.
        let bytes_per_sample = sample_type.bytes_per_sample().get();
        let frame_bytes = channels as usize * bytes_per_sample;
        let frames_in_sound = sound_range.len() / frame_bytes.max(1);
        let num_frames = num_frames.min(frames_in_sound);
        sound_range.end = sound_range.start + num_frames * frame_bytes;

        Ok(AiffFile {
            data_source,
            file_path,
            form_type,
            compression,
            available_chunks,
            channels,
            num_frames,
            bits_per_sample,
            sample_rate,
            sample_type,
            encoding,
            sound_range,
        })
    }

    /// The native sample type of the sound data.
    pub const fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    /// Number of sample frames (samples per channel).
    pub const fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Decode the sound data into planar samples of type `T`.
    fn decode_planar<T>(&self) -> AudioIOResult<Vec<T>>
    where
        T: StandardSample + 'static,
    {
        let sound = &self.data_source.as_bytes()[self.sound_range.clone()];
        let channels = self.channels as usize;
        let frames = self.num_frames;
        let bps = self.sample_type.bytes_per_sample().get();

        // Planar output: channel-major, matching AudioSamples' multi-channel layout.
        let mut out: Vec<T> = Vec::with_capacity(channels * frames);

        for ch in 0..channels {
            for frame in 0..frames {
                let at = (frame * channels + ch) * bps;
                let sample = &sound[at..at + bps];
                let value: T = match self.encoding {
                    SoundEncoding::PcmBigEndian => match self.sample_type {
                        // AIFF 8-bit is signed; shift to the crate's unsigned-u8 convention.
                        ValidatedSampleType::U8 => T::convert_from((sample[0] as i8 as i16 + 128) as u8),
                        ValidatedSampleType::I16 => T::convert_from(i16::from_be_bytes([sample[0], sample[1]])),
                        ValidatedSampleType::I24 => {
                            let wide = i32::from_be_bytes([sample[0], sample[1], sample[2], 0]) >> 8;
                            T::convert_from(I24::wrapping_from_i32(wide))
                        },
                        ValidatedSampleType::I32 => {
                            T::convert_from(i32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]]))
                        },
                        _ => unreachable!("PCM encoding never maps to float sample types"),
                    },
                    SoundEncoding::PcmLittleEndian => T::convert_from(i16::from_le_bytes([sample[0], sample[1]])),
                    SoundEncoding::Float32 => {
                        T::convert_from(f32::from_be_bytes([sample[0], sample[1], sample[2], sample[3]]))
                    },
                    SoundEncoding::Float64 => {
                        T::convert_from(f64::from_be_bytes(sample.try_into().expect("8-byte slice")))
                    },
                };
                out.push(value);
            }
        }

        Ok(out)
    }
}

impl<'a> AudioFile for AiffFile<'a> {
    fn open<P: AsRef<Path>>(fp: P) -> AudioIOResult<Self> {
        Self::open_with_options(fp, OpenOptions::default())
    }

    fn open_with_options<P: AsRef<Path>>(fp: P, options: OpenOptions) -> AudioIOResult<Self> {
        let path = fp.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let file_size = file.metadata()?.len();

        let data_source = if options.use_memory_map && file_size <= crate::MAX_MMAP_SIZE {
            // SAFETY: the file is open for the lifetime of the mmap.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            AudioDataSource::MemoryMapped(mmap)
        } else {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut std::io::BufReader::new(file), &mut bytes)?;
            AudioDataSource::Owned(bytes)
        };

        Self::parse(data_source, path)
    }

    fn len(&self) -> u64 {
        self.data_source.len() as u64
    }

    fn is_empty(&self) -> bool {
        self.data_source.len() == 0
    }
}

impl<'a> AudioFileMetadata for AiffFile<'a> {
    fn open_metadata<P: AsRef<Path>>(path: P) -> AudioIOResult<Self> {
        Self::open_with_options(path, OpenOptions::default())
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let bytes_per_sample = self.sample_type.bytes_per_sample().get() as u16;
        let block_align = self.channels * bytes_per_sample;
        let byte_rate = self.sample_rate.get() * block_align as u32;
        let duration = Duration::from_secs_f64(self.num_frames as f64 / self.sample_rate.get() as f64);

        Ok(BaseAudioInfo::new(
            self.sample_rate,
            self.channels,
            self.bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            self.total_samples(),
            duration,
            FileType::AIFF,
            self.sample_type.into(),
        ))
    }

    fn specific_info(&self) -> impl AudioInfoMarker {
        AiffFileInfo {
            form_type: self.form_type,
            compression: self.compression,
            available_chunks: self.available_chunks.clone(),
        }
    }

    fn file_type(&self) -> FileType {
        FileType::AIFF
    }

    fn file_path(&self) -> &Path {
        &self.file_path
    }

    fn total_samples(&self) -> usize {
        self.num_frames * self.channels as usize
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        Ok(Duration::from_secs_f64(
            self.num_frames as f64 / self.sample_rate.get() as f64,
        ))
    }

    fn samples_per_channel(&self) -> AudioIOResult<usize> {
        Ok(self.num_frames)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.channels
    }
}

impl<'a> AudioFileRead<'a> for AiffFile<'a> {
    fn read<T>(&'a self) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: StandardSample + 'static,
    {
        let flat = self.decode_planar::<T>()?;

        if self.channels == 1 {
            let arr = ndarray::Array1::from_vec(flat);
            AudioSamples::new_mono(arr, self.sample_rate).map_err(Into::into)
        } else {
            let arr = ndarray::Array2::from_shape_vec((self.channels as usize, self.num_frames), flat)
                .map_err(|e| AudioIOError::corrupted_data_simple("Array shape error", e.to_string()))?;
            AudioSamples::new_multi_channel(arr, self.sample_rate).map_err(Into::into)
        }
    }

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: StandardSample + 'static,
    {
        let decoded = self.read::<T>()?;
        if decoded.total_samples() != audio.total_samples() {
            return Err(AudioIOError::corrupted_data_simple(
                "Sample count mismatch",
                format!(
                    "AIFF has {} samples, buffer has {}",
                    decoded.total_samples(),
                    audio.total_samples()
                ),
            ));
        }
        *audio = decoded;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// AIFC version 1 timestamp, required in the FVER chunk (AIFF-C specification).
const AIFC_VERSION_1: u32 = 0xA280_5140;

/// Write `audio` as an AIFF (integer types) or AIFF-C (`fl32`/`fl64` for float
/// types) file. Samples are written interleaved big-endian, the format's
/// native layout; `u8` audio is converted to AIFF's signed 8-bit convention.
pub fn write_aiff<W, T>(mut writer: W, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    W: std::io::Write,
    T: StandardSample + 'static,
{
    use std::any::TypeId;

    let channels = audio.num_channels().get() as u16;
    let frames = audio.samples_per_channel().get();
    let sample_rate = audio.sample_rate().get();

    let t = TypeId::of::<T>();
    let (sample_type, is_float) = if t == TypeId::of::<u8>() {
        (ValidatedSampleType::U8, false)
    } else if t == TypeId::of::<i16>() {
        (ValidatedSampleType::I16, false)
    } else if t == TypeId::of::<I24>() {
        (ValidatedSampleType::I24, false)
    } else if t == TypeId::of::<i32>() {
        (ValidatedSampleType::I32, false)
    } else if t == TypeId::of::<f32>() {
        (ValidatedSampleType::F32, true)
    } else if t == TypeId::of::<f64>() {
        (ValidatedSampleType::F64, true)
    } else {
        return Err(AudioIOError::unsupported_format(
            "No AIFF encoding for this sample type".to_string(),
        ));
    };

    let bits = sample_type.bits_per_sample().get() as u16;
    let bytes_per_sample = sample_type.bytes_per_sample().get();
    let data_size = frames * channels as usize * bytes_per_sample;

    // AIFF-C is only needed for float payloads; integer audio stays plain AIFF
    // for maximum compatibility with older readers.
    let comm_body_len: usize = if is_float { 18 + 4 + 2 } else { 18 }; // + compression code + empty pstring
    let fver_len: usize = if is_float { 8 + 4 } else { 0 };
    let ssnd_body_len = 8 + data_size;
    let ssnd_pad = ssnd_body_len & 1;
    let form_size = 4 + fver_len + (8 + comm_body_len) + (8 + ssnd_body_len) + ssnd_pad;

    writer.write_all(b"FORM")?;
    writer.write_all(&(form_size as u32).to_be_bytes())?;
    writer.write_all(if is_float { b"AIFC" } else { b"AIFF" })?;

    if is_float {
        writer.write_all(b"FVER")?;
        writer.write_all(&4u32.to_be_bytes())?;
        writer.write_all(&AIFC_VERSION_1.to_be_bytes())?;
    }

    writer.write_all(b"COMM")?;
    writer.write_all(&(comm_body_len as u32).to_be_bytes())?;
    writer.write_all(&channels.to_be_bytes())?;
    writer.write_all(&(frames as u32).to_be_bytes())?;
    writer.write_all(&bits.to_be_bytes())?;
    writer.write_all(&encode_extended(sample_rate as f64))?;
    if is_float {
        writer.write_all(if bits == 32 { b"fl32" } else { b"fl64" })?;
        writer.write_all(&[0u8, 0u8])?; // empty pstring (length 0 + pad)
    }

    writer.write_all(b"SSND")?;
    writer.write_all(&(ssnd_body_len as u32).to_be_bytes())?;
    writer.write_all(&0u32.to_be_bytes())?; // offset
    writer.write_all(&0u32.to_be_bytes())?; // blockSize

    // Interleave and serialise big-endian frame by frame.
    let interleaved: Vec<T> = audio.to_interleaved_vec().into_iter().collect();
    let mut buf = Vec::with_capacity(data_size);
    for s in interleaved {
        write_sample_be(&mut buf, s, sample_type);
    }
    writer.write_all(&buf)?;
    if ssnd_pad == 1 {
        writer.write_all(&[0u8])?;
    }
    writer.flush()?;
    Ok(())
}

/// Append one sample in AIFF's on-disk encoding (big-endian, signed 8-bit).
fn write_sample_be<T>(out: &mut Vec<u8>, sample: T, sample_type: ValidatedSampleType)
where
    T: StandardSample + 'static,
{
    match sample_type {
        ValidatedSampleType::U8 => {
            let v: u8 = sample.convert_to();
            out.push((v as i16 - 128) as i8 as u8);
        },
        ValidatedSampleType::I16 => {
            let v: i16 = sample.convert_to();
            out.extend_from_slice(&v.to_be_bytes());
        },
        ValidatedSampleType::I24 => {
            let v: I24 = sample.convert_to();
            let wide = v.to_i32();
            out.extend_from_slice(&wide.to_be_bytes()[1..4]);
        },
        ValidatedSampleType::I32 => {
            let v: i32 = sample.convert_to();
            out.extend_from_slice(&v.to_be_bytes());
        },
        ValidatedSampleType::F32 => {
            let v: f32 = sample.convert_to();
            out.extend_from_slice(&v.to_be_bytes());
        },
        ValidatedSampleType::F64 => {
            let v: f64 = sample.convert_to();
            out.extend_from_slice(&v.to_be_bytes());
        },
    }
}
