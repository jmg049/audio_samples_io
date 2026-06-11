use core::fmt::{Display, Formatter, Result as FmtResult};
use std::{
    any::TypeId,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    mem,
    num::NonZeroU32,
    ops::Range,
    path::{Path, PathBuf},
    time::Duration,
};

use audio_samples::{
    AudioData, AudioSamples, I24, SampleType,
    traits::{ConvertFrom, StandardSample},
};
use memmap2::MmapOptions;
use ndarray::{Array1, Array2, ShapeBuilder};
use non_empty_slice::NonEmptyVec;

use crate::{
    MAX_MMAP_SIZE, MAX_WAV_SIZE,
    error::{AudioIOError, AudioIOResult, ErrorPosition},
    traits::{AudioFile, AudioFileMetadata, AudioFileRead, AudioFileWrite, AudioInfoMarker},
    types::{AudioDataSource, BaseAudioInfo, FileType, OpenOptions, ValidatedSampleType, WriteOptions},
    wav::{
        Companding, FormatCode,
        bext::BextChunk,
        chunks::{
            BEXT_CHUNK, BW64_CHUNK, CUE_CHUNK, ChunkDesc, ChunkID, DATA_CHUNK, DS64_CHUNK, FACT_CHUNK, FMT_CHUNK,
            LIST_CHUNK, RF64_CHUNK, RIFF_CHUNK, SMPL_CHUNK, WAVE_CHUNK,
        },
        cue::{CueChunk, CuePoint, cue_chunk_bytes},
        data::DataChunk,
        ds64::Ds64,
        error::WavError,
        fact::FactChunk,
        fmt::FmtChunk,
        list_info::{InfoMetadata, ListChunk},
        smpl::SmplChunk,
    },
};

#[derive(Debug, Clone)]
pub struct WavFileInfo {
    pub available_chunks: Vec<ChunkID>,
    pub encoding: FormatCode,
    /// Sample frame count from the FACT chunk, if present.
    pub fact_num_samples: Option<u32>,
    /// Metadata tags from the LIST/INFO chunk, if present and parseable.
    pub info_metadata: Option<InfoMetadata>,
}

impl Display for WavFileInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "WAV File Info:")?;
        writeln!(f, "Encoding: {}", self.encoding)?;
        writeln!(f, "Available Chunks: {:?}", self.available_chunks)?;
        if let Some(n) = self.fact_num_samples {
            writeln!(f, "Fact Sample Count: {n}")?;
        }
        if let Some(ref meta) = self.info_metadata {
            writeln!(f, "Info Metadata:")?;
            write!(f, "{meta}")?;
        }
        Ok(())
    }
}

impl AudioInfoMarker for WavFileInfo {}

/// High-level WAV file representation
#[derive(Debug)]
pub struct WavFile<'a> {
    /// Data source (owned or memory-mapped)
    data_source: AudioDataSource<'a>,
    /// File path if loaded from file
    file_path: PathBuf,
    /// Chunk Storage
    chunks: Vec<ChunkDesc>,
    /// fmt chunk byte range (excludes 8-byte chunk header)
    fmt_range: Range<usize>,
    /// data chunk byte range (excludes 8-byte chunk header)
    data_range: Range<usize>,
    /// Validated sample type
    sample_type: ValidatedSampleType,
    /// Companding scheme (mu-law / a-law) when the data is 8-bit companded, else None.
    companding: Option<Companding>,
    total_samples: usize,
}

impl<'a> WavFile<'a> {
    /// Get the file path if available
    pub fn file_path(&self) -> Option<&Path> {
        Some(self.file_path.as_path())
    }

    /// Get the FMT chunk
    ///
    /// # Panics
    ///
    /// This function will panic if the FMT chunk is not valid.
    /// However, this should never happen as it will have been validated during file opening.
    pub fn fmt_chunk(&self) -> FmtChunk<'_> {
        FmtChunk::from_bytes(self.fmt_bytes()).expect("fmt chunk validated during open")
    }

    /// Get the DATA chunk
    pub fn data(&self) -> DataChunk<'_> {
        DataChunk::from_bytes(self.data_bytes())
    }

    #[inline]
    fn fmt_bytes(&self) -> &[u8] {
        &self.data_source[self.fmt_range.clone()]
    }

    #[inline]
    fn data_bytes(&self) -> &[u8] {
        &self.data_source[self.data_range.clone()]
    }

    /// Parse the FACT chunk, if present.
    ///
    /// Returns `None` when the file does not contain a FACT chunk.
    pub fn fact(&self) -> AudioIOResult<Option<FactChunk<'_>>> {
        self.chunks
            .iter()
            .find(|c| c.id == FACT_CHUNK)
            .map(|desc| FactChunk::from_bytes(&self.data_source[desc.data_range()]).map_err(AudioIOError::WavError))
            .transpose()
    }

    /// Parse the first LIST chunk, if present.
    ///
    /// Returns `None` when the file does not contain a LIST chunk.
    /// Call [`ListChunk::info_metadata`] on the result to access INFO tags.
    pub fn list(&self) -> AudioIOResult<Option<ListChunk<'_>>> {
        self.chunks
            .iter()
            .find(|c| c.id == LIST_CHUNK)
            .map(|desc| ListChunk::from_bytes(&self.data_source[desc.data_range()]).map_err(AudioIOError::WavError))
            .transpose()
    }

    /// Parse the CUE chunk, if present.
    ///
    /// Returns `None` when the file does not contain a CUE chunk.
    pub fn cue(&self) -> AudioIOResult<Option<CueChunk<'_>>> {
        self.chunks
            .iter()
            .find(|c| c.id == CUE_CHUNK)
            .map(|desc| CueChunk::from_bytes(&self.data_source[desc.data_range()]).map_err(AudioIOError::WavError))
            .transpose()
    }

    /// Parse the SMPL chunk, if present.
    ///
    /// Returns `None` when the file does not contain a SMPL chunk.
    pub fn smpl(&self) -> AudioIOResult<Option<SmplChunk<'_>>> {
        self.chunks
            .iter()
            .find(|c| c.id == SMPL_CHUNK)
            .map(|desc| SmplChunk::from_bytes(&self.data_source[desc.data_range()]).map_err(AudioIOError::WavError))
            .transpose()
    }

    /// Parse the BEXT chunk, if present.
    ///
    /// Returns `None` when the file does not contain a BEXT chunk.
    pub fn bext(&self) -> AudioIOResult<Option<BextChunk<'_>>> {
        self.chunks
            .iter()
            .find(|c| c.id == BEXT_CHUNK)
            .map(|desc| BextChunk::from_bytes(&self.data_source[desc.data_range()]).map_err(AudioIOError::WavError))
            .transpose()
    }

    #[allow(dead_code)]
    fn chunk_bytes(&self, chunk: &ChunkDesc) -> &[u8] {
        // Return entire chunk including header (8 bytes) + logical data (no padding)
        &self.data_source[chunk.offset..chunk.offset + 8 + chunk.logical_size]
    }

    pub const fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.fmt_chunk().sample_rate()
    }

    pub fn is_mono(&self) -> bool {
        self.fmt_chunk().channels() == 1
    }

    pub fn is_mulit_channel(&self) -> bool {
        self.fmt_chunk().channels() > 1
    }
}

impl<'a> AudioFileMetadata for WavFile<'a> {
    fn open_metadata<P: AsRef<Path>>(path: P) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        // For metadata-only operations, we can use the same implementation as regular open
        // but with memory mapping enabled by default for efficiency
        Self::open_with_options(path, OpenOptions::default())
    }

    fn base_info(&self) -> AudioIOResult<BaseAudioInfo> {
        let fmt_chunk = self.fmt_chunk();
        let (_, channels, sample_rate, byte_rate, block_align, bits_per_sample) = fmt_chunk.fmt_chunk();
        let sample_rate = match NonZeroU32::new(sample_rate) {
            Some(sr) => sr,
            None => {
                return Err(AudioIOError::corrupted_data_simple(
                    "Invalid sample rate in FMT chunk",
                    "Sample rate cannot be zero",
                ));
            },
        };
        let (total_samples, duration) = {
            let data_chunk = self.data();
            // safety: channels is non-zero as per WAV spec
            let total_frames =
                data_chunk.total_frames(self.sample_type, unsafe { NonZeroU32::new_unchecked(channels as u32) });
            // Duration is based on frames, not total samples
            let duration = Duration::from_secs_f64(total_frames as f64 / sample_rate.get() as f64);
            (self.total_samples(), duration)
        };
        let file_type = FileType::WAV;
        let sample_type = self.sample_type();
        let bytes_per_sample = (bits_per_sample as usize / 8) as u16;

        Ok(BaseAudioInfo::new(
            sample_rate,
            channels,
            bits_per_sample,
            bytes_per_sample,
            byte_rate,
            block_align,
            total_samples,
            duration,
            file_type,
            sample_type.into(),
        ))
    }

    #[allow(refining_impl_trait)]
    fn specific_info(&self) -> WavFileInfo {
        let fact_num_samples = self.fact().ok().flatten().map(|f| f.num_samples());
        let info_metadata = self
            .list()
            .ok()
            .flatten()
            .and_then(|l| l.info_metadata())
            .and_then(|r| r.ok());
        WavFileInfo {
            available_chunks: self.chunks.iter().map(|c| c.id).collect(),
            encoding: self.fmt_chunk().format_code(),
            fact_num_samples,
            info_metadata,
        }
    }

    fn file_type(&self) -> FileType {
        FileType::WAV
    }

    fn file_path(&self) -> &Path {
        self.file_path.as_path()
    }

    fn total_samples(&self) -> usize {
        self.total_samples
    }

    fn duration(&self) -> AudioIOResult<Duration> {
        let base_info = self.base_info()?;
        Ok(base_info.duration)
    }

    fn sample_type(&self) -> ValidatedSampleType {
        self.sample_type
    }

    fn num_channels(&self) -> u16 {
        self.fmt_chunk().channels()
    }
}

impl<'a> AudioFile for WavFile<'a> {
    fn open_with_options<P: AsRef<Path>>(fp: P, options: OpenOptions) -> AudioIOResult<Self>
    where
        Self: Sized,
    {
        let path = fp.as_ref().to_path_buf();
        let file = File::open(&path)?;

        let file_size = file.metadata()?.len();

        if file_size > MAX_WAV_SIZE {
            return Err(AudioIOError::corrupted_data_simple(
                "File too large",
                format!("File size {file_size} exceeds maximum {MAX_WAV_SIZE} bytes"),
            ));
        }

        let use_mmap = options.use_memory_map && file_size <= MAX_MMAP_SIZE;

        let audio_data_source: AudioDataSource<'a> = if use_mmap {
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            // Hint to the OS that pages will be accessed sequentially, enabling aggressive
            // read-ahead prefetching.  Best-effort: ignore errors (e.g. on platforms that
            // don't support it).
            let _ = mmap.advise(memmap2::Advice::Sequential);
            AudioDataSource::MemoryMapped(mmap)
        } else {
            // Fallback to buffered read for large files or when mmap is disabled
            let mut buf_reader = BufReader::new(file);
            let mut bytes = Vec::new();
            buf_reader.read_to_end(&mut bytes)?;
            AudioDataSource::Owned(bytes)
        };
        let bytes = audio_data_source.as_bytes();

        if bytes.len() < 12 {
            return Err(AudioIOError::corrupted_data(
                "File too small to be a valid WAV file",
                format!("File size: {}", audio_data_source.len()),
                ErrorPosition::new(0).with_description("start of file"),
            ));
        }

        // 1. Parse the RIFF header + File size + WAVE identifier
        // Assume the first 12 bytes are the RIFF header. RF64/BW64 are the 64-bit
        // variants of RIFF (EBU Tech 3306 / ITU-R BS.2088): same layout, but the
        // 32-bit size fields hold 0xFFFFFFFF and the true sizes live in a `ds64`
        // chunk that must directly follow the WAVE identifier.
        let riff = ChunkID::new(bytes[0..4].try_into().expect("Guaranteed to be at least 12 bytes now"));
        let is_rf64 = riff == RF64_CHUNK || riff == BW64_CHUNK;

        if riff != RIFF_CHUNK && !is_rf64 {
            return Err(AudioIOError::corrupted_data(
                "Data does not start with RIFF header",
                format!("Found: {riff:?}"),
                ErrorPosition::new(0).with_description("RIFF/RF64/BW64 header at file start"),
            ));
        }

        let declared_file_size_32 =
            u32::from_le_bytes(bytes[4..8].try_into().expect("Guaranteed to be at least 12 bytes now"));

        let wave = ChunkID::new(bytes[8..12].try_into().expect("Guaranteed to be at least 12 bytes now"));

        if wave != WAVE_CHUNK {
            return Err(AudioIOError::corrupted_data(
                "Data does not contain WAVE identifier after RIFF header",
                format!("Found: {wave:?}"),
                ErrorPosition::new(8).with_description("WAVE identifier after RIFF header"),
            ));
        }

        // RF64/BW64: the mandatory ds64 chunk is the first chunk after WAVE.
        let ds64 = if is_rf64 {
            if bytes.len() < 20 || ChunkID::new(bytes[12..16].try_into().expect("4-byte slice")) != DS64_CHUNK {
                return Err(AudioIOError::corrupted_data(
                    "RF64/BW64 file is missing the mandatory ds64 chunk",
                    "The first chunk after WAVE must be ds64",
                    ErrorPosition::new(12).with_description("ds64 chunk header"),
                ));
            }
            let ds64_size = u32::from_le_bytes(bytes[16..20].try_into().expect("4-byte slice")) as usize;
            let end = (20 + ds64_size).min(bytes.len());
            Some(Ds64::from_bytes(&bytes[20..end])?)
        } else {
            None
        };

        let declared_file_size = match &ds64 {
            Some(ds) if declared_file_size_32 == u32::MAX => usize::try_from(ds.riff_size).unwrap_or(usize::MAX),
            _ => declared_file_size_32 as usize,
        };

        // Streaming producers (ffmpeg, live capture) cannot know the final length up front and
        // write a placeholder RIFF size — commonly 0xFFFFFFFF — that overruns the real byte count.
        // Such files play fine everywhere else, so we tolerate them: clamp the declared size to the
        // bytes actually present rather than rejecting the file outright.
        let file_size = declared_file_size.min(bytes.len().saturating_sub(8));

        let mut chunks: Vec<ChunkDesc> = Vec::new();
        chunks.push(ChunkDesc {
            id: riff,
            offset: 0,
            logical_size: file_size,
            total_size: file_size + 8, // RIFF header + data
        });
        chunks.push(ChunkDesc {
            id: wave,
            offset: 8,
            logical_size: 4,
            total_size: 4,
        });

        // 2. Iterate through the rest of the file to find chunks
        let mut offset = 12;

        while offset + 8 <= bytes.len() {
            let id = ChunkID::new(bytes[offset..offset + 4].try_into().map_err(|_| {
                AudioIOError::corrupted_data(
                    "Cannot read chunk ID",
                    "Insufficient data for chunk header",
                    ErrorPosition::new(offset).with_description("chunk ID bytes"),
                )
            })?);
            let size_bytes = bytes[offset + 4..offset + 8].try_into().map_err(|_| {
                AudioIOError::corrupted_data(
                    "Cannot read chunk size",
                    "Insufficient data for chunk header",
                    ErrorPosition::new(offset + 4).with_description("chunk size bytes"),
                )
            })?;
            let declared_size_32 = u32::from_le_bytes(size_bytes);
            // In an RF64/BW64 file, a chunk declaring 0xFFFFFFFF stores its true
            // 64-bit size in the ds64 chunk (the data chunk always; others via table).
            let declared_size = match &ds64 {
                Some(ds) => usize::try_from(ds.resolve(id, declared_size_32)).unwrap_or(usize::MAX),
                None => declared_size_32 as usize,
            };

            // Bytes physically available for this chunk's body (after its 8-byte header).
            let avail = bytes.len() - (offset + 8);

            // Tolerate over-declared chunk sizes the same way we tolerate over-declared RIFF
            // sizes: a `data` chunk written by a streaming encoder often carries 0xFFFFFFFF (or
            // any value larger than what was ultimately produced).  Clamp to the bytes actually
            // present and treat such a chunk as the final one in the file.
            let size = declared_size.min(avail);
            let was_clamped = size < declared_size;

            let padded = size + (size & 1); // RIFF chunks are word-aligned; size + size&1 <= avail+1
            // The trailing pad byte may itself run one past EOF on an unpadded final chunk; cap it.
            let end = (offset + 8 + padded).min(bytes.len());

            chunks.push(ChunkDesc {
                id,
                offset,
                logical_size: size,       // Original chunk size without padding
                total_size: end - offset, // Header + data + padding for file positioning
            });

            if was_clamped {
                // Declared size overran the file: there is nothing valid after this chunk.
                break;
            }

            offset = end;
        }

        // 3. Ensure fmt and data chunks are present
        let fmt_chunk_desc = chunks.iter().find(|c| c.id == FMT_CHUNK);
        let data_chunk_desc = chunks.iter().find(|c| c.id == DATA_CHUNK);

        let (fmt_range, sample_type, companding, is_adpcm) = match fmt_chunk_desc {
            Some(fmt_chunk) => {
                let start = fmt_chunk.offset + 8; // skip 8-byte header
                let end = start + fmt_chunk.logical_size; // exclude padding if any
                let fmt_chunk = FmtChunk::from_bytes_validated(&bytes[start..end]).map_err(AudioIOError::WavError)?;
                let sample_type = fmt_chunk.actual_sample_type()?;
                let companding = fmt_chunk.companding();
                let is_adpcm = fmt_chunk.format_code().is_adpcm();
                (start..end, sample_type, companding, is_adpcm)
            },
            None => {
                return Err(AudioIOError::corrupted_data(
                    "FMT chunk not found in WAV file",
                    format!("Found chunks: {:?}", chunks.iter().map(|c| c.id).collect::<Vec<_>>()),
                    ErrorPosition::new(12).with_description("chunk data section"),
                ));
            },
        };

        let data_range = match data_chunk_desc {
            Some(data_chunk) => {
                let start = data_chunk.offset + 8; // skip 8-byte header
                let end = start + data_chunk.logical_size; // exclude padding byte
                start..end
            },
            None => {
                return Err(AudioIOError::corrupted_data(
                    "DATA chunk not found in WAV file",
                    format!("Found chunks: {:?}", chunks.iter().map(|c| c.id).collect::<Vec<_>>()),
                    ErrorPosition::new(12).with_description("chunk data section"),
                ));
            },
        };

        let total_samples = {
            let data_chunk = DataChunk::from_bytes(&bytes[data_range.clone()]);
            if is_adpcm {
                // ADPCM expands a variable number of samples per block.
                let fmt_chunk =
                    FmtChunk::from_bytes(&bytes[fmt_range.clone()]).expect("fmt chunk validated during open");
                crate::wav::adpcm::decoded_sample_count(&fmt_chunk, data_chunk.len())
            } else if companding.is_some() {
                // Companded data is one byte per (decoded) sample.
                data_chunk.len()
            } else {
                data_chunk.total_samples(sample_type)
            }
        };

        let wav_file = WavFile {
            data_source: audio_data_source,
            file_path: path,
            chunks,
            fmt_range,
            data_range,
            sample_type,
            companding,
            total_samples,
        };

        Ok(wav_file)
    }

    fn len(&self) -> u64 {
        self.data_source.len() as u64
    }
}

/// Parse just the WAV header from a seekable reader, returning audio metadata and the byte offset
/// where the `data` chunk payload begins.
///
/// Unlike [`WavFile::open_with_options`] this never loads or maps the audio payload — it stops
/// scanning as soon as it has seen the `fmt ` and `data` chunk headers.  On a typical WAV file the
/// header fits in the first 8 KiB read by the `BufReader`, so this triggers at most one read
/// syscall regardless of file size.
#[inline]
pub fn parse_wav_header_streaming<R: Read + Seek>(reader: &mut R) -> AudioIOResult<(BaseAudioInfo, u64)> {
    use crate::wav::chunks::RIFF_CHUNK;

    // ---- RIFF + WAVE header (12 bytes) ----
    let mut buf12 = [0u8; 12];
    reader.read_exact(&mut buf12).map_err(AudioIOError::from)?;
    let is_rf64 = &buf12[0..4] == b"RF64" || &buf12[0..4] == b"BW64";
    if &buf12[0..4] != RIFF_CHUNK.as_bytes() && !is_rf64 {
        return Err(AudioIOError::corrupted_data_simple(
            "Not a RIFF file",
            "First 4 bytes are not 'RIFF', 'RF64', or 'BW64'",
        ));
    }
    if &buf12[8..12] != b"WAVE" {
        return Err(AudioIOError::corrupted_data_simple(
            "Not a WAV file",
            "Bytes 8-12 are not 'WAVE'",
        ));
    }

    // ---- scan sub-chunks until we find fmt  and data ----
    let mut fmt_buf = [0u8; 40]; // enough for base (16) or extensible (40) fmt chunk
    let mut fmt_size: usize = 0;
    let mut have_fmt = false;
    let mut data_byte_offset: Option<u64> = None;
    let mut data_byte_count: Option<usize> = None;
    // RF64/BW64: true 64-bit data size from the mandatory ds64 chunk.
    let mut ds64_data_size: Option<u64> = None;

    let mut chunk_hdr = [0u8; 8];

    loop {
        match reader.read_exact(&mut chunk_hdr) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(AudioIOError::from(e)),
            Ok(_) => {},
        }
        let id = &chunk_hdr[0..4];
        let size = u32::from_le_bytes(match chunk_hdr[4..8].try_into() {
            Ok(b) => b,
            Err(_) => {
                unreachable!("chunk header is 8 bytes long, 4..8 is a 4-byte slice, try_into will always succeed")
            },
        }) as usize;
        let padded = size + (size & 1); // RIFF chunks are padded to even sizes

        if id == b"fmt " {
            if !(16..=40).contains(&size) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Invalid fmt chunk size",
                    format!("Expected 16 or 40, got {size}"),
                ));
            }
            reader.read_exact(&mut fmt_buf[..size]).map_err(AudioIOError::from)?;
            if size & 1 != 0 {
                reader.read_exact(&mut [0u8; 1]).map_err(AudioIOError::from)?;
            }
            fmt_size = size;
            have_fmt = true;
        } else if is_rf64 && id == b"ds64" {
            // Read just the fixed 28-byte prefix (we only need the data size here)
            // and skip the chunk-size table that may follow it.
            let mut ds64_buf = [0u8; crate::wav::ds64::DS64_MIN_BODY_LEN];
            if size < ds64_buf.len() {
                return Err(AudioIOError::corrupted_data_simple(
                    "ds64 chunk too small",
                    format!("Expected at least {} bytes, got {size}", ds64_buf.len()),
                ));
            }
            reader.read_exact(&mut ds64_buf).map_err(AudioIOError::from)?;
            ds64_data_size = Some(u64::from_le_bytes(ds64_buf[8..16].try_into().expect("8-byte slice")));
            let remaining = padded - ds64_buf.len();
            reader
                .seek(SeekFrom::Current(remaining as i64))
                .map_err(AudioIOError::from)?;
        } else if id == b"data" {
            // Current stream position is now at the first byte of audio data.
            let offset = reader.stream_position().map_err(AudioIOError::from)?;
            // RF64/BW64 store the true 64-bit data size in ds64 and 0xFFFFFFFF here.
            let size = match ds64_data_size {
                Some(ds64_size) if size == u32::MAX as usize => usize::try_from(ds64_size).unwrap_or(usize::MAX),
                _ => size,
            };
            // Streaming encoders write a placeholder data size (often 0xFFFFFFFF) when the final
            // length is unknown. Clamp the declared size to the bytes actually present so the
            // reported sample count stays accurate instead of astronomically large.
            let stream_end = reader.seek(SeekFrom::End(0)).map_err(AudioIOError::from)?;
            let avail = stream_end.saturating_sub(offset) as usize;
            // Restore the documented contract: the reader is left positioned at the first
            // byte of audio data, so callers can read the payload without seeking.
            reader.seek(SeekFrom::Start(offset)).map_err(AudioIOError::from)?;
            data_byte_offset = Some(offset);
            data_byte_count = Some(size.min(avail));
            break; // we have everything we need
        } else {
            // Skip this chunk (including any padding byte).
            reader
                .seek(SeekFrom::Current(padded as i64))
                .map_err(AudioIOError::from)?;
        }
    }

    if !have_fmt {
        return Err(AudioIOError::corrupted_data_simple(
            "No fmt chunk found",
            "WAV file must contain a fmt  chunk before data",
        ));
    }
    let data_byte_offset = data_byte_offset
        .ok_or_else(|| AudioIOError::corrupted_data_simple("No data chunk found", "WAV file has no data chunk"))?;

    let data_byte_count = data_byte_count.ok_or_else(|| {
        AudioIOError::corrupted_data_simple(
            "Data chunk size missing",
            "Could not determine size of audio data from data chunk header",
        )
    })?;

    // ---- parse fmt bytes ----
    let fmt_chunk = FmtChunk::from_bytes_validated(&fmt_buf[..fmt_size]).map_err(AudioIOError::WavError)?;
    let sample_type: ValidatedSampleType = fmt_chunk.actual_sample_type().map_err(AudioIOError::WavError)?;

    let (_, channels, sample_rate, byte_rate, block_align, bits_per_sample) = fmt_chunk.fmt_chunk();
    let sample_rate = NonZeroU32::new(sample_rate)
        .ok_or_else(|| AudioIOError::corrupted_data_simple("Invalid sample rate", "Sample rate cannot be zero"))?;

    let bytes_per_sample = fmt_chunk.bytes_per_sample();
    let total_samples = if bytes_per_sample > 0 {
        data_byte_count / bytes_per_sample as usize
    } else {
        0
    };
    let total_frames = if channels > 0 {
        total_samples / channels as usize
    } else {
        0
    };
    let duration = Duration::from_secs_f64(total_frames as f64 / sample_rate.get() as f64);

    let info = BaseAudioInfo::new(
        sample_rate,
        channels,
        bits_per_sample,
        bytes_per_sample,
        byte_rate,
        block_align,
        total_samples,
        duration,
        FileType::WAV,
        sample_type.into(),
    );

    Ok((info, data_byte_offset))
}

impl<'a> AudioFileRead<'a> for WavFile<'a> {
    /// Reads all samples from the audio file
    fn read<T>(&'a self) -> AudioIOResult<AudioSamples<'a, T>>
    where
        T: StandardSample + 'static,
    {
        let data_chunk = self.data();
        let fmt_chunk = self.fmt_chunk();

        let sample_type = self.sample_type;
        let sample_rate = fmt_chunk.sample_rate();
        // safety: sample_rate is guaranteed to be non-zero due to WAV spec validation during fmt chunk parsing
        let sample_rate = unsafe { NonZeroU32::new_unchecked(sample_rate) };
        let num_channels = fmt_chunk.channels() as u32;
        // safety: num_channels == 0 would have been rejected during fmt chunk parsing.
        let num_channels = unsafe {
            NonZeroU32::new_unchecked(num_channels) // WAV spec requires channels > 0
        };

        // ADPCM blocks are decoded to 16-bit linear PCM before conversion.
        if fmt_chunk.format_code().is_adpcm() {
            return read_adpcm::<T>(&fmt_chunk, &data_chunk, num_channels, sample_rate);
        }

        // Companded (mu-law / a-law) data is expanded to 16-bit linear PCM before conversion.
        if let Some(comp) = self.companding {
            return read_companded::<T>(&data_chunk, comp, num_channels, sample_rate);
        }

        match sample_type {
            ValidatedSampleType::U8 => read_typed_internal::<u8, T>(&data_chunk, num_channels, sample_rate),
            ValidatedSampleType::I16 => read_typed_internal::<i16, T>(&data_chunk, num_channels, sample_rate),
            ValidatedSampleType::I24 => read_typed_internal::<I24, T>(&data_chunk, num_channels, sample_rate),
            ValidatedSampleType::I32 => read_typed_internal::<i32, T>(&data_chunk, num_channels, sample_rate),
            ValidatedSampleType::F32 => read_typed_internal::<f32, T>(&data_chunk, num_channels, sample_rate),
            ValidatedSampleType::F64 => read_typed_internal::<f64, T>(&data_chunk, num_channels, sample_rate),
        }
    }

    fn read_into<T>(&'a self, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
    where
        T: StandardSample + 'static,
    {
        let data_chunk = self.data();

        if self.fmt_chunk().format_code().is_adpcm() {
            return read_adpcm_into::<T>(&self.fmt_chunk(), &data_chunk, audio);
        }

        if let Some(comp) = self.companding {
            return read_companded_into::<T>(&data_chunk, comp, audio);
        }

        match self.sample_type {
            ValidatedSampleType::U8 => read_into_typed_internal::<u8, T>(&data_chunk, audio), /* technicaly not part
            * of the wav spec,
             * but it does not
             * disallow it either,
             * so we support it */
            ValidatedSampleType::I16 => read_into_typed_internal::<i16, T>(&data_chunk, audio),
            ValidatedSampleType::I24 => read_into_typed_internal::<I24, T>(&data_chunk, audio),
            ValidatedSampleType::I32 => read_into_typed_internal::<i32, T>(&data_chunk, audio),
            ValidatedSampleType::F32 => read_into_typed_internal::<f32, T>(&data_chunk, audio),
            ValidatedSampleType::F64 => read_into_typed_internal::<f64, T>(&data_chunk, audio),
        }
    }
}

/// Wrap interleaved WAV samples into an AudioSamples without a deinterleave copy.
///
/// WAV stores samples interleaved: [L0, R0, L1, R1, …].  Rather than physically rearranging
/// memory into planar layout, multi-channel audio is represented as an F-order (column-major)
/// Array2 with shape (channels, frames) and strides (1, channels), which matches the
/// interleaved memory layout exactly.
fn build_samples_from_interleaved_vec<'a, T>(
    interleaved_data: NonEmptyVec<T>,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    T: StandardSample + 'static,
{
    // SAFETY: sample_rate comes from validated WAV header which requires non-zero sample rate
    if num_channels.get() == 1 {
        // Mono: data is already in correct format
        AudioSamples::new_mono(Array1::from_vec(interleaved_data.into_vec()), sample_rate).map_err(Into::into)
    } else {
        let total_samples = interleaved_data.len();
        let frames = total_samples.get() / num_channels.get() as usize;

        if frames == 0 {
            return Err(AudioIOError::corrupted_data_simple(
                "No frames in audio data",
                format!("total_samples={total_samples}, channels={num_channels}"),
            ));
        }

        // Fix C: wrap the interleaved Vec directly as an F-order (column-major) Array2.
        // An F-order array with logical shape (channels, frames) stores data as
        // [s[0,0], s[1,0], s[0,1], s[1,1], …] — exactly WAV's interleaved layout.
        // This avoids the deinterleave allocation and scatter-write entirely.
        let arr = Array2::from_shape_vec(
            (num_channels.get() as usize, frames).f(),
            interleaved_data.into_vec(), // Fix B: move, no copy
        )
        .map_err(|e| AudioIOError::corrupted_data_simple("Array shape error", e.to_string()))?;

        AudioSamples::new_multi_channel(arr, sample_rate).map_err(Into::into)
    }
}

fn read_into_typed_internal<'a, S, T>(data_chunk: &DataChunk<'a>, audio: &mut AudioSamples<'a, T>) -> AudioIOResult<()>
where
    S: StandardSample + 'static,
    T: StandardSample + ConvertFrom<S> + 'static,
{
    let bytes_per_sample = S::BITS as usize / 8;
    let num_channels = audio.num_channels();
    let frame_bytes = bytes_per_sample * num_channels.get() as usize;

    // Decode only whole frames: a trailing partial frame (ragged data chunk, odd trailing byte,
    // or a clamped streaming size) is dropped rather than rejected, matching ffmpeg/VLC behaviour.
    let usable = (data_chunk.len() / frame_bytes) * frame_bytes;
    let data_chunk = DataChunk::from_bytes(&data_chunk.as_bytes()[..usable]);

    let converted = data_chunk.read_samples::<S, T>()?;

    if converted.len() != audio.total_samples() {
        return Err(AudioIOError::corrupted_data_simple(
            "Sample count mismatch",
            format!(
                "Converted sample count {} does not match target audio sample count {}",
                converted.len(),
                audio.total_samples(),
            ),
        ));
    }

    // For mono audio, data is already in correct format
    if audio.is_mono() {
        audio.replace_with_vec(&converted).map_err(|e| e.into())
    } else {
        // Multi-channel: deinterleave the converted data before replacing
        // Use optimized deinterleave from audio_samples
        let planar_data = audio_samples::simd_conversions::deinterleave_multi_vec(&converted, num_channels)
            .map_err(|e| AudioIOError::corrupted_data_simple("Deinterleave failed", e.to_string()))?;
        audio.replace_with_vec(&planar_data).map_err(|e| e.into())
    }
}

fn read_typed_internal<'a, S, T>(
    data_chunk: &DataChunk<'a>,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    S: StandardSample + 'static,
    T: StandardSample + ConvertFrom<S> + 'static,
{
    let bytes_per_sample = S::BITS as usize / 8;
    let frame_bytes = bytes_per_sample * num_channels.get() as usize;

    // Decode only whole frames; drop a ragged trailing partial frame rather than rejecting the
    // file, matching ffmpeg/VLC.
    let usable = (data_chunk.len() / frame_bytes) * frame_bytes;
    let data_chunk = DataChunk::from_bytes(&data_chunk.as_bytes()[..usable]);

    let converted = data_chunk.read_samples::<S, T>()?;

    build_samples_from_interleaved_vec(converted, num_channels, sample_rate)
}

/// Decode 8-bit companded (mu-law / a-law) data to linear PCM and build an `AudioSamples<T>`.
///
/// Each byte expands to one 16-bit linear sample which is then converted to `T`. Only whole
/// frames are decoded; a ragged trailing partial frame is dropped.
fn read_companded<'a, T>(
    data_chunk: &DataChunk<'a>,
    companding: Companding,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let decoded = decode_companded::<T>(data_chunk, companding, num_channels)?;
    build_samples_from_interleaved_vec(decoded, num_channels, sample_rate)
}

/// `read_companded` for the `read_into` path: decode into an existing buffer.
fn read_companded_into<'a, T>(
    data_chunk: &DataChunk<'a>,
    companding: Companding,
    audio: &mut AudioSamples<'a, T>,
) -> AudioIOResult<()>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let num_channels = audio.num_channels();
    let decoded = decode_companded::<T>(data_chunk, companding, num_channels)?;

    if decoded.len() != audio.total_samples() {
        return Err(AudioIOError::corrupted_data_simple(
            "Sample count mismatch",
            format!(
                "Decoded sample count {} does not match target audio sample count {}",
                decoded.len(),
                audio.total_samples(),
            ),
        ));
    }

    if audio.is_mono() {
        audio.replace_with_vec(&decoded).map_err(|e| e.into())
    } else {
        let planar_data = audio_samples::simd_conversions::deinterleave_multi_vec(&decoded, num_channels)
            .map_err(|e| AudioIOError::corrupted_data_simple("Deinterleave failed", e.to_string()))?;
        audio.replace_with_vec(&planar_data).map_err(|e| e.into())
    }
}

/// Shared mu-law/a-law expansion: one companded byte → one `T` sample, whole frames only.
fn decode_companded<T>(
    data_chunk: &DataChunk<'_>,
    companding: Companding,
    num_channels: NonZeroU32,
) -> AudioIOResult<NonEmptyVec<T>>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let bytes = data_chunk.as_bytes();
    let channels = num_channels.get() as usize;
    // Companded data is one byte per sample; keep only whole frames.
    let usable = (bytes.len() / channels) * channels;
    if usable == 0 {
        return Err(AudioIOError::corrupted_data_simple(
            "No frames in companded audio data",
            format!("{} bytes, {channels} channels", bytes.len()),
        ));
    }
    let decoded: Vec<T> = bytes[..usable]
        .iter()
        .map(|&b| T::convert_from(companding.decode(b)))
        .collect();
    // SAFETY: usable > 0 guarantees at least one decoded sample.
    Ok(unsafe { NonEmptyVec::new_unchecked(decoded) })
}

/// Decode an ADPCM `data` chunk to linear PCM and build an `AudioSamples<T>`.
fn read_adpcm<'a, T>(
    fmt_chunk: &FmtChunk<'_>,
    data_chunk: &DataChunk<'a>,
    num_channels: NonZeroU32,
    sample_rate: NonZeroU32,
) -> AudioIOResult<AudioSamples<'a, T>>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let decoded = decode_adpcm::<T>(fmt_chunk, data_chunk)?;
    build_samples_from_interleaved_vec(decoded, num_channels, sample_rate)
}

/// `read_adpcm` for the `read_into` path: decode into an existing buffer.
fn read_adpcm_into<'a, T>(
    fmt_chunk: &FmtChunk<'_>,
    data_chunk: &DataChunk<'a>,
    audio: &mut AudioSamples<'a, T>,
) -> AudioIOResult<()>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let num_channels = audio.num_channels();
    let decoded = decode_adpcm::<T>(fmt_chunk, data_chunk)?;

    if decoded.len() != audio.total_samples() {
        return Err(AudioIOError::corrupted_data_simple(
            "Sample count mismatch",
            format!(
                "Decoded ADPCM sample count {} does not match target audio sample count {}",
                decoded.len(),
                audio.total_samples(),
            ),
        ));
    }

    if audio.is_mono() {
        audio.replace_with_vec(&decoded).map_err(|e| e.into())
    } else {
        let planar_data = audio_samples::simd_conversions::deinterleave_multi_vec(&decoded, num_channels)
            .map_err(|e| AudioIOError::corrupted_data_simple("Deinterleave failed", e.to_string()))?;
        audio.replace_with_vec(&planar_data).map_err(|e| e.into())
    }
}

/// Shared ADPCM expansion: decode blocks to 16-bit linear PCM, then convert to `T`.
fn decode_adpcm<T>(fmt_chunk: &FmtChunk<'_>, data_chunk: &DataChunk<'_>) -> AudioIOResult<NonEmptyVec<T>>
where
    T: StandardSample + ConvertFrom<i16> + 'static,
{
    let decoded_i16 = crate::wav::adpcm::decode(fmt_chunk, data_chunk.as_bytes())?;
    if decoded_i16.is_empty() {
        return Err(AudioIOError::corrupted_data_simple(
            "No samples decoded from ADPCM data",
            format!("{} data bytes", data_chunk.len()),
        ));
    }
    let converted: Vec<T> = decoded_i16.into_iter().map(T::convert_from).collect();
    // SAFETY: non-empty checked above.
    Ok(unsafe { NonEmptyVec::new_unchecked(converted) })
}

// Helper functions for WAV writing

/// Maps SampleType to WAV FormatCode
///
/// Returns `None` if the SampleType is not supported for WAV writing.
/// This function validates that the sample type is valid before conversion.
const fn sample_type_to_format(sample_type: SampleType) -> Option<FormatCode> {
    match sample_type {
        SampleType::U8 | SampleType::I16 | SampleType::I24 | SampleType::I32 => Some(FormatCode::Pcm),
        SampleType::F32 | SampleType::F64 => Some(FormatCode::IeeeFloat),
        _ => None,
    }
}

/// Get SampleType from AudioSample type parameter
const fn get_sample_type<T>() -> SampleType
where
    T: StandardSample,
{
    T::SAMPLE_TYPE
}

/// Write 16-byte base FMT chunk
fn write_base_fmt<T, W>(writer: &mut W, channels: u16, sample_rate: u32) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_type = get_sample_type::<T>();
    let format_code =
        sample_type_to_format(sample_type).ok_or(AudioIOError::WavError(WavError::UnsupportedSampleType))?;
    let bits_per_sample = T::BITS as u16;
    let bytes_per_sample = T::BYTES as u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;

    // FMT chunk header
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Chunk size (16 bytes)

    // FMT chunk data (16 bytes)
    writer.write_all(&format_code.as_u16().to_le_bytes())?; // Format code
    writer.write_all(&channels.to_le_bytes())?; // Channels
    writer.write_all(&sample_rate.to_le_bytes())?; // Sample rate
    writer.write_all(&byte_rate.to_le_bytes())?; // Byte rate
    writer.write_all(&block_align.to_le_bytes())?; // Block align
    writer.write_all(&bits_per_sample.to_le_bytes())?; // Bits per sample

    Ok(())
}

/// Determine if extensible format is needed
const fn needs_extensible_format<T>(channels: u16) -> bool
where
    T: StandardSample,
{
    // Use extensible format for more than 2 channels or non-standard bit depths.
    // 8-bit, 16-bit, and 32-bit PCM/float formats use base format for 1-2 channels
    // for maximum compatibility (e.g. soundfile/libsndfile doesn't support
    // WAVE_FORMAT_EXTENSIBLE with 8-bit PCM).
    channels > 2 || (T::BITS != 8 && T::BITS != 16 && T::BITS != 32)
}

/// Write 40-byte extensible FMT chunk
fn write_extensible_fmt<T, W>(writer: &mut W, channels: u16, sample_rate: u32) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_type = get_sample_type::<T>();
    let format_code =
        sample_type_to_format(sample_type).ok_or(AudioIOError::WavError(WavError::UnsupportedSampleType))?;
    let bits_per_sample = T::BITS as u16;
    let bytes_per_sample = T::BYTES as u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;

    // Windows standard speaker position channel masks
    // Based on WAVE_FORMAT_EXTENSIBLE specification
    let channel_mask: u32 = match channels {
        1 => 0x4,   // SPEAKER_FRONT_CENTER
        2 => 0x3,   // SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT
        3 => 0x7,   // FRONT_LEFT | FRONT_RIGHT | FRONT_CENTER
        4 => 0x33,  // FRONT_LEFT | FRONT_RIGHT | BACK_LEFT | BACK_RIGHT
        5 => 0x37,  // 4.0 + FRONT_CENTER
        6 => 0x3F,  // 5.0 + LFE (5.1)
        7 => 0x13F, // 5.1 + BACK_CENTER
        8 => 0x63F, // 5.1 + SIDE_LEFT | SIDE_RIGHT (7.1)
        _ => {
            // For >8 channels, saturate to all bits set
            // This is a reasonable fallback for non-standard channel configurations
            if channels < 32 {
                (1u32 << channels) - 1
            } else {
                0xFFFFFFFF
            }
        },
    };

    // FMT chunk header
    writer.write_all(b"fmt ")?;
    writer.write_all(&40u32.to_le_bytes())?; // Chunk size (40 bytes)

    // Base FMT chunk data (16 bytes)
    writer.write_all(&FormatCode::Extensible.as_u16().to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // Extension size (2 bytes)
    writer.write_all(&22u16.to_le_bytes())?;

    // Extended format info (22 bytes)
    writer.write_all(&bits_per_sample.to_le_bytes())?; // Valid bits per sample
    writer.write_all(&channel_mask.to_le_bytes())?; // Channel mask

    // Sub-format GUID (16 bytes)
    // Standard KSDATAFORMAT_SUBTYPE_PCM:       {00000001-0000-0010-8000-00AA00389B71}
    // Standard KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: {00000003-0000-0010-8000-00AA00389B71}
    // In memory: Data1 (4 bytes LE), Data2 (2 bytes LE), Data3 (2 bytes LE), Data4 (8 bytes BE)
    let mut sub_format = [0u8; 16];
    // Data1: format_code as u32 LE (e.g. 1 for PCM, 3 for IEEE float)
    sub_format[0..4].copy_from_slice(&(u32::from(format_code.as_u16())).to_le_bytes());
    // Data2: 0x0000
    sub_format[4..6].copy_from_slice(&0u16.to_le_bytes());
    // Data3: 0x0010
    sub_format[6..8].copy_from_slice(&0x0010u16.to_le_bytes());
    // Data4: 0x8000-00AA00389B71 (big-endian)
    sub_format[8..16].copy_from_slice(&[0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71]);
    writer.write_all(&sub_format)?;

    Ok(())
}

/// Build an interleaved byte buffer for WAV output and write it in one go.
/// Mono fast-path uses the underlying contiguous bytes view; multi-channel
/// uses optimized interleave functions for better cache locality.
fn write_audio_data_interleaved<T, W>(writer: &mut W, audio: &AudioSamples<T>) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let num_channels = audio.num_channels();

    // Mono data is already laid out correctly; respect I24 packing via AudioSamples::bytes
    if audio.is_mono() {
        let bytes = audio.bytes()?;
        writer.write_all(bytes.as_slice())?;
        return Ok(());
    }

    // Fast path: F-order (Fortran-contiguous) multi-channel arrays already have interleaved
    // layout in memory — [L0, R0, L1, R1, …] — matching WAV's on-disk format exactly.
    // On little-endian platforms the in-memory bytes need no reordering for any integer or
    // float type (except I24 which uses a non-standard 4-byte in-memory representation).
    #[cfg(target_endian = "little")]
    if TypeId::of::<T>() != TypeId::of::<I24>()
        && let AudioData::Multi(ref m) = audio.data
    {
        let view = m.as_view();
        if !view.is_standard_layout()
            && let Some(slice) = view.as_slice_memory_order()
        {
            // Safety: T is a plain numeric type (StandardSample, not I24).
            // Casting any &[T] to &[u8] is valid — bytes are always initialised.
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) };
            writer.write_all(bytes)?;
            return Ok(());
        }
    }

    let bytes_per_sample = if TypeId::of::<T>() == TypeId::of::<I24>() {
        3usize
    } else {
        mem::size_of::<T>()
    };

    // C-order (standard-layout) path: stream interleaved bytes via a small tile buffer.
    //
    // The naive approach (view.t().as_standard_layout()) allocates a 10 MB intermediate Vec
    // upfront, incurring ~2500 OS page faults (~2.5 ms) before any data moves.  Instead, we
    // reuse a single fixed-size tile buffer (~64 KB), fill it with interleaved samples one
    // tile at a time, and write each tile to the output — zero large allocations, sequential
    // reads within each channel's tile window, sequential writes.
    #[cfg(target_endian = "little")]
    if TypeId::of::<T>() != TypeId::of::<I24>()
        && let AudioData::Multi(ref m) = audio.data
    {
        let view = m.as_view();
        if view.is_standard_layout() {
            use std::mem::MaybeUninit;

            let channels = view.shape()[0];
            let frames = view.shape()[1];
            let sample_size = mem::size_of::<T>();

            // Tile size: aim for ~64 KB of output data to stay in L1/L2 cache.
            const TARGET_TILE_BYTES: usize = 512 * 1024;
            let tile_frames = (TARGET_TILE_BYTES / (channels * sample_size)).max(1);

            // One contiguous slice per channel (rows are contiguous in C-order).
            let rows: Vec<&[T]> = (0..channels)
                .map(|c| view.row(c).to_slice().expect("C-order row is contiguous"))
                .collect();

            // Pre-allocate tile buffer once — avoids per-tile page faults.
            let tile_capacity = tile_frames * channels;

            // Allocate uninitialised buffer
            let mut tile: Vec<MaybeUninit<T>> = Vec::with_capacity(tile_capacity);

            // SAFETY: we will initialise every element before reading
            unsafe { tile.set_len(tile_capacity) };

            for tile_start in (0..frames).step_by(tile_frames) {
                let tile_end = (tile_start + tile_frames).min(frames);
                let actual_frames = tile_end - tile_start;
                // Channels-outer loop: each channel's tile window is read sequentially,
                // letting the HW prefetcher stream contiguous cache lines.  Writes land
                // at stride=channels in the tile buffer (still within a small working set).
                for c in 0..channels {
                    for (i, f) in (tile_start..tile_end).enumerate() {
                        // Safety: f < frames and c < channels, both in bounds.
                        tile[i * channels + c].write(rows[c][f]);
                    }
                }
                // Safety: T is a plain numeric type (not I24), LE platform; tile
                // slice is fully initialised above.
                let bytes = unsafe {
                    std::slice::from_raw_parts(tile.as_ptr().cast::<u8>(), actual_frames * channels * sample_size)
                };
                writer.write_all(bytes)?;
            }
            return Ok(());
        }
    }

    // Fallback: materialise an interleaved Vec then serialise sample-by-sample.
    // Handles I24, big-endian platforms, and non-contiguous array layouts.
    let interleaved = audio.data.as_interleaved_vec();
    let total_samples = interleaved.len();

    const TARGET_CHUNK_BYTES: usize = 256 * 1024;
    let chunk_samples = TARGET_CHUNK_BYTES
        .checked_div(bytes_per_sample)
        .ok_or(AudioIOError::corrupted_data_simple(
            "Chunk size calculation overflow",
            "TARGET_CHUNK_BYTES / bytes_per_sample",
        ))?
        .max(num_channels.get() as usize);

    let mut buf = vec![0u8; chunk_samples * bytes_per_sample];

    let mut sample_start = 0;
    while sample_start < total_samples.get() {
        let remaining = total_samples.get() - sample_start;
        let samples_this_chunk = remaining.min(chunk_samples);
        let bytes_this_chunk = samples_this_chunk * bytes_per_sample;

        let mut write_idx = 0;
        for sample in interleaved.iter().skip(sample_start).take(samples_this_chunk) {
            let bytes = sample.to_le_bytes();
            let dst = &mut buf[write_idx..write_idx + bytes_per_sample];
            dst.copy_from_slice(bytes.as_ref());
            write_idx += bytes_per_sample;
        }

        debug_assert_eq!(write_idx, bytes_this_chunk);
        writer.write_all(&buf[..bytes_this_chunk])?;
        sample_start += samples_this_chunk;
    }

    Ok(())
}

/// Optional metadata chunks written after the audio `data` chunk.
///
/// Lets callers persist tags and markers that would otherwise be lost on a read→write round-trip.
/// Build one and pass it to [`write_wav_with_metadata`] / [`crate::write_with_metadata`].
#[derive(Debug, Default, Clone)]
pub struct WavMetadata {
    /// LIST/INFO tags (title, artist, …).
    pub info: Option<InfoMetadata>,
    /// Cue points / markers.
    pub cue_points: Vec<CuePoint>,
}

impl WavMetadata {
    /// True if there is nothing to write.
    pub fn is_empty(&self) -> bool {
        self.cue_points.is_empty() && self.info.as_ref().is_none_or(|i| i.to_list_chunk().is_none())
    }

    /// Serialise all present metadata into one contiguous, word-aligned buffer of complete chunks.
    fn to_chunk_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if let Some(ref info) = self.info
            && let Some(list) = info.to_list_chunk()
        {
            out.extend_from_slice(&list);
        }
        if let Some(cue) = cue_chunk_bytes(&self.cue_points) {
            out.extend_from_slice(&cue);
        }
        out
    }
}

// Write complete WAV file to a writer
pub(crate) fn write_wav<T, W>(writer: W, audio: &AudioSamples<T>, opts: WriteOptions) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    write_wav_with_metadata(writer, audio, opts, &WavMetadata::default())
}

/// Write a complete WAV file, appending optional metadata chunks (LIST/INFO, cue) after the data.
pub(crate) fn write_wav_with_metadata<T, W>(
    writer: W,
    audio: &AudioSamples<T>,
    opts: WriteOptions,
    metadata: &WavMetadata,
) -> AudioIOResult<()>
where
    T: StandardSample + 'static,
    W: Write,
{
    let sample_rate = audio.sample_rate().get();
    let channels = audio.num_channels().get() as u16;
    let bytes_per_sample_disk = if TypeId::of::<T>() == TypeId::of::<I24>() {
        3usize
    } else {
        mem::size_of::<T>()
    };
    let data_size = audio
        .samples_per_channel()
        .get()
        .checked_mul(audio.num_channels().get() as usize)
        .and_then(|v| v.checked_mul(bytes_per_sample_disk))
        .ok_or_else(|| {
            AudioIOError::corrupted_data_simple(
                "Byte size overflow during header calculation",
                format!(
                    "channels={}, samples_per_channel={}, bytes_per_sample={}",
                    channels,
                    audio.samples_per_channel().get(),
                    bytes_per_sample_disk
                ),
            )
        })?;

    // Calculate padded data size (must be even)
    let padded_data_size = if data_size % 2 == 1 { data_size + 1 } else { data_size };

    // Determine FMT chunk size
    let fmt_chunk_size = if needs_extensible_format::<T>(channels) { 40 } else { 16 };
    let fmt_total_size = 8 + fmt_chunk_size; // chunk header + data

    // Serialise trailing metadata chunks up front so their bytes are counted in the RIFF size.
    let metadata_bytes = metadata.to_chunk_bytes();

    // Calculate total file size
    let file_size = 4 + fmt_total_size + 8 + padded_data_size + metadata_bytes.len(); // WAVE + FMT + DATA + metadata

    let mut writer = BufWriter::with_capacity(opts.write_buf_capacity, writer);

    // Write RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // Write FMT chunk
    if needs_extensible_format::<T>(channels) {
        write_extensible_fmt::<T, _>(&mut writer, channels, sample_rate)?;
    } else {
        write_base_fmt::<T, _>(&mut writer, channels, sample_rate)?;
    }

    // Write DATA chunk header
    writer.write_all(b"data")?;
    writer.write_all(&(data_size as u32).to_le_bytes())?;

    // Write audio data (interleaved for multi-channel)
    write_audio_data_interleaved(&mut writer, audio)?;

    // Add padding byte if needed
    if data_size % 2 == 1 {
        writer.write_all(&[0])?;
    }

    // Append metadata chunks (LIST/INFO, cue) after the data chunk.
    if !metadata_bytes.is_empty() {
        writer.write_all(&metadata_bytes)?;
    }

    writer.flush()?;
    Ok(())
}

impl<'a> AudioFileWrite for WavFile<'a> {
    fn write<P, T>(&mut self, out_fp: P) -> AudioIOResult<()>
    where
        P: AsRef<Path>,
        T: StandardSample + 'static,
    {
        let audio = self.read::<T>()?;
        let file = File::create(out_fp)?;
        write_wav(file, &audio, WriteOptions::default())?;
        Ok(())
    }
}

impl Display for WavFile<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "WAV File:")?;
        writeln!(f, "File Path: {:?}", self.file_path)?;
        writeln!(f, "Chunks:")?;
        for chunk in &self.chunks {
            writeln!(
                f,
                "  ID: {}, Offset: {}, Logical Size: {}, Total Size: {}",
                chunk.id, chunk.offset, chunk.logical_size, chunk.total_size
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod wav_tests {
    use audio_samples::sample_rate;
    use non_empty_slice::NonEmptySlice;

    use super::*;
    use crate::wav::FormatCode;

    #[allow(dead_code)]
    fn make_base_fmt_bytes(
        format_code: u16,
        channels: u16,
        sample_rate: u32,
        byte_rate: u32,
        block_align: u16,
        bits_per_sample: u16,
    ) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..2].copy_from_slice(&format_code.to_le_bytes());
        bytes[2..4].copy_from_slice(&channels.to_le_bytes());
        bytes[4..8].copy_from_slice(&sample_rate.to_le_bytes());
        bytes[8..12].copy_from_slice(&byte_rate.to_le_bytes());
        bytes[12..14].copy_from_slice(&block_align.to_le_bytes());
        bytes[14..16].copy_from_slice(&bits_per_sample.to_le_bytes());
        bytes
    }

    #[test]
    fn test_wav_open() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file = WavFile::open_with_options(wav_path, OpenOptions::default());
        assert!(wav_file.is_ok(), "Failed to open test WAV file");
    }

    #[test]
    fn test_wav_fmt_chunk() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file =
            WavFile::open_with_options(wav_path, OpenOptions::default()).expect("Failed to open test WAV file");
        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::Pcm, "Format code mismatch");
        assert_eq!(fmt_chunk.sample_rate(), 44100, "Sample rate mismatch");
        assert_eq!(fmt_chunk.channels(), 2, "Channel count mismatch");
    }

    #[test]
    fn test_wav_data_chunk() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file =
            WavFile::open_with_options(wav_path, OpenOptions::default()).expect("Failed to open test WAV file");

        let audio = wav_file.read::<i16>();
        assert!(audio.is_ok(), "Failed to read audio samples from DATA chunk");
    }

    #[test]
    fn test_wav_properties() {
        let wav_path = Path::new("resources/test.wav");
        let wav_file =
            WavFile::open_with_options(wav_path, OpenOptions::default()).expect("Failed to open test WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100), "Sample rate mismatch");
        assert_eq!(base_info.channels, 2, "Channel count mismatch");
        assert_eq!(base_info.bits_per_sample, 16, "Bits per sample mismatch");

        let specific_info = wav_file.specific_info();
        assert_eq!(specific_info.encoding, FormatCode::Pcm, "Encoding mismatch");
        assert!(
            specific_info.available_chunks.contains(&FMT_CHUNK),
            "FMT chunk should be available"
        );
        assert!(
            specific_info.available_chunks.contains(&DATA_CHUNK),
            "DATA chunk should be available"
        );

        println!("Base Info: {base_info:#}");
    }

    #[test]
    fn test_wav_write_i16() {
        use std::fs;

        use audio_samples::{AudioTypeConversion, sine_wave};

        // Generate a sine wave
        let sample_rate = sample_rate!(44100);
        let frequency = 440.0;
        let duration = Duration::from_secs_f64(1.0); // 1 second
        let amplitude = 0.5;
        let sine_samples = sine_wave::<f32>(frequency, duration, sample_rate, amplitude);
        let sine_i16 = sine_samples.to_format::<i16>();

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_i16.wav");
        write_wav(
            std::fs::File::create(&output_path).expect("Failed to create output file"),
            &sine_i16,
            WriteOptions::default(),
        )
        .expect("Failed to write WAV file");

        // Verify file was created and has reasonable size
        let metadata = fs::metadata(&output_path).expect("Failed to get file metadata");
        assert!(metadata.len() > 44, "WAV file too small"); // At least header size

        // Read back and verify
        let wav_file =
            WavFile::open_with_options(&output_path, OpenOptions::default()).expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100));
        assert_eq!(base_info.channels, 1);
        assert_eq!(base_info.bits_per_sample, 16);

        let read_samples = wav_file.read::<i16>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), sine_i16.total_samples());

        // // Clean up
        // fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_f32() {
        use std::fs;

        use audio_samples::sine_wave;

        // Generate a sine wave
        let sample_rate = sample_rate!(48000);
        let frequency = 1000.0;
        let duration = Duration::from_secs_f64(0.5); // 0.5 seconds
        let amplitude = 0.8;
        let sine_samples = sine_wave::<f32>(frequency, duration, sample_rate, amplitude);

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_f32.wav");
        write_wav(
            std::fs::File::create(&output_path).expect("Failed to create output file"),
            &sine_samples,
            WriteOptions::default(),
        )
        .expect("Failed to write WAV file");

        // Read back and verify
        let wav_file =
            WavFile::open_with_options(&output_path, OpenOptions::default()).expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(48000));
        assert_eq!(base_info.channels, 1);
        assert_eq!(base_info.bits_per_sample, 32);

        let read_samples = wav_file.read::<f32>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), sine_samples.total_samples());

        // Verify format is IEEE Float
        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::IeeeFloat);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_read_i24_roundtrip() {
        use std::fs;

        use audio_samples::sine_wave;

        let sample_rate = sample_rate!(48_000);
        let duration = Duration::from_millis(20);
        let audio = sine_wave::<I24>(440.0, duration, sample_rate, 0.5);

        let output_path = std::env::temp_dir().join(format!("test_read_i24_roundtrip_{}.wav", std::process::id()));
        write_wav(
            std::fs::File::create(&output_path).expect("Failed to create output file"),
            &audio,
            WriteOptions::default(),
        )
        .expect("Failed to write I24 WAV file");

        let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
            .expect("Failed to reopen written WAV file");
        let read_samples = wav_file.read::<I24>().expect("Failed to read I24 samples");

        assert_eq!(
            read_samples.total_samples(),
            audio.total_samples(),
            "Roundtrip sample count mismatch"
        );
        assert_eq!(read_samples.num_channels(), audio.num_channels());
        assert_eq!(read_samples.sample_rate(), audio.sample_rate());

        let original = audio.to_interleaved_vec();
        let roundtrip = read_samples.to_interleaved_vec();
        assert_eq!(original, roundtrip, "I24 roundtrip data mismatch");

        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_stereo() {
        use std::fs;

        use audio_samples::sine_wave;

        // Generate stereo sine waves (left: 440Hz, right: 880Hz)
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f64(0.25);
        let left = sine_wave::<f32>(440.0, duration, sample_rate, 0.6);
        let right = sine_wave::<f32>(880.0, duration, sample_rate, 0.4);

        // Combine into stereo
        let stereo = audio_samples::AudioEditing::stack(NonEmptySlice::new(&[left, right]).expect("two channels"))
            .expect("Failed to create stereo");

        // Write to file
        let output_path = std::env::temp_dir().join("test_write_stereo.wav");
        write_wav(
            std::fs::File::create(&output_path).expect("Failed to create output file"),
            &stereo,
            WriteOptions::default(),
        )
        .expect("Failed to write stereo WAV file");

        // Read back and verify
        let wav_file =
            WavFile::open_with_options(&output_path, OpenOptions::default()).expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.sample_rate, sample_rate!(44100));
        assert_eq!(base_info.channels, 2);
        assert_eq!(base_info.bits_per_sample, 32);

        let read_samples = wav_file.read::<f32>().expect("Failed to read samples");
        assert_eq!(read_samples.total_samples(), stereo.total_samples());
        assert_eq!(read_samples.num_channels().get(), 2);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_type_conversion() {
        use std::fs;

        use audio_samples::{AudioTypeConversion, sine_wave};

        // Generate f32 sine wave
        let sample_rate = sample_rate!(44100);
        let sine_f32 = sine_wave::<f32>(440.0, Duration::from_secs_f64(0.1), sample_rate, 0.7);

        // Write as i16 (should convert)
        let output_path = std::env::temp_dir().join("test_conversion.wav");
        let sine_i16 = sine_f32.to_format::<i16>();
        write_wav(
            std::fs::File::create(&output_path).expect("Failed to create output file"),
            &sine_i16,
            WriteOptions::default(),
        )
        .expect("Failed to write converted WAV file");

        // Verify it's written as i16 PCM
        let wav_file =
            WavFile::open_with_options(&output_path, OpenOptions::default()).expect("Failed to open written WAV file");

        let base_info = wav_file.base_info().expect("Failed to get base info");
        assert_eq!(base_info.bits_per_sample, 16);

        let fmt_chunk = wav_file.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::Pcm);

        // Clean up
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_audiofilewrite_trait() {
        use std::fs;

        use audio_samples::sine_wave;

        // Create a test WAV file first
        let sample_rate = sample_rate!(22050);
        let sine_samples = sine_wave::<i16>(330.0, Duration::from_secs_f64(0.2), sample_rate, 0.5);
        let input_path = std::env::temp_dir().join("test_input.wav");
        write_wav(
            std::fs::File::create(&input_path).expect("Failed to create input file"),
            &sine_samples,
            WriteOptions::default(),
        )
        .expect("Failed to write input WAV file");

        // Open the WAV file and use the trait method to write as f32
        let mut wav_file =
            WavFile::open_with_options(&input_path, OpenOptions::default()).expect("Failed to open input WAV file");

        let output_path = std::env::temp_dir().join("test_trait_output.wav");
        wav_file
            .write::<_, f32>(&output_path)
            .expect("Failed to write using trait method");

        // Verify the output is f32
        let output_wav =
            WavFile::open_with_options(&output_path, OpenOptions::default()).expect("Failed to open output WAV file");

        let base_info = output_wav.base_info().expect("Failed to get base info");
        assert_eq!(base_info.bits_per_sample, 32);
        assert_eq!(base_info.sample_rate, sample_rate!(22050));

        let fmt_chunk = output_wav.fmt_chunk();
        assert_eq!(fmt_chunk.format_code(), FormatCode::IeeeFloat);

        // Clean up
        fs::remove_file(&input_path).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn test_wav_write_read_roundtrip_validation() {
        use std::fs;

        use audio_samples::{AudioTypeConversion, sine_wave};

        // Test multiple sample types with comprehensive validation
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f64(0.5);
        let base_sine = sine_wave::<f32>(440.0, duration, sample_rate, 0.5);

        // Test cases: (type_name, bits_per_sample, format_code)
        let test_cases = [
            ("i16", 16, FormatCode::Pcm),
            ("i32", 32, FormatCode::Pcm),
            ("f32", 32, FormatCode::IeeeFloat),
        ];

        for (type_name, expected_bits, expected_format) in test_cases.iter() {
            let output_path = std::env::temp_dir().join(format!("test_roundtrip_{type_name}.wav"));

            match *type_name {
                "i16" => {
                    let samples = base_sine.to_format::<i16>();
                    write_wav(
                        std::fs::File::create(&output_path).expect("Failed to create output file"),
                        &samples,
                        WriteOptions::default(),
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity
                    let read_samples = wav_file.read::<i16>().expect("Failed to read WAV samples");
                    let read_bytes = read_samples.bytes().expect("Failed to get bytes from read samples");
                    let written_bytes = samples.bytes().expect("Failed to get bytes from written samples");
                    assert_eq!(read_bytes.as_slice(), written_bytes.as_slice());
                },
                "i32" => {
                    let samples = base_sine.to_format::<i32>();
                    write_wav(
                        std::fs::File::create(&output_path).expect("Failed to create output file"),
                        &samples,
                        WriteOptions::default(),
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity
                    let read_samples = wav_file.read::<i32>().expect("Failed to read WAV samples");
                    let read_bytes = read_samples.bytes().expect("Failed to get bytes from read samples");
                    let written_bytes = samples.bytes().expect("Failed to get bytes from written samples");
                    assert_eq!(read_bytes.as_slice(), written_bytes.as_slice());
                },
                "f32" => {
                    write_wav(
                        std::fs::File::create(&output_path).expect("Failed to create output file"),
                        &base_sine,
                        WriteOptions::default(),
                    )
                    .expect("Failed to write WAV file");

                    // Validate WAV structure
                    let wav_file = WavFile::open_with_options(&output_path, OpenOptions::default())
                        .expect("Failed to open WAV file");
                    let base_info = wav_file.base_info().expect("Failed to get WAV base info");
                    let fmt_chunk = wav_file.fmt_chunk();

                    assert_eq!(base_info.sample_rate, sample_rate!(44100));
                    assert_eq!(base_info.bits_per_sample, *expected_bits);
                    assert_eq!(fmt_chunk.format_code(), *expected_format);

                    // Read back and verify data integrity (with small tolerance for f32)
                    let read_samples = wav_file.read::<f32>().expect("Failed to read WAV samples");
                    let orig_bytes = base_sine.bytes().expect("Failed to get bytes from original samples");
                    let read_bytes = read_samples.bytes().expect("Failed to get bytes from read samples");

                    let orig_f32: &[f32] = bytemuck::cast_slice(orig_bytes.as_slice());
                    let read_f32: &[f32] = bytemuck::cast_slice(read_bytes.as_slice());

                    for (orig, read) in orig_f32.iter().zip(read_f32.iter()) {
                        assert!((orig - read).abs() < 1e-6, "f32 samples should be nearly identical");
                    }
                },
                _ => unreachable!(),
            }

            // Verify file is readable by external tools (basic structure check)
            let file_bytes = std::fs::read(&output_path).expect("Failed to read output file");
            assert!(file_bytes.len() > 44, "WAV file should have proper header + data");
            assert_eq!(&file_bytes[0..4], b"RIFF");
            assert_eq!(&file_bytes[8..12], b"WAVE");

            // Find and validate FMT and DATA chunks exist
            let mut has_fmt = false;
            let mut has_data = false;
            let mut pos = 12;

            while pos + 8 <= file_bytes.len() {
                let chunk_id = &file_bytes[pos..pos + 4];
                let chunk_size = u32::from_le_bytes([
                    file_bytes[pos + 4],
                    file_bytes[pos + 5],
                    file_bytes[pos + 6],
                    file_bytes[pos + 7],
                ]) as usize;

                if chunk_id == b"fmt " {
                    has_fmt = true;
                    assert!(chunk_size >= 16, "FMT chunk should be at least 16 bytes");
                } else if chunk_id == b"data" {
                    has_data = true;
                    assert!(chunk_size > 0, "DATA chunk should contain audio data");
                }

                pos += 8 + chunk_size + (chunk_size % 2); // Add padding if odd size
            }

            assert!(has_fmt, "WAV file should have FMT chunk");
            assert!(has_data, "WAV file should have DATA chunk");

            // Clean up
            fs::remove_file(&output_path).ok();
        }
    }

    /// Build a minimal WAV in memory, then append a LIST/INFO chunk and optionally a FACT chunk,
    /// then verify that `WavFile::list()` / `WavFile::fact()` parse them correctly.
    fn build_wav_with_chunks(include_fact: bool, info_tags: &[(&[u8; 4], &str)]) -> Vec<u8> {
        // 1. Core WAV: RIFF + fmt  + data (silence, 8-bit PCM mono, 8 kHz)
        let sample_rate: u32 = 8000;
        let num_samples: u32 = 8000; // 1 second of silence
        let data_bytes = vec![128u8; num_samples as usize]; // 8-bit PCM silence

        // FMT chunk (16 bytes)
        let mut fmt_data = Vec::new();
        fmt_data.extend_from_slice(&1u16.to_le_bytes()); // PCM
        fmt_data.extend_from_slice(&1u16.to_le_bytes()); // 1 channel
        fmt_data.extend_from_slice(&sample_rate.to_le_bytes());
        fmt_data.extend_from_slice(&sample_rate.to_le_bytes()); // byte rate
        fmt_data.extend_from_slice(&1u16.to_le_bytes()); // block align
        fmt_data.extend_from_slice(&8u16.to_le_bytes()); // 8 bits

        // Build optional FACT chunk
        let mut fact_chunk = Vec::new();
        if include_fact {
            fact_chunk.extend_from_slice(b"fact");
            fact_chunk.extend_from_slice(&4u32.to_le_bytes());
            fact_chunk.extend_from_slice(&num_samples.to_le_bytes());
        }

        // Build LIST/INFO chunk
        let mut info_subchunks: Vec<u8> = Vec::new();
        for &(id, value) in info_tags {
            let mut payload = value.as_bytes().to_vec();
            payload.push(0); // null terminator
            info_subchunks.extend_from_slice(id);
            info_subchunks.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            info_subchunks.extend_from_slice(&payload);
            if !payload.len().is_multiple_of(2) {
                info_subchunks.push(0); // padding
            }
        }

        let mut list_chunk = Vec::new();
        if !info_tags.is_empty() {
            let list_data_len = 4 + info_subchunks.len(); // "INFO" + subchunks
            list_chunk.extend_from_slice(b"LIST");
            list_chunk.extend_from_slice(&(list_data_len as u32).to_le_bytes());
            list_chunk.extend_from_slice(b"INFO");
            list_chunk.extend_from_slice(&info_subchunks);
        }

        // Assemble the full RIFF file
        let wave_payload_size = 8 + fmt_data.len()  // "fmt " header + data
            + 8 + data_bytes.len()                   // "data" header + samples
            + fact_chunk.len()
            + list_chunk.len()
            + 4; // "WAVE" tag

        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(wave_payload_size as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&(fmt_data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&fmt_data);
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data_bytes);
        wav.extend_from_slice(&fact_chunk);
        wav.extend_from_slice(&list_chunk);
        wav
    }

    #[test]
    fn test_fact_chunk_absent_for_plain_pcm() {
        let path = std::env::temp_dir().join(format!("test_no_fact_{}.wav", std::process::id()));
        let wav_bytes = build_wav_with_chunks(false, &[]);
        std::fs::write(&path, &wav_bytes).expect("write test wav");
        let wav = WavFile::open_with_options(&path, OpenOptions::default()).expect("open test wav");
        assert!(wav.fact().expect("fact() should not error").is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fact_chunk_parsed() {
        let path = std::env::temp_dir().join(format!("test_fact_{}.wav", std::process::id()));
        let wav_bytes = build_wav_with_chunks(true, &[]);
        std::fs::write(&path, &wav_bytes).expect("write test wav");

        let wav = WavFile::open_with_options(&path, OpenOptions::default()).expect("open test wav");
        let fact = wav
            .fact()
            .expect("fact() should not error")
            .expect("FACT chunk should be present");
        assert_eq!(fact.num_samples(), 8000);

        let info = wav.specific_info();
        assert_eq!(info.fact_num_samples, Some(8000));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_list_info_chunk_parsed() {
        let tags: &[(&[u8; 4], &str)] = &[
            (b"INAM", "Test Track"),
            (b"IART", "Test Artist"),
            (b"IPRD", "Test Album"),
        ];
        let path = std::env::temp_dir().join(format!("test_list_info_{}.wav", std::process::id()));
        let wav_bytes = build_wav_with_chunks(false, tags);
        std::fs::write(&path, &wav_bytes).expect("write test wav");

        let wav = WavFile::open_with_options(&path, OpenOptions::default()).expect("open test wav");
        let list = wav
            .list()
            .expect("list() should not error")
            .expect("LIST chunk should be present");
        assert!(list.is_info());
        let meta = list
            .info_metadata()
            .expect("is INFO type")
            .expect("parses without error");
        assert_eq!(meta.title.as_deref(), Some("Test Track"));
        assert_eq!(meta.artist.as_deref(), Some("Test Artist"));
        assert_eq!(meta.album.as_deref(), Some("Test Album"));
        assert!(meta.genre.is_none());

        // Also check specific_info exposes the same data
        let info = wav.specific_info();
        let si_meta = info.info_metadata.expect("info_metadata should be populated");
        assert_eq!(si_meta.title.as_deref(), Some("Test Track"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fact_and_list_together() {
        let tags: &[(&[u8; 4], &str)] = &[(b"IGNR", "Ambient"), (b"ICRD", "2024")];
        let path = std::env::temp_dir().join(format!("test_fact_and_list_{}.wav", std::process::id()));
        let wav_bytes = build_wav_with_chunks(true, tags);
        std::fs::write(&path, &wav_bytes).expect("write test wav");

        let wav = WavFile::open_with_options(&path, OpenOptions::default()).expect("open test wav");
        assert_eq!(
            wav.fact()
                .expect("fact() should not error")
                .expect("FACT chunk should be present")
                .num_samples(),
            8000
        );
        let meta = wav
            .list()
            .expect("list() should not error")
            .expect("LIST chunk should be present")
            .info_metadata()
            .expect("is INFO type")
            .expect("parses without error");
        assert_eq!(meta.genre.as_deref(), Some("Ambient"));
        assert_eq!(meta.date.as_deref(), Some("2024"));

        std::fs::remove_file(&path).ok();
    }
}
