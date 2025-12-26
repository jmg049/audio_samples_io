//! FLAC metadata block parsing and serialization.
//!
//! FLAC files begin with a stream marker followed by one or more metadata blocks.
//! The first metadata block must be STREAMINFO. This module handles parsing and
//! writing all standard metadata block types.

use core::fmt::{Display, Formatter, Result as FmtResult};
use std::collections::HashMap;

use crate::error::{AudioIOError, AudioIOResult, ErrorPosition};
use crate::flac::constants::{MD5_SIZE, STREAMINFO_SIZE};
use crate::flac::error::FlacError;
use crate::traits::AudioInfoMarker;
use crate::types::ValidatedSampleType;

/// Metadata block types defined by the FLAC specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MetadataBlockType {
    /// STREAMINFO: mandatory, must be first
    StreamInfo = 0,
    /// PADDING: placeholder for future use
    Padding = 1,
    /// APPLICATION: third-party application data
    Application = 2,
    /// SEEKTABLE: seek points for fast seeking
    SeekTable = 3,
    /// VORBIS_COMMENT: Vorbis-style comments (tags)
    VorbisComment = 4,
    /// CUESHEET: CD cue sheet information
    CueSheet = 5,
    /// PICTURE: embedded picture (album art, etc.)
    Picture = 6,
    /// Reserved or invalid block type
    Reserved(u8),
}

impl MetadataBlockType {
    /// Parse a block type from its numeric value.
    pub const fn from_byte(value: u8) -> Self {
        match value {
            0 => MetadataBlockType::StreamInfo,
            1 => MetadataBlockType::Padding,
            2 => MetadataBlockType::Application,
            3 => MetadataBlockType::SeekTable,
            4 => MetadataBlockType::VorbisComment,
            5 => MetadataBlockType::CueSheet,
            6 => MetadataBlockType::Picture,
            n => MetadataBlockType::Reserved(n),
        }
    }

    /// Convert to the numeric value.
    pub const fn as_byte(self) -> u8 {
        match self {
            MetadataBlockType::StreamInfo => 0,
            MetadataBlockType::Padding => 1,
            MetadataBlockType::Application => 2,
            MetadataBlockType::SeekTable => 3,
            MetadataBlockType::VorbisComment => 4,
            MetadataBlockType::CueSheet => 5,
            MetadataBlockType::Picture => 6,
            MetadataBlockType::Reserved(n) => n,
        }
    }
}

impl Display for MetadataBlockType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            MetadataBlockType::StreamInfo => write!(f, "STREAMINFO"),
            MetadataBlockType::Padding => write!(f, "PADDING"),
            MetadataBlockType::Application => write!(f, "APPLICATION"),
            MetadataBlockType::SeekTable => write!(f, "SEEKTABLE"),
            MetadataBlockType::VorbisComment => write!(f, "VORBIS_COMMENT"),
            MetadataBlockType::CueSheet => write!(f, "CUESHEET"),
            MetadataBlockType::Picture => write!(f, "PICTURE"),
            MetadataBlockType::Reserved(n) => write!(f, "RESERVED({})", n),
        }
    }
}

/// Header for a metadata block (1 byte type + 3 bytes length).
#[derive(Debug, Clone, Copy)]
pub struct MetadataBlockHeader {
    /// Whether this is the last metadata block
    pub is_last: bool,
    /// Block type
    pub block_type: MetadataBlockType,
    /// Length of block data in bytes (not including header)
    pub length: u32,
}

impl MetadataBlockHeader {
    /// Parse a metadata block header from 4 bytes.
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        let is_last = (bytes[0] & 0x80) != 0;
        let block_type = MetadataBlockType::from_byte(bytes[0] & 0x7F);
        let length = u32::from_be_bytes([0, bytes[1], bytes[2], bytes[3]]);

        MetadataBlockHeader {
            is_last,
            block_type,
            length,
        }
    }

    /// Serialize to 4 bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        let type_byte = self.block_type.as_byte() | if self.is_last { 0x80 } else { 0 };
        let len_bytes = self.length.to_be_bytes();
        [type_byte, len_bytes[1], len_bytes[2], len_bytes[3]]
    }

    /// Total size of header + data
    pub const fn total_size(&self) -> usize {
        4 + self.length as usize
    }
}

/// STREAMINFO metadata block - mandatory, contains essential stream parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
    /// Minimum block size in samples (16-65535)
    pub min_block_size: u16,
    /// Maximum block size in samples (16-65535)
    pub max_block_size: u16,
    /// Minimum frame size in bytes (0 = unknown)
    pub min_frame_size: u32,
    /// Maximum frame size in bytes (0 = unknown)
    pub max_frame_size: u32,
    /// Sample rate in Hz (1-655350)
    pub sample_rate: u32,
    /// Number of channels (1-8)
    pub channels: u8,
    /// Bits per sample (4-32)
    pub bits_per_sample: u8,
    /// Total samples per channel (0 = unknown)
    pub total_samples: u64,
    /// MD5 signature of unencoded audio data
    pub md5_signature: [u8; MD5_SIZE],
}

impl StreamInfo {
    /// Parse STREAMINFO from exactly 34 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FlacError> {
        if bytes.len() != STREAMINFO_SIZE {
            return Err(FlacError::InvalidStreamInfoSize(bytes.len()));
        }

        // Bytes 0-1: minimum block size
        let min_block_size = u16::from_be_bytes([bytes[0], bytes[1]]);

        // Bytes 2-3: maximum block size
        let max_block_size = u16::from_be_bytes([bytes[2], bytes[3]]);

        // Bytes 4-6: minimum frame size (24 bits)
        let min_frame_size = u32::from_be_bytes([0, bytes[4], bytes[5], bytes[6]]);

        // Bytes 7-9: maximum frame size (24 bits)
        let max_frame_size = u32::from_be_bytes([0, bytes[7], bytes[8], bytes[9]]);

        // Bytes 10-13: sample rate (20 bits), channels (3 bits), bits per sample (5 bits)
        // Byte 10: sample rate bits 19-12
        // Byte 11: sample rate bits 11-4
        // Byte 12: sample rate bits 3-0, channels bits 2-0
        // Byte 13: bits per sample bits 4-0, total samples bits 35-32
        let sample_rate =
            ((bytes[10] as u32) << 12) | ((bytes[11] as u32) << 4) | ((bytes[12] as u32) >> 4);

        let channels = ((bytes[12] & 0x0E) >> 1) + 1;

        let bits_per_sample = (((bytes[12] & 0x01) << 4) | ((bytes[13] & 0xF0) >> 4)) + 1;

        // Bytes 13-17: total samples (36 bits, upper 4 bits in byte 13)
        let total_samples = ((bytes[13] as u64 & 0x0F) << 32)
            | ((bytes[14] as u64) << 24)
            | ((bytes[15] as u64) << 16)
            | ((bytes[16] as u64) << 8)
            | (bytes[17] as u64);

        // Bytes 18-33: MD5 signature
        let mut md5_signature = [0u8; MD5_SIZE];
        md5_signature.copy_from_slice(&bytes[18..34]);

        Ok(StreamInfo {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5_signature,
        })
    }

    /// Serialize to 34 bytes.
    pub fn to_bytes(&self) -> [u8; STREAMINFO_SIZE] {
        let mut bytes = [0u8; STREAMINFO_SIZE];

        // Bytes 0-1: minimum block size
        bytes[0..2].copy_from_slice(&self.min_block_size.to_be_bytes());

        // Bytes 2-3: maximum block size
        bytes[2..4].copy_from_slice(&self.max_block_size.to_be_bytes());

        // Bytes 4-6: minimum frame size (24 bits)
        let min_frame = self.min_frame_size.to_be_bytes();
        bytes[4..7].copy_from_slice(&min_frame[1..4]);

        // Bytes 7-9: maximum frame size (24 bits)
        let max_frame = self.max_frame_size.to_be_bytes();
        bytes[7..10].copy_from_slice(&max_frame[1..4]);

        // Bytes 10-13: sample rate (20), channels (3), bits (5), total samples upper (4)
        let channels_minus_1 = self.channels - 1;
        let bits_minus_1 = self.bits_per_sample - 1;

        bytes[10] = (self.sample_rate >> 12) as u8;
        bytes[11] = (self.sample_rate >> 4) as u8;
        bytes[12] = ((self.sample_rate & 0x0F) << 4) as u8
            | ((channels_minus_1 & 0x07) << 1)
            | ((bits_minus_1 >> 4) & 0x01);
        bytes[13] = ((bits_minus_1 & 0x0F) << 4) | ((self.total_samples >> 32) as u8 & 0x0F);

        // Bytes 14-17: total samples lower 32 bits
        let total_lower = (self.total_samples & 0xFFFFFFFF) as u32;
        bytes[14..18].copy_from_slice(&total_lower.to_be_bytes());

        // Bytes 18-33: MD5 signature
        bytes[18..34].copy_from_slice(&self.md5_signature);

        bytes
    }

    /// Get the validated sample type for this stream.
    pub fn sample_type(&self) -> AudioIOResult<ValidatedSampleType> {
        match self.bits_per_sample {
            1..=16 => Ok(ValidatedSampleType::I16),
            17..=24 => Ok(ValidatedSampleType::I24),
            25..=32 => Ok(ValidatedSampleType::I32),
            _ => Err(AudioIOError::corrupted_data_simple(
                "Invalid bits per sample",
                format!("{} bits", self.bits_per_sample),
            )),
        }
    }

    /// Calculate block align (bytes per frame, all channels)
    pub const fn block_align(&self) -> u16 {
        let bytes_per_sample = (self.bits_per_sample as u16 + 7) / 8;
        bytes_per_sample * self.channels as u16
    }

    /// Calculate byte rate (for uncompressed, used in base info)
    pub const fn byte_rate(&self) -> u32 {
        self.block_align() as u32 * self.sample_rate
    }

    /// Total samples across all channels
    pub const fn total_samples_all_channels(&self) -> u64 {
        self.total_samples * self.channels as u64
    }

    /// Check if MD5 is present (not all zeros)
    pub fn has_md5(&self) -> bool {
        self.md5_signature.iter().any(|&b| b != 0)
    }
}

impl Display for StreamInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "STREAMINFO:")?;
        writeln!(
            f,
            "  Block size: {}-{} samples",
            self.min_block_size, self.max_block_size
        )?;
        writeln!(
            f,
            "  Frame size: {}-{} bytes",
            self.min_frame_size, self.max_frame_size
        )?;
        writeln!(f, "  Sample rate: {} Hz", self.sample_rate)?;
        writeln!(f, "  Channels: {}", self.channels)?;
        writeln!(f, "  Bits per sample: {}", self.bits_per_sample)?;
        writeln!(f, "  Total samples: {}", self.total_samples)?;
        write!(f, "  MD5: ")?;
        for b in &self.md5_signature {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

/// A single seek point in a SEEKTABLE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekPoint {
    /// Sample number of target frame (first sample in frame)
    pub sample_number: u64,
    /// Byte offset from first frame to target frame
    pub stream_offset: u64,
    /// Number of samples in target frame
    pub frame_samples: u16,
}

impl SeekPoint {
    /// Size of a seek point in bytes
    pub const SIZE: usize = 18;

    /// Placeholder value for empty seek points
    pub const PLACEHOLDER_SAMPLE: u64 = 0xFFFFFFFFFFFFFFFF;

    /// Parse a seek point from 18 bytes.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        let sample_number = u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let stream_offset = u64::from_be_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let frame_samples = u16::from_be_bytes([bytes[16], bytes[17]]);

        SeekPoint {
            sample_number,
            stream_offset,
            frame_samples,
        }
    }

    /// Serialize to 18 bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..8].copy_from_slice(&self.sample_number.to_be_bytes());
        bytes[8..16].copy_from_slice(&self.stream_offset.to_be_bytes());
        bytes[16..18].copy_from_slice(&self.frame_samples.to_be_bytes());
        bytes
    }

    /// Check if this is a placeholder seek point.
    pub const fn is_placeholder(&self) -> bool {
        self.sample_number == Self::PLACEHOLDER_SAMPLE
    }
}

/// SEEKTABLE metadata block.
#[derive(Debug, Clone, Default)]
pub struct SeekTable {
    pub points: Vec<SeekPoint>,
}

impl SeekTable {
    /// Parse a seek table from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FlacError> {
        if bytes.len() % SeekPoint::SIZE != 0 {
            return Err(FlacError::InvalidMetadataBlockSize {
                size: bytes.len() as u32,
            });
        }

        let num_points = bytes.len() / SeekPoint::SIZE;
        let mut points = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let start = i * SeekPoint::SIZE;
            let point_bytes: &[u8; SeekPoint::SIZE] = bytes[start..start + SeekPoint::SIZE]
                .try_into()
                .map_err(|_| FlacError::UnexpectedEof)?;
            points.push(SeekPoint::from_bytes(point_bytes));
        }

        Ok(SeekTable { points })
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.points.len() * SeekPoint::SIZE);
        for point in &self.points {
            bytes.extend_from_slice(&point.to_bytes());
        }
        bytes
    }

    /// Find the seek point for a given sample.
    pub fn find_seek_point(&self, sample: u64) -> Option<&SeekPoint> {
        // Find the largest seek point <= sample
        self.points
            .iter()
            .filter(|p| !p.is_placeholder() && p.sample_number <= sample)
            .max_by_key(|p| p.sample_number)
    }

    /// Validate that seek points are sorted and within bounds.
    pub fn validate(&self, total_samples: u64) -> Result<(), FlacError> {
        let mut prev_sample = None;
        for point in &self.points {
            if point.is_placeholder() {
                continue;
            }
            if let Some(prev) = prev_sample {
                if point.sample_number <= prev {
                    return Err(FlacError::SeekTableNotSorted);
                }
            }
            if point.sample_number > total_samples {
                return Err(FlacError::InvalidSeekPoint {
                    sample: point.sample_number,
                    total: total_samples,
                });
            }
            prev_sample = Some(point.sample_number);
        }
        Ok(())
    }
}

/// VORBIS_COMMENT metadata block (tags).
#[derive(Debug, Clone, Default)]
pub struct VorbisComment {
    /// Vendor string (encoder identification)
    pub vendor: String,
    /// Comments as key-value pairs (keys are uppercase by convention)
    pub comments: HashMap<String, Vec<String>>,
}

impl VorbisComment {
    /// Parse a Vorbis comment block from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FlacError> {
        if bytes.len() < 8 {
            return Err(FlacError::UnexpectedEof);
        }

        let mut pos = 0;

        // Vendor string length (little-endian!)
        let vendor_len =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        if pos + vendor_len > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }

        let vendor = String::from_utf8_lossy(&bytes[pos..pos + vendor_len]).to_string();
        pos += vendor_len;

        if pos + 4 > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }

        // Number of comments
        let num_comments =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        let mut comments: HashMap<String, Vec<String>> = HashMap::new();

        for _ in 0..num_comments {
            if pos + 4 > bytes.len() {
                return Err(FlacError::UnexpectedEof);
            }

            let comment_len =
                u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            pos += 4;

            if pos + comment_len > bytes.len() {
                return Err(FlacError::UnexpectedEof);
            }

            let comment = String::from_utf8_lossy(&bytes[pos..pos + comment_len]);
            pos += comment_len;

            // Split on first '='
            if let Some(eq_pos) = comment.find('=') {
                let key = comment[..eq_pos].to_uppercase();
                let value = comment[eq_pos + 1..].to_string();
                comments.entry(key).or_default().push(value);
            }
        }

        Ok(VorbisComment { vendor, comments })
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Vendor string
        let vendor_bytes = self.vendor.as_bytes();
        bytes.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(vendor_bytes);

        // Count total comments
        let total_comments: usize = self.comments.values().map(|v| v.len()).sum();
        bytes.extend_from_slice(&(total_comments as u32).to_le_bytes());

        // Each comment
        for (key, values) in &self.comments {
            for value in values {
                let comment = format!("{}={}", key, value);
                let comment_bytes = comment.as_bytes();
                bytes.extend_from_slice(&(comment_bytes.len() as u32).to_le_bytes());
                bytes.extend_from_slice(comment_bytes);
            }
        }

        bytes
    }

    /// Get a tag value (first value if multiple).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.comments
            .get(&key.to_uppercase())
            .and_then(|v| v.first())
            .map(|s| s.as_str())
    }

    /// Get all values for a tag.
    pub fn get_all(&self, key: &str) -> Option<&Vec<String>> {
        self.comments.get(&key.to_uppercase())
    }

    /// Set a tag value (replaces existing).
    pub fn set(&mut self, key: &str, value: impl Into<String>) {
        self.comments.insert(key.to_uppercase(), vec![value.into()]);
    }

    /// Add a tag value (appends to existing).
    pub fn add(&mut self, key: &str, value: impl Into<String>) {
        self.comments
            .entry(key.to_uppercase())
            .or_default()
            .push(value.into());
    }
}

impl Display for VorbisComment {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "VORBIS_COMMENT:")?;
        writeln!(f, "  Vendor: {}", self.vendor)?;
        for (key, values) in &self.comments {
            for value in values {
                writeln!(f, "  {}={}", key, value)?;
            }
        }
        Ok(())
    }
}

/// Picture type for PICTURE metadata blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PictureType {
    Other = 0,
    FileIcon = 1,
    OtherFileIcon = 2,
    FrontCover = 3,
    BackCover = 4,
    LeafletPage = 5,
    Media = 6,
    LeadArtist = 7,
    Artist = 8,
    Conductor = 9,
    Band = 10,
    Composer = 11,
    Lyricist = 12,
    RecordingLocation = 13,
    DuringRecording = 14,
    DuringPerformance = 15,
    ScreenCapture = 16,
    BrightFish = 17,
    Illustration = 18,
    BandLogo = 19,
    PublisherLogo = 20,
}

impl PictureType {
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0 => PictureType::Other,
            1 => PictureType::FileIcon,
            2 => PictureType::OtherFileIcon,
            3 => PictureType::FrontCover,
            4 => PictureType::BackCover,
            5 => PictureType::LeafletPage,
            6 => PictureType::Media,
            7 => PictureType::LeadArtist,
            8 => PictureType::Artist,
            9 => PictureType::Conductor,
            10 => PictureType::Band,
            11 => PictureType::Composer,
            12 => PictureType::Lyricist,
            13 => PictureType::RecordingLocation,
            14 => PictureType::DuringRecording,
            15 => PictureType::DuringPerformance,
            16 => PictureType::ScreenCapture,
            17 => PictureType::BrightFish,
            18 => PictureType::Illustration,
            19 => PictureType::BandLogo,
            20 => PictureType::PublisherLogo,
            _ => PictureType::Other,
        }
    }
}

/// PICTURE metadata block.
#[derive(Debug, Clone)]
pub struct Picture {
    /// Picture type
    pub picture_type: PictureType,
    /// MIME type string
    pub mime_type: String,
    /// Description
    pub description: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Color depth in bits per pixel
    pub color_depth: u32,
    /// Number of colors (for indexed images, 0 otherwise)
    pub num_colors: u32,
    /// Picture data
    pub data: Vec<u8>,
}

impl Picture {
    /// Parse a picture block from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FlacError> {
        if bytes.len() < 32 {
            return Err(FlacError::UnexpectedEof);
        }

        let mut pos = 0;

        let picture_type = PictureType::from_u32(u32::from_be_bytes([
            bytes[pos],
            bytes[pos + 1],
            bytes[pos + 2],
            bytes[pos + 3],
        ]));
        pos += 4;

        let mime_len =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        if pos + mime_len > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }
        let mime_type = String::from_utf8_lossy(&bytes[pos..pos + mime_len]).to_string();
        pos += mime_len;

        if pos + 4 > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }
        let desc_len =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        if pos + desc_len > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }
        let description = String::from_utf8_lossy(&bytes[pos..pos + desc_len]).to_string();
        pos += desc_len;

        if pos + 20 > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }

        let width =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        let height =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        let color_depth =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        let num_colors =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        pos += 4;

        let data_len =
            u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        pos += 4;

        if pos + data_len > bytes.len() {
            return Err(FlacError::UnexpectedEof);
        }
        let data = bytes[pos..pos + data_len].to_vec();

        Ok(Picture {
            picture_type,
            mime_type,
            description,
            width,
            height,
            color_depth,
            num_colors,
            data,
        })
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&(self.picture_type as u32).to_be_bytes());

        let mime_bytes = self.mime_type.as_bytes();
        bytes.extend_from_slice(&(mime_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(mime_bytes);

        let desc_bytes = self.description.as_bytes();
        bytes.extend_from_slice(&(desc_bytes.len() as u32).to_be_bytes());
        bytes.extend_from_slice(desc_bytes);

        bytes.extend_from_slice(&self.width.to_be_bytes());
        bytes.extend_from_slice(&self.height.to_be_bytes());
        bytes.extend_from_slice(&self.color_depth.to_be_bytes());
        bytes.extend_from_slice(&self.num_colors.to_be_bytes());

        bytes.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&self.data);

        bytes
    }
}

/// A parsed metadata block.
#[derive(Debug, Clone)]
pub enum MetadataBlock {
    StreamInfo(StreamInfo),
    Padding(usize),
    Application { id: [u8; 4], data: Vec<u8> },
    SeekTable(SeekTable),
    VorbisComment(VorbisComment),
    CueSheet(Vec<u8>), // Raw bytes for now
    Picture(Picture),
    Unknown { block_type: u8, data: Vec<u8> },
}

impl MetadataBlock {
    /// Parse a metadata block from its header and data.
    pub fn parse(header: &MetadataBlockHeader, data: &[u8]) -> Result<Self, FlacError> {
        match header.block_type {
            MetadataBlockType::StreamInfo => {
                Ok(MetadataBlock::StreamInfo(StreamInfo::from_bytes(data)?))
            }
            MetadataBlockType::Padding => Ok(MetadataBlock::Padding(data.len())),
            MetadataBlockType::Application => {
                if data.len() < 4 {
                    return Err(FlacError::UnexpectedEof);
                }
                let mut id = [0u8; 4];
                id.copy_from_slice(&data[0..4]);
                Ok(MetadataBlock::Application {
                    id,
                    data: data[4..].to_vec(),
                })
            }
            MetadataBlockType::SeekTable => {
                Ok(MetadataBlock::SeekTable(SeekTable::from_bytes(data)?))
            }
            MetadataBlockType::VorbisComment => Ok(MetadataBlock::VorbisComment(
                VorbisComment::from_bytes(data)?,
            )),
            MetadataBlockType::CueSheet => Ok(MetadataBlock::CueSheet(data.to_vec())),
            MetadataBlockType::Picture => Ok(MetadataBlock::Picture(Picture::from_bytes(data)?)),
            MetadataBlockType::Reserved(n) => Ok(MetadataBlock::Unknown {
                block_type: n,
                data: data.to_vec(),
            }),
        }
    }

    /// Get the block type.
    pub fn block_type(&self) -> MetadataBlockType {
        match self {
            MetadataBlock::StreamInfo(_) => MetadataBlockType::StreamInfo,
            MetadataBlock::Padding(_) => MetadataBlockType::Padding,
            MetadataBlock::Application { .. } => MetadataBlockType::Application,
            MetadataBlock::SeekTable(_) => MetadataBlockType::SeekTable,
            MetadataBlock::VorbisComment(_) => MetadataBlockType::VorbisComment,
            MetadataBlock::CueSheet(_) => MetadataBlockType::CueSheet,
            MetadataBlock::Picture(_) => MetadataBlockType::Picture,
            MetadataBlock::Unknown { block_type, .. } => MetadataBlockType::Reserved(*block_type),
        }
    }

    /// Serialize the block data (without header).
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            MetadataBlock::StreamInfo(info) => info.to_bytes().to_vec(),
            MetadataBlock::Padding(size) => vec![0u8; *size],
            MetadataBlock::Application { id, data } => {
                let mut bytes = id.to_vec();
                bytes.extend_from_slice(data);
                bytes
            }
            MetadataBlock::SeekTable(table) => table.to_bytes(),
            MetadataBlock::VorbisComment(comment) => comment.to_bytes(),
            MetadataBlock::CueSheet(data) => data.clone(),
            MetadataBlock::Picture(picture) => picture.to_bytes(),
            MetadataBlock::Unknown { data, .. } => data.clone(),
        }
    }
}

/// Parsed metadata from a FLAC file.
#[derive(Debug, Clone)]
pub struct FlacMetadata {
    /// STREAMINFO block (always present)
    pub stream_info: StreamInfo,
    /// All metadata blocks in order
    pub blocks: Vec<MetadataBlock>,
    /// Byte offset where audio frames begin
    pub audio_offset: usize,
}

impl FlacMetadata {
    /// Parse all metadata blocks from FLAC data.
    ///
    /// Assumes the "fLaC" marker has already been validated and
    /// the data starts at the first metadata block header.
    pub fn parse(data: &[u8], start_offset: usize) -> AudioIOResult<Self> {
        let mut offset = start_offset;
        let mut blocks = Vec::new();
        let mut stream_info: Option<StreamInfo> = None;

        loop {
            if offset + 4 > data.len() {
                return Err(AudioIOError::corrupted_data(
                    "Unexpected end of metadata",
                    "Not enough bytes for metadata block header",
                    ErrorPosition::new(offset),
                ));
            }

            let header_bytes: [u8; 4] = data[offset..offset + 4]
                .try_into()
                .map_err(|_| FlacError::UnexpectedEof)?;
            let header = MetadataBlockHeader::from_bytes(&header_bytes);
            offset += 4;

            if header.length > 16 * 1024 * 1024 {
                return Err(AudioIOError::from(FlacError::InvalidMetadataBlockSize {
                    size: header.length,
                }));
            }

            if offset + header.length as usize > data.len() {
                return Err(AudioIOError::corrupted_data(
                    "Metadata block extends beyond file",
                    format!(
                        "Block {} at offset {}, length {} exceeds file size {}",
                        header.block_type,
                        offset - 4,
                        header.length,
                        data.len()
                    ),
                    ErrorPosition::new(offset - 4),
                ));
            }

            let block_data = &data[offset..offset + header.length as usize];
            let block = MetadataBlock::parse(&header, block_data).map_err(AudioIOError::from)?;

            if let MetadataBlock::StreamInfo(ref info) = block {
                stream_info = Some(*info);
            }

            blocks.push(block);
            offset += header.length as usize;

            if header.is_last {
                break;
            }
        }

        let stream_info =
            stream_info.ok_or_else(|| AudioIOError::from(FlacError::MissingStreamInfo))?;

        Ok(FlacMetadata {
            stream_info,
            blocks,
            audio_offset: offset,
        })
    }

    /// Get the seek table if present.
    pub fn seek_table(&self) -> Option<&SeekTable> {
        self.blocks.iter().find_map(|b| {
            if let MetadataBlock::SeekTable(table) = b {
                Some(table)
            } else {
                None
            }
        })
    }

    /// Get the Vorbis comment if present.
    pub fn vorbis_comment(&self) -> Option<&VorbisComment> {
        self.blocks.iter().find_map(|b| {
            if let MetadataBlock::VorbisComment(comment) = b {
                Some(comment)
            } else {
                None
            }
        })
    }

    /// Get all pictures.
    pub fn pictures(&self) -> Vec<&Picture> {
        self.blocks
            .iter()
            .filter_map(|b| {
                if let MetadataBlock::Picture(pic) = b {
                    Some(pic)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl AudioInfoMarker for FlacMetadata {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaminfo_roundtrip() {
        let info = StreamInfo {
            min_block_size: 4096,
            max_block_size: 4096,
            min_frame_size: 1234,
            max_frame_size: 5678,
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 441000,
            md5_signature: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        };

        let bytes = info.to_bytes();
        assert_eq!(bytes.len(), STREAMINFO_SIZE);

        let parsed = StreamInfo::from_bytes(&bytes).expect("Parse failed");
        assert_eq!(parsed.min_block_size, info.min_block_size);
        assert_eq!(parsed.max_block_size, info.max_block_size);
        assert_eq!(parsed.min_frame_size, info.min_frame_size);
        assert_eq!(parsed.max_frame_size, info.max_frame_size);
        assert_eq!(parsed.sample_rate, info.sample_rate);
        assert_eq!(parsed.channels, info.channels);
        assert_eq!(parsed.bits_per_sample, info.bits_per_sample);
        assert_eq!(parsed.total_samples, info.total_samples);
        assert_eq!(parsed.md5_signature, info.md5_signature);
    }

    #[test]
    fn test_seekpoint_roundtrip() {
        let point = SeekPoint {
            sample_number: 0x123456789ABCDEF0,
            stream_offset: 0xFEDCBA9876543210,
            frame_samples: 4096,
        };

        let bytes = point.to_bytes();
        let parsed = SeekPoint::from_bytes(&bytes);

        assert_eq!(parsed.sample_number, point.sample_number);
        assert_eq!(parsed.stream_offset, point.stream_offset);
        assert_eq!(parsed.frame_samples, point.frame_samples);
    }

    #[test]
    fn test_vorbis_comment_roundtrip() {
        let mut comment = VorbisComment {
            vendor: "audio_samples_io/0.1.0".to_string(),
            comments: HashMap::new(),
        };
        comment.set("TITLE", "Test Song");
        comment.set("ARTIST", "Test Artist");
        comment.add("GENRE", "Electronic");
        comment.add("GENRE", "Ambient");

        let bytes = comment.to_bytes();
        let parsed = VorbisComment::from_bytes(&bytes).expect("Parse failed");

        assert_eq!(parsed.vendor, comment.vendor);
        assert_eq!(parsed.get("TITLE"), Some("Test Song"));
        assert_eq!(parsed.get("ARTIST"), Some("Test Artist"));
        assert_eq!(parsed.get_all("GENRE").map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_metadata_block_header() {
        let header = MetadataBlockHeader {
            is_last: true,
            block_type: MetadataBlockType::StreamInfo,
            length: 34,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes[0], 0x80); // is_last + STREAMINFO
        assert_eq!(bytes[1], 0);
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 34);

        let parsed = MetadataBlockHeader::from_bytes(&bytes);
        assert!(parsed.is_last);
        assert_eq!(parsed.block_type, MetadataBlockType::StreamInfo);
        assert_eq!(parsed.length, 34);
    }
}
