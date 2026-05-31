use core::fmt::{Display, Formatter, Result as FmtResult};

use super::{chunks::ChunkID, error::WavError};

/// The list type for metadata tag subchunks.
pub const INFO_TYPE: ChunkID = ChunkID::new(b"INFO");

/// Metadata extracted from a LIST/INFO chunk.
///
/// All fields are optional because INFO subchunks are individually optional.
/// Unknown subchunks land in `extra`.
#[derive(Debug, Default, Clone)]
pub struct InfoMetadata {
    /// INAM — clip title
    pub title: Option<String>,
    /// IART — artist / performer
    pub artist: Option<String>,
    /// IPRD — album / product name
    pub album: Option<String>,
    /// ICRD — creation date (typically YYYY-MM-DD or YYYY)
    pub date: Option<String>,
    /// ICMT — free-form comment
    pub comment: Option<String>,
    /// IGNR — genre
    pub genre: Option<String>,
    /// ISFT — software that created the file
    pub software: Option<String>,
    /// ICOP — copyright notice
    pub copyright: Option<String>,
    /// IENG — engineer
    pub engineer: Option<String>,
    /// ISBJ — subject
    pub subject: Option<String>,
    /// ISRC — source medium (e.g. "CD")
    pub source: Option<String>,
    /// IKEY — semicolon-separated keywords
    pub keywords: Option<String>,
    /// Any INFO subchunks not recognised above.
    pub extra: Vec<(ChunkID, String)>,
}

impl InfoMetadata {
    fn new() -> Self {
        InfoMetadata {
            extra: Vec::with_capacity(4),
            ..InfoMetadata::default()
        }
    }
}

impl Display for InfoMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        macro_rules! show {
            ($label:expr, $field:expr) => {
                if let Some(ref v) = $field {
                    writeln!(f, "  {}: {}", $label, v)?;
                }
            };
        }
        show!("Title", self.title);
        show!("Artist", self.artist);
        show!("Album", self.album);
        show!("Date", self.date);
        show!("Comment", self.comment);
        show!("Genre", self.genre);
        show!("Software", self.software);
        show!("Copyright", self.copyright);
        show!("Engineer", self.engineer);
        show!("Subject", self.subject);
        show!("Source", self.source);
        show!("Keywords", self.keywords);
        for (id, value) in &self.extra {
            writeln!(f, "  {id}: {value}")?;
        }
        Ok(())
    }
}

/// Append one INFO subchunk (`id` + size + NUL-terminated value, word-aligned) to `body`.
fn append_info_subchunk(body: &mut Vec<u8>, id: &[u8; 4], value: &str) {
    body.extend_from_slice(id);
    let value_bytes = value.as_bytes();
    let size = value_bytes.len() + 1; // include the NUL terminator
    body.extend_from_slice(&(size as u32).to_le_bytes());
    body.extend_from_slice(value_bytes);
    body.push(0); // NUL terminator
    if size % 2 == 1 {
        body.push(0); // word-align the subchunk
    }
}

impl InfoMetadata {
    /// Serialise to a complete `LIST`/`INFO` chunk (chunk header + `INFO` tag + subchunks,
    /// word-aligned). Returns `None` when there is no metadata to write, so callers can skip
    /// emitting an empty chunk. This is the write-side counterpart of [`InfoMetadata::parse`].
    pub fn to_list_chunk(&self) -> Option<Vec<u8>> {
        let mut body = Vec::new();
        body.extend_from_slice(INFO_TYPE.as_bytes());

        // Known tags, in canonical order.
        for (id, field) in [
            (b"INAM", &self.title),
            (b"IART", &self.artist),
            (b"IPRD", &self.album),
            (b"ICRD", &self.date),
            (b"ICMT", &self.comment),
            (b"IGNR", &self.genre),
            (b"ISFT", &self.software),
            (b"ICOP", &self.copyright),
            (b"IENG", &self.engineer),
            (b"ISBJ", &self.subject),
            (b"ISRC", &self.source),
            (b"IKEY", &self.keywords),
        ] {
            if let Some(value) = field {
                append_info_subchunk(&mut body, id, value);
            }
        }
        // Preserve any unrecognised tags captured on read.
        for (id, value) in &self.extra {
            append_info_subchunk(&mut body, id.as_bytes(), value);
        }

        // Only the 4-byte "INFO" tag and no subchunks → nothing worth writing.
        if body.len() <= 4 {
            return None;
        }

        let mut chunk = Vec::with_capacity(8 + body.len());
        chunk.extend_from_slice(b"LIST");
        chunk.extend_from_slice(&(body.len() as u32).to_le_bytes());
        chunk.extend_from_slice(&body);
        // body is always even (each subchunk is word-aligned and "INFO" is 4 bytes), but pad
        // defensively to keep the invariant explicit.
        if body.len() % 2 == 1 {
            chunk.push(0);
        }
        Some(chunk)
    }

    fn parse(bytes: &[u8]) -> Result<Self, WavError> {
        let mut metadata = InfoMetadata::new();
        let mut offset = 0;

        while offset + 8 <= bytes.len() {
            // Loop guard ensures offset+8 <= bytes.len(), so both 4-byte reads are in bounds.
            let id = ChunkID::new(&[bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]);
            let size = u32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as usize;
            let padded = size + (size & 1);
            let data_start = offset + 8;
            let data_end = data_start + size;
            let next_offset = offset + 8 + padded;

            if data_end > bytes.len() {
                return Err(WavError::chunk_parsing(
                    "INFO",
                    offset.to_string(),
                    format!(
                        "subchunk {id} data ({data_end} bytes) extends beyond chunk ({} bytes)",
                        bytes.len()
                    ),
                ));
            }

            let value = decode_info_string(&bytes[data_start..data_end]);

            match id.as_str() {
                Some("INAM") => metadata.title = Some(value),
                Some("IART") => metadata.artist = Some(value),
                Some("IPRD") => metadata.album = Some(value),
                Some("ICRD") => metadata.date = Some(value),
                Some("ICMT") => metadata.comment = Some(value),
                Some("IGNR") => metadata.genre = Some(value),
                Some("ISFT") => metadata.software = Some(value),
                Some("ICOP") => metadata.copyright = Some(value),
                Some("IENG") => metadata.engineer = Some(value),
                Some("ISBJ") => metadata.subject = Some(value),
                Some("ISRC") => metadata.source = Some(value),
                Some("IKEY") => metadata.keywords = Some(value),
                _ => metadata.extra.push((id, value)),
            }

            offset = next_offset;
        }

        Ok(metadata)
    }
}

/// Decode a null-terminated, potentially truncated INFO string.
///
/// INFO strings are nominally null-terminated ASCII, but real files sometimes
/// omit the terminator or use Windows-1252. We truncate at the first NUL and
/// lossily decode whatever remains.
fn decode_info_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_owned()
}

/// Parsed LIST chunk.
///
/// A LIST chunk wraps a 4-byte type tag followed by zero or more sub-chunks.
/// The most common type is `INFO`, which contains human-readable metadata tags.
pub struct ListChunk<'a> {
    list_type: ChunkID,
    /// Bytes following the 4-byte list type tag.
    data: &'a [u8],
}

impl<'a> ListChunk<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, WavError> {
        if bytes.len() < 4 {
            return Err(WavError::chunk_parsing(
                "LIST",
                "0",
                format!("LIST chunk must be at least 4 bytes, got {}", bytes.len()),
            ));
        }
        let list_type = ChunkID::new(&[bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok(ListChunk {
            list_type,
            data: &bytes[4..],
        })
    }

    /// The 4-byte type tag (e.g. `INFO`, `adtl`).
    pub const fn list_type(&self) -> ChunkID {
        self.list_type
    }

    /// True if this is a LIST/INFO chunk containing metadata tags.
    pub fn is_info(&self) -> bool {
        self.list_type == INFO_TYPE
    }

    /// Parse the INFO subchunks into [`InfoMetadata`].
    ///
    /// Returns `None` if the list type is not `INFO`.
    pub fn info_metadata(&self) -> Option<Result<InfoMetadata, WavError>> {
        if !self.is_info() {
            return None;
        }
        Some(InfoMetadata::parse(self.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info_subchunk(id: &[u8; 4], value: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(id);
        let size = value.len() as u32;
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(value);
        if !value.len().is_multiple_of(2) {
            buf.push(0); // padding
        }
        buf
    }

    fn make_list_info_bytes(subchunks: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"INFO");
        buf.extend_from_slice(subchunks);
        buf
    }

    #[test]
    fn test_parse_title_and_artist() {
        let mut subchunks = Vec::new();
        subchunks.extend(make_info_subchunk(b"INAM", b"My Track\0"));
        subchunks.extend(make_info_subchunk(b"IART", b"Some Artist\0"));

        let list_bytes = make_list_info_bytes(&subchunks);
        let chunk = ListChunk::from_bytes(&list_bytes).expect("valid LIST chunk");

        assert!(chunk.is_info());
        let meta = chunk
            .info_metadata()
            .expect("is INFO type")
            .expect("parses without error");

        assert_eq!(meta.title.as_deref(), Some("My Track"));
        assert_eq!(meta.artist.as_deref(), Some("Some Artist"));
        assert!(meta.album.is_none());
    }

    #[test]
    fn test_extra_subchunk_preserved() {
        let mut subchunks = Vec::new();
        subchunks.extend(make_info_subchunk(b"INAM", b"Title\0"));
        subchunks.extend(make_info_subchunk(b"IXXX", b"custom\0"));

        let list_bytes = make_list_info_bytes(&subchunks);
        let chunk = ListChunk::from_bytes(&list_bytes).expect("valid LIST chunk");
        let meta = chunk
            .info_metadata()
            .expect("is INFO type")
            .expect("parses without error");

        assert_eq!(meta.title.as_deref(), Some("Title"));
        assert_eq!(meta.extra.len(), 1);
        assert_eq!(meta.extra[0].0, ChunkID::new(b"IXXX"));
        assert_eq!(meta.extra[0].1, "custom");
    }

    #[test]
    fn test_non_info_list_returns_none() {
        let bytes = b"adtl"; // not INFO
        let chunk = ListChunk::from_bytes(bytes).expect("valid LIST chunk");
        assert!(!chunk.is_info());
        assert!(chunk.info_metadata().is_none());
    }

    #[test]
    fn test_list_chunk_too_small() {
        assert!(ListChunk::from_bytes(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_info_string_strips_null_and_whitespace() {
        assert_eq!(decode_info_string(b"hello \0garbage"), "hello");
        assert_eq!(decode_info_string(b"  trimmed  \0"), "trimmed");
        assert_eq!(decode_info_string(b"no null"), "no null");
        assert_eq!(decode_info_string(b""), "");
    }

    #[test]
    fn test_all_known_fields() {
        let mut subchunks = Vec::new();
        for (id, val) in [
            (b"INAM", b"Title" as &[u8]),
            (b"IART", b"Artist"),
            (b"IPRD", b"Album"),
            (b"ICRD", b"2024"),
            (b"ICMT", b"A comment"),
            (b"IGNR", b"Electronic"),
            (b"ISFT", b"MyDAW"),
            (b"ICOP", b"2024 Author"),
            (b"IENG", b"Bob"),
            (b"ISBJ", b"Sound design"),
            (b"ISRC", b"CD"),
            (b"IKEY", b"ambient; drone"),
        ] {
            let mut v = val.to_vec();
            v.push(0);
            subchunks.extend(make_info_subchunk(id, &v));
        }

        let list_bytes = make_list_info_bytes(&subchunks);
        let chunk = ListChunk::from_bytes(&list_bytes).expect("valid LIST chunk");
        let meta = chunk
            .info_metadata()
            .expect("is INFO type")
            .expect("parses without error");

        assert_eq!(meta.title.as_deref(), Some("Title"));
        assert_eq!(meta.artist.as_deref(), Some("Artist"));
        assert_eq!(meta.album.as_deref(), Some("Album"));
        assert_eq!(meta.date.as_deref(), Some("2024"));
        assert_eq!(meta.comment.as_deref(), Some("A comment"));
        assert_eq!(meta.genre.as_deref(), Some("Electronic"));
        assert_eq!(meta.software.as_deref(), Some("MyDAW"));
        assert_eq!(meta.copyright.as_deref(), Some("2024 Author"));
        assert_eq!(meta.engineer.as_deref(), Some("Bob"));
        assert_eq!(meta.subject.as_deref(), Some("Sound design"));
        assert_eq!(meta.source.as_deref(), Some("CD"));
        assert_eq!(meta.keywords.as_deref(), Some("ambient; drone"));
        assert!(meta.extra.is_empty());
    }
}
