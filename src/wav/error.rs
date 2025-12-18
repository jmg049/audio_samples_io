use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum WavError {
    #[error("WAV chunk parsing error: Chunk '{0}', Position {1}, Details: {2}")]
    ChunkParsingError(String, String, String),
    #[error("Invalid sub-format in WAVE_FORMAT_EXTENSIBLE")]
    InvalidSubFormat,
    #[error("Invalid FMT chunk size: found {0} bytes")]
    InvalidFmtChunkSize(usize),
    #[error("Unsupported sample type")]
    UnsupportedSampleType,
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

impl WavError {
    pub fn chunk_parsing<S1: Into<String>, S2: Into<String>, S3: Into<String>>(
        chunk_id: S1,
        position: S2,
        details: S3,
    ) -> Self {
        WavError::ChunkParsingError(chunk_id.into(), position.into(), details.into())
    }

    pub const fn invalid_subformat() -> Self {
        WavError::InvalidSubFormat
    }

    pub fn invalid_format<S: Into<String>>(message: S) -> Self {
        WavError::InvalidFormat(message.into())
    }
}
