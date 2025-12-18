use crate::{
    error::{AudioIOError, AudioIOResult},
    types::ValidatedSampleType,
};
use audio_samples::{AudioSample, ConvertTo, I24};
use std::{any::TypeId, mem};

#[derive(Debug, Clone)]
pub struct DataChunk<'a> {
    bytes: &'a [u8],
}

impl<'a> AsRef<[u8]> for DataChunk<'a> {
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}

impl<'a> DataChunk<'a> {
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub const fn from_bytes(bytes: &'a [u8]) -> DataChunk<'a> {
        DataChunk { bytes }
    }

    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub(crate) const fn total_samples(&self, sample_type: ValidatedSampleType) -> usize {
        (self.bytes.len() as f64 / sample_type.bytes_per_sample() as f64) as usize
    }

    pub(crate) const fn total_frames(
        &self,
        sample_type: ValidatedSampleType,
        num_channels: usize,
    ) -> usize {
        let total_samples = self.total_samples(sample_type);
        total_samples / num_channels
    }

    /// Parse the data chunk into a vector of samples of type `S`, handling alignment safely.
    ///
    /// # Returns
    ///
    /// Ok(Vec<S>) if successful, or an AudioIOError if the data is corrupted or unsupported.
    ///
    /// *Note:* This function does not handle 24-bit samples; instead this is done before calling this function.
    fn to_sample_vec<S: AudioSample>(&self) -> AudioIOResult<Vec<S>> {
        let sample_size = S::BITS as usize / 8;
        if !self.bytes.len().is_multiple_of(sample_size) {
            return Err(AudioIOError::corrupted_data_simple(
                "Data size not aligned to sample boundaries",
                format!(
                    "Data size {} is not a multiple of sample size {} for type",
                    self.bytes.len(),
                    sample_size
                ),
            ));
        }

        // Fast-path: common 16/32-bit types can be copied with native alignment if available.
        let ptr = self.bytes.as_ptr();
        let aligned = (ptr as usize).is_multiple_of(mem::align_of::<S>());
        if aligned && (S::BITS == 16 || S::BITS == 32) {
            let num_samples = self.bytes.len() / sample_size;
            // Safety: alignment and size multiples already checked.
            let slice = unsafe { core::slice::from_raw_parts(ptr as *const S, num_samples) };
            return Ok(slice.to_vec());
        }

        // Fallback: decode little-endian bytes per sample.
        let mut out = Vec::with_capacity(self.bytes.len() / sample_size);
        match S::BITS {
            16 => {
                for chunk in self.bytes.chunks_exact(2) {
                    let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                    // Safety: i16 and S share size (16 bits enforced above).
                    let s: S = unsafe { mem::transmute_copy(&v) };
                    out.push(s);
                }
            }
            32 => {
                if TypeId::of::<S>() == TypeId::of::<f32>() {
                    for chunk in self.bytes.chunks_exact(4) {
                        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        let s: S = unsafe { mem::transmute_copy(&v) };
                        out.push(s);
                    }
                } else {
                    for chunk in self.bytes.chunks_exact(4) {
                        let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        let s: S = unsafe { mem::transmute_copy(&v) };
                        out.push(s);
                    }
                }
            }
            _ => {
                return Err(AudioIOError::corrupted_data_simple(
                    "Unsupported sample width",
                    format!("{} bits", S::BITS),
                ));
            }
        }

        Ok(out)
    }

    /// A necessary workaround to read 24-bit integer samples, since on disk they are stored as 3 bytes, but in memory they are represented as a 4-byte struct.
    fn read_i24_vec(&self) -> AudioIOResult<Vec<I24>> {
        // bytes multiple of 3 already validated
        let chunks = self.bytes.chunks_exact(3);
        Ok(chunks
            .into_iter()
            .map(|c| I24::from_le_bytes([c[0], c[1], c[2]]))
            .collect())
    }

    /// Read samples from the data chunk, converting from stored type `S` to desired type `T`.
    pub fn read_samples<S: AudioSample + ConvertTo<T>, T: AudioSample>(
        &self,
    ) -> AudioIOResult<Vec<T>>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
    {
        let sample_size = S::BYTES;

        if S::BITS == 24 {
            if !self.bytes.len().is_multiple_of(3) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Data size not aligned to 24-bit samples",
                    format!("Data size {} is not a multiple of 3", self.bytes.len()),
                ));
            }

            return Ok(self
                .read_i24_vec()?
                .into_iter()
                .map(S::convert_from)
                .collect());
        } else if S::BITS == 64 {
            if !self.bytes.len().is_multiple_of(sample_size) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Data size not aligned to 64-bit samples",
                    format!("Data size {} is not a multiple of 8", self.bytes.len()),
                ));
            }

            let chunks = self.bytes.chunks_exact(sample_size);
            return Ok(chunks
                .into_iter()
                .map(|c| {
                    let f: f64 =
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);

                    S::convert_from(f)
                })
                .collect());
        }

        Ok(self
            .to_sample_vec::<S>()?
            .into_iter()
            .map(T::convert_from)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_chunk_rejects_unaligned_i16() {
        let data = [0u8; 1];
        let chunk = DataChunk::from_bytes(&data);
        let err = chunk.read_samples::<i16, i16>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to sample boundaries"));
    }

    #[test]
    fn test_data_chunk_rejects_unaligned_i24() {
        let data = [0u8; 4]; // not a multiple of 3
        let chunk = DataChunk::from_bytes(&data);
        let err = chunk.read_samples::<I24, I24>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to 24-bit samples"));
    }

    #[test]
    fn test_data_chunk_rejects_unaligned_f64() {
        let data = [0u8; 10]; // not a multiple of 8
        let chunk = DataChunk::from_bytes(&data);
        let err = chunk.read_samples::<f64, f64>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to 64-bit samples"));
    }

    #[test]
    fn test_data_chunk_read_i24_with_padding() {
        // Single 24-bit sample (odd frame count is fine as long as size is multiple of 3)
        let bytes = [0x01u8, 0x02, 0x03];
        let chunk = DataChunk::from_bytes(&bytes);
        let samples = chunk
            .read_samples::<I24, I24>()
            .expect("Expected aligned I24 read to succeed");
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0], I24::from_le_bytes([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_data_chunk_read_f64_roundtrip() {
        let values = [1.25f64, -0.5f64];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let chunk = DataChunk::from_bytes(&bytes);
        let samples = chunk
            .read_samples::<f64, f64>()
            .expect("Expected aligned f64 read to succeed");
        assert_eq!(samples, values);
    }
}
