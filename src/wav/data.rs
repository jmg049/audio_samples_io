use crate::{
    error::{AudioIOError, AudioIOResult},
    types::ValidatedSampleType,
};
use audio_samples::{
    I24,
    traits::{ConvertFrom, StandardSample},
};
use non_empty_iter::{IntoNonEmptyIterator, NonEmptyIterator};
use non_empty_slice::NonEmptyVec;
use std::{any::TypeId, mem, num::NonZeroU32};

#[derive(Debug, Clone)]
pub struct DataChunk<'a> {
    bytes: &'a [u8], // Raw audio data bytes can be empty
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
        let sample_size = sample_type.bytes_per_sample().get();
        self.bytes.len() / sample_size
    }

    pub(crate) const fn total_frames(
        &self,
        sample_type: ValidatedSampleType,
        num_channels: NonZeroU32,
    ) -> usize {
        let total_samples = self.total_samples(sample_type);
        let num_channels = num_channels.get() as usize;
        total_samples / num_channels
    }

    /// Parse the data chunk into a vector of samples of type `S`, handling alignment safely.
    ///
    /// # Returns
    ///
    /// Ok(Vec<S>) if successful, or an AudioIOError if the data is corrupted or unsupported.
    ///
    /// *Note:* This function does not handle 24-bit samples; instead this is done before calling this function.
    fn to_sample_vec<S>(&self) -> AudioIOResult<NonEmptyVec<S>>
    where
        S: StandardSample,
    {
        let sample_size = S::BYTES;
        if !(self.bytes.len()).is_multiple_of(sample_size as usize) {
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
            let num_samples = self.bytes.len() / sample_size as usize;
            // Safety: alignment and size multiples already checked.
            let slice = unsafe { core::slice::from_raw_parts(ptr as *const S, num_samples) };
            // SAFETY: non-empty by design since data chunk must contain at least one sample to be valid
            return Ok(unsafe { NonEmptyVec::new_unchecked(slice.to_vec()) });
        }

        // Fallback: decode little-endian bytes per sample.
        let mut out = Vec::with_capacity(self.bytes.len() / sample_size as usize);
        match S::BITS {
            8 => {
                for &b in self.bytes {
                    let v = b as i8;
                    // Safety: i8 and S share size (8 bits enforced above).
                    let s: S = unsafe { mem::transmute_copy(&v) };
                    out.push(s);
                }
            }
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
        // safety: non-empty by design since data chunk must contain at least one sample to be valid
        let out = unsafe { NonEmptyVec::new_unchecked(out) };
        Ok(out)
    }

    /// A necessary workaround to read 24-bit integer samples, since on disk they are stored as 3 bytes, but in memory they are represented as a 4-byte struct.
    fn read_i24_vec(&self) -> NonEmptyVec<I24> {
        // bytes multiple of 3 already validated
        let chunks = self.bytes.chunks_exact(3);
        let v = chunks
            .into_iter()
            .map(|c| I24::from_le_bytes([c[0], c[1], c[2]]))
            .collect();
        // SAFETY: non-empty by design since data chunk must contain at least one sample to be valid
        unsafe { NonEmptyVec::new_unchecked(v) }
    }

    /// Read samples from the data chunk, converting from stored type `S` to desired type `T`.
    pub fn read_samples<S, T>(&self) -> AudioIOResult<NonEmptyVec<T>>
    where
        S: StandardSample + 'static,
        T: StandardSample + ConvertFrom<S> + 'static,
    {
        if self.bytes.is_empty() {
            return Err(AudioIOError::corrupted_data_simple(
                "Tried to read samples from empty data chunk",
                "data chunk is empty",
            ));
        }
        let sample_size = S::BYTES as usize;

        if S::BITS == 24 {
            if !self.bytes.len().is_multiple_of(3) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Data size not aligned to 24-bit samples",
                    format!("Data size {} is not a multiple of 3", self.bytes.len()),
                ));
            }

            return Ok(self
                .read_i24_vec()
                .into_non_empty_iter()
                .map(T::convert_from)
                .collect_non_empty());
        } else if S::BITS == 64 {
            if !self.bytes.len().is_multiple_of(sample_size as usize) {
                return Err(AudioIOError::corrupted_data_simple(
                    "Data size not aligned to 64-bit samples",
                    format!("Data size {} is not a multiple of 8", self.bytes.len()),
                ));
            }

            let chunks = self.bytes.chunks_exact(sample_size as usize);
            let v = chunks
                .into_iter()
                .map(|c| {
                    let f: f64 =
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);

                    T::convert_from(f)
                })
                .collect();
            // safety: non-empty by design since data chunk must contain at least one sample to be valid
            return Ok(unsafe { NonEmptyVec::new_unchecked(v) });
        }

        // Fast path: when S == T the conversion is an identity — skip the collect entirely.
        if TypeId::of::<S>() == TypeId::of::<T>() {
            let samples = self.to_sample_vec::<S>()?;
            // Safety: TypeId equality guarantees S and T are the same concrete type with
            // identical in-memory layout.  NonEmptyVec<S> and NonEmptyVec<T> are both
            // (ptr, len, cap) wrappers so their sizes always match — transmute is sound.
            return Ok(unsafe { mem::transmute::<NonEmptyVec<S>, NonEmptyVec<T>>(samples) });
        }

        Ok(self
            .to_sample_vec::<S>()?
            .into_non_empty_iter()
            .map(T::convert_from)
            .collect_non_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_chunk_rejects_unaligned_i16() {
        let data = [0u8; 1];
        let data_slice = non_empty_slice::non_empty_slice!(&data);
        let chunk = DataChunk::from_bytes(data_slice);
        let err = chunk
            .read_samples::<i16, i16>()
            .expect_err("Expected unaligned read to fail");
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to sample boundaries"));
    }

    #[test]
    fn test_data_chunk_rejects_unaligned_i24() {
        let data = [0u8; 4]; // not a multiple of 3
        let data_slice = non_empty_slice::non_empty_slice!(&data);
        let chunk = DataChunk::from_bytes(data_slice);
        let err = chunk
            .read_samples::<I24, I24>()
            .expect_err("Expected unaligned read to fail");
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to 24-bit samples"));
    }

    #[test]
    fn test_data_chunk_rejects_unaligned_f64() {
        let data = [0u8; 10]; // not a multiple of 8
        let data_slice = non_empty_slice::non_empty_slice!(&data);
        let chunk = DataChunk::from_bytes(data_slice);
        let err = chunk
            .read_samples::<f64, f64>()
            .expect_err("Expected unaligned read to fail");
        let msg = err.to_string();
        assert!(msg.contains("Data size not aligned to 64-bit samples"));
    }

    #[test]
    fn test_data_chunk_read_i24_with_padding() {
        // Single 24-bit sample (odd frame count is fine as long as size is multiple of 3)
        let bytes = [0x01u8, 0x02, 0x03];
        let data_slice = non_empty_slice::non_empty_slice!(&bytes);
        let chunk = DataChunk::from_bytes(data_slice);
        let samples = chunk
            .read_samples::<I24, I24>()
            .expect("Expected aligned I24 read to succeed");
        assert_eq!(samples.len().get(), 1);
        assert_eq!(samples[0], I24::from_le_bytes([0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_data_chunk_read_f64_roundtrip() {
        let values = [1.25f64, -0.5f64];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let data_slice = non_empty_slice::non_empty_slice!(&bytes);
        let chunk = DataChunk::from_bytes(data_slice);
        let samples = chunk
            .read_samples::<f64, f64>()
            .expect("Expected aligned f64 read to succeed");
        let values = NonEmptyVec::new(values.to_vec()).unwrap();
        assert_eq!(samples, values);
    }
}
