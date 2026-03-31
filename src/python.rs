//! Python NumPy integration for optimized audio file reading.
//! This module provides optimized read functions that create NumPy arrays
//! directly from WAV data using Fortran (column-major) layout to eliminate
//! deinterleaving overhead.

use audio_samples::traits::StandardSample;
use non_empty_slice::NonEmptyVec;
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::any::TypeId;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::traits::{AudioFile, AudioFileMetadata};
use crate::types::{OpenOptions, ValidatedSampleType};
use crate::wav::{WavFile, wav_file::parse_wav_header_streaming};
use crate::{AudioIOError, AudioIOResult, BaseAudioInfo, FileType};
use audio_samples::I24;

/// Read audio file directly into NumPy array.
///
/// For multichannel audio, creates Fortran-layout (column-major) array that matches
/// WAV interleaved format, eliminating deinterleaving overhead.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `fp` - Path to audio file
///
/// # Returns
///
/// `Ok((PyArray2<T>, BaseAudioInfo))` where the PyArray has:
/// - Shape: `(channels, frames)` for multi-channel, `(1, frames)` for mono
/// - Layout: Fortran (column-major) for multi-channel
/// - Data: Samples converted to type `T`
pub fn read_pyarray<P, T>(py: Python<'_>, fp: P) -> PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>
where
    P: AsRef<Path>,
    T: StandardSample + Element + 'static,
{
    let path = fp.as_ref();

    // Read file without GIL (expensive I/O operation)
    // Opens file once and extracts both data and metadata
    let (interleaved_vec, info) = py
        .detach(|| read_interleaved_with_info::<_, T>(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Create PyArray with optimal layout (requires GIL, but fast ~2ms)
    let pyarray = create_pyarray_fortran(
        py,
        interleaved_vec,
        info.channels as usize,
        info.total_samples,
    )?;

    Ok((pyarray, info))
}

/// Read audio file into interleaved Vec with metadata,
fn read_interleaved_with_info<P, T>(fp: P) -> AudioIOResult<(NonEmptyVec<T>, BaseAudioInfo)>
where
    P: AsRef<Path>,
    T: StandardSample + 'static,
{
    let path = fp.as_ref();

    match FileType::from_path(path) {
        FileType::WAV => {
            // Fast path: T is a plain numeric type (not I24), little-endian platform.
            // Parse only the WAV header, then read_exact directly into a Vec<T> —
            // one syscall for the header (BufReader), one for the payload.  Avoids
            // the mmap + intermediate-Vec copy that the mmap path requires.
            #[cfg(target_endian = "little")]
            if TypeId::of::<T>() != TypeId::of::<I24>() {
                if let Ok(result) = read_wav_direct::<T>(path) {
                    return Ok(result);
                }
                // Any error falls through to the mmap path below.
            }

            // Mmap path: general fallback (type conversion, I24, big-endian, etc.)
            let wav_file = WavFile::open_with_options(path, OpenOptions::default())?;
            let info = wav_file.base_info()?;
            let data_chunk = wav_file.data();
            let sample_type = wav_file.sample_type();

            let interleaved_vec = match sample_type {
                ValidatedSampleType::U8 => data_chunk.read_samples::<u8, T>(),
                ValidatedSampleType::I16 => data_chunk.read_samples::<i16, T>(),
                ValidatedSampleType::I24 => data_chunk.read_samples::<I24, T>(),
                ValidatedSampleType::I32 => data_chunk.read_samples::<i32, T>(),
                ValidatedSampleType::F32 => data_chunk.read_samples::<f32, T>(),
                ValidatedSampleType::F64 => data_chunk.read_samples::<f64, T>(),
            }?;

            Ok((interleaved_vec, info))
        }
        #[cfg(feature = "flac")]
        FileType::FLAC => Err(AudioIOError::unsupported_format(
            "FLAC support with PyArray not yet implemented",
        )),
        other => Err(AudioIOError::unsupported_format(format!(
            "Unsupported file format: {:?}",
            other
        ))),
    }
}

/// Direct-read fast path for WAV files: parse header with BufReader (one read syscall for the
/// first ≤8 KiB), then `read_exact` the audio payload straight into a fresh `Vec<T>`.
///
/// Only used when:
/// - The file is a WAV file
/// - `T` matches the file's native sample type (no conversion needed)
/// - Platform is little-endian
/// - `T` is not `I24` (I24 has non-standard in-memory layout)
///
/// Returns `Err` if any of these preconditions aren't met, so the caller can fall back
/// to the general mmap path.
#[cfg(target_endian = "little")]
fn read_wav_direct<T>(path: &Path) -> AudioIOResult<(NonEmptyVec<T>, BaseAudioInfo)>
where
    T: StandardSample + 'static,
{
    use std::mem::size_of;

    let file = std::fs::File::open(path).map_err(AudioIOError::from)?;
    let mut reader = BufReader::new(file);

    let (info, data_byte_offset) = parse_wav_header_streaming(&mut reader)?;

    // Verify T matches the file's native sample type — otherwise fall back to mmap path.
    let native_matches = match info.sample_type {
        audio_samples::SampleType::U8 => TypeId::of::<T>() == TypeId::of::<u8>(),
        audio_samples::SampleType::I16 => TypeId::of::<T>() == TypeId::of::<i16>(),
        audio_samples::SampleType::I32 => TypeId::of::<T>() == TypeId::of::<i32>(),
        audio_samples::SampleType::F32 => TypeId::of::<T>() == TypeId::of::<f32>(),
        audio_samples::SampleType::F64 => TypeId::of::<T>() == TypeId::of::<f64>(),
        _ => false,
    };
    if !native_matches {
        return Err(AudioIOError::unsupported_format(
            "Type mismatch — use mmap path for conversion",
        ));
    }

    let total_samples = info.total_samples;
    let byte_count = total_samples * size_of::<T>();

    // Allocate Vec<T> and read the entire payload directly into it.
    // Safety: T: Copy; every byte is overwritten by read_exact before it is read.
    let mut vec: Vec<T> = Vec::with_capacity(total_samples);
    unsafe { vec.set_len(total_samples) };

    let bytes =
        unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr().cast::<u8>(), byte_count) };

    // No explicit seek needed: parse_wav_header_streaming stops reading at the first byte of the
    // data payload, leaving the BufReader positioned there.  Its internal buffer already holds
    // the first ≤8 KiB of audio data (read as part of the initial header scan), so we drain
    // those cached bytes first and then read the remaining payload in one large kernel call.
    let _ = data_byte_offset; // used only for the type-conversion fallback check above
    reader.read_exact(bytes).map_err(AudioIOError::from)?;

    let nev = NonEmptyVec::try_from(vec)
        .map_err(|_| AudioIOError::corrupted_data_simple("Empty WAV file", "No audio samples"))?;

    Ok((nev, info))
}

/// Create PyArray from interleaved Vec with Fortran layout.
fn create_pyarray_fortran<T>(
    py: Python<'_>,
    interleaved_vec: NonEmptyVec<T>,
    channels: usize,
    total_samples: usize,
) -> PyResult<Py<PyArray2<T>>>
where
    T: StandardSample + Element,
{
    if channels == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "channels must be non-zero",
        ));
    }

    if total_samples % channels != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "total_samples ({}) not divisible by channels ({})",
            total_samples, channels
        )));
    }

    if interleaved_vec.len().get() != total_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Vec length ({}) does not match total_samples ({})",
            interleaved_vec.len(),
            total_samples
        )));
    }

    let frames = total_samples / channels;
    let shape = (channels, frames);

    // Create 1D array from Vec (takes ownership, zero-copy transfer to Python)
    let array1 = PyArray1::from_vec(py, interleaved_vec.into_vec());

    // Reshape to 2D with Fortran order
    // This is the key: Fortran layout interprets the same memory as column-major,
    // which matches the interleaved format perfectly
    let array2 = array1
        .reshape_with_order(shape, numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to reshape array: {}",
                e
            ))
        })?;

    Ok(array2.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use non_empty_slice::non_empty_vec;
    use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

    #[test]
    #[cfg(feature = "numpy")]
    fn test_create_pyarray_fortran_stereo() {
        Python::initialize();
        Python::attach(|py| {
            // Interleaved stereo: [L0, R0, L1, R1, L2, R2]
            let interleaved = non_empty_vec![1i16, 2, 3, 4, 5, 6];
            let channels = 2;
            let total_samples = 6;

            let arr = create_pyarray_fortran(py, interleaved, channels, total_samples)
                .expect("Failed to create PyArray");
            let bound = arr.bind(py);

            // Should be shape (2 channels, 3 frames)
            assert_eq!(bound.shape(), &[2, 3]);

            // Verify Fortran layout
            assert!(bound.is_fortran_contiguous());
            assert!(!bound.is_c_contiguous());

            // Verify correct indexing (column-major interpretation)
            // arr[channel, frame]
            // Left channel (ch0): indices [0, 0], [0, 1], [0, 2] → values 1, 3, 5
            // Right channel (ch1): indices [1, 0], [1, 1], [1, 2] → values 2, 4, 6
            let ro: PyReadonlyArray2<i16> = bound.readonly();
            let nd = ro.as_array();
            assert_eq!(nd[[0, 0]], 1);
            assert_eq!(nd[[1, 0]], 2);
            assert_eq!(nd[[0, 1]], 3);
            assert_eq!(nd[[1, 1]], 4);
            assert_eq!(nd[[0, 2]], 5);
            assert_eq!(nd[[1, 2]], 6);
        });
    }

    #[test]
    #[cfg(feature = "numpy")]
    fn test_create_pyarray_fortran_mono() {
        Python::initialize();
        Python::attach(|py| {
            // Mono: [S0, S1, S2, S3]
            let mono = non_empty_vec![10i16, 20, 30, 40];
            let channels = 1;
            let total_samples = 4;

            let arr = create_pyarray_fortran(py, mono, channels, total_samples)
                .expect("Failed to create PyArray");
            let bound = arr.bind(py);

            // Should be shape (1 channel, 4 frames)
            assert_eq!(bound.shape(), &[1, 4]);

            // Verify values
            let ro: PyReadonlyArray2<i16> = bound.readonly();
            let nd = ro.as_array();
            assert_eq!(nd[[0, 0]], 10);
            assert_eq!(nd[[0, 1]], 20);
            assert_eq!(nd[[0, 2]], 30);
            assert_eq!(nd[[0, 3]], 40);
        });
    }

    #[test]
    #[cfg(feature = "numpy")]
    fn test_create_pyarray_fortran_validation() {
        Python::initialize();
        Python::attach(|py| {
            // Test zero channels
            let result = create_pyarray_fortran(py, non_empty_vec![1i16, 2], 0, 2);
            assert!(result.is_err());

            // Test mismatched total_samples
            let result = create_pyarray_fortran(py, non_empty_vec![1i16, 2, 3], 2, 2);
            assert!(result.is_err());

            // Test non-divisible total_samples
            let result = create_pyarray_fortran(py, non_empty_vec![1i16, 2, 3], 2, 3);
            assert!(result.is_err());
        });
    }
}
