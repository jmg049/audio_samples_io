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

/// Type-erased container for a natively-typed audio array produced by [`read_pyarray_native`].
///
/// Each variant holds a Fortran-layout `PyArray2<T>` matching the file's native sample type,
/// so callers can dispatch on the type without a second file open or header parse.
#[cfg(target_endian = "little")]
pub enum NativeAudioArray {
    U8(Py<PyArray2<u8>>, BaseAudioInfo),
    I16(Py<PyArray2<i16>>, BaseAudioInfo),
    I32(Py<PyArray2<i32>>, BaseAudioInfo),
    F32(Py<PyArray2<f32>>, BaseAudioInfo),
    F64(Py<PyArray2<f64>>, BaseAudioInfo),
}

/// Single-pass WAV read: one `File::open`, one header parse, one `read_exact` into a
/// freshly-allocated `PyArray2<T>` whose type matches the file's native sample type.
///
/// Returns `None` when the fast path does not apply (non-WAV, I24, big-endian).
/// Returns `Some(Err(...))` for I/O errors encountered after the file was opened.
///
/// The happy path for the common `read(fp)` call (no type conversion) should always hit
/// this function, saving ~3 µs per call by avoiding the redundant `peek_native_type` open.
#[cfg(target_endian = "little")]
pub fn read_pyarray_native(
    py: Python<'_>,
    path: &Path,
) -> Option<PyResult<NativeAudioArray>> {
    // Only WAV files are handled by the single-pass direct path.
    if !matches!(FileType::from_path(path), FileType::WAV) {
        return None;
    }

    // Step 1: open + header (GIL released — pure I/O)
    let parse_result: AudioIOResult<(BaseAudioInfo, BufReader<std::fs::File>)> = py.detach(|| {
        let file = std::fs::File::open(path).map_err(AudioIOError::from)?;
        let mut reader = BufReader::with_capacity(65536, file);
        let (info, _) = parse_wav_header_streaming(&mut reader)?;
        Ok((info, reader))
    });

    // Return None on any parse error — the caller will try the slower fallback path.
    let (info, reader) = parse_result.ok()?;

    // I24 has a non-standard in-memory layout — exclude from the direct path.
    if matches!(info.sample_type, audio_samples::SampleType::I24) {
        return None;
    }

    let channels = info.channels as usize;
    let total_samples = info.total_samples;
    if channels == 0 || total_samples == 0 || total_samples % channels != 0 {
        return None;
    }

    // Step 2 + 3: allocate PyArray (GIL held) then fill it (GIL released), dispatched per type.
    let result = match info.sample_type {
        audio_samples::SampleType::U8 => {
            alloc_and_fill::<u8>(py, reader, info).map(|(a, i)| NativeAudioArray::U8(a, i))
        }
        audio_samples::SampleType::I16 => {
            alloc_and_fill::<i16>(py, reader, info).map(|(a, i)| NativeAudioArray::I16(a, i))
        }
        audio_samples::SampleType::I32 => {
            alloc_and_fill::<i32>(py, reader, info).map(|(a, i)| NativeAudioArray::I32(a, i))
        }
        audio_samples::SampleType::F32 => {
            alloc_and_fill::<f32>(py, reader, info).map(|(a, i)| NativeAudioArray::F32(a, i))
        }
        audio_samples::SampleType::F64 => {
            alloc_and_fill::<f64>(py, reader, info).map(|(a, i)| NativeAudioArray::F64(a, i))
        }
        _ => return None,
    };

    Some(result)
}

/// Allocate a Fortran-layout `PyArray2<T>` and fill it from the open `BufReader`.
///
/// GIL is held for the allocation, released for the `read_exact` call.
#[cfg(target_endian = "little")]
fn alloc_and_fill<T>(
    py: Python<'_>,
    reader: BufReader<std::fs::File>,
    info: BaseAudioInfo,
) -> PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>
where
    T: Element + 'static,
{
    use std::mem::size_of;

    let channels = info.channels as usize;
    let total_samples = info.total_samples;
    let frames = total_samples / channels;
    let byte_count = total_samples * size_of::<T>();

    // Fortran (column-major) layout: element [c, f] → c + f*channels, matching WAV interleaved.
    let array = unsafe { PyArray2::<T>::new(py, [channels, frames], true) };
    let data_ptr_usize = array.data() as usize;

    let read_result: AudioIOResult<()> = py.detach(|| {
        let mut r = reader;
        // Safety: pointer valid (just allocated, not yet shared), CPython is non-moving.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(data_ptr_usize as *mut u8, byte_count) };
        r.read_exact(bytes).map_err(AudioIOError::from)
    });

    read_result
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok((array.unbind(), info))
}

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

    // Fast path: parse header once, allocate numpy array directly, read into its buffer.
    // Eliminates the intermediate Vec allocation and the extra mmap(MAP_ANONYMOUS) zero-fill
    // overhead that Vec::with_capacity triggers for large allocations.
    #[cfg(target_endian = "little")]
    if TypeId::of::<T>() != TypeId::of::<I24>() {
        if let Some(result) = read_pyarray_direct::<T>(py, path) {
            return result;
        }
        // Falls through to the Vec path below if direct path is not applicable.
    }

    // Vec path: reads into intermediate Vec<T>, then hands ownership to numpy (zero-copy).
    let (interleaved_vec, info) = py
        .detach(|| read_interleaved_with_info::<_, T>(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let pyarray = create_pyarray_fortran(
        py,
        interleaved_vec,
        info.channels as usize,
        info.total_samples,
    )?;

    Ok((pyarray, info))
}

/// Fast path for WAV loading: parse header, allocate PyArray2 with Fortran layout, read
/// audio payload directly into numpy's buffer — no intermediate Vec allocation.
///
/// Returns `None` when the fast path does not apply (type mismatch, I24, big-endian, or
/// any I/O error), letting the caller fall back to the Vec path.
///
/// # Safety
///
/// The raw data pointer obtained from the freshly-allocated, unshared PyArray is used while
/// the GIL is released.  This is sound because:
/// 1. The array was just created and has not been returned to Python — no other thread holds it.
/// 2. CPython uses reference-counted, non-moving GC; object addresses are stable.
/// 3. We hold a strong reference (`Bound<'_, PyArray2<T>>`) that prevents deallocation.
#[cfg(target_endian = "little")]
fn read_pyarray_direct<T>(
    py: Python<'_>,
    path: &Path,
) -> Option<PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>>
where
    T: StandardSample + Element + 'static,
{
    use std::mem::size_of;

    // Step 1: Parse WAV header with GIL released (I/O).  A 64 KB BufReader reads the header
    // and the first chunk of audio data in a single syscall, reducing round-trips.
    let parse_result: AudioIOResult<(BaseAudioInfo, BufReader<std::fs::File>)> = py.detach(|| {
        let file = std::fs::File::open(path).map_err(AudioIOError::from)?;
        let mut reader = BufReader::with_capacity(65536, file);
        let (info, _) = parse_wav_header_streaming(&mut reader)?;
        Ok((info, reader))
    });

    let (info, reader) = match parse_result {
        Ok(v) => v,
        Err(_) => return None,
    };

    // Fast path applies only when T matches the file's native sample type.
    let native_matches = match info.sample_type {
        audio_samples::SampleType::U8 => TypeId::of::<T>() == TypeId::of::<u8>(),
        audio_samples::SampleType::I16 => TypeId::of::<T>() == TypeId::of::<i16>(),
        audio_samples::SampleType::I32 => TypeId::of::<T>() == TypeId::of::<i32>(),
        audio_samples::SampleType::F32 => TypeId::of::<T>() == TypeId::of::<f32>(),
        audio_samples::SampleType::F64 => TypeId::of::<T>() == TypeId::of::<f64>(),
        _ => false,
    };
    if !native_matches {
        return None;
    }

    let channels = info.channels as usize;
    let total_samples = info.total_samples;
    if channels == 0 || total_samples == 0 || total_samples % channels != 0 {
        return None;
    }
    let frames = total_samples / channels;
    let byte_count = total_samples * size_of::<T>();

    // Step 2: Allocate PyArray2 with Fortran (column-major) layout (GIL held, fast).
    // Fortran layout for shape (channels, frames) maps element [c, f] to index c + f*channels,
    // which is exactly WAV's interleaved on-disk format — no deinterleave needed.
    // `PyArray2::new` is the numpy equivalent of `numpy.empty(..., order='F')`.
    let array = unsafe { PyArray2::<T>::new(py, [channels, frames], true) };

    // Step 3: Read audio payload directly into the PyArray's buffer (GIL released).
    // Convert the raw data pointer to usize so it can cross the `py.detach()` Ungil boundary.
    let data_ptr_usize = array.data() as usize;

    let read_result: AudioIOResult<()> = py.detach(|| {
        let mut r = reader;
        // Safety: pointer was valid when captured (step 2), array is held alive by `array`,
        // CPython does not move objects, no other thread has seen this array.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data_ptr_usize as *mut u8, byte_count) };
        r.read_exact(bytes).map_err(AudioIOError::from)
    });

    if let Err(e) = read_result {
        return Some(Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())));
    }

    Some(Ok((array.unbind(), info)))
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
        FileType::FLAC => {
            use crate::flac::FlacFile;
            use crate::traits::{AudioFile, AudioFileRead};

            let flac_file = FlacFile::open_with_options(path, OpenOptions::default())?;
            let info = flac_file.base_info()?;
            let channels = info.channels as usize;
            let total_samples = info.total_samples;
            let frames = total_samples / channels;

            // FLAC decode → planar layout: [ch0[0..frames], ch1[0..frames], ...]
            // Stored as C-order Array2(channels, frames).
            let audio = flac_file.read::<T>()?.into_owned();
            let planar = audio.as_slice().ok_or_else(|| {
                AudioIOError::corrupted_data_simple(
                    "FLAC decode produced non-contiguous data",
                    "Cannot extract samples",
                )
            })?;

            // Convert planar → interleaved so create_pyarray_fortran can reuse the WAV path.
            let mut interleaved: Vec<T> = Vec::with_capacity(total_samples);
            for f in 0..frames {
                for c in 0..channels {
                    interleaved.push(planar[c * frames + f]);
                }
            }

            let nev = NonEmptyVec::try_from(interleaved).map_err(|_| {
                AudioIOError::corrupted_data_simple("Empty FLAC file", "No samples decoded")
            })?;

            Ok((nev, info))
        }
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
    let mut reader = BufReader::with_capacity(65536, file);

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
