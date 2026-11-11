//! Python NumPy integration for optimized audio file reading.
//! This module provides optimized read functions that create NumPy arrays
//! directly from WAV data using Fortran (column-major) layout to eliminate
//! deinterleaving overhead.

use std::any::TypeId;
use std::io::{BufReader, Read};
use std::path::Path;

use audio_samples::I24;
use audio_samples::traits::StandardSample;
use non_empty_slice::NonEmptyVec;
use numpy::{Element, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::traits::{AudioFile, AudioFileMetadata};
use crate::types::{OpenOptions, ValidatedSampleType};
use crate::wav::{WavFile, wav_file::parse_wav_header_streaming};
use crate::{AudioIOError, AudioIOResult, BaseAudioInfo, FileType};

/// Arrays of at least this many bytes start at the page offset of the file they are read from.
const PAGE_OFFSET_MATCH_MIN_BYTES: usize = 1024 * 1024;

/// An uninitialised `(channels, frames)` array in interleaved (`fortran`) or planar memory, with a
/// pointer to its data. A large array starts at the same offset within a page as the samples do in
/// the file (`file_offset`), where the sample type's alignment allows. The kernel's copy is faster
/// then: writing such an array to a new WAV file takes 0.92 ms instead of 1.20 for 10 MiB on ext4, and
/// 0.57 instead of 0.82 on xfs, because a source just below the file's offset within a page is slow.
fn new_output_array<T>(
    py: Python<'_>,
    channels: usize,
    frames: usize,
    fortran: bool,
    file_offset: u64,
) -> (Bound<'_, PyArray2<T>>, usize)
where
    T: Element,
{
    const PAGE: usize = 4096;
    let bytes = channels * frames * std::mem::size_of::<T>();
    let wanted = (file_offset as usize) % PAGE;
    if bytes >= PAGE_OFFSET_MATCH_MIN_BYTES && wanted.is_multiple_of(std::mem::align_of::<T>()) {
        // SAFETY: the bytes are only reached through the array below, which its writer fills.
        let block = unsafe { PyArray1::<u8>::new(py, [bytes + PAGE], false) };
        let start = block.data() as usize;
        let data = start + (wanted + PAGE - start % PAGE) % PAGE;
        use ndarray::ShapeBuilder;
        // SAFETY: the block holds `bytes` from `data`, which is aligned for T since the block's start
        // and `wanted` are, and the block becomes the array's base object, so it outlives the array.
        let array = unsafe {
            if fortran {
                let view = ndarray::ArrayView2::from_shape_ptr((channels, frames).f(), data as *const T);
                PyArray2::borrow_from_array(&view, block.into_any())
            } else {
                let view = ndarray::ArrayView2::from_shape_ptr((channels, frames), data as *const T);
                PyArray2::borrow_from_array(&view, block.into_any())
            }
        };
        return (array, data);
    }
    // SAFETY: the array is filled by the caller before anything else sees it.
    let array = unsafe { PyArray2::<T>::new(py, [channels, frames], fortran) };
    let data = array.data() as usize;
    (array, data)
}

/// A WAV file whose first bytes have been read into a block with room for the whole file: the header
/// is parsed from them, and they hold the first samples. See [`crate::fastio::whole_read_plan`].
#[cfg(target_endian = "little")]
struct FirstRead<'py> {
    block: Bound<'py, PyArray1<u8>>,
    /// Page-aligned address of the file's first byte in the block.
    start: usize,
    /// Bytes read so far, from the file's first.
    filled: usize,
    info: BaseAudioInfo,
    /// Where the samples start in the file.
    offset: usize,
}

/// Open a WAV file and read its first bytes, from which the header comes: one read for a small file
/// off the disk, which is read whole. `None` if the header cannot be parsed from them.
#[cfg(target_endian = "little")]
fn read_first<'py>(py: Python<'py>, path: &Path) -> Option<PyResult<(std::fs::File, FirstRead<'py>)>> {
    const PAGE: usize = 4096;
    let opened = py.detach(|| -> AudioIOResult<_> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        let (first, will_need) = crate::fastio::whole_read_plan(&file, len);
        Ok((file, len, first, will_need))
    });
    let (file, len, first, will_need) = opened.ok()?;
    let len_bytes = usize::try_from(len).ok()?;
    // SAFETY: the bytes are only read after the reads below fill them.
    let block = unsafe { PyArray1::<u8>::new(py, [len_bytes + PAGE], false) };
    let start = (block.data() as usize).next_multiple_of(PAGE);
    let read = py.detach(|| {
        if will_need {
            crate::fastio::will_need(&file);
        }
        // SAFETY: `start` is at most PAGE - 1 bytes into the block, which holds len + PAGE bytes, and
        // nothing else has seen the block.
        let bytes = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, first) };
        crate::fastio::read_exact_at(&file, 0, bytes)
    });
    if let Err(e) = read {
        return Some(Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())));
    }
    // SAFETY: the read above filled these bytes.
    let prefix = unsafe { std::slice::from_raw_parts(start as *const u8, first) };
    let (info, offset) = crate::wav::wav_file::wav_header_from(prefix, &file, len).ok()?;
    Some(Ok((
        file,
        FirstRead {
            block,
            start,
            filled: first,
            info,
            offset: usize::try_from(offset).ok()?,
        },
    )))
}

/// Whether `T` is the sample type of the file, so that its samples can be used in place.
fn is_native<T: 'static>(sample_type: audio_samples::SampleType) -> bool {
    match sample_type {
        audio_samples::SampleType::U8 => TypeId::of::<T>() == TypeId::of::<u8>(),
        audio_samples::SampleType::I16 => TypeId::of::<T>() == TypeId::of::<i16>(),
        audio_samples::SampleType::I32 => TypeId::of::<T>() == TypeId::of::<i32>(),
        audio_samples::SampleType::F32 => TypeId::of::<T>() == TypeId::of::<f32>(),
        audio_samples::SampleType::F64 => TypeId::of::<T>() == TypeId::of::<f64>(),
        _ => false,
    }
}

fn io_err(e: AudioIOError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
}

/// The samples of a file as an interleaved `(channels, frames)` array, starting from its first read.
///
/// When `T` is the file's type, the samples are read on into the block behind the first ones and
/// used in place (or, where the sample type's alignment does not allow that, into an array of their
/// own). Otherwise those read so far are converted, and the rest converted as they are read.
#[cfg(target_endian = "little")]
fn first_into_array<T>(
    py: Python<'_>,
    file: &std::fs::File,
    first: FirstRead<'_>,
) -> Option<PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>>
where
    T: StandardSample + Element + 'static,
{
    let info = first.info;
    let file_type = ValidatedSampleType::try_from(info.sample_type).ok()?;
    let channels = info.channels as usize;
    let total = info.total_samples;
    if channels == 0 || total == 0 || !total.is_multiple_of(channels) {
        return None;
    }
    let frames = total / channels;
    let size = file_type.bytes_per_sample().get();
    let byte_count = total.checked_mul(size)?;
    let data_end = first.offset.checked_add(byte_count)?;
    let have = first.filled.saturating_sub(first.offset).min(byte_count);
    let data = first.start + first.offset;

    if is_native::<T>(info.sample_type) && size == std::mem::size_of::<T>() {
        if data.is_multiple_of(std::mem::align_of::<T>()) {
            if first.filled < data_end {
                let rest = data_end - first.filled;
                let at = first.start + first.filled;
                let filled = first.filled as u64;
                // SAFETY: the block holds the whole file from `start`, and nothing else has seen it.
                let read = py.detach(|| {
                    let bytes = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, rest) };
                    crate::fastio::read_exact_at_parallel(file, filled, bytes)
                });
                if let Err(e) = read {
                    return Some(Err(io_err(e.into())));
                }
            }
            use ndarray::ShapeBuilder;
            // SAFETY: the block holds `byte_count` bytes of samples of type T from `data`, which is
            // aligned for T, and the block becomes the array's base object, so it outlives the array.
            let array = unsafe {
                let view = ndarray::ArrayView2::from_shape_ptr((channels, frames).f(), data as *const T);
                PyArray2::borrow_from_array(&view, first.block.into_any())
            };
            return Some(Ok((array.unbind(), info)));
        }
        let (array, out) = new_output_array::<T>(py, channels, frames, true, first.offset as u64);
        let read = py.detach(|| {
            // SAFETY: the array holds `byte_count` bytes and nothing else has seen it; the first `have`
            // bytes of samples were read into the block.
            let bytes = unsafe { std::slice::from_raw_parts_mut(out as *mut u8, byte_count) };
            bytes[..have].copy_from_slice(unsafe { std::slice::from_raw_parts(data as *const u8, have) });
            crate::fastio::read_exact_at_parallel(file, (first.offset + have) as u64, &mut bytes[have..])
        });
        return Some(read.map(|()| (array.unbind(), info)).map_err(|e| io_err(e.into())));
    }

    // Converted: whole frames from the first read, then the rest from the file.
    let frame_bytes = channels * size;
    let done = have / frame_bytes * channels;
    let (array, out) = new_output_array::<T>(py, channels, frames, true, 0);
    let offset = first.offset;
    let read = py.detach(|| {
        // SAFETY: the array holds `total` values and nothing else has seen it; the source bytes were
        // read into the block.
        let out = unsafe { std::slice::from_raw_parts_mut(out as *mut T, total) };
        let source = unsafe { std::slice::from_raw_parts(data as *const u8, done * size) };
        crate::wav::streaming::convert_bytes_into(file_type, source, &mut out[..done]);
        if done == total {
            return Ok(());
        }
        let read = crate::wav::streaming::read_converted_parallel(
            file,
            (offset + done * size) as u64,
            file_type,
            &mut out[done..],
            channels,
        )?;
        if done + read == total {
            Ok(())
        } else {
            Err(AudioIOError::corrupted_data_simple(
                "Truncated WAV data",
                format!("expected {total} samples, found {}", done + read),
            ))
        }
    });
    drop(first.block);
    Some(read.map(|()| (array.unbind(), info)).map_err(io_err))
}

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
pub fn read_pyarray_native(py: Python<'_>, path: &Path) -> Option<PyResult<NativeAudioArray>> {
    // Only WAV files are handled by the single-pass direct path.
    if !matches!(FileType::from_path(path), FileType::WAV) {
        return None;
    }

    // The header comes from the file's first read, which also holds the first samples.
    let (file, first) = match read_first(py, path)? {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let read = match first.info.sample_type {
        audio_samples::SampleType::U8 => {
            first_into_array::<u8>(py, &file, first)?.map(|(a, i)| NativeAudioArray::U8(a, i))
        },
        audio_samples::SampleType::I16 => {
            first_into_array::<i16>(py, &file, first)?.map(|(a, i)| NativeAudioArray::I16(a, i))
        },
        audio_samples::SampleType::I32 => {
            first_into_array::<i32>(py, &file, first)?.map(|(a, i)| NativeAudioArray::I32(a, i))
        },
        audio_samples::SampleType::F32 => {
            first_into_array::<f32>(py, &file, first)?.map(|(a, i)| NativeAudioArray::F32(a, i))
        },
        audio_samples::SampleType::F64 => {
            first_into_array::<f64>(py, &file, first)?.map(|(a, i)| NativeAudioArray::F64(a, i))
        },
        // I24 has a layout of its own in memory; the caller's general path reads it.
        _ => return None,
    };
    Some(read)
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
    if let Some(result) = read_pyarray_direct::<T>(py, path) {
        return result;
    }
    // Falls through to the Vec path below if direct path is not applicable.

    #[cfg(feature = "flac")]
    if let Some(result) = read_flac_planar_pyarray::<T>(py, path) {
        return result;
    }

    // Vec path: reads into intermediate Vec<T>, then hands ownership to numpy (zero-copy).
    let (interleaved_vec, info) = py
        .detach(|| read_interleaved_with_info::<_, T>(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let pyarray = create_pyarray_fortran(py, interleaved_vec, info.channels as usize, info.total_samples)?;

    Ok((pyarray, info))
}

/// Fast path for WAV loading: parse header, allocate PyArray2 with Fortran layout, read
/// audio payload directly into numpy's buffer — no intermediate Vec allocation.
///
/// Returns `None` when the fast path does not apply (big-endian, or
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
fn read_pyarray_direct<T>(py: Python<'_>, path: &Path) -> Option<PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>>
where
    T: StandardSample + Element + 'static,
{
    let (file, first) = match read_first(py, path)? {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    first_into_array::<T>(py, &file, first)
}

/// FLAC: decode into a C-order `(channels, frames)` NumPy array in one pass, with the GIL
/// released. Frames decode to planar samples, so the array needs no reordering.
///
/// Returns `None` when this does not apply (not FLAC, or STREAMINFO's sample count does not
/// hold), letting the caller fall back to the Vec path.
#[cfg(feature = "flac")]
fn read_flac_planar_pyarray<T>(py: Python<'_>, path: &Path) -> Option<PyResult<(Py<PyArray2<T>>, BaseAudioInfo)>>
where
    T: StandardSample + Element + 'static,
{
    use crate::flac::StreamedFlacFile;

    if !matches!(FileType::from_path(path), FileType::FLAC) {
        return None;
    }
    let opened = py.detach(|| {
        let file = std::fs::File::open(path).map_err(AudioIOError::from)?;
        let flac = StreamedFlacFile::new_with_path(file, path.to_path_buf())?.without_seek_index();
        let info = flac.base_info()?;
        Ok::<_, AudioIOError>((flac, info))
    });
    let (mut flac, info) = match opened {
        Ok(v) => v,
        Err(e) => return Some(Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))),
    };
    let channels = info.channels as usize;
    let total_samples = info.total_samples;
    if channels == 0 || total_samples == 0 || total_samples % channels != 0 {
        return None;
    }
    let frames = total_samples / channels;

    let (array, data_ptr_usize) = new_output_array::<T>(py, channels, frames, false, 0);
    let decoded = py.detach(|| {
        // Whole-file decoding can divide the stream between threads; it reports failure for any
        // stream that needs the error handling of the sequential decoder below.
        let (stream_info, data_start) = flac.stream_parameters();
        if let Ok(metadata) = std::fs::metadata(path)
            // SAFETY: the array holds `frames` samples for each of its channels.
            && unsafe {
                crate::flac::parallel::decode_planar_file(
                    path,
                    data_start,
                    metadata.len(),
                    stream_info,
                    data_ptr_usize as *mut T,
                    frames,
                )
            }
        {
            return Ok(frames);
        }
        // SAFETY: as in `read_pyarray_direct`: the fresh array is kept alive by `array` and no
        // other thread has seen it; it holds `total_samples` values.
        let out = unsafe { std::slice::from_raw_parts_mut(data_ptr_usize as *mut T, total_samples) };
        flac.read_planar_into(out)
    });
    match decoded {
        Ok(written) if written == frames => Some(Ok((array.unbind(), info))),
        // Fewer samples than STREAMINFO says.
        Ok(_) => None,
        Err(e) => Some(Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))),
    }
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
            // Any error falls through to the mmap path below.
            if TypeId::of::<T>() != TypeId::of::<I24>()
                && let Ok(result) = read_wav_direct::<T>(path)
            {
                return Ok(result);
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
        },
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

            let nev = NonEmptyVec::try_from(interleaved)
                .map_err(|_| AudioIOError::corrupted_data_simple("Empty FLAC file", "No samples decoded"))?;

            Ok((nev, info))
        },
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

    // Allocate Vec<T> and read the entire payload directly into its spare capacity.
    let mut vec: Vec<T> = Vec::with_capacity(total_samples);

    let bytes = unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr().cast::<u8>(), byte_count) };

    // No explicit seek needed: parse_wav_header_streaming leaves the BufReader positioned at
    // the first byte of the data payload.
    let _ = data_byte_offset; // used only for the type-conversion fallback check above
    reader.read_exact(bytes).map_err(AudioIOError::from)?;
    // Safety: T: Copy; read_exact has written every byte of the first total_samples elements.
    unsafe { vec.set_len(total_samples) };

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

    if !total_samples.is_multiple_of(channels) {
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
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to reshape array: {}", e)))?;

    Ok(array2.unbind())
}

#[cfg(test)]
mod tests {
    use non_empty_slice::non_empty_vec;
    use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

    use super::*;

    #[test]
    #[cfg(feature = "numpy")]
    fn test_create_pyarray_fortran_stereo() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipped: the embedded interpreter has no numpy");
                return;
            }
            // Interleaved stereo: [L0, R0, L1, R1, L2, R2]
            let interleaved = non_empty_vec![1i16, 2, 3, 4, 5, 6];
            let channels = 2;
            let total_samples = 6;

            let arr =
                create_pyarray_fortran(py, interleaved, channels, total_samples).expect("Failed to create PyArray");
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
            if py.import("numpy").is_err() {
                eprintln!("skipped: the embedded interpreter has no numpy");
                return;
            }
            // Mono: [S0, S1, S2, S3]
            let mono = non_empty_vec![10i16, 20, 30, 40];
            let channels = 1;
            let total_samples = 4;

            let arr = create_pyarray_fortran(py, mono, channels, total_samples).expect("Failed to create PyArray");
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
