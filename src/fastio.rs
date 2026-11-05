//! File transfers shared by the read and write paths.
//!
//! Large reads and decodes run on a few short-lived threads, which end before the call returns.
//! The number follows the process's CPU affinity and is capped; setting the environment variable
//! `AUDIO_SAMPLES_IO_THREADS` replaces the cap, and `1` keeps everything on the calling thread.

use std::any::TypeId;
use std::fs::File;
use std::io::{self, Seek, Write};
use std::mem;
use std::path::Path;
use std::sync::OnceLock;

use audio_samples::{
    I24,
    traits::{ConvertTo, StandardSample},
};
use ndarray::{ArrayView2, Axis};

use crate::error::AudioIOResult;

/// Each thread of a file transfer gets at least this many bytes: below it, starting a thread
/// costs more than it saves.
#[cfg(target_os = "linux")]
pub(crate) const TRANSFER_BYTES_PER_THREAD: usize = 512 * 1024;
#[cfg(not(target_os = "linux"))]
pub(crate) const TRANSFER_BYTES_PER_THREAD: usize = usize::MAX;

/// Most threads a file transfer uses: the copy is limited by memory bandwidth beyond this.
const MAX_TRANSFER_THREADS: usize = 8;
/// Most threads a decode uses.
const MAX_DECODE_THREADS: usize = 8;

fn available_threads() -> (usize, bool) {
    static THREADS: OnceLock<(usize, bool)> = OnceLock::new();
    *THREADS.get_or_init(|| {
        if let Some(n) = std::env::var("AUDIO_SAMPLES_IO_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
        {
            return (n, true);
        }
        (std::thread::available_parallelism().map_or(1, |n| n.get()), false)
    })
}

/// Threads for moving bytes between a file and memory.
pub(crate) fn transfer_threads() -> usize {
    match available_threads() {
        (n, true) => n,
        (n, false) => n.min(MAX_TRANSFER_THREADS),
    }
}

/// Threads for CPU-bound decoding.
pub(crate) fn decode_threads() -> usize {
    match available_threads() {
        (n, true) => n,
        (n, false) => n.min(MAX_DECODE_THREADS),
    }
}

/// Run `work(0..count)`, each index on its own thread except the last, which runs here. An index
/// whose thread cannot be started runs here as well. True if every call returned true.
pub(crate) fn run_on_threads<F>(count: usize, work: F) -> bool
where
    F: Fn(usize) -> bool + Sync,
{
    let work = &work;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count.saturating_sub(1));
        let mut inline = Vec::new();
        for index in 0..count.saturating_sub(1) {
            match std::thread::Builder::new().spawn_scoped(scope, move || work(index)) {
                Ok(handle) => handles.push(handle),
                Err(_) => inline.push(index),
            }
        }
        let mut ok = work(count - 1);
        for index in inline {
            ok &= work(index);
        }
        for handle in handles {
            ok &= handle.join().unwrap_or(false);
        }
        ok
    })
}

/// Fill `buf` from `file` at `offset`, leaving the file position alone.
#[cfg(unix)]
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
pub(crate) fn read_exact_at(mut file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// [`read_exact_at`], on several threads for a large `buf`.
pub(crate) fn read_exact_at_parallel(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let threads = transfer_threads().min(buf.len() / TRANSFER_BYTES_PER_THREAD);
    if threads <= 1 {
        return read_exact_at(file, offset, buf);
    }
    let len = buf.len();
    let chunk = len.div_ceil(threads).next_multiple_of(64 * 1024);
    let base = buf.as_mut_ptr() as usize;
    let failed = std::sync::Mutex::new(None);
    let count = len.div_ceil(chunk);
    run_on_threads(count, |index| {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        // SAFETY: indices give disjoint ranges inside `buf`, which outlives the threads.
        let part = unsafe { std::slice::from_raw_parts_mut((base + start) as *mut u8, end - start) };
        match read_exact_at(file, offset + start as u64, part) {
            Ok(()) => true,
            Err(e) => {
                if let Ok(mut slot) = failed.lock() {
                    *slot = Some(e);
                }
                false
            },
        }
    });
    match failed.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
}

/// A reader over part of a file that leaves the file position alone, so that several threads
/// can read one file.
#[cfg(any(unix, windows))]
pub(crate) struct FileAt<'a> {
    pub file: &'a File,
    pub offset: u64,
}

#[cfg(any(unix, windows))]
impl io::Read for FileAt<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(self.file, buf, self.offset)?;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(self.file, buf, self.offset)?;
        self.offset += n as u64;
        Ok(n)
    }
}

/// Open `path` for a whole-file write, keeping any existing contents until
/// [`finish_overwrite`] cuts the file to what was written.
///
/// Truncating first costs more than the write in two common cases: tmpfs frees every page and
/// allocates it again, and btrfs flushes a file truncated to zero when it is closed. Overwriting
/// reuses the pages. Permissions, ownership and hard links stay as they are, as with truncation.
pub(crate) fn open_for_overwrite(path: &Path) -> io::Result<File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// Cut a file opened by [`open_for_overwrite`] at the current write position. Anything that is
/// not a regular file, such as a pipe, is left alone.
pub(crate) fn finish_overwrite(file: &mut File) -> io::Result<()> {
    let metadata = file.metadata()?;
    if metadata.is_file() {
        let end = file.stream_position()?;
        if metadata.len() != end {
            file.set_len(end)?;
        }
    }
    Ok(())
}

/// Open `path` for a streamed write, whose length is unknown until it ends. The old contents
/// shrink to one byte instead of none: see [`open_for_overwrite`] for what zero costs, and every
/// streamed format writes its header over that byte at once.
pub(crate) fn open_for_streaming(path: &Path) -> io::Result<File> {
    let file = open_for_overwrite(path)?;
    let metadata = file.metadata()?;
    if metadata.is_file() && metadata.len() > 1 {
        file.set_len(1)?;
    }
    Ok(file)
}

/// Bytes of interleaved samples converted at a time. The buffer lives on the stack, so writing
/// a file makes no allocation that grows with its length. Smaller tiles mean more write calls, each
/// of which costs about as much as converting a few KiB.
const TILE_BYTES: usize = 64 * 1024;

/// Convert `count` frames from the channel runs in `runs` to `U`, writing them interleaved
/// little-endian at `out`. Each run holds one channel, or several already interleaved.
///
/// # Safety
///
/// `out` must be valid for `count * runs.len() * U::BYTES` bytes, and each run for `count` samples.
#[inline(always)]
unsafe fn pack_frames<T, U>(runs: &[&[T]], count: usize, out: *mut u8)
where
    T: StandardSample + ConvertTo<U>,
    U: StandardSample,
{
    let size = U::BYTES as usize;
    let stride = runs.len() * size;
    // One channel at a time: each is read in order, and the tile stays in the cache.
    for (c, run) in runs.iter().enumerate() {
        // SAFETY: the last value lands at `(count - 1) * stride + c * size`, which ends within
        // the caller's bound.
        let mut p = unsafe { out.add(c * size) };
        for &sample in &run[..count] {
            let converted: U = sample.convert_to();
            let bytes = converted.to_le_bytes();
            // SAFETY: as above. `wrapping_add` steps past the end after the last value only.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ref().as_ptr(), p, size) };
            p = p.wrapping_add(stride);
        }
    }
}

/// Write the samples of `view`, shape `(channels, frames)`, interleaved and little-endian as `U`.
///
/// Samples whose memory is already the file's layout go straight to `writer`. Anything else
/// passes through a buffer of [`TILE_BYTES`], so no copy of the signal is made. Returns the
/// number of bytes written.
pub(crate) fn write_interleaved_le<T, U, W>(writer: &mut W, view: ArrayView2<'_, T>) -> AudioIOResult<usize>
where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
    W: Write,
{
    let (channels, frames) = view.dim();
    let size = U::BYTES as usize;
    let total = channels * frames * size;
    if total == 0 {
        return Ok(0);
    }

    // Interleaved memory: channel stride 1, frame stride `channels`.
    let interleaved = if channels == 1 || view.strides() == [1, channels as isize] {
        view.as_slice_memory_order()
    } else {
        None
    };

    #[cfg(target_endian = "little")]
    if let Some(slice) = interleaved
        && TypeId::of::<T>() == TypeId::of::<U>()
        && TypeId::of::<T>() != TypeId::of::<I24>()
    {
        // SAFETY: T is a plain numeric type, for which any initialised memory is valid bytes,
        // and its in-memory layout on a little-endian machine is the file's.
        let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), total) };
        writer.write_all(bytes)?;
        return Ok(total);
    }

    let frame_bytes = channels * size;
    #[repr(align(64))]
    struct Tile([mem::MaybeUninit<u8>; TILE_BYTES]);
    let mut stack = Tile([mem::MaybeUninit::uninit(); TILE_BYTES]);
    // A frame larger than the stack buffer is only possible with over a thousand channels.
    let mut heap: Vec<u8> = Vec::new();
    let tile: *mut u8 = if frame_bytes > TILE_BYTES {
        heap.reserve_exact(frame_bytes);
        heap.as_mut_ptr()
    } else {
        stack.0.as_mut_ptr().cast()
    };
    let tile_frames = (TILE_BYTES / frame_bytes).max(1);

    let row_slices: Option<Vec<&[T]>> = if interleaved.is_none() {
        (0..channels).map(|c| view.index_axis(Axis(0), c).to_slice()).collect()
    } else {
        None
    };

    let packed_i24 = TypeId::of::<T>() == TypeId::of::<I24>() && TypeId::of::<U>() == TypeId::of::<I24>();
    let mut tails: Vec<&[T]> = Vec::with_capacity(channels);
    let mut start = 0usize;
    while start < frames {
        let count = tile_frames.min(frames - start);
        let bytes = count * frame_bytes;
        if let Some(slice) = interleaved
            && packed_i24
        {
            let samples = &slice[start * channels..(start + count) * channels];
            // SAFETY: T is I24, checked above, and the tile holds three bytes per sample.
            unsafe {
                crate::types::pack_i24(
                    std::slice::from_raw_parts(samples.as_ptr().cast::<I24>(), samples.len()),
                    tile,
                );
            }
        } else if let Some(slice) = interleaved {
            // SAFETY: the tile holds `count * channels * size` bytes, the slice that many samples
            // from `start * channels`.
            unsafe { pack_frames::<T, U>(&[&slice[start * channels..]], count * channels, tile) };
        } else if let Some(rows) = &row_slices {
            tails.clear();
            tails.extend(rows.iter().map(|r| &r[start..]));
            // SAFETY: the tile holds `count` frames and each tail that many samples.
            unsafe { pack_frames::<T, U>(&tails, count, tile) };
        } else {
            let mut p = tile;
            for f in start..start + count {
                for c in 0..channels {
                    let converted: U = view[[c, f]].convert_to();
                    let encoded = converted.to_le_bytes();
                    // SAFETY: at most `count * frame_bytes` bytes are written to the tile.
                    unsafe {
                        std::ptr::copy_nonoverlapping(encoded.as_ref().as_ptr(), p, size);
                        p = p.add(size);
                    }
                }
            }
        }
        // SAFETY: every one of these bytes was written above.
        writer.write_all(unsafe { std::slice::from_raw_parts(tile, bytes) })?;
        start += count;
    }
    Ok(total)
}
