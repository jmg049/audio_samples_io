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
use std::sync::atomic::{AtomicBool, Ordering};

use audio_samples::{
    I24,
    traits::{ConvertTo, StandardSample},
};
use ndarray::{ArrayView2, Axis};

use crate::error::AudioIOResult;

/// Each thread of a file transfer gets at least this many bytes: below it, starting a thread
/// costs more than it saves.
pub(crate) const TRANSFER_BYTES_PER_THREAD: usize = 512 * 1024;

/// A transfer below this stays on the calling thread: two threads would not pay for themselves.
pub(crate) const TRANSFER_MIN_BYTES: usize = 3 * 512 * 1024;

/// Most threads a file transfer uses: the copy is limited by memory bandwidth beyond this.
const MAX_TRANSFER_THREADS: usize = 8;
/// Most threads a decode uses.
const MAX_DECODE_THREADS: usize = 8;

/// Stack of a transfer thread: the work on them uses little, and a stack of a huge page or more
/// could be backed by one.
const THREAD_STACK_BYTES: usize = 256 * 1024;

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

/// Held by the one transfer allowed to use several threads at a time. A program that reads many
/// files on its own threads already has the parallelism, and more would only contend.
pub(crate) struct ThreadsPermit(bool);

static THREADED_TRANSFER: AtomicBool = AtomicBool::new(false);

impl ThreadsPermit {
    /// Claim the permit if it is free.
    pub(crate) fn try_acquire() -> Self {
        Self(!THREADED_TRANSFER.swap(true, Ordering::Acquire))
    }

    /// The threads this transfer may use, out of `wanted`.
    pub(crate) const fn threads(&self, wanted: usize) -> usize {
        if self.0 { wanted } else { 1 }
    }
}

impl Drop for ThreadsPermit {
    fn drop(&mut self) {
        if self.0 {
            THREADED_TRANSFER.store(false, Ordering::Release);
        }
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
///
/// On Linux the threads are plain pthreads. A `std::thread` also sets up a signal stack, a thread
/// handle and thread-local destructors, which came to one more resident page per thread.
#[cfg(target_os = "linux")]
pub(crate) fn run_on_threads<F>(count: usize, work: F) -> bool
where
    F: Fn(usize) -> bool + Sync,
{
    struct Job<'a, F> {
        work: &'a F,
        index: usize,
    }
    extern "C" fn start<F: Fn(usize) -> bool + Sync>(job: *mut libc::c_void) -> *mut libc::c_void {
        // SAFETY: `job` points to a `Job` in the frame below, which joins before returning.
        let job = unsafe { &*job.cast::<Job<'_, F>>() };
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.work)(job.index))).unwrap_or(false);
        usize::from(ok) as *mut libc::c_void
    }

    let spawned = count.saturating_sub(1);
    let jobs: Vec<Job<'_, F>> = (0..spawned).map(|index| Job { work: &work, index }).collect();
    let mut handles: Vec<libc::pthread_t> = Vec::with_capacity(spawned);
    let mut inline: Vec<usize> = Vec::new();
    // SAFETY: an all-zero value is valid storage for pthread_attr_init to fill.
    let mut attr: libc::pthread_attr_t = unsafe { mem::zeroed() };
    // SAFETY: `attr` is valid storage, initialised once and destroyed below.
    unsafe {
        libc::pthread_attr_init(&raw mut attr);
        libc::pthread_attr_setstacksize(&raw mut attr, THREAD_STACK_BYTES);
    }
    for job in &jobs {
        let mut handle: libc::pthread_t = 0;
        // SAFETY: `start::<F>` matches the argument, and each job outlives its thread: every started
        // thread is joined below, before `jobs` is dropped.
        let result = unsafe {
            libc::pthread_create(
                &raw mut handle,
                &raw const attr,
                start::<F>,
                std::ptr::from_ref(job).cast_mut().cast(),
            )
        };
        if result == 0 {
            handles.push(handle);
        } else {
            inline.push(job.index);
        }
    }
    // SAFETY: `attr` was initialised above.
    unsafe { libc::pthread_attr_destroy(&raw mut attr) };
    let mut ok = count == 0 || work(count - 1);
    for index in inline {
        ok &= work(index);
    }
    for handle in handles {
        let mut result: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: the handle came from pthread_create and is joined once.
        unsafe { libc::pthread_join(handle, &raw mut result) };
        ok &= !result.is_null();
    }
    ok
}

/// Run `work(0..count)`, each index on its own thread except the last, which runs here. An index
/// whose thread cannot be started runs here as well. True if every call returned true.
#[cfg(not(target_os = "linux"))]
pub(crate) fn run_on_threads<F>(count: usize, work: F) -> bool
where
    F: Fn(usize) -> bool + Sync,
{
    let work = &work;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count.saturating_sub(1));
        let mut inline = Vec::new();
        for index in 0..count.saturating_sub(1) {
            match std::thread::Builder::new()
                .stack_size(THREAD_STACK_BYTES)
                .spawn_scoped(scope, move || work(index))
            {
                Ok(handle) => handles.push(handle),
                Err(_) => inline.push(index),
            }
        }
        let mut ok = count == 0 || work(count - 1);
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

#[cfg(windows)]
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], offset + done as u64) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {},
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Without positioned reads, the single file position rules out threads.
#[cfg(not(any(unix, windows)))]
pub(crate) fn read_exact_at(mut file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// Whether a range of `file` must come from the device. Threads then only split one sequential
/// stream into several, which reads 3 to 8% slower on a SATA SSD and worse on a hard disk, and the
/// copy they speed up is a small part of the time. Linux tells, from 6.5; elsewhere, assume cached.
#[cfg(target_os = "linux")]
pub(crate) fn range_is_cold(file: &File, offset: u64, len: usize) -> bool {
    use std::os::fd::AsRawFd;

    #[repr(C)]
    struct Range {
        offset: u64,
        len: u64,
    }
    #[repr(C)]
    #[derive(Default)]
    struct Stat {
        nr_cache: u64,
        nr_dirty: u64,
        nr_writeback: u64,
        nr_evicted: u64,
        nr_recently_evicted: u64,
    }
    const SYS_CACHESTAT: libc::c_long = 451;
    let range = Range {
        offset,
        len: len as u64,
    };
    let mut stat = Stat::default();
    // SAFETY: the descriptor is open, and both pointers are to live values of the layout the call uses.
    let result = unsafe {
        libc::syscall(
            SYS_CACHESTAT,
            file.as_raw_fd(),
            &raw const range,
            &raw mut stat,
            0 as libc::c_uint,
        )
    };
    let pages = (len as u64).div_ceil(4096).max(1);
    result == 0 && stat.nr_cache < pages - pages / 8
}

#[cfg(not(target_os = "linux"))]
pub(crate) const fn range_is_cold(_file: &File, _offset: u64, _len: usize) -> bool {
    false
}

/// The threads a transfer of `len` bytes from `file` should use, given a permit.
pub(crate) fn threads_for_read(permit: &ThreadsPermit, file: &File, offset: u64, len: usize) -> usize {
    let threads = permit.threads(transfer_threads().min(len / TRANSFER_BYTES_PER_THREAD));
    if threads <= 1 || len < TRANSFER_MIN_BYTES || range_is_cold(file, offset, len) {
        1
    } else {
        threads
    }
}

/// [`read_exact_at`], on several threads for a large `buf`.
pub(crate) fn read_exact_at_parallel(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let permit = ThreadsPermit::try_acquire();
    let threads = threads_for_read(&permit, file, offset, buf.len());
    if threads <= 1 || cfg!(not(any(unix, windows))) {
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
/// number of bytes written. `file_offset` is where the samples start in the destination: each tile
/// is placed a little above that offset within a page, which makes the kernel's copy into the
/// page cache about 15% faster than when the source's page offset is the lower of the two.
pub(crate) fn write_interleaved_le<T, U, W>(
    writer: &mut W,
    view: ArrayView2<'_, T>,
    file_offset: u64,
) -> AudioIOResult<usize>
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
    const PAGE: usize = 4096;
    const ABOVE: usize = 64;
    #[repr(align(4096))]
    struct Tile([mem::MaybeUninit<u8>; TILE_BYTES + PAGE + ABOVE]);
    let mut stack = Tile([mem::MaybeUninit::uninit(); TILE_BYTES + PAGE + ABOVE]);
    // A frame larger than the stack buffer is only possible with over a thousand channels.
    let mut heap: Vec<u8> = Vec::new();
    let tile_base: *mut u8 = if frame_bytes > TILE_BYTES {
        heap.reserve_exact(frame_bytes + PAGE + ABOVE);
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
        let position = file_offset as usize + start * frame_bytes;
        // SAFETY: the buffer has room for a page and a bit beyond the tile.
        let tile = unsafe { tile_base.add(position % PAGE + ABOVE) };
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
