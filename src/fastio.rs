//! File transfers shared by the read and write paths.
//!
//! Large reads and decodes run on a few short-lived threads, which end before the call returns.
//! The number follows the process's CPU affinity and is capped; [`set_thread_limit`] or the
//! environment variable `AUDIO_SAMPLES_IO_THREADS` replaces the cap, and `1` keeps everything on the
//! calling thread.

#[cfg(any(feature = "wav", feature = "flac"))]
use std::any::TypeId;
#[cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]
use std::fs::File;
#[cfg(any(feature = "wav", feature = "flac"))]
use std::io::Write;
#[cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]
use std::io::{self, Seek};
#[cfg(any(feature = "wav", feature = "flac"))]
use std::marker::PhantomData;
#[cfg(any(feature = "wav", feature = "flac"))]
use std::mem;
#[cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]
use std::path::Path;
use std::sync::OnceLock;
#[cfg(any(feature = "wav", feature = "flac"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(feature = "wav", feature = "flac"))]
use audio_samples::{
    I24,
    traits::{ConvertTo, StandardSample},
};
#[cfg(any(feature = "wav", feature = "flac"))]
use ndarray::{ArrayView2, Axis};

#[cfg(any(feature = "wav", feature = "flac"))]
use crate::error::AudioIOResult;

/// Each thread of a file transfer gets at least this many bytes: below it, starting a thread
/// costs more than it saves.
#[cfg(feature = "wav")]
pub(crate) const TRANSFER_BYTES_PER_THREAD: usize = 512 * 1024;

/// A transfer below this stays on the calling thread: two threads would not pay for themselves.
#[cfg(feature = "wav")]
pub(crate) const TRANSFER_MIN_BYTES: usize = 3 * 512 * 1024;

/// Most threads a file transfer uses: the copy is limited by memory bandwidth beyond this.
const MAX_TRANSFER_THREADS: usize = 8;
/// Most threads a decode uses.
#[cfg(feature = "flac")]
const MAX_DECODE_THREADS: usize = 8;

/// Stack of a transfer thread: the work on them uses little, and a stack of a huge page or more
/// could be backed by one.
#[cfg(any(feature = "wav", feature = "flac"))]
const THREAD_STACK_BYTES: usize = 256 * 1024;

/// The limit set by [`set_thread_limit`]; zero when there is none.
static THREAD_LIMIT: AtomicUsize = AtomicUsize::new(0);

/// How to stream a file from start to end: see [`stream_plan`].
#[cfg(feature = "wav")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPlan {
    /// Bytes read at a time into the buffer.
    pub buffer: usize,
    /// Requests of at least this many bytes are read straight into the caller's memory, past the
    /// buffer, once the buffer is empty.
    pub direct_min: usize,
    /// The file's length.
    pub len: u64,
}

/// Choose how to stream `file` from start to end, from its size, whether it is in the page cache
/// and its file system, and prepare it: a small file off the disk is asked for whole first
/// (`posix_fadvise(WILLNEED)`), so that the first read, which tells the format and holds the header,
/// and the reads after it are not separate round trips to the disk.
///
/// | file | buffer | read straight into the caller's memory | asked for whole |
/// |---|---|---|---|
/// | in the page cache | 256 KiB | requests of 16 KiB and more | no |
/// | off the disk, ext4 or xfs | 1 MiB | never | up to 1 MiB |
/// | off the disk, btrfs | 64 KiB | requests of 16 KiB and more | up to 768 KiB |
/// | off the disk, elsewhere | 64 KiB | requests of 16 KiB and more | no |
///
/// Measured in C over caller blocks of 1 KiB to 1 MiB and files of 0.5 to 106 MiB, against the
/// faster of an 8 KiB buffer (wave) and reads of the caller's block (soundfile):
/// - cached: a 256 KiB buffer took 0.83 to 0.92 times as long for blocks of 1 and 4 KiB, where
///   for blocks of 16 KiB and more the copy through any buffer cost 1.2 to 1.8 times, so those go
///   straight to the caller;
/// - off the disk on ext4 and xfs, reads of 1 MiB took 0.93 to 0.98 times as long as reads of 8 to
///   256 KiB, at every block size;
/// - off the disk on btrfs, files of 512 KiB asked for whole took 0.86 to 0.90 times as long; from
///   2 MiB a 64 KiB buffer was the fastest plan (0.987 to 0.995), and reads of 1 MiB or more 2 to 5%
///   slower. In Python against soundfile, a 256 KiB first read lost 4 to 5% on a cold 1.76 MB file.
#[cfg(feature = "wav")]
pub fn stream_plan(file: &File) -> StreamPlan {
    const KIB: usize = 1024;
    let len = file.metadata().map_or(0, |m| m.len());
    let cached = StreamPlan {
        buffer: 256 * KIB,
        direct_min: 16 * KIB,
        len,
    };
    if len == 0 || !range_is_cold(file, 0, len as usize) {
        return cached;
    }
    let fs = file_system(file);
    if whole_read_plan_on(file, len, fs) == Some(true) {
        will_need(file);
    }
    if fs.is_ext4_or_xfs() {
        StreamPlan {
            buffer: 1024 * KIB,
            direct_min: usize::MAX,
            len,
        }
    } else {
        StreamPlan {
            buffer: 64 * KIB,
            ..cached
        }
    }
}

/// A buffered reader that follows a [`StreamPlan`]: refills of `buffer` bytes, and requests of
/// `direct_min` bytes or more read straight into the caller's memory once the buffer is empty.
#[cfg(feature = "wav")]
pub struct StreamReader<R> {
    inner: R,
    buf: Box<[u8]>,
    pos: usize,
    filled: usize,
    direct_min: usize,
}

#[cfg(feature = "wav")]
impl<R: io::Read> StreamReader<R> {
    /// A reader over `inner`, positioned where `inner` is.
    pub fn new(inner: R, plan: StreamPlan) -> Self {
        Self::with_buffer(inner, plan.buffer, plan.direct_min)
    }

    /// A reader with a buffer of `buffer` bytes that reads requests of `direct_min` bytes or more
    /// straight into the caller's memory.
    pub fn with_buffer(inner: R, buffer: usize, direct_min: usize) -> Self {
        Self {
            inner,
            buf: vec![0u8; buffer.max(1)].into_boxed_slice(),
            pos: 0,
            filled: 0,
            direct_min,
        }
    }

    /// The source.
    pub const fn get_ref(&self) -> &R {
        &self.inner
    }
}

#[cfg(feature = "wav")]
impl<R: io::Read> io::Read for StreamReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos == self.filled && out.len() >= self.direct_min {
            return self.inner.read(out);
        }
        let available = io::BufRead::fill_buf(self)?;
        let n = available.len().min(out.len());
        out[..n].copy_from_slice(&available[..n]);
        io::BufRead::consume(self, n);
        Ok(n)
    }
}

#[cfg(feature = "wav")]
impl<R: io::Read> io::BufRead for StreamReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos == self.filled {
            self.filled = self.inner.read(&mut self.buf)?;
            self.pos = 0;
        }
        Ok(&self.buf[self.pos..self.filled])
    }

    fn consume(&mut self, amount: usize) {
        self.pos = (self.pos + amount).min(self.filled);
    }
}

#[cfg(feature = "wav")]
impl<R: io::Read + io::Seek> io::Seek for StreamReader<R> {
    fn seek(&mut self, to: io::SeekFrom) -> io::Result<u64> {
        let buffered = (self.filled - self.pos) as i64;
        let result = match to {
            // Relative to the position seen by the caller, which is behind the source's.
            io::SeekFrom::Current(d) => self.inner.seek(io::SeekFrom::Current(d - buffered))?,
            other => self.inner.seek(other)?,
        };
        self.pos = 0;
        self.filled = 0;
        Ok(result)
    }
}

/// Set how many threads a file transfer or decode may use, in place of the default cap and of
/// `AUDIO_SAMPLES_IO_THREADS`. `Some(1)` keeps everything on the calling thread. `None` returns to
/// the default. Transfers already running keep the threads they have.
pub fn set_thread_limit(limit: Option<std::num::NonZeroUsize>) {
    THREAD_LIMIT.store(limit.map_or(0, std::num::NonZeroUsize::get), Ordering::Relaxed);
}

/// The most threads a large file read will use at present.
#[must_use]
pub fn thread_limit() -> usize {
    transfer_threads()
}

fn available_threads() -> (usize, bool) {
    let set = THREAD_LIMIT.load(Ordering::Relaxed);
    if set > 0 {
        return (set, true);
    }
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
#[cfg(any(feature = "wav", feature = "flac"))]
pub(crate) struct ThreadsPermit(bool);

#[cfg(any(feature = "wav", feature = "flac"))]
static THREADED_TRANSFER: AtomicBool = AtomicBool::new(false);

#[cfg(any(feature = "wav", feature = "flac"))]
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

#[cfg(any(feature = "wav", feature = "flac"))]
impl Drop for ThreadsPermit {
    fn drop(&mut self) {
        if self.0 {
            THREADED_TRANSFER.store(false, Ordering::Release);
        }
    }
}

/// Threads for CPU-bound decoding.
#[cfg(feature = "flac")]
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
#[cfg(any(feature = "wav", feature = "flac"))]
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
#[cfg(any(feature = "wav", feature = "flac"))]
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
#[cfg(feature = "wav")]
#[cfg(unix)]
pub(crate) fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(feature = "wav")]
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
#[cfg(feature = "wav")]
#[cfg(not(any(unix, windows)))]
pub(crate) fn read_exact_at(mut file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// Whether a range of `file` must come from the device. Threads then only split one sequential
/// stream into several, which reads 3 to 8% slower on a SATA SSD and worse on a hard disk, and the
/// copy they speed up is a small part of the time. Linux tells, from 6.5; elsewhere, assume cached.
#[cfg(feature = "wav")]
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

#[cfg(feature = "wav")]
#[cfg(not(target_os = "linux"))]
pub(crate) const fn range_is_cold(_file: &File, _offset: u64, _len: usize) -> bool {
    false
}

/// The threads a transfer of `len` bytes from `file` should use, given a permit.
#[cfg(feature = "wav")]
pub(crate) fn threads_for_read(permit: &ThreadsPermit, file: &File, offset: u64, len: usize) -> usize {
    let threads = permit.threads(transfer_threads().min(len / TRANSFER_BYTES_PER_THREAD));
    if threads <= 1 || len < TRANSFER_MIN_BYTES || range_is_cold(file, offset, len) {
        1
    } else {
        threads
    }
}

/// [`read_exact_at`], on several threads for a large `buf`.
#[cfg(feature = "wav")]
pub(crate) fn read_exact_at_parallel(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    read_exact_at_parallel_known(file, offset, buf, None)
}

/// [`read_exact_at_parallel`] for a range already known to be off the disk or not (`cold`), which
/// saves asking the kernel again.
#[cfg(feature = "wav")]
pub(crate) fn read_exact_at_parallel_known(
    file: &File,
    offset: u64,
    buf: &mut [u8],
    cold: Option<bool>,
) -> io::Result<()> {
    let permit = ThreadsPermit::try_acquire();
    let threads = match cold {
        Some(true) => 1,
        Some(false) => {
            let threads = permit.threads(transfer_threads().min(buf.len() / TRANSFER_BYTES_PER_THREAD));
            if buf.len() < TRANSFER_MIN_BYTES { 1 } else { threads }
        },
        None => threads_for_read(&permit, file, offset, buf.len()),
    };
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
#[cfg(feature = "wav")]
#[cfg(any(unix, windows))]
pub(crate) struct FileAt<'a> {
    pub file: &'a File,
    pub offset: u64,
}

#[cfg(feature = "wav")]
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
#[cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]
pub(crate) fn open_for_overwrite(path: &Path) -> io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(false);
    // Read access as well if the file allows it, which a mapping of it needs.
    let mut readable = options.clone();
    readable.read(true);
    readable.open(path).or_else(|_| options.open(path))
}

/// Cut a file opened by [`open_for_overwrite`] at the current write position. Anything that is
/// not a regular file, such as a pipe, is left alone.
#[cfg(any(feature = "wav", feature = "flac", feature = "aiff"))]
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
#[cfg(any(feature = "wav", feature = "flac"))]
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
#[cfg(any(feature = "wav", feature = "flac"))]
const TILE_BYTES: usize = 64 * 1024;

/// Convert `count` frames from the channel runs in `runs` to `U`, writing them interleaved
/// little-endian at `out`. Each run holds one channel, or several already interleaved.
///
/// # Safety
///
/// `out` must be valid for `count * runs.len() * U::BYTES` bytes, and each run for `count` samples.
#[cfg(any(feature = "wav", feature = "flac"))]
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
/// passes through a buffer of [`TILE_BYTES`], so no copy of the signal is made. `prefix`, such as a
/// header, is written in the same call as the first samples. Returns the number of sample bytes
/// written. `file_offset` is where the samples start in the destination: each tile
/// is placed a little above that offset within a page, which makes the kernel's copy into the
/// page cache about 15% faster than when the source's page offset is the lower of the two.
#[cfg(any(feature = "wav", feature = "flac"))]
pub(crate) fn write_interleaved_le<T, U, W>(
    writer: &mut W,
    prefix: &[u8],
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
        writer.write_all(prefix)?;
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
        write_both(writer, prefix, bytes)?;
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

    let source = FrameSource::<T, U>::new(view);
    let mut tails: Vec<&[T]> = Vec::with_capacity(channels);
    let mut start = 0usize;
    let mut prefix = prefix;
    while start < frames {
        let count = tile_frames.min(frames - start);
        let bytes = count * frame_bytes;
        let position = file_offset as usize + start * frame_bytes;
        // SAFETY: the buffer has room for a page and a bit beyond the tile.
        let tile = unsafe { tile_base.add(position % PAGE + ABOVE) };
        // SAFETY: the tile holds `count` frames, and they lie within the view.
        unsafe { source.pack(start, count, tile, &mut tails) };
        // SAFETY: every one of these bytes was written above.
        write_both(writer, prefix, unsafe { std::slice::from_raw_parts(tile, bytes) })?;
        prefix = &[];
        start += count;
    }
    Ok(total)
}

/// Write `first` and then `second`, in one call where the writer takes both.
#[cfg(any(feature = "wav", feature = "flac"))]
fn write_both<W: Write>(writer: &mut W, first: &[u8], second: &[u8]) -> io::Result<()> {
    let (mut first, mut second) = (first, second);
    while !first.is_empty() {
        let n = writer.write_vectored(&[io::IoSlice::new(first), io::IoSlice::new(second)])?;
        if n == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        if n < first.len() {
            first = &first[n..];
        } else {
            second = &second[n - first.len()..];
            first = &[];
        }
    }
    writer.write_all(second)
}

/// Frames of a view, for packing into the file's layout in any order.
#[cfg(any(feature = "wav", feature = "flac"))]
struct FrameSource<'a, T, U> {
    view: ArrayView2<'a, T>,
    /// The view's memory, when interleaved.
    interleaved: Option<&'a [T]>,
    /// Each channel's memory, when planar.
    rows: Option<Vec<&'a [T]>>,
    /// In memory the samples already are as the file has them.
    same_layout: bool,
    packed_i24: bool,
    _to: PhantomData<U>,
}

#[cfg(any(feature = "wav", feature = "flac"))]
impl<'a, T, U> FrameSource<'a, T, U>
where
    T: StandardSample + ConvertTo<U> + 'static,
    U: StandardSample + 'static,
{
    fn new(view: ArrayView2<'a, T>) -> Self {
        let (channels, _) = view.dim();
        // Interleaved memory: channel stride 1, frame stride `channels`.
        let interleaved = if channels == 1 || view.strides() == [1, channels as isize] {
            view.to_slice_memory_order()
        } else {
            None
        };
        let rows = if interleaved.is_none() {
            (0..channels)
                .map(|c| view.index_axis_move(Axis(0), c).to_slice())
                .collect()
        } else {
            None
        };
        Self {
            view,
            interleaved,
            rows,
            same_layout: cfg!(target_endian = "little")
                && interleaved.is_some()
                && TypeId::of::<T>() == TypeId::of::<U>()
                && TypeId::of::<T>() != TypeId::of::<I24>(),
            packed_i24: TypeId::of::<T>() == TypeId::of::<I24>() && TypeId::of::<U>() == TypeId::of::<I24>(),
            _to: PhantomData,
        }
    }

    /// Write `count` frames from frame `start` to `out`, as the file has them.
    ///
    /// # Safety
    ///
    /// `out` holds `count` frames of `U`, and the frames lie within the view. `tails` is scratch.
    unsafe fn pack(&self, start: usize, count: usize, out: *mut u8, tails: &mut Vec<&'a [T]>) {
        let (channels, _) = self.view.dim();
        let size = U::BYTES as usize;
        if let Some(slice) = self.interleaved {
            let samples = &slice[start * channels..(start + count) * channels];
            if self.same_layout {
                // SAFETY: T is plain numeric and already laid out as the file has it.
                unsafe { std::ptr::copy_nonoverlapping(samples.as_ptr().cast::<u8>(), out, samples.len() * size) };
            } else if self.packed_i24 {
                // SAFETY: T is I24, checked in `new`, and `out` holds three bytes per sample.
                unsafe {
                    crate::types::pack_i24(
                        std::slice::from_raw_parts(samples.as_ptr().cast::<I24>(), samples.len()),
                        out,
                    );
                }
            } else {
                // SAFETY: `out` holds `count * channels` samples, and so does the slice.
                unsafe { pack_frames::<T, U>(&[samples], count * channels, out) };
            }
        } else if let Some(rows) = &self.rows {
            tails.clear();
            tails.extend(rows.iter().map(|r| &r[start..]));
            // SAFETY: `out` holds `count` frames and each tail that many samples.
            unsafe { pack_frames::<T, U>(tails, count, out) };
        } else {
            let mut p = out;
            for f in start..start + count {
                for c in 0..channels {
                    let converted: U = self.view[[c, f]].convert_to();
                    let encoded = converted.to_le_bytes();
                    // SAFETY: at most `count` frames of bytes are written to `out`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(encoded.as_ref().as_ptr(), p, size);
                        p = p.add(size);
                    }
                }
            }
        }
    }
}

/// Bytes of samples from which [`write_samples_mapped`] is used on tmpfs: at 7 MiB it was level
/// with one write.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
const MAPPED_MIN_BYTES: usize = 8 * 1024 * 1024;

/// Bytes of samples from which [`write_samples_mapped`] is used on ext4: at 4 MiB it took 1.06 times
/// as long as one write, at 7 MiB 0.79 times, and a 6.7 MiB save 0.80 times as long as one write did.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
const MAPPED_EXT4_MIN_BYTES: usize = 6 * 1024 * 1024;

/// Bytes of samples from which [`write_samples_mapped`] is used on xfs, where a 6.7 MiB save took
/// 1.09 times and a 10 MiB save 0.95 times as long as one write did, and saves of 20 MiB and more
/// 0.74 to 0.83 times.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
const MAPPED_XFS_MIN_BYTES: usize = 16 * 1024 * 1024;

/// The file system holding a file, where that changes how the file is best read or written.
#[cfg(feature = "wav")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FileSystem {
    /// tmpfs, in memory; a write on it runs on one thread at a time.
    Tmpfs,
    /// ext4.
    Ext4,
    /// xfs.
    Xfs,
    /// btrfs.
    Btrfs,
    /// Anything else.
    Other,
}

#[cfg(feature = "wav")]
impl FileSystem {
    pub(crate) const fn is_ext4_or_xfs(self) -> bool {
        matches!(self, Self::Ext4 | Self::Xfs)
    }
}

#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
pub(crate) fn file_system(file: &File) -> FileSystem {
    use std::os::fd::AsRawFd;
    const TMPFS_MAGIC: i64 = 0x0102_1994;
    const EXT4_SUPER_MAGIC: i64 = 0xEF53;
    const XFS_SUPER_MAGIC: i64 = 0x5846_5342;
    const BTRFS_SUPER_MAGIC: i64 = 0x9123_683E;
    // SAFETY: an all-zero value is valid storage for fstatfs to fill.
    let mut info: libc::statfs = unsafe { mem::zeroed() };
    // SAFETY: the descriptor is open and `info` is valid storage.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &raw mut info) } != 0 {
        return FileSystem::Other;
    }
    match info.f_type as i64 {
        TMPFS_MAGIC => FileSystem::Tmpfs,
        EXT4_SUPER_MAGIC => FileSystem::Ext4,
        XFS_SUPER_MAGIC => FileSystem::Xfs,
        BTRFS_SUPER_MAGIC => FileSystem::Btrfs,
        _ => FileSystem::Other,
    }
}

#[cfg(feature = "wav")]
#[cfg(not(target_os = "linux"))]
pub(crate) const fn file_system(_file: &File) -> FileSystem {
    FileSystem::Other
}

/// How a whole-file read takes a WAV file: how many bytes the first read, from byte 0, takes (the
/// header comes from it, with the first samples), and whether to ask the kernel for the whole file
/// first (`posix_fadvise(WILLNEED)`). The rest of the file follows in one more read, on several
/// threads where it is in the page cache.
///
/// A file in the page cache costs no trip to the disk, and its first read is only the header's 4 KiB,
/// so that the threads take the rest. A small file off the disk is read whole; a larger one also
/// starts with 4 KiB, since a first read of 1 MiB made cold reads of 3.5 to 10 MB files on btrfs 0.6
/// to 3% slower. The third value says whether the file is off the disk.
///
/// Measured over ten files per size, against a header read followed by a read of the samples:
/// - ext4 and xfs up to 1 MiB: asking first and reading whole took 0.86 to 0.94 times as long;
/// - ext4 and xfs from 1 to 2 MiB: reading whole took 0.97 to 0.98 times as long, where asking
///   first was slower; at 3.5 MiB one read of the whole file took 1.01 to 1.03 times as long;
/// - btrfs at 512 KiB: asking first and reading whole took 0.92 times as long, but from 1 MiB both
///   were level or slower, up to 1.09 times at 4 MiB; files up to 768 KiB are asked for whole.
#[cfg(all(feature = "wav", feature = "numpy"))]
pub(crate) fn whole_read_plan(file: &File, len: u64) -> (usize, bool, bool) {
    const HEADER: u64 = 4096;
    let cold = len > 0 && range_is_cold(file, 0, len as usize);
    if !cold {
        return (len.min(HEADER) as usize, false, false);
    }
    match whole_read_plan_on(file, len, file_system(file)) {
        Some(will_need) => (len as usize, will_need, true),
        None => (len.min(HEADER) as usize, false, true),
    }
}

/// Whether a cold file of `len` bytes on `fs` is read whole, and asked for whole first.
#[cfg(feature = "wav")]
fn whole_read_plan_on(_file: &File, len: u64, fs: FileSystem) -> Option<bool> {
    const MIB: u64 = 1024 * 1024;
    if len == 0 || len > 2 * MIB {
        return None;
    }
    match fs {
        FileSystem::Ext4 | FileSystem::Xfs => Some(len <= MIB),
        FileSystem::Btrfs if len <= 3 * MIB / 4 => Some(true),
        _ => None,
    }
}

/// Ask the kernel to read all of `file` into the page cache.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
pub(crate) fn will_need(file: &File) {
    use std::os::fd::AsRawFd;
    // SAFETY: the descriptor is open; the call only gives advice.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED) };
}

#[cfg(feature = "wav")]
#[cfg(not(target_os = "linux"))]
pub(crate) const fn will_need(_file: &File) {}

/// How [`write_samples_mapped`] prepares a file.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum MappedFill {
    /// Size the file and populate each part before filling it.
    Memory,
    /// Allocate the file's blocks, then fill it.
    Disk,
}

/// Fill a large new file through a mapping, from several threads, with the samples of `view` in
/// the file's layout at `file_offset`, preceded by `header`, which is that long. True if done; on false
/// nothing is written that counts, the position is unchanged, and the ordinary write is left to
/// report what is wrong.
///
/// On tmpfs a write runs on one thread at a time and takes 20 to 60% longer; each thread populates
/// its pages before touching them, so a full filesystem is an error here and not a signal. On ext4
/// and xfs the file's blocks are allocated first, so the threads only copy into the page cache: a
/// write of 10 to 100 MiB takes 1.3 to 1.8 times as long, and reaching the disk takes about as long. The
/// file must hold no more than its header there, since a page of old data would be read in before
/// being overwritten. Only another process truncating the file during the write can raise a signal.
#[cfg(feature = "wav")]
#[cfg(target_os = "linux")]
pub(crate) fn write_samples_mapped<T, U>(
    file: &mut File,
    header: &[u8],
    view: ArrayView2<'_, T>,
    file_offset: u64,
) -> io::Result<bool>
where
    T: StandardSample + ConvertTo<U> + Sync + 'static,
    U: StandardSample + 'static,
{
    let (channels, frames) = view.dim();
    let frame_bytes = channels * U::BYTES as usize;
    let total = frames * frame_bytes;
    let permit = ThreadsPermit::try_acquire();
    let threads = permit.threads(transfer_threads());
    if total < MAPPED_EXT4_MIN_BYTES.min(MAPPED_MIN_BYTES) {
        return Ok(false);
    }
    let (fill, least) = match file_system(file) {
        FileSystem::Tmpfs => (MappedFill::Memory, MAPPED_MIN_BYTES),
        FileSystem::Ext4 => (MappedFill::Disk, MAPPED_EXT4_MIN_BYTES),
        FileSystem::Xfs => (MappedFill::Disk, MAPPED_XFS_MIN_BYTES),
        FileSystem::Btrfs | FileSystem::Other => return Ok(false),
    };
    if total < least || threads < 4 {
        return Ok(false);
    }
    use std::os::fd::AsRawFd;
    let offset = file_offset as usize;
    if header.len() != offset {
        return Ok(false);
    }
    let end = offset + total;
    let before = file.metadata()?.len();
    match fill {
        MappedFill::Memory => {
            if before < end as u64 {
                file.set_len(end as u64)?;
            }
        },
        MappedFill::Disk => {
            if before > file_offset {
                return Ok(false);
            }
            // SAFETY: the descriptor is open; the call allocates blocks and extends the file.
            if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, end as libc::off_t) } != 0 {
                file.set_len(before)?;
                return Ok(false);
            }
        },
    }
    // SAFETY: a new shared mapping of the open file, within its length, unmapped below.
    let map = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            end,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if map == libc::MAP_FAILED {
        file.set_len(before)?;
        return Ok(false);
    }
    let base = map as usize;
    // The header goes into the mapping too, so that the file takes no write call at all.
    // SAFETY: the first page lies in the mapping; populating it changes no contents.
    if fill == MappedFill::Memory && unsafe { libc::madvise(map, 4096.min(end), libc::MADV_POPULATE_WRITE) } != 0 {
        // SAFETY: the mapping was made above.
        unsafe { libc::munmap(map, end) };
        file.set_len(before)?;
        return Ok(false);
    }
    // SAFETY: the header is `offset` bytes, checked above, and the mapping holds `end` > `offset`.
    unsafe { std::ptr::copy_nonoverlapping(header.as_ptr(), base as *mut u8, header.len()) };
    let source = FrameSource::<T, U>::new(view);
    let chunk = frames.div_ceil(threads).max(1);
    let source = &source;
    let done = run_on_threads(frames.div_ceil(chunk), |index| {
        let first = index * chunk;
        let last = (first + chunk).min(frames);
        let from = (base + offset + first * frame_bytes) & !4095;
        let to = (base + offset + last * frame_bytes)
            .next_multiple_of(4096)
            .min(base + end.next_multiple_of(4096));
        // SAFETY: the range lies in the mapping, and populating pages changes no contents. Disk
        // blocks were allocated above, and populating them first makes the copy slower.
        if fill == MappedFill::Memory
            && unsafe { libc::madvise(from as *mut libc::c_void, to - from, libc::MADV_POPULATE_WRITE) } != 0
        {
            return false;
        }
        let mut tails = Vec::with_capacity(channels);
        let step = (TILE_BYTES / frame_bytes).max(1);
        let mut at = first;
        while at < last {
            let count = step.min(last - at);
            // SAFETY: these frames lie in the part of the mapping given to this index alone.
            unsafe { source.pack(at, count, (base + offset + at * frame_bytes) as *mut u8, &mut tails) };
            at += count;
        }
        true
    });
    // SAFETY: the mapping was made above, and the threads have gone.
    unsafe { libc::munmap(map, end) };
    if !done {
        file.set_len(before)?;
        return Ok(false);
    }
    file.seek(io::SeekFrom::Start(end as u64))?;
    Ok(true)
}

#[cfg(feature = "wav")]
#[cfg(not(target_os = "linux"))]
pub(crate) fn write_samples_mapped<T, U>(
    _file: &mut File,
    _header: &[u8],
    _view: ArrayView2<'_, T>,
    _file_offset: u64,
) -> io::Result<bool>
where
    T: StandardSample + ConvertTo<U> + Sync + 'static,
    U: StandardSample + 'static,
{
    Ok(false)
}
