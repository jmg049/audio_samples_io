//! Decoding a whole FLAC file on several threads, each reading its part of the file through a
//! small window instead of a mapping.

use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use audio_samples::traits::StandardSample;

use super::{MIN_BYTES_PER_THREAD, sample_from_flac};
use crate::fastio::{ThreadsPermit, decode_threads, run_on_threads};
use crate::flac::frame::{decode_frame_into_scratch, parse_frame_position};
use crate::flac::metadata::StreamInfo;

/// Compressed bytes read at a time beyond what a frame can need.
const READ_AHEAD_BYTES: usize = 8 * 1024;

/// One thread's compressed bytes: a part of the file it reads ahead of where it decodes.
struct Window {
    bytes: Vec<u8>,
    /// Bytes a frame can take; less than this before the end of the file calls for a read.
    frame_bytes: usize,
    /// Offset in the file of the first byte.
    at: u64,
}

impl Window {
    /// The bytes from `offset` on, a frame's worth of them unless the file ends first.
    fn from(&mut self, file: &File, offset: u64) -> &[u8] {
        let end = self.at + self.bytes.len() as u64;
        let keep = if offset >= self.at && offset <= end {
            (end - offset) as usize
        } else {
            0
        };
        if keep < self.frame_bytes {
            let held = self.bytes.len();
            self.bytes.copy_within(held - keep.., 0);
            let room = self.bytes.capacity();
            // SAFETY: the capacity is `room`; bytes past `keep` are written before `set_len` counts them.
            let target = unsafe { std::slice::from_raw_parts_mut(self.bytes.as_mut_ptr().add(keep), room - keep) };
            let mut got = 0;
            while got < target.len() {
                match read_at(file, offset + (keep + got) as u64, &mut target[got..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => got += n,
                }
            }
            // SAFETY: `keep + got` bytes are initialised.
            unsafe { self.bytes.set_len(keep + got) };
            self.at = offset;
        }
        &self.bytes[(offset - self.at) as usize..]
    }
}

#[cfg(unix)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _offset: u64, _buf: &mut [u8]) -> std::io::Result<usize> {
    Err(std::io::ErrorKind::Unsupported.into())
}

/// The first frame header at or after `from` in the file, with its first sample.
fn next_frame_in_file(file: &File, window: &mut Window, from: u64, end: u64, info: &StreamInfo) -> Option<(u64, u64)> {
    const HEADER_BYTES: usize = 32;
    let fixed = u32::from(info.max_block_size);
    let channels = info.channels as usize;
    let mut at = from;
    while at + 1 < end {
        let data = window.from(file, at);
        if data.len() < 2 {
            return None;
        }
        let limit = data.len().saturating_sub(HEADER_BYTES).max(1);
        for offset in 0..limit {
            if data[offset] == 0xFF
                && (data[offset + 1] & 0xFE) == 0xF8
                && let Some((first, _)) = parse_frame_position(&data[offset..], fixed, channels, info.bits_per_sample)
            {
                return Some((at + offset as u64, first));
            }
        }
        at += limit as u64;
    }
    None
}

/// What one thread decodes with.
struct Workspace {
    window: Window,
    scratch: Vec<Vec<i32>>,
}

impl Workspace {
    fn new(channels: usize, block: usize, bits: u8) -> Self {
        let frame_bytes = channels * block * (bits as usize + 1) / 8 + 64 * channels;
        Self {
            window: Window {
                bytes: Vec::with_capacity(frame_bytes + READ_AHEAD_BYTES),
                frame_bytes,
                at: 0,
            },
            scratch: (0..channels).map(|_| Vec::with_capacity(block)).collect(),
        }
    }
}

/// Decode the frames from `at` up to `stop`, the first holding sample `sample`, into channels of
/// `total` samples at `base`, until sample `end_sample` has been written. A `stop` of `u64::MAX` leaves
/// open where the last frame ends, since a file can carry other bytes after it.
///
/// # Safety
///
/// `base` is valid for `channels * total` values of `T`, and no one else uses the range of samples.
#[allow(clippy::too_many_arguments)]
unsafe fn decode_range<T>(
    file: &File,
    work: &mut Workspace,
    info: &StreamInfo,
    mut at: u64,
    stop: u64,
    mut sample: usize,
    end_sample: usize,
    base: usize,
    total: usize,
) -> bool
where
    T: StandardSample + 'static,
{
    let bits = info.bits_per_sample;
    while at < stop && sample < end_sample {
        let data = work.window.from(file, at);
        let Ok(used) = decode_frame_into_scratch(data, info.sample_rate, bits, &mut work.scratch) else {
            return false;
        };
        let block = work.scratch[0].len();
        if sample + block > end_sample || work.scratch.iter().any(|c| c.len() != block) {
            return false;
        }
        for (ch, samples) in work.scratch.iter().enumerate() {
            // SAFETY: sample + block <= end_sample <= total, so this is inside channel `ch`.
            let dst = unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(ch * total + sample), block) };
            for (d, &s) in dst.iter_mut().zip(samples) {
                *d = sample_from_flac::<T>(s, bits);
            }
        }
        sample += block;
        at += used as u64;
    }
    sample == end_sample && (at == stop || stop == u64::MAX)
}

/// Decode the frames of a file between `start` and `end` into planar output, read through small
/// windows instead of a mapping, on threads when the file is large enough. False means the stream
/// needs the sequential decoder's error handling, and the output is to be overwritten.
///
/// # Safety
///
/// `out` must be valid for writes of `channels * total` values of `T`, and aligned.
pub(crate) unsafe fn decode_planar_file<T>(
    path: &Path,
    start: u64,
    end: u64,
    info: &StreamInfo,
    out: *mut T,
    total: usize,
) -> bool
where
    T: StandardSample + 'static,
{
    let channels = info.channels as usize;
    let block = info.max_block_size as usize;
    if channels == 0 || block == 0 {
        return false;
    }
    let Ok(file) = File::open(path) else {
        return false;
    };
    let compressed = end.saturating_sub(start);
    let base = out as usize;
    let bits = info.bits_per_sample;
    let permit = ThreadsPermit::try_acquire();
    let wanted = decode_threads().min(compressed as usize / MIN_BYTES_PER_THREAD).max(1);
    let threads = if crate::fastio::range_is_cold(&file, start, compressed as usize) {
        1
    } else {
        permit.threads(wanted)
    };
    let mut own = Workspace::new(channels, block, bits);
    if threads < 2 {
        // SAFETY: the caller's guarantee covers the whole output.
        return unsafe { decode_range::<T>(&file, &mut own, info, start, u64::MAX, 0, total, base, total) };
    }

    let mut starts: Vec<(u64, u64)> = Vec::with_capacity(threads + 1);
    starts.push((start, 0));
    for t in 1..threads as u64 {
        let Some(found) = next_frame_in_file(
            &file,
            &mut own.window,
            start + compressed / threads as u64 * t,
            end,
            info,
        ) else {
            return false;
        };
        if found.0 > starts[starts.len() - 1].0 {
            starts.push(found);
        }
    }
    let ranges = starts.len();
    let spots = starts.as_slice();
    // Set up here, so that no thread's first allocation creates a malloc arena.
    let workers: Vec<Mutex<Workspace>> = (0..ranges)
        .map(|_| Mutex::new(Workspace::new(channels, block, bits)))
        .collect();
    let workers = workers.as_slice();
    let file_ref = &file;

    run_on_threads(ranges, |index| {
        let (at, first) = spots[index];
        let (stop, end_sample) = spots
            .get(index + 1)
            .map_or((u64::MAX, total), |&(o, s)| (o, s as usize));
        let Ok(mut work) = workers[index].lock() else {
            return false;
        };
        // SAFETY: threads get disjoint ranges of samples.
        unsafe {
            decode_range::<T>(
                file_ref,
                &mut work,
                info,
                at,
                stop,
                first as usize,
                end_sample,
                base,
                total,
            )
        }
    })
}
