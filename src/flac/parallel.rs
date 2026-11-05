//! Decoding a whole FLAC stream on several threads.
//!
//! Frames are self-contained, so the compressed bytes are divided at frame boundaries found by
//! scanning for a frame header. Each thread decodes its range into its own part of the output.
//! Anything unexpected makes the whole attempt return false, and the caller decodes in sequence,
//! which reports errors properly.

use audio_samples::{I24, traits::StandardSample};

use crate::fastio::{ThreadsPermit, decode_threads, run_on_threads};
use crate::flac::frame::{decode_frame_into_scratch, parse_frame_position};
use crate::flac::metadata::StreamInfo;

/// Compressed bytes per thread below which a thread would not pay for itself.
const MIN_BYTES_PER_THREAD: usize = 256 * 1024;

/// One FLAC sample as `T`.
#[inline(always)]
pub(crate) fn sample_from_flac<T: StandardSample + 'static>(s: i32, bits: u8) -> T {
    match bits {
        1..=8 => T::convert_from((s << (16 - u32::from(bits))) as i16),
        9..=16 => T::convert_from(s as i16),
        17..=24 => T::convert_from(I24::wrapping_from_i32(s)),
        _ => T::convert_from(s),
    }
}

/// The offset of the first frame header at or after `from`, with its first sample.
fn next_frame(data: &[u8], from: usize, info: &StreamInfo) -> Option<(usize, u64)> {
    let fixed = u32::from(info.max_block_size);
    let channels = info.channels as usize;
    let mut at = from;
    while at + 1 < data.len() {
        // The sync code is 14 ones and a reserved zero, then the blocking strategy.
        let offset = data[at..data.len() - 1]
            .iter()
            .zip(&data[at + 1..])
            .position(|(&a, &b)| a == 0xFF && (b & 0xFE) == 0xF8)?;
        at += offset;
        if let Some((first, _)) = parse_frame_position(&data[at..], fixed, channels, info.bits_per_sample) {
            return Some((at, first));
        }
        at += 1;
    }
    None
}

/// Whether `data` is large enough for [`decode_planar_parallel`] to help.
pub(crate) fn worth_decoding_in_parallel(data: &[u8]) -> bool {
    decode_threads() > 1 && data.len() >= 2 * MIN_BYTES_PER_THREAD
}

/// Decode the frames in `data` into `out`, planar with `total` samples per channel. True if every
/// frame decoded and together they cover exactly `total` samples. `consumed(range)` is called now
/// and then with a range of `data` that thread has finished with.
///
/// # Safety
///
/// `out` must be valid for writes of `channels * total` values of `T`.
pub(crate) unsafe fn decode_planar_parallel<T>(
    data: &[u8],
    info: &StreamInfo,
    out: *mut T,
    total: usize,
    consumed: &(dyn Fn(std::ops::Range<usize>) + Sync),
) -> bool
where
    T: StandardSample + 'static,
{
    /// Bytes a thread decodes between calls to `consumed`.
    const RELEASE_BYTES: usize = 32 * 1024;

    let channels = info.channels as usize;
    let permit = ThreadsPermit::try_acquire();
    let threads = permit.threads(decode_threads().min(data.len() / MIN_BYTES_PER_THREAD).max(1));
    if channels == 0 || threads < 2 {
        return false;
    }

    // Range starts: the first frame, then the first frame after each even division of the bytes.
    let mut starts: Vec<(usize, u64)> = Vec::with_capacity(threads + 1);
    starts.push((0, 0));
    for t in 1..threads {
        let Some(found) = next_frame(data, data.len() / threads * t, info) else {
            return false;
        };
        if found.0 > starts[starts.len() - 1].0 {
            starts.push(found);
        }
    }
    let ranges = starts.len();
    let starts = starts.as_slice();
    let base = out as usize;
    let bits = info.bits_per_sample;

    run_on_threads(ranges, |index| {
        let (mut at, first) = starts[index];
        let (end, end_sample) = match starts.get(index + 1) {
            Some(&(offset, sample)) => (offset, sample as usize),
            None => (data.len(), total),
        };
        let mut sample = first as usize;
        let mut released = at;
        let block_hint = info.max_block_size as usize;
        let mut scratch: Vec<Vec<i32>> = (0..channels).map(|_| Vec::with_capacity(block_hint)).collect();
        while at < end && sample < end_sample {
            let Ok(used) = decode_frame_into_scratch(&data[at..], info.sample_rate, bits, &mut scratch) else {
                return false;
            };
            let block = scratch[0].len();
            if sample + block > end_sample || scratch.iter().any(|c| c.len() != block) {
                return false;
            }
            for (ch, samples) in scratch.iter().enumerate() {
                // SAFETY: sample + block <= total, so this is inside channel `ch` of the output,
                // and ranges of samples are disjoint between threads.
                let dst = unsafe { std::slice::from_raw_parts_mut((base as *mut T).add(ch * total + sample), block) };
                for (d, &s) in dst.iter_mut().zip(samples) {
                    *d = sample_from_flac::<T>(s, bits);
                }
            }
            sample += block;
            at += used;
            if at - released >= RELEASE_BYTES {
                consumed(released..at);
                released = at;
            }
        }
        sample == end_sample && (index + 1 == ranges || at == end)
    })
}
