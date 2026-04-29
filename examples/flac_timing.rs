//! Quick wall-clock timing for the FLAC write hot path.
//! Run with: cargo run --example flac_timing --features "wav,flac" --release

#[cfg(feature = "flac")]
use std::{hint::black_box, io::Cursor, num::NonZeroU32, time::Instant};

#[cfg(feature = "flac")]
use audio_samples::sine_wave;

#[cfg(feature = "flac")]
use audio_samples_io::flac::{CompressionLevel, write_flac};

#[cfg(feature = "flac")]
fn main() {
    let sr = NonZeroU32::new(44100).unwrap();
    let duration = std::time::Duration::from_millis(250);
    let audio = sine_wave::<i16>(110.0, duration, sr, 0.5);

    // warm up
    for _ in 0..200 {
        let w = Cursor::new(Vec::with_capacity(32768));
        black_box(write_flac(w, &audio, CompressionLevel::DEFAULT).unwrap());
    }

    // measure
    let iters = 500u64;
    let t0 = Instant::now();
    for _ in 0..iters {
        let w = Cursor::new(Vec::with_capacity(32768));
        black_box(write_flac(w, &audio, CompressionLevel::DEFAULT).unwrap());
    }
    let elapsed = t0.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iters as f64;
    println!("flac_write 44100hz 1ch i16 250ms: {avg_us:.1} µs/iter  ({iters} iters, {elapsed:.2?} total)");
}

#[cfg(not(feature = "flac"))]
fn main() {
    println!("FLAC feature not enabled");
}