
#[cfg(feature = "flac")]
use audio_samples_io::error::AudioIOResult;

#[cfg(feature = "flac")]
pub fn main() -> AudioIOResult<()> {
    use std::time::Duration;

    use audio_samples::{AudioSamples, sample_rate, sine_wave};

    // create and write a basic signal and read it back
    let sine_wave: AudioSamples<f32> = sine_wave::<f32>(
        440.0,
        Duration::from_secs_f64(10.0),
        sample_rate!(44100),
        1.0,
    );
    println!("Original: {sine_wave:#}");
    audio_samples_io::write("./sine_test.flac", &sine_wave)?;
    let read_sine_wave: AudioSamples<f32> = audio_samples_io::read("./sine_test.flac")?;
    
    // FLAC is integer-only (24-bit). f32 → 24-bit int → f32 is lossy by ~1.2e-7 per sample.
    // Use a tolerance matching 1 LSB of 24-bit quantization (1/8388607 ≈ 1.19e-7).
    const TOLERANCE: f32 = 2e-7;
    let mut mismatches = 0usize;
    for (i, (a, b)) in sine_wave.frames().zip(read_sine_wave.frames()).enumerate() {
        let a_val = a.to_interleaved_vec()[0];
        let b_val = b.to_interleaved_vec()[0];
        let diff = (a_val - b_val).abs();
        if diff > TOLERANCE {
            eprintln!("Frame {i}: {a_val:.8} vs {b_val:.8} (diff={diff:.2e})");
            mismatches += 1;
            if mismatches >= 10 { break; }
        }
    }
    if mismatches == 0 {
        println!("All frames match within {TOLERANCE:.1e} tolerance.");
    }
    println!("{read_sine_wave:#}");
    println!("Duration: {:.1}s", read_sine_wave.duration_seconds());
    Ok(())
}

#[cfg(not(feature = "flac"))]
pub fn main() -> () {
    use std::process::exit;
    eprintln!("FLAC feature not enabled");
    exit(1);
}