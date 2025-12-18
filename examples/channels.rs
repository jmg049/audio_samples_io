use std::time::{Duration, Instant};

use audio_samples_io::error::AudioIOResult;
use audio_samples::{AudioEditing, AudioSamples, AudioStatistics, cosine_wave, sine_wave};

pub fn main() -> AudioIOResult<()> {
    let test_signal =
        sine_wave::<i16, f32>(440.0, Duration::from_secs_f64(60.0 * 60.0), 44100, 0.5);
    let test_signal_two =
        cosine_wave::<i16, f32>(440.0, Duration::from_secs_f64(60.0 * 60.0), 44100, 1.0);
    let test_signal = AudioEditing::stack(&[test_signal, test_signal_two])?;
    audio_samples_io::write("channels.wav", &test_signal)?;
    let t0 = Instant::now();
    let wav: AudioSamples<i16> = audio_samples_io::read("channels.wav")?;

    for (idx, ch) in wav.channels().enumerate() {
        let max = ch.max_sample();
        println!("Channel {}: Max sample value: {}", idx + 1, max);
    }

    let t1 = Instant::now();
    let dt = t1 - t0;
    println!("Elapsed {:.3}ms", dt.as_millis());
    Ok(())
}
