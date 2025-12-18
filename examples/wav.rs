use std::time::Duration;

use audio_samples_io::error::AudioIOResult;
use audio_samples::{AudioSamples, sine_wave};

pub fn main() -> AudioIOResult<()> {
    // create and write a basic signal and read it back
    let sine_wave: AudioSamples<f32> =
        sine_wave::<_, f32>(440.0, Duration::from_secs_f64(10.0), 44100, 1.0);
    audio_samples_io::write("./sine_wave.wav", &sine_wave)?;
    let read_sine_wave: AudioSamples<f32> = audio_samples_io::read("./sine_wave.wav")?;
    assert_eq!(
        sine_wave, read_sine_wave,
        "Written and read sine waves are not equal!"
    );
    println!("{:#}", read_sine_wave);
    println!("Duration: {:.1}s", read_sine_wave.duration_seconds::<f32>());
    Ok(())
}
