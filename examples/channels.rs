use std::time::{Duration, Instant};

use audio_samples::{
    AudioEditing, AudioSamples, AudioStatistics, cosine_wave, sample_rate, sine_wave,
};
use audio_samples_io::error::AudioIOResult;
use non_empty_slice::NonEmptyVec;

pub fn main() -> AudioIOResult<()> {
    let test_signal = sine_wave::<i16>(
        440.0,
        Duration::from_secs_f64(60.0 * 60.0),
        sample_rate!(44100),
        0.5,
    );
    let test_signal_two = cosine_wave::<i16>(
        440.0,
        Duration::from_secs_f64(60.0 * 60.0),
        sample_rate!(44100),
        1.0,
    );
    let sources =
        NonEmptyVec::new(vec![test_signal, test_signal_two]).expect("sources must be non-empty");
    let test_signal = AudioEditing::stack(&sources)?;
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
