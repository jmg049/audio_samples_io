//! A Rice quotient that ends at the last bit of the 64-bit read buffer.

#![cfg(all(feature = "flac", feature = "wav"))]

/// Two seconds of low-level noise with sparse large spikes, encoded by the reference
/// `flac` encoder at level 0. Its large residuals produce such a quotient.
#[test]
fn flac_decodes_to_the_source_pcm() {
    let flac = audio_samples_io::read::<_, i16>("resources/rice_quotient_at_buffer_end.flac").unwrap();
    let wav = audio_samples_io::read::<_, i16>("resources/rice_quotient_at_buffer_end.wav").unwrap();
    assert_eq!(flac.to_interleaved_vec(), wav.to_interleaved_vec());
}
