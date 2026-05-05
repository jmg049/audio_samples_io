/// WAV write tests — two layers:
///   1. ours → hound: we write, hound reads back and asserts exact values.
///   2. ours → ours: roundtrip with exact value comparison.
///
/// Cross-reading with hound validates that our output is spec-compliant.
#[cfg(feature = "wav")]
mod wav_write {
    use audio_samples::traits::StandardSample;
    use audio_samples::{AudioSamples, I24, nzu, sample_rate};
    use audio_samples_io::traits::{AudioFile, AudioFileRead};
    use audio_samples_io::{OpenOptions, WavFile, write};
    use hound::{SampleFormat, WavReader};
    use std::io::Cursor;
    use std::path::Path;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("asio_write_test_{name}_{}.wav", std::process::id()))
    }

    fn mono<T: StandardSample + 'static>(samples: &[T], sr: u32) -> AudioSamples<'static, T> {
        let sr = std::num::NonZeroU32::new(sr).unwrap();
        let len = std::num::NonZeroUsize::new(samples.len()).unwrap();
        let mut audio = AudioSamples::zeros_mono(len, sr);
        if let Some(slice) = audio.as_slice_mut() {
            slice.copy_from_slice(samples);
        }
        audio
    }

    fn open(path: &Path) -> WavFile<'static> {
        <WavFile as AudioFile>::open_with_options(path, OpenOptions::default())
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()))
    }

    fn interleaved<T: StandardSample + 'static>(wav: &WavFile<'_>) -> Vec<T> {
        let audio = <WavFile as AudioFileRead>::read::<T>(wav).expect("read failed");
        audio.to_interleaved_vec().into_vec()
    }

    // ── write i16, verify with hound ──────────────────────────────────────────

    #[test]
    fn write_i16_mono_hound_reads_exact_values() {
        let path = temp_path("i16_mono_hound");
        let audio = mono(&[2i16, -3, 5, -7], 44100);
        write(&path, &audio).expect("write failed");

        let mut reader = WavReader::open(&path).expect("hound open failed");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 44100);
        assert_eq!(reader.spec().bits_per_sample, 16);
        assert_eq!(reader.spec().sample_format, SampleFormat::Int);
        let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples, vec![2, -3, 5, -7]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_i16_stereo_hound_reads_exact_values() {
        let path = temp_path("i16_stereo_hound");
        let sr = sample_rate!(44100);
        let left: AudioSamples<i16> = mono(&[2i16, 5, 11, 17], 44100);
        let right: AudioSamples<i16> = mono(&[-3i16, -7, -13, -19], 44100);
        let stereo = audio_samples::AudioEditing::stack(
            non_empty_slice::NonEmptySlice::new(&[left, right]).unwrap(),
        )
        .expect("stack failed");
        write(&path, &stereo).expect("write failed");

        let mut reader = WavReader::open(&path).expect("hound open failed");
        assert_eq!(reader.spec().channels, 2);
        assert_eq!(reader.spec().sample_rate, 44100);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
        // Interleaved: [L0,R0, L1,R1, ...]
        assert_eq!(samples, vec![2, -3, 5, -7, 11, -13, 17, -19]);

        let _ = sr; // silence unused warning
        std::fs::remove_file(&path).ok();
    }

    // ── write i32, verify with hound ──────────────────────────────────────────

    #[test]
    fn write_i32_mono_hound_reads_exact_values() {
        let path = temp_path("i32_mono_hound");
        let audio = mono(&[19i32, -229_373, 33_587_161, -2_147_483_497], 48000);
        write(&path, &audio).expect("write failed");

        let mut reader = WavReader::open(&path).expect("hound open failed");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 48000);
        assert_eq!(reader.spec().bits_per_sample, 32);
        let samples: Vec<i32> = reader.samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples, vec![19, -229_373, 33_587_161, -2_147_483_497]);

        std::fs::remove_file(&path).ok();
    }

    // ── write f32, verify with hound ──────────────────────────────────────────

    #[test]
    fn write_f32_mono_hound_reads_exact_values() {
        let path = temp_path("f32_mono_hound");
        let audio = mono(&[2.0f32, 3.0, -16411.0, 1019.0], 44100);
        write(&path, &audio).expect("write failed");

        let mut reader = WavReader::open(&path).expect("hound open failed");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 44100);
        assert_eq!(reader.spec().bits_per_sample, 32);
        assert_eq!(reader.spec().sample_format, SampleFormat::Float);
        let samples: Vec<f32> = reader.samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples, vec![2.0f32, 3.0, -16411.0, 1019.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_f32_min_max_hound_reads_exact() {
        let path = temp_path("f32_minmax_hound");
        let audio = mono(&[1.0f32, -1.0, 0.0, 0.5], 44100);
        write(&path, &audio).expect("write failed");

        let mut reader = WavReader::open(&path).expect("hound open failed");
        assert_eq!(reader.spec().sample_format, SampleFormat::Float);
        let samples: Vec<f32> = reader.samples().map(|s| s.unwrap()).collect();
        assert_eq!(samples, vec![1.0f32, -1.0, 0.0, 0.5]);

        std::fs::remove_file(&path).ok();
    }

    // ── write I24, roundtrip (hound doesn't support 24-bit) ───────────────────

    #[test]
    fn write_i24_roundtrip_exact_values() {
        let path = temp_path("i24_roundtrip");
        let sr = sample_rate!(48000);
        let len = nzu!(4);

        let values_i32: &[i32] = &[-17, 4_194_319, -6_291_437, 8_355_817];
        let i24_values: Vec<I24> = values_i32
            .iter()
            .map(|&v| {
                let bytes = v.to_le_bytes();
                I24::from_le_bytes([bytes[0], bytes[1], bytes[2]])
            })
            .collect();

        let mut audio = AudioSamples::<I24>::zeros_mono(len, sr);
        if let Some(s) = audio.as_slice_mut() {
            s.copy_from_slice(&i24_values);
        }
        write(&path, &audio).expect("write failed");

        let wav = open(&path);
        let read_back = <WavFile as AudioFileRead>::read::<I24>(&wav).expect("read failed");
        let read_i32: Vec<i32> = read_back
            .to_interleaved_vec()
            .iter()
            .map(|s| s.to_i32())
            .collect();
        assert_eq!(read_i32, values_i32);

        std::fs::remove_file(&path).ok();
    }

    // ── in-memory write (Cursor) ───────────────────────────────────────────────

    #[test]
    fn write_to_cursor_produces_valid_riff_header() {
        let audio = mono(&[1i16, 2, 3, 4], 44100);
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        audio_samples_io::write_with(cursor, &audio, audio_samples_io::FileType::WAV)
            .expect("write_with failed");

        assert!(buffer.len() > 44, "file too small");
        assert_eq!(&buffer[0..4], b"RIFF");
        assert_eq!(&buffer[8..12], b"WAVE");
    }

    #[test]
    fn write_to_cursor_then_read_back() {
        let original = mono(&[100i16, -200, 300, -400], 22050);
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        audio_samples_io::write_with(cursor, &original, audio_samples_io::FileType::WAV)
            .expect("write_with failed");

        let path = temp_path("cursor_readback");
        std::fs::write(&path, &buffer).expect("write to temp file failed");
        let wav = open(&path);
        assert_eq!(interleaved::<i16>(&wav), vec![100, -200, 300, -400]);

        std::fs::remove_file(&path).ok();
    }

    // ── RIFF chunk structure validation ───────────────────────────────────────

    #[test]
    fn written_file_has_fmt_and_data_chunks() {
        let path = temp_path("chunk_structure");
        let audio = mono(&[1i16, 2, 3, 4], 44100);
        write(&path, &audio).expect("write failed");

        let bytes = std::fs::read(&path).expect("read back failed");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        let mut found_fmt = false;
        let mut found_data = false;
        let mut pos = 12usize;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            if id == b"fmt " {
                found_fmt = true;
                assert!(size >= 16, "fmt chunk too small: {size}");
            }
            if id == b"data" {
                found_data = true;
                assert!(size > 0, "data chunk is empty");
            }
            pos += 8 + size + (size % 2);
        }
        assert!(found_fmt, "no fmt chunk in written file");
        assert!(found_data, "no data chunk in written file");

        std::fs::remove_file(&path).ok();
    }

    // ── roundtrip byte-exact for integer types ────────────────────────────────

    #[test]
    fn i16_roundtrip_bytes_identical() {
        let path = temp_path("i16_byte_roundtrip");
        let audio = mono(&[32767i16, -32768, 1000, -1000], 44100);
        write(&path, &audio).expect("write failed");

        let wav = open(&path);
        let read_back = interleaved::<i16>(&wav);
        assert_eq!(read_back, vec![32767, -32768, 1000, -1000]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn i32_roundtrip_bytes_identical() {
        let path = temp_path("i32_byte_roundtrip");
        let audio = mono(&[i32::MIN, i32::MAX, 0, -1], 44100);
        write(&path, &audio).expect("write failed");

        let wav = open(&path);
        let read_back = interleaved::<i32>(&wav);
        assert_eq!(read_back, vec![i32::MIN, i32::MAX, 0, -1]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_roundtrip_exact_bitwise() {
        let path = temp_path("f32_byte_roundtrip");
        let values = [2.0f32, 3.0, -16411.0, 1019.0];
        let audio = mono(&values, 44100);
        write(&path, &audio).expect("write failed");

        let wav = open(&path);
        let read_back = interleaved::<f32>(&wav);
        assert_eq!(read_back, values.to_vec());

        std::fs::remove_file(&path).ok();
    }

    // ── sample rate and channel count preserved ───────────────────────────────

    #[test]
    fn write_preserves_sample_rate() {
        for &sr in &[8000u32, 11025, 16000, 22050, 44100, 48000, 96000, 192000] {
            let path = temp_path(&format!("sr_{sr}"));
            let audio = mono(&[1i16, 2], sr);
            write(&path, &audio).expect("write failed");

            let reader = WavReader::open(&path).expect("hound open failed");
            assert_eq!(
                reader.spec().sample_rate,
                sr,
                "sample rate mismatch for {sr}"
            );

            std::fs::remove_file(&path).ok();
        }
    }

    // ── regression: WAVEFORMATEXTENSIBLE f32 correctly identified as F32 ──────

    #[test]
    fn extensible_f32_roundtrip_sample_type_is_f32() {
        use audio_samples::SampleType;
        use audio_samples_io::traits::AudioFileMetadata;

        // Our crate writes f32 > 2ch as WAVEFORMATEXTENSIBLE.
        // This test guards against the bug where the sub-format GUID was parsed
        // at the wrong byte offset (valid-bits field instead of GUID data).
        let path = temp_path("ext_f32_type");
        let sr = sample_rate!(44100);

        let ch1 = mono(&[1.0f32, 0.5, 0.0, -0.5], 44100);
        let ch2 = mono(&[0.5f32, 1.0, -0.5, 0.0], 44100);
        let ch3 = mono(&[0.0f32, 0.5, 1.0, -1.0], 44100);
        let three_ch = audio_samples::AudioEditing::stack(
            non_empty_slice::NonEmptySlice::new(&[ch1, ch2, ch3]).unwrap(),
        )
        .expect("stack failed");
        write(&path, &three_ch).expect("write failed");

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(
            info.sample_type,
            SampleType::F32,
            "extensible f32 parsed as wrong type"
        );
        assert_eq!(info.channels, 3);
        assert_eq!(info.bits_per_sample, 32);

        let _ = sr;
        std::fs::remove_file(&path).ok();
    }
}
