/// WAV read tests — two layers:
///   1. hound → ours: hound writes known-value fixtures, we read and assert exact values.
///   2. ours → ours: we write a fixture, read it back, assert exact values.
///
/// Using hound as the fixture generator cross-validates against the de-facto WAV reference.
#[cfg(feature = "wav")]
mod wav_read {
    use std::path::Path;

    use audio_samples::SampleType;
    use audio_samples_io::traits::{AudioFile, AudioFileMetadata, AudioFileRead};
    use audio_samples_io::{OpenOptions, WavFile};
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("asio_test_{name}_{}.wav", std::process::id()))
    }

    fn open(path: &Path) -> WavFile<'static> {
        <WavFile as AudioFile>::open_with_options(path, OpenOptions::default())
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()))
    }

    fn interleaved_i16(wav: &WavFile<'_>) -> Vec<i16> {
        let audio = <WavFile as AudioFileRead>::read::<i16>(wav).expect("read::<i16> failed");
        audio.to_interleaved_vec().into_vec()
    }

    fn interleaved_i32(wav: &WavFile<'_>) -> Vec<i32> {
        let audio = <WavFile as AudioFileRead>::read::<i32>(wav).expect("read::<i32> failed");
        audio.to_interleaved_vec().into_vec()
    }

    fn interleaved_f32(wav: &WavFile<'_>) -> Vec<f32> {
        let audio = <WavFile as AudioFileRead>::read::<f32>(wav).expect("read::<f32> failed");
        audio.to_interleaved_vec().into_vec()
    }

    // ── 16-bit PCM mono ──────────────────────────────────────────────────────

    #[test]
    fn read_16bit_pcm_mono_metadata() {
        let path = temp_path("16bit_mono_meta");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2i16, -3, 5, -7] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 44100);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.sample_type, SampleType::I16);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_16bit_pcm_mono_exact_values() {
        let path = temp_path("16bit_mono_vals");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2i16, -3, 5, -7] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_i16(&wav), vec![2, -3, 5, -7]);

        std::fs::remove_file(&path).ok();
    }

    // ── 16-bit PCM stereo ────────────────────────────────────────────────────

    #[test]
    fn read_16bit_pcm_stereo_metadata() {
        let path = temp_path("16bit_stereo_meta");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2i16, -3, 5, -7, 11, -13, 17, -19] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.sample_type, SampleType::I16);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_16bit_pcm_stereo_exact_values() {
        let path = temp_path("16bit_stereo_vals");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2i16, -3, 5, -7, 11, -13, 17, -19] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_i16(&wav), vec![2, -3, 5, -7, 11, -13, 17, -19]);

        std::fs::remove_file(&path).ok();
    }

    // ── 32-bit PCM mono ──────────────────────────────────────────────────────

    #[test]
    fn read_32bit_pcm_mono_metadata() {
        let path = temp_path("32bit_mono_meta");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[19i32, -229_373, 33_587_161, -2_147_483_497] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 48000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 32);
        assert_eq!(info.sample_type, SampleType::I32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_32bit_pcm_mono_exact_values() {
        let path = temp_path("32bit_mono_vals");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[19i32, -229_373, 33_587_161, -2_147_483_497] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_i32(&wav), vec![19, -229_373, 33_587_161, -2_147_483_497]);

        std::fs::remove_file(&path).ok();
    }

    // ── 32-bit PCM stereo ────────────────────────────────────────────────────

    #[test]
    fn read_32bit_pcm_stereo_exact_values() {
        let path = temp_path("32bit_stereo_vals");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[19i32, -229_373, 33_587_161, -2_147_483_497] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.channels, 2);
        assert_eq!(interleaved_i32(&wav), vec![19, -229_373, 33_587_161, -2_147_483_497]);

        std::fs::remove_file(&path).ok();
    }

    // ── IEEE float 32-bit mono ───────────────────────────────────────────────

    #[test]
    fn read_f32_mono_metadata() {
        let path = temp_path("f32_mono_meta");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2.0f32, 3.0, -16411.0, 1019.0] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 44100);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 32);
        assert_eq!(info.sample_type, SampleType::F32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_f32_mono_exact_values() {
        let path = temp_path("f32_mono_vals");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[2.0f32, 3.0, -16411.0, 1019.0] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_f32(&wav), vec![2.0f32, 3.0, -16411.0, 1019.0]);

        std::fs::remove_file(&path).ok();
    }

    // ── 8-bit PCM mono ───────────────────────────────────────────────────────
    // 8-bit WAV is unsigned (0-255, midpoint 128). Hound writes i16 values and
    // the 8-bit encoding uses an unsigned container.

    #[test]
    fn read_8bit_pcm_mono_metadata() {
        let path = temp_path("8bit_mono_meta");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 8,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        // Hound writes 8-bit as i16 (it handles the unsigned bias internally)
        for &s in &[19i16, -53, 89, -127] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 44100);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 8);
        assert_eq!(info.sample_type, SampleType::U8);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_8bit_pcm_mono_opens_and_has_samples() {
        let path = temp_path("8bit_mono_open");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 8,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[19i16, -53, 89, -127] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let audio = <WavFile as AudioFileRead>::read::<u8>(&wav).expect("read::<u8> failed");
        assert_eq!(audio.to_interleaved_vec().len().get(), 4);

        std::fs::remove_file(&path).ok();
    }

    // ── non-44100 sample rates ────────────────────────────────────────────────

    #[test]
    fn read_192khz_mono() {
        let path = temp_path("192khz_mono");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 192_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[100i16, -200, 300, -400] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 192_000);
        assert_eq!(interleaved_i16(&wav), vec![100, -200, 300, -400]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_11025hz_mono() {
        let path = temp_path("11025hz_mono");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 11025,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[1000i16, -2000, 3000, -4000] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.sample_rate.get(), 11025);
        assert_eq!(interleaved_i16(&wav), vec![1000, -2000, 3000, -4000]);

        std::fs::remove_file(&path).ok();
    }

    // ── existing real-world file ──────────────────────────────────────────────

    #[test]
    fn read_test_wav_opens_without_error() {
        let wav = open(Path::new("resources/test.wav"));
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert!(info.sample_rate.get() > 0);
        assert!(info.channels > 0);
        assert!(info.total_samples > 0);
    }

    #[test]
    fn read_test_wav_sample_count_matches_metadata() {
        let wav = open(Path::new("resources/test.wav"));
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        let audio = <WavFile as AudioFileRead>::read::<f32>(&wav).expect("read failed");
        let total = audio.to_interleaved_vec().len().get();
        assert_eq!(total, info.total_samples);
    }

    // ── boundary values ───────────────────────────────────────────────────────

    #[test]
    fn read_16bit_min_max_values() {
        let path = temp_path("16bit_minmax");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[i16::MIN, i16::MAX, 0, -1] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_i16(&wav), vec![i16::MIN, i16::MAX, 0, -1]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_32bit_min_max_values() {
        let path = temp_path("32bit_minmax");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[i32::MIN, i32::MAX, 0, -1] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_i32(&wav), vec![i32::MIN, i32::MAX, 0, -1]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn read_f32_special_values() {
        let path = temp_path("f32_special");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[1.0f32, -1.0, 0.0, 0.5] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        assert_eq!(interleaved_f32(&wav), vec![1.0f32, -1.0, 0.0, 0.5]);

        std::fs::remove_file(&path).ok();
    }

    // ── duration and sample count consistency ─────────────────────────────────

    #[test]
    fn duration_matches_sample_count() {
        // 4 samples at 44100 Hz mono ≈ 4/44100 seconds
        let path = temp_path("duration_check");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[1i16, 2, 3, 4] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.total_samples, 4);
        let expected_duration_secs = 4.0 / 44100.0f64;
        let actual_secs = info.duration.as_secs_f64();
        assert!(
            (actual_secs - expected_duration_secs).abs() < 1e-6,
            "duration {actual_secs} vs expected {expected_duration_secs}"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stereo_total_samples_includes_all_channels() {
        // 4 frames of stereo = 8 total samples
        let path = temp_path("stereo_sample_count");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for &s in &[1i16, 2, 3, 4, 5, 6, 7, 8] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let wav = open(&path);
        let info = <WavFile as AudioFileMetadata>::base_info(&wav).unwrap();
        assert_eq!(info.total_samples, 8);
        assert_eq!(info.channels, 2);

        std::fs::remove_file(&path).ok();
    }
}

#[cfg(feature = "wav")]
mod sample_iter {
    use std::{fs::File, io::BufReader};

    use audio_samples_io::wav::StreamedWavFile;
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("asio_siter_{name}_{}.wav", std::process::id()))
    }

    #[test]
    fn mono_i16_sample_count_matches() {
        let path = temp_path("mono_count");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for s in 0i16..128 {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let mut reader = StreamedWavFile::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let count = reader.samples::<i16>().filter(|r| r.is_ok()).count();
        assert_eq!(count, 128);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stereo_i16_values_interleaved() {
        let path = temp_path("stereo_interleaved");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        // Write 4 frames: (1,2), (3,4), (5,6), (7,8)
        for &s in &[1i16, 2, 3, 4, 5, 6, 7, 8] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let mut reader = StreamedWavFile::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(|r| r.expect("sample error")).collect();
        assert_eq!(samples, [1, 2, 3, 4, 5, 6, 7, 8]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn sum_via_iterator_adaptor() {
        let path = temp_path("sum");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for s in 1i16..=10 {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let mut reader = StreamedWavFile::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let sum: i16 = reader.samples::<i16>().map(|r| r.expect("sample error")).sum();
        assert_eq!(sum, 55); // 1+2+…+10
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f32_conversion_from_i16_file() {
        let path = temp_path("f32_conv");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        w.write_sample(i16::MAX).unwrap();
        w.write_sample(0i16).unwrap();
        w.write_sample(i16::MIN).unwrap();
        w.finalize().unwrap();

        let mut reader = StreamedWavFile::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let samples: Vec<f32> = reader.samples::<f32>().map(|r| r.expect("sample error")).collect();
        assert_eq!(samples.len(), 3);
        assert!(samples[0] > 0.0);
        assert_eq!(samples[1], 0.0);
        assert!(samples[2] < 0.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn size_hint_exact() {
        let path = temp_path("size_hint");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::create(&path, spec).unwrap();
        for s in 0i16..64 {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        let mut reader = StreamedWavFile::new(BufReader::new(File::open(&path).unwrap())).unwrap();
        let mut iter = reader.samples::<i16>();
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 64);
        assert_eq!(hi, Some(64));
        // Consume one sample and check the hint decrements
        iter.next();
        let (lo2, _) = iter.size_hint();
        assert!(lo2 < 64);
        std::fs::remove_file(&path).ok();
    }
}
