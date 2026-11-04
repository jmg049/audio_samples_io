#![cfg(feature = "flac")]
//! Encoded streams decode to the input, through this crate's decoder and the reference `flac`
//! tool when it is installed.

use std::io::{Cursor, Write};
use std::process::{Command, Stdio};

use audio_samples::{AudioSamples, I24, sample_rate};
use audio_samples_io::flac::{CompressionLevel, FlacFile, StreamedFlacWriter, write_flac};
use audio_samples_io::traits::{AudioFile, AudioFileRead, AudioStreamWrite, AudioStreamWriter};
use audio_samples_io::types::{OpenOptions, ValidatedSampleType};
use ndarray::{Array1, Array2, ShapeBuilder};

fn noise(state: &mut u64) -> i64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as i64
}

/// Signals that reach the encoder's corner cases, as `(channels, frames)` samples of `bits` bits.
fn signals(bits: u32, channels: usize, frames: usize) -> Vec<(&'static str, Array2<i32>)> {
    let top = 1i64 << (bits - 1);
    let clamp = |v: i64| v.clamp(-top, top - 1) as i32;
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::new();
    out.push((
        "tone and noise",
        Array2::from_shape_fn((channels, frames), |(c, f)| {
            let t = f as f64 / 44100.0;
            let tone = (std::f64::consts::TAU * (220.0 + 110.0 * c as f64) * t).sin() * 0.4 * top as f64;
            clamp(tone as i64 + noise(&mut state) % (top / 64 + 1))
        }),
    ));
    out.push((
        "full scale noise",
        Array2::from_shape_fn((channels, frames), |_| clamp(noise(&mut state) % top)),
    ));
    out.push(("silence", Array2::zeros((channels, frames))));
    out.push(("constant", Array2::from_elem((channels, frames), clamp(top / 3))));
    out.push((
        "extremes",
        Array2::from_shape_fn((channels, frames), |(c, f)| {
            if (f + c) % 2 == 0 { clamp(top - 1) } else { clamp(-top) }
        }),
    ));
    out.push((
        "wasted bits",
        Array2::from_shape_fn((channels, frames), |_| clamp((noise(&mut state) % (top / 16)) * 16)),
    ));
    // Zero runs with an isolated full-scale spike: a very long unary code next to a parameter of
    // zero, and partitions that are all zeros.
    out.push((
        "spikes in silence",
        Array2::from_shape_fn(
            (channels, frames),
            |(_, f)| {
                if f % 3001 == 1500 { clamp(top - 1) } else { 0 }
            },
        ),
    ));
    out
}

/// Decode with the reference tool to interleaved integers. `None` when the tool is not installed.
fn reference_decode(bytes: &[u8], bits: u32) -> Option<Vec<i32>> {
    let mut child = Command::new("flac")
        .args([
            "-d",
            "-s",
            "-c",
            "--force-raw-format",
            "--endian=little",
            "--sign=signed",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take().unwrap();
    let input = bytes.to_vec();
    let feeder = std::thread::spawn(move || stdin.write_all(&input));
    let output = child.wait_with_output().unwrap();
    feeder.join().unwrap().unwrap();
    assert!(
        output.status.success(),
        "flac: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let width = (bits / 8) as usize;
    Some(
        output
            .stdout
            .chunks_exact(width)
            .map(|b| {
                let mut w = [0u8; 4];
                w[4 - width..].copy_from_slice(b);
                i32::from_le_bytes(w) >> (32 - bits)
            })
            .collect(),
    )
}

/// Assert that `bytes` decodes to `expected` through the reference tool, if present.
fn check_reference(bytes: &[u8], bits: u32, expected: &[i32], context: &str) {
    if let Some(decoded) = reference_decode(bytes, bits) {
        assert!(decoded == expected, "reference decoder: {context}");
    }
}

fn interleaved(a: &Array2<i32>) -> Vec<i32> {
    let (channels, frames) = a.dim();
    let mut v = Vec::with_capacity(channels * frames);
    for f in 0..frames {
        for c in 0..channels {
            v.push(a[[c, f]]);
        }
    }
    v
}

#[test]
fn i16_decodes_to_the_input_at_every_level() {
    let dir = std::env::temp_dir();
    for channels in [1usize, 2, 3, 6, 8] {
        for frames in [1usize, 2, 5, 16, 33, 1151, 4096, 4097, 10_000] {
            for (what, signal) in signals(16, channels, frames) {
                for level in 0..=8u8 {
                    let audio16 = signal.mapv(|s| s as i16);
                    let audio = if channels == 1 {
                        AudioSamples::new_mono(Array1::from_iter(audio16.iter().copied()), sample_rate!(44100)).unwrap()
                    } else {
                        AudioSamples::new_multi_channel(audio16.clone(), sample_rate!(44100)).unwrap()
                    };
                    let mut bytes = Vec::new();
                    write_flac(&mut bytes, &audio, CompressionLevel::new(level)).unwrap();
                    let context = format!("{what}, {channels} channels, {frames} frames, level {level}");

                    if level % 4 == 0 && frames != 2 {
                        check_reference(&bytes, 16, &interleaved(&signal), &context);
                    }

                    let path = dir.join(format!("flac_lossless_{channels}_{frames}_{level}.flac"));
                    std::fs::write(&path, &bytes).unwrap();
                    let flac = FlacFile::open_with_options(&path, OpenOptions::default()).unwrap();
                    let back = flac.read::<i16>().unwrap();
                    assert!(
                        back.to_interleaved_vec()
                            .iter()
                            .map(|&s| i32::from(s))
                            .eq(interleaved(&signal)),
                        "our decoder: {context}"
                    );
                    std::fs::remove_file(&path).ok();
                }
            }
        }
    }
}

#[test]
fn i24_writes_the_sample_values() {
    for channels in [1usize, 2, 6] {
        for (what, signal) in signals(24, channels, 10_000) {
            for level in [0u8, 3, 5, 8] {
                let values = signal.mapv(I24::wrapping_from_i32);
                let audio = if channels == 1 {
                    AudioSamples::new_mono(Array1::from_iter(values.iter().copied()), sample_rate!(44100)).unwrap()
                } else {
                    AudioSamples::new_multi_channel(values.clone(), sample_rate!(44100)).unwrap()
                };
                let mut bytes = Vec::new();
                write_flac(&mut bytes, &audio, CompressionLevel::new(level)).unwrap();
                let context = format!("{what}, {channels} channels, level {level}");
                check_reference(&bytes, 24, &interleaved(&signal), &context);
                let path = std::env::temp_dir().join(format!("flac_lossless_i24_{channels}_{level}.flac"));
                std::fs::write(&path, &bytes).unwrap();
                let flac = FlacFile::open_with_options(&path, OpenOptions::default()).unwrap();
                let back = flac.read::<I24>().unwrap();
                assert!(
                    back.to_interleaved_vec()
                        .iter()
                        .map(|s| s.to_i32())
                        .eq(interleaved(&signal)),
                    "our decoder: {context}"
                );
                std::fs::remove_file(&path).ok();
            }
        }
    }
}

#[test]
fn i32_keeps_the_top_24_bits_in_any_layout() {
    let frames = 9_000;
    let c_order = Array2::from_shape_fn((2, frames), |(c, f)| {
        ((f as i32 * 7919 + c as i32 * 104_729) % 8_000_000) << 8
    });
    let f_order = Array2::from_shape_fn((2, frames).f(), |(c, f)| {
        ((f as i32 * 7919 + c as i32 * 104_729) % 8_000_000) << 8
    });
    let encode = |a: Array2<i32>| {
        let audio = AudioSamples::new_multi_channel(a, sample_rate!(44100)).unwrap();
        let mut bytes = Vec::new();
        write_flac(&mut bytes, &audio, CompressionLevel::DEFAULT).unwrap();
        bytes
    };
    let from_c = encode(c_order.clone());
    let from_f = encode(f_order);
    assert_eq!(from_c, from_f);
    let expected: Vec<i32> = interleaved(&c_order).iter().map(|&s| s >> 8).collect();
    check_reference(&from_c, 24, &expected, "i32");
}

#[test]
fn float_rounds_to_24_bits_and_back() {
    // Values of the form v / 8388607 and v / 8388608, as decoding produces.
    let values: Vec<i32> = (0..20_000).map(|i| ((i * 4001) % 16_000_000) - 8_000_000).collect();
    let floats: Vec<f32> = values
        .iter()
        .map(|&v| {
            if v < 0 {
                v as f32 / 8_388_608.0
            } else {
                v as f32 / 8_388_607.0
            }
        })
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(floats), sample_rate!(44100)).unwrap();
    let mut bytes = Vec::new();
    write_flac(&mut bytes, &audio, CompressionLevel::DEFAULT).unwrap();
    check_reference(&bytes, 24, &values, "f32");
}

#[test]
fn streamed_chunks_match_the_bulk_encoder() {
    let frames = 30_011;
    let signal = Array2::from_shape_fn((2, frames), |(c, f)| {
        (((f as f64 * 0.01 * (c + 1) as f64).sin() * 12_000.0) as i32 + ((f * 7) % 91) as i32) as i16
    });
    let audio = AudioSamples::new_multi_channel(signal.clone(), sample_rate!(44100)).unwrap();
    let mut bulk = Vec::new();
    write_flac(&mut bulk, &audio, CompressionLevel::DEFAULT).unwrap();

    for chunk in [1000usize, 4096, 5000] {
        let mut out = Vec::new();
        {
            let mut writer = StreamedFlacWriter::new(
                Cursor::new(&mut out),
                2,
                44100,
                ValidatedSampleType::I16,
                CompressionLevel::DEFAULT,
            )
            .unwrap();
            let mut start = 0;
            while start < frames {
                let end = (start + chunk).min(frames);
                let part = signal.slice(ndarray::s![.., start..end]).to_owned();
                writer
                    .write_frames(&AudioSamples::new_multi_channel(part, sample_rate!(44100)).unwrap())
                    .unwrap();
                start = end;
            }
            writer.finalize().unwrap();
        }
        assert!(out == bulk, "chunks of {chunk}");
    }
}
