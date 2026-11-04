#![cfg(feature = "wav")]
//! Every layout of every sample type writes the same interleaved little-endian bytes.

use std::io::Cursor;

use audio_samples::{AudioData, AudioSamples, I24, sample_rate, traits::StandardSample};
use audio_samples_io::{
    create_streamed_writer,
    traits::{AudioStreamWrite, AudioStreamWriter},
    types::FileType,
    write_with,
};
use ndarray::{Array1, Array2, ShapeBuilder, s};

/// A value for channel `c`, frame `f` that differs across channels, frames and bytes.
fn value(c: usize, f: usize) -> i64 {
    let x = (f as i64 * 2654435761 + c as i64 * 40503) & 0xFF_FFFF;
    x - 0x80_0000
}

fn expected_data<T: StandardSample>(audio: &Array2<T>) -> Vec<u8> {
    let (channels, frames) = audio.dim();
    let mut out = Vec::new();
    for f in 0..frames {
        for c in 0..channels {
            out.extend_from_slice(audio[[c, f]].to_le_bytes().as_ref());
        }
    }
    out
}

fn data_chunk(file: &[u8]) -> &[u8] {
    let mut pos = 12;
    loop {
        let size = u32::from_le_bytes(file[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &file[pos..pos + 4] == b"data" {
            return &file[pos + 8..pos + 8 + size];
        }
        pos += 8 + size + (size & 1);
    }
}

fn check_layouts<T: StandardSample + 'static>(make: impl Fn(i64) -> T) {
    // 20_011 frames: more than one 8 KiB tile, and an odd count for the pad byte.
    for channels in [1usize, 2, 3, 6] {
        let frames = 20_011usize;
        let sr = sample_rate!(44100);
        let c_order = Array2::from_shape_fn((channels, frames), |(c, f)| make(value(c, f)));
        let f_order = Array2::from_shape_fn((channels, frames).f(), |(c, f)| make(value(c, f)));
        let doubled = Array2::from_shape_fn((channels, frames * 2), |(c, f)| make(value(c, f / 2)));
        let expected = expected_data(&c_order);

        let bulk = |audio: AudioSamples<'_, T>, what: &str| {
            let mut out = Vec::new();
            write_with(Cursor::new(&mut out), &audio, FileType::WAV).unwrap();
            assert!(data_chunk(&out) == expected.as_slice(), "bulk {what}, {channels} channels");

            let mut out = Vec::new();
            let mut w = create_streamed_writer::<_, T>(Cursor::new(&mut out), channels as u16, 44100).unwrap();
            w.write_frames(&audio).unwrap();
            w.finalize().unwrap();
            drop(w);
            assert!(data_chunk(&out) == expected.as_slice(), "streamed {what}, {channels} channels");
        };

        if channels == 1 {
            bulk(
                AudioSamples::new_mono(Array1::from_iter(c_order.iter().copied()), sr).unwrap(),
                "mono",
            );
            let strided = Array1::from_iter(doubled.iter().copied());
            let view = strided.slice(s![..;2]).to_owned();
            bulk(AudioSamples::new_mono(view, sr).unwrap(), "mono copy");
        } else {
            bulk(AudioSamples::new_multi_channel(c_order.clone(), sr).unwrap(), "C order");
            bulk(AudioSamples::new_multi_channel(f_order, sr).unwrap(), "F order");
            let strided = doubled.slice(s![.., ..;2]);
            bulk(
                AudioSamples::new(AudioData::from_borrowed_array2(strided).unwrap(), sr),
                "strided",
            );
        }
    }
}

#[test]
fn u8_layouts() {
    check_layouts(|v| (v & 0xFF) as u8);
}

#[test]
fn i16_layouts() {
    check_layouts(|v| (v >> 8) as i16);
}

#[test]
fn i24_layouts() {
    check_layouts(|v| I24::wrapping_from_i32(v as i32));
}

#[test]
fn i32_layouts() {
    check_layouts(|v| (v << 8) as i32);
}

#[test]
fn f32_layouts() {
    check_layouts(|v| v as f32 / 8388608.0);
}

#[test]
fn f64_layouts() {
    check_layouts(|v| v as f64 / 8388608.0);
}
