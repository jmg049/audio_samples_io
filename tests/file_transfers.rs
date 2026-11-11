#![cfg(all(feature = "wav", feature = "flac"))]
//! Whole-file reads and writes: overwriting, and transfers large enough to use several threads.

use std::path::PathBuf;

use audio_samples::{AudioSamples, I24, nzu, sample_rate};
use audio_samples_io::{
    create_streamed,
    flac::{CompressionLevel, write_flac},
    read,
    traits::{AudioStreamWrite, AudioStreamWriter},
    write,
};
use ndarray::{Array1, Array2};

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("aus_file_transfers_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn noise(channels: usize, frames: usize) -> Array2<i16> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    Array2::from_shape_fn((channels, frames), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 48) as i16
    })
}

#[test]
fn overwriting_leaves_exactly_the_new_file() {
    let path = temp("overwrite.wav");
    let long = AudioSamples::new_multi_channel(noise(2, 100_000), sample_rate!(44100)).unwrap();
    let short = AudioSamples::new_multi_channel(noise(2, 1_000), sample_rate!(44100)).unwrap();
    write(&path, &long).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 400_000);
    write(&path, &short).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 4_000);
    let back = read::<_, i16>(&path).unwrap();
    assert_eq!(back.to_interleaved_vec(), short.to_interleaved_vec());

    let flac = temp("overwrite.flac");
    write(&flac, &long).unwrap();
    write(&flac, &short).unwrap();
    assert_eq!(
        read::<_, i16>(&flac).unwrap().to_interleaved_vec(),
        short.to_interleaved_vec()
    );
}

#[test]
fn streaming_over_a_longer_file_leaves_exactly_the_new_file() {
    let path = temp("stream_over.wav");
    let long = AudioSamples::new_multi_channel(noise(2, 100_000), sample_rate!(44100)).unwrap();
    write(&path, &long).unwrap();
    let chunk = AudioSamples::<i16>::zeros_mono(nzu!(1000), sample_rate!(44100));
    let mut writer = create_streamed::<_, i16>(&path, 1, 44100).unwrap();
    writer.write_frames(&chunk).unwrap();
    writer.finalize().unwrap();
    drop(writer);
    let back = read::<_, i16>(&path).unwrap();
    assert_eq!(back.samples_per_channel().get(), 1000);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 44 + 2_000);
}

#[cfg(unix)]
#[test]
fn overwriting_keeps_links_and_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let path = temp("linked.wav");
    let link = temp("link.wav");
    let hard = temp("hard.wav");
    let audio = AudioSamples::new_multi_channel(noise(2, 1_000), sample_rate!(44100)).unwrap();
    write(&path, &audio).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    let _ = std::fs::remove_file(&link);
    let _ = std::fs::remove_file(&hard);
    std::os::unix::fs::symlink(&path, &link).unwrap();
    std::fs::hard_link(&path, &hard).unwrap();
    let other = AudioSamples::new_multi_channel(noise(2, 500), sample_rate!(44100)).unwrap();
    write(&link, &other).unwrap();
    assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
    assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o640);
    assert_eq!(read::<_, i16>(&hard).unwrap().samples_per_channel().get(), 500);
}

/// Whole-file FLAC writes work out the signature on another thread and write it afterwards.
#[test]
fn large_flac_file_has_a_valid_signature() {
    let audio = AudioSamples::new_multi_channel(noise(2, 800_000), sample_rate!(44100)).unwrap();
    let path = temp("signed.flac");
    write(&path, &audio).unwrap();
    let Ok(status) = std::process::Command::new("flac")
        .args(["-t", "-w", "-s"])
        .arg(&path)
        .status()
    else {
        return;
    };
    assert!(status.success());
    assert!(read::<_, i16>(&path).unwrap().to_interleaved_vec() == audio.to_interleaved_vec());
}

#[test]
fn large_wav_reads_match_in_every_sample_type() {
    // Large enough to be divided between threads.
    let audio = noise(2, 2_000_000);
    let path = temp("large.wav");
    let source = AudioSamples::new_multi_channel(audio, sample_rate!(44100)).unwrap();
    write(&path, &source).unwrap();
    let back = read::<_, i16>(&path).unwrap();
    assert!(back.to_interleaved_vec() == source.to_interleaved_vec());

    let values: Vec<f32> = source
        .to_interleaved_vec()
        .iter()
        .map(|&s| f32::from(s) / 40_000.0)
        .collect();
    let floats = AudioSamples::new_mono(Array1::from(values.clone()), sample_rate!(44100)).unwrap();
    let path = temp("large_f32.wav");
    write(&path, &floats).unwrap();
    assert!(read::<_, f32>(&path).unwrap().to_interleaved_vec().as_slice() == values.as_slice());
}

#[test]
fn large_flac_decodes_the_same_on_several_threads() {
    let source = noise(2, 1_500_000);
    let audio = AudioSamples::new_multi_channel(source, sample_rate!(44100)).unwrap();
    let path = temp("large.flac");
    let mut bytes = Vec::new();
    write_flac(&mut bytes, &audio, CompressionLevel::FASTEST).unwrap();
    assert!(bytes.len() > 4 * 1024 * 1024);
    std::fs::write(&path, &bytes).unwrap();
    let expected = audio.to_interleaved_vec();
    assert!(read::<_, i16>(&path).unwrap().to_interleaved_vec() == expected);

    // Trailing bytes after the last frame, as a tag would leave.
    let mut tagged = bytes.clone();
    tagged.extend_from_slice(b"TAG not audio not audio not audio");
    let path = temp("tagged.flac");
    std::fs::write(&path, &tagged).unwrap();
    assert!(read::<_, i16>(&path).unwrap().to_interleaved_vec() == expected);

    // 24-bit, through a conversion.
    let values = audio.to_interleaved_vec();
    let i24 = Array2::from_shape_fn((2, 1_500_000), |(c, f)| {
        I24::wrapping_from_i32(i32::from(values[f * 2 + c]) << 8)
    });
    let audio = AudioSamples::new_multi_channel(i24, sample_rate!(44100)).unwrap();
    let path = temp("large24.flac");
    let file = std::fs::File::create(&path).unwrap();
    write_flac(file, &audio, CompressionLevel::FASTEST).unwrap();
    let back = read::<_, I24>(&path).unwrap();
    assert!(back.to_interleaved_vec() == audio.to_interleaved_vec());
}

/// Large saves on tmpfs go through a mapping: bytes and position must match an ordinary save, for
/// interleaved and planar memory, a new file and a longer old one.
#[cfg(target_os = "linux")]
#[test]
fn tmpfs_mapped_save_matches_plain_save() {
    use audio_samples::{AudioSamples, sample_rate};
    use ndarray::Array2;

    let shm = std::path::Path::new("/dev/shm");
    if !shm.is_dir() {
        return;
    }
    let frames = 1_200_000usize;
    let planar = Array2::from_shape_fn((2, frames), |(c, f)| (f as i32).wrapping_mul(31).wrapping_add(c as i32));
    let audio = AudioSamples::new_multi_channel(planar.clone(), sample_rate!(44100)).unwrap();

    let id = std::process::id();
    let mapped = shm.join(format!("asio_mapped_{id}.wav"));
    // A build directory rather than the temporary one, which can itself be tmpfs.
    let plain = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("asio_plain_{id}.wav"));
    audio_samples_io::write(&mapped, &audio).unwrap();
    audio_samples_io::write(&plain, &audio).unwrap();
    assert!(std::fs::read(&mapped).unwrap() == std::fs::read(&plain).unwrap());

    // Over a longer file, which the save must cut.
    std::fs::write(&mapped, vec![7u8; 30_000_000]).unwrap();
    audio_samples_io::write(&mapped, &audio).unwrap();
    assert!(std::fs::read(&mapped).unwrap() == std::fs::read(&plain).unwrap());

    let _ = std::fs::remove_file(&mapped);
    let _ = std::fs::remove_file(&plain);
}

/// A header longer than one read, with chunks before the samples and an odd-sized chunk that
/// carries a pad byte, reads the same samples.
#[test]
fn long_header_reads_the_samples() {
    let frames = 700_000usize;
    let samples: Vec<i16> = (0..frames * 2)
        .map(|i| (i as i32 * 31 % 65_536 - 32_768) as i16)
        .collect();
    let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut chunks = Vec::new();
    chunks.extend_from_slice(b"fmt ");
    chunks.extend_from_slice(&16u32.to_le_bytes());
    chunks.extend_from_slice(&1u16.to_le_bytes());
    chunks.extend_from_slice(&2u16.to_le_bytes());
    chunks.extend_from_slice(&44_100u32.to_le_bytes());
    chunks.extend_from_slice(&(44_100u32 * 4).to_le_bytes());
    chunks.extend_from_slice(&4u16.to_le_bytes());
    chunks.extend_from_slice(&16u16.to_le_bytes());
    for (id, len) in [(b"LIST", 6_001u32), (b"junk", 3u32)] {
        chunks.extend_from_slice(id);
        chunks.extend_from_slice(&len.to_le_bytes());
        chunks.extend(std::iter::repeat_n(0x5Au8, len as usize + (len as usize & 1)));
    }
    chunks.extend_from_slice(b"data");
    chunks.extend_from_slice(&(data.len() as u32).to_le_bytes());
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&((4 + chunks.len() + data.len()) as u32).to_le_bytes());
    file.extend_from_slice(b"WAVE");
    file.extend_from_slice(&chunks);
    file.extend_from_slice(&data);
    let path = temp("long_header.wav");
    std::fs::write(&path, &file).unwrap();

    let audio = read::<_, i16>(&path).unwrap();
    assert_eq!(audio.samples_per_channel().get(), frames);
    assert!(audio.to_interleaved_vec() == samples);
    let _ = std::fs::remove_file(&path);
}

/// A streamed reader built over a buffer that already holds the file's first bytes reads the same
/// samples as a whole-file read, and seeking back still works.
#[test]
fn streamed_reader_from_a_filled_buffer() {
    use std::io::BufRead;

    use audio_samples_io::wav::StreamedWavFile;

    let frames = 50_000usize;
    let audio = AudioSamples::new_multi_channel(noise(2, frames), sample_rate!(44100)).unwrap();
    let path = temp("filled_buffer.wav");
    write(&path, &audio).unwrap();
    let expected = read::<_, i16>(&path).unwrap().to_interleaved_vec();

    let file = std::fs::File::open(&path).unwrap();
    let len = file.metadata().unwrap().len();
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    assert!(!reader.fill_buf().unwrap().is_empty());
    let mut streamed = StreamedWavFile::new_with_len(reader, path.clone(), len).unwrap();
    let got: Vec<i16> = streamed.samples::<i16>().map(Result::unwrap).collect();
    assert!(got == expected);
    streamed.seek_to_frame(10).unwrap();
    let again: Vec<i16> = streamed.samples::<i16>().take(4).map(Result::unwrap).collect();
    assert_eq!(again, expected[20..24]);
    let _ = std::fs::remove_file(&path);
}

/// The planned stream reader returns the source's bytes in order whether requests go through its
/// buffer or straight past it, and seeks relative to what the caller has read.
#[test]
fn stream_reader_reads_and_seeks_like_its_source() {
    use std::io::{Cursor, Read, Seek, SeekFrom};

    use audio_samples_io::StreamReader;

    let data: Vec<u8> = (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect();
    let mut reader = StreamReader::with_buffer(Cursor::new(data.clone()), 4096, 1024);
    let mut got = Vec::new();
    let mut piece = vec![0u8; 3000];
    for size in [10usize, 2000, 3000, 1, 1023, 1024, 3000] {
        let n = reader.read(&mut piece[..size]).unwrap();
        got.extend_from_slice(&piece[..n]);
    }
    assert!(got == data[..got.len()]);
    let at = got.len() as u64;
    assert_eq!(reader.seek(SeekFrom::Current(-5)).unwrap(), at - 5);
    let mut five = [0u8; 5];
    reader.read_exact(&mut five).unwrap();
    assert_eq!(five, data[(at - 5) as usize..at as usize]);
    reader.seek(SeekFrom::Start(150_000)).unwrap();
    let mut rest = Vec::new();
    reader.read_to_end(&mut rest).unwrap();
    assert!(rest == data[150_000..]);
}
