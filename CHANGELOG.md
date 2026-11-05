# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- FLAC: `write_flac` scaled I24 samples to 32 bits before encoding, so files decoded to wrong values or failed to decode.
- FLAC: float input truncated to 24 bits and now rounds, so decoded audio encodes back to the same integers. `u8` uses the crate's `u8` to 24-bit conversion.
- WAV: `create_streamed::<u8>` wrote 16-bit samples; it writes 8-bit samples.
- WAV: the header parser accepted RF64 without a `ds64` chunk.
- WAV: writing more than 4 GiB returns an error instead of a header with wrapped sizes.
- FLAC: decoding failed with a CRC error or returned wrong samples when a Rice quotient ended at the last bit of the 64-bit read buffer, because the buffer was shifted by 64 bits.
- WAV: `parse_wav_header_streaming` left the reader at the stream end (instead of the first audio byte) for files with a ragged or over-declared `data` chunk
- WAV: streaming writer wrote an 8-byte bits-per-sample value into the 2-byte fmt chunk field, corrupting every file produced by `StreamedWavWriter`/`create_streamed`
- WAV: tolerate real-world/streaming files — over-declared RIFF/`data` size fields (including the `0xFFFFFFFF` placeholder written by streaming encoders) are clamped to the bytes actually present, and a ragged `data` chunk decodes its whole frames instead of erroring
- Streaming: `AudioStreamReader` now exposes `num_channels()`, allowing `Box<dyn AudioStreamReader>` callers to allocate read buffers without a separate `info()` call (eliminating the probe → drop → reopen two-pass latency)

### Changed
- `DEFAULT_WRITE_BUF_CAPACITY` is 256 bytes, enough for a header. Samples go straight to the destination, or through a 64 KiB stack buffer when their memory is not the file's layout.
- FLAC: files carry the MD5 signature of the audio. `write_flac_without_md5` leaves it out.
- FLAC: levels use the conventional block sizes, LPC orders and partition ranges, and `qlp_precision` follows the block size, so levels 2 and 3 produce different frames. The encoder uses windowed LPC analysis, and output is within 0.2% of the reference encoder on real music at each level.
- FLAC: with `flac-parallel`, `write_flac` encodes whole frames in parallel instead of subframes.
- Writes to a path no longer truncate an existing file first; they overwrite it and cut it to length, which keeps its pages and avoids the flush btrfs does when a file truncated to zero is closed. Streamed writes shrink the old file to one byte for the same reason.
- WAV saves of 8 MiB or more on tmpfs fill the file through a mapping from up to 8 threads, 18 to 40% faster than one `write`. Pages are populated first, so a full filesystem returns the usual error.
- Whole-file FLAC reads use positioned reads through small windows instead of a mapping, about 5% faster with 70% fewer pages beyond the output.
- Reads of 1.5 MiB or more that are in the page cache use up to 8 short-lived threads, since threads slow a read from the device, and FLAC files of 512 KiB or more decode on up to 8. Whole-file FLAC writes work out the MD5 on one extra thread. They follow the process's CPU affinity; `AUDIO_SAMPLES_IO_THREADS` sets the limit, and `1` keeps all work on the calling thread. Only one such transfer runs at a time.
- Rust `read()` of WAV reads into memory directly for every sample type instead of mapping the file.
- Integer to float conversion uses AVX2 where available, with the same values as `ConvertFrom`. 24-bit samples pack and unpack with SSSE3.
- FLAC: `StreamedFlacFile` decodes through a small sliding window, keeps an index of decoded frames and bisects by frame header on seek, so seeking no longer decodes from the start.
- WAV and FLAC streaming readers write directly into a caller's buffer without intermediate copies.

### Removed
- Dead `format_conversion` module (never compiled into the crate); transcoding is covered by `read()` + `write()`

### Changed
- Docs: FLAC streaming examples now compile — they import the `AudioStreamRead`/`AudioStreamReader` traits their method calls require

### Added
- `flac::write_flac_without_md5`, `StreamedWavWriter::new_u8`, `CompressionLevel::window_parts`.
- Dependency: `libc` on Linux.
- `benches/python_io_bench.py`: reads and writes against other Python libraries on Linux, macOS and Windows, warm and cold, with fsync and CPU time.
- `StreamedWavFile::new_forward_only`/`StreamedAiffFile::new_forward_only`/`StreamedFlacFile::new` (relaxed): decode wav/flac/aiff incrementally over a plain `Read` source (no `Seek` required) — for network transports and other non-seekable streams. `seek_to_frame`/`reset` (and the `AudioStreamReader`/`AudioStreamRead` trait impls) still require `Read + Seek`, but construction and `read_frames_into` no longer do. WAV and AIFF carry over any header bytes their scan buffered past the `data`/`SSND` chunk boundary as a `leftover` prefix rather than seeking back to discard them; FLAC's metadata-block parser already stops exactly at the frame boundary, so it needed no equivalent. All three verified over a `Read`-only (deliberately non-`Seek`) test wrapper, matching a `Read + Seek` decode exactly.
- `StreamedAiffFile` (`aiff` feature): incremental AIFF/AIFF-C reader mirroring `StreamedWavFile`/`StreamedFlacFile` — parses `COMM`/`SSND` at construction, reads frames on demand, supports `seek_to_frame`/`reset`, and works over any `Read + Seek` source (not just files)
- AIFF/AIFF-C support (`aiff` feature): read plain AIFF and the practical AIFF-C codecs (`sowt`, `fl32`, `fl64`); write AIFF for integer audio and AIFF-C for floats — verified against ffmpeg in both directions
- WAV: read RF64/BW64 files (the 64-bit RIFF variants used for >4 GiB broadcast audio) — all read paths resolve the `0xFFFFFFFF` size placeholders through the `ds64` chunk
- Format detection: `read()`, `info()`, `peek_native_type()`, `open()`, and the streamed-open functions now sniff the file's magic bytes (`FileType::from_magic_bytes`/`FileType::detect`) and fall back to the extension, so mismatched or missing extensions no longer break reads
- FLAC: streaming writer (`create_streamed_flac`/`create_streamed_flac_writer`) for incremental, low-memory encoding, plus a format-agnostic `StreamedAudioWriter` so `create_streamed` selects WAV or FLAC by file extension (`create_streamed_with` for an explicit format)
- WAV: decode mu-law/a-law (G.711) and Microsoft/IMA ADPCM to 16-bit linear PCM; reader also accepts 18-byte PCM `fmt ` headers
- WAV: write LIST/INFO tags and cue markers via `WavMetadata` and `write_with_metadata`/`write_with_metadata_to`
- WAV: `WavSink` — a non-seekable streaming writer (`create_streamed_sink`) for stdout, pipes, and sockets that needs only `Write`, not `Seek`
- WAV: precompute headers and final file size without the sample data via `build_wav_header`, `build_wav_header_infinite`, and `wav_file_len`/`wav_header_len`/`wav_data_len`
- WAV: read LIST/INFO metadata tags (title, artist, album, date, genre, software, copyright, and more) via `WavFile::list()` and `ListChunk::info_metadata()`; also read FACT chunk sample count via `WavFile::fact()`; both are surfaced in `WavFileInfo` returned by `specific_info()`
- WAV: read CUE chunk (named markers) via `WavFile::cue()` returning `CueChunk` with `cue_points()`
- WAV: read SMPL chunk (MIDI sampler metadata: pitch, unity note, loop points) via `WavFile::smpl()` returning `SmplChunk` with `loops()` and `LoopType` enum
- WAV: read BEXT chunk (Broadcast Wave Format metadata: originator, timecode, SMPTE UMID, EBU R128 loudness) via `WavFile::bext()` returning `BextChunk`

## 0.3.0

### Added
- WAV: `StreamedWavFile::samples::<T>()` — a sample-level iterator yielding `Result<T, AudioIOError>` in interleaved channel order, composable with `.map()`, `.filter()`, `.sum()` and other standard adaptors

### Changed
- Release profile: explicit `opt-level = 3` and `panic = "abort"` for smaller binaries and lower error-path overhead
