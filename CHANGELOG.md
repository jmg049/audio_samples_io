# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- WAV: streaming writer wrote an 8-byte bits-per-sample value into the 2-byte fmt chunk field, corrupting every file produced by `StreamedWavWriter`/`create_streamed`
- WAV: tolerate real-world/streaming files — over-declared RIFF/`data` size fields (including the `0xFFFFFFFF` placeholder written by streaming encoders) are clamped to the bytes actually present, and a ragged `data` chunk decodes its whole frames instead of erroring
- Streaming: `AudioStreamReader` now exposes `num_channels()`, allowing `Box<dyn AudioStreamReader>` callers to allocate read buffers without a separate `info()` call (eliminating the probe → drop → reopen two-pass latency)

### Removed
- Dead `format_conversion` module (never compiled into the crate); transcoding is covered by `read()` + `write()`

### Changed
- Docs: FLAC streaming examples now compile — they import the `AudioStreamRead`/`AudioStreamReader` traits their method calls require

### Added
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
