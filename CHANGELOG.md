# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- WAV: read LIST/INFO metadata tags (title, artist, album, date, genre, software, copyright, and more) via `WavFile::list()` and `ListChunk::info_metadata()`; also read FACT chunk sample count via `WavFile::fact()`; both are surfaced in `WavFileInfo` returned by `specific_info()`
- WAV: read CUE chunk (named markers) via `WavFile::cue()` returning `CueChunk` with `cue_points()`
- WAV: read SMPL chunk (MIDI sampler metadata: pitch, unity note, loop points) via `WavFile::smpl()` returning `SmplChunk` with `loops()` and `LoopType` enum
- WAV: read BEXT chunk (Broadcast Wave Format metadata: originator, timecode, SMPTE UMID, EBU R128 loudness) via `WavFile::bext()` returning `BextChunk`

## 0.3.0

### Added
- WAV: `StreamedWavFile::samples::<T>()` — a sample-level iterator yielding `Result<T, AudioIOError>` in interleaved channel order, composable with `.map()`, `.filter()`, `.sum()` and other standard adaptors

### Changed
- Release profile: explicit `opt-level = 3` and `panic = "abort"` for smaller binaries and lower error-path overhead
