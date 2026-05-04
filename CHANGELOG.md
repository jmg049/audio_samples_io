# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- WAV: `StreamedWavFile::samples::<T>()` — a sample-level iterator yielding `Result<T, AudioIOError>` in interleaved channel order, composable with `.map()`, `.filter()`, `.sum()` and other standard adaptors

### Changed
- Release profile: explicit `opt-level = 3` and `panic = "abort"` for smaller binaries and lower error-path overhead
