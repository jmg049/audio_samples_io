# FLAC Read and Write Benchmarks

Comparative benchmarks for FLAC decode/encode across `audio_samples_io`, Symphonia, and libFLAC.

Each benchmark processes **0.25 seconds of audio** at the given sample rate and channel count. All timings are in **microseconds (µs)**. The fastest result per row is **bold**. Lower is better.

Benchmarks run via Criterion (Rust) and a custom harness (C/libFLAC) on the same machine.

---

## FLAC Read (decode)

| Case | **AudioSamples i16** | **AudioSamples i32** | **AudioSamples f32** | Symphonia i32 | libFLAC i16 | libFLAC i32 |
|:-----|-------------:|-------------:|-------------:|--------------:|------------:|------------:|
| 44100 Hz 1ch | **32.4** | 32.5 | 34.7 | 39.7 | 44.5 | 48.0 |
| 44100 Hz 2ch | **75.8** | 76.0 | 79.1 | 80.0 | 83.9 | 91.0 |
| 44100 Hz 6ch | **216.4** | 216.8 | 227.6 | 236.0 | 221.6 | 258.6 |
| 96000 Hz 1ch | **56.4** | 56.7 | 60.7 | 79.1 | 74.8 | 91.2 |
| 96000 Hz 2ch | 158.3 | **154.8** | 164.3 | 173.3 | 155.8 | 185.8 |
| 96000 Hz 6ch | **381.5** | 386.5 | 410.4 | 453.4 | 445.1 | 545.9 |

**vs libFLAC i16 (our i16):** −27% / −10% / −2% / −25% / +2% / −14%  
(positive = we are slower; negative = we are faster)

---

## FLAC Write (encode)

| Case | **AudioSamples i16** | **AudioSamples i32** | libFLAC i16 | libFLAC i32 |
|:-----|-------------:|-------------:|------------:|------------:|
| 44100 Hz 1ch | **67.2** | 73.1 | 69.7 | 90.7 |
| 44100 Hz 2ch | **151.1** | 154.0 | 173.7 | 216.6 |
| 44100 Hz 6ch | 438.8 | 474.9 | **364.7** | 487.7 |
| 96000 Hz 1ch | **139.8** | 145.0 | 150.8 | 177.6 |
| 96000 Hz 2ch | **348.3** | 341.7 | 369.2 | 457.6 |
| 96000 Hz 6ch | 910.9 | 948.1 | **781.6** | 1040.4 |

**Notes:** The write encoder has not been optimised. The 6ch gap vs libFLAC i16 (438.8 vs 364.7 µs, ~17% slower) is the most significant remaining opportunity.

---

## System Environment

### CPU
- **Model**: 13th Gen Intel Core i7-13700KF
- **Cores**: 16 physical, 24 logical
- **Architecture**: x86_64

### Operating System
- **Distribution**: CachyOS (Arch-based)
- **Kernel**: 6.19.9-2-cachyos

### Compilation
- **Profile**: `release` with `lto = "thin"`, `codegen-units = 1`
- **Rust**: stable

### Benchmark dates
- Rust (Criterion) benchmarks: 2026-04-23
- C (libFLAC) benchmarks: 2026-04-23
