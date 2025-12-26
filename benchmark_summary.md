# `audio_samples_io` Read and Write Benchmarks

This document reports a set of comparative benchmarks evaluating audio read and write performance across three audio libraries: `audio_samples_io`, `hound`, and `libsndfile`.

All benchmarks were executed on the same machine under identical conditions (as close as possible for libsndfile). Each benchmark processes **0.25 seconds of audio**, and all timings are reported in **microseconds (μs)**. For each configuration, the fastest implementation is highlighted in **bold**. Lower values indicate better performance.

## Reading

### i16

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |     **1.82** |     55.50 |           5.22 |
| 44100           | 2                |         8.13 |    111.44 |       **5.67** |
| 44100           | 6                |        21.65 |         - |       **7.49** |
| 96000           | 1                |     **2.86** |    121.24 |           5.60 |
| 96000           | 2                |        17.77 |    242.70 |       **6.58** |
| 96000           | 6                |        43.06 |         - |      **10.55** |

### i24

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |     **3.57** |         - |          10.28 |
| 44100           | 2                |    **12.91** |         - |          15.99 |
| 44100           | 6                |    **31.48** |         - |          39.76 |
| 96000           | 1                |     **6.82** |         - |          16.67 |
| 96000           | 2                |    **26.29** |         - |          29.37 |
| 96000           | 6                |    **67.41** |         - |          82.56 |

### i32

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |     **2.72** |     45.43 |           5.80 |
| 44100           | 2                |        12.17 |     90.64 |       **6.50** |
| 44100           | 6                |        27.46 |         - |      **10.20** |
| 96000           | 1                |     **5.58** |     98.38 |           6.57 |
| 96000           | 2                |        23.31 |    196.38 |       **8.58** |
| 96000           | 6                |        59.50 |         - |      **16.82** |

### f32

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |     **2.79** |     35.30 |           7.89 |
| 44100           | 2                |        10.29 |     70.14 |       **9.27** |
| 44100           | 6                |        30.82 |         - |      **14.95** |
| 96000           | 1                |     **5.61** |     76.33 |           8.80 |
| 96000           | 2                |        19.79 |    152.23 |      **11.42** |
| 96000           | 6                |        67.11 |         - |      **21.73** |

### f64

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |     **5.40** |         - |           8.71 |
| 44100           | 2                |        17.29 |         - |      **11.29** |
| 44100           | 6                |        48.03 |         - |      **20.65** |
| 96000           | 1                |     **8.86** |         - |          10.86 |
| 96000           | 2                |        37.34 |         - |      **15.43** |
| 96000           | 6                |       132.45 |         - |      **44.81** |

## Writing

### i16

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |         4.82 |     17.61 |       **0.67** |
| 44100           | 2                |        18.08 |     35.42 |       **1.05** |
| 44100           | 6                |        61.85 |         - |       **2.30** |
| 96000           | 1                |        10.76 |     38.70 |       **1.14** |
| 96000           | 2                |        36.04 |     77.54 |       **1.78** |
| 96000           | 6                |       134.34 |         - |       **4.47** |

### i24

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |        12.36 |         - |       **4.57** |
| 44100           | 2                |        28.87 |         - |       **8.97** |
| 44100           | 6                |        93.99 |         - |      **26.71** |
| 96000           | 1                |        27.05 |         - |       **9.78** |
| 96000           | 2                |        59.55 |         - |      **19.64** |
| 96000           | 6                |       213.69 |         - |      **59.19** |

### i32

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |         9.78 |     22.84 |       **1.05** |
| 44100           | 2                |        30.19 |     46.18 |       **1.68** |
| 44100           | 6                |        98.78 |         - |       **4.16** |
| 96000           | 1                |        21.97 |     50.09 |       **1.79** |
| 96000           | 2                |        63.22 |    100.94 |       **3.17** |
| 96000           | 6                |       230.47 |         - |       **8.53** |

### f32

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |         9.84 |     20.59 |       **9.34** |
| 44100           | 2                |        30.28 |     41.99 |      **18.54** |
| 44100           | 6                |        97.33 |         - |      **54.52** |
| 96000           | 1                |        21.93 |     45.40 |      **19.93** |
| 96000           | 2                |        62.89 |     92.11 |      **39.57** |
| 96000           | 6                |       228.16 |         - |     **118.25** |

### f64

| **Sample Rate** | **Num Channels** | **audio_samples_io** | **hound** | **libsndfile** |
| :-------------- | :--------------- | -----------: | --------: | -------------: |
| 44100           | 1                |        20.05 |         - |       **5.15** |
| 44100           | 2                |        54.38 |         - |       **9.58** |
| 44100           | 6                |       179.29 |         - |      **30.09** |
| 96000           | 1                |        44.19 |         - |      **10.25** |
| 96000           | 2                |       120.10 |         - |      **19.87** |
| 96000           | 6                |       436.28 |         - |      **73.03** |

## System Environment

### CPU

* **Model**: 13th Gen Intel(R) Core(TM) i7-13700KF
* **Cores**: 16 physical, 24 logical
* **Base Frequency**: 1.9 GHz
* **Architecture**: x86_64

### Memory

* **Total RAM**: 62.6 GB
* **Available RAM**: 40.0 GB
* **Used RAM**: 22.7 GB (36.2%)

### Operating System

* **Distribution**: Ubuntu 24.04
* **Kernel**: 6.14.0-37-generic

### Storage

* **Disk Type**: SSD
* **File System**: ext4

### System Load

* **Load Average (1 / 5 / 15 min)**: 0.78 / 0.66 / 0.79

### Compilation Target

* **Target Triple**: x86_64-unknown-linux-gnu
* **Target Family**: unix
* **Rust Compiler**: `rustc 1.94.0-nightly (ba2142a19 2025-12-07)`