//! FLAC constants and magic numbers.

/// FLAC stream marker "fLaC"
pub const FLAC_MARKER: [u8; 4] = *b"fLaC";

/// Frame sync code (14 bits: 0b11111111111110)
pub const FRAME_SYNC_CODE: u16 = 0x3FFE;

/// Maximum LPC order
pub const MAX_LPC_ORDER: usize = 32;

/// STREAMINFO block size (always 34 bytes)
pub const STREAMINFO_SIZE: usize = 34;

/// MD5 signature size
pub const MD5_SIZE: usize = 16;

/// Fixed predictor orders (0-4 supported)
pub const MAX_FIXED_ORDER: usize = 4;


/// Sample rate lookup table for frame header
/// Index 0-11 are predefined, 12-14 read from end of header
pub const SAMPLE_RATE_TABLE: [u32; 12] = [
    0,      // 0: get from STREAMINFO
    88200,  // 1
    176400, // 2
    192000, // 3
    8000,   // 4
    16000,  // 5
    22050,  // 6
    24000,  // 7
    32000,  // 8
    44100,  // 9
    48000,  // 10
    96000,  // 11
];

/// Block size lookup table for frame header
/// Values with special meaning: 0 = reserved, 1 = 192, 6 = get 8-bit, 7 = get 16-bit
pub const BLOCK_SIZE_TABLE: [u32; 16] = [
    0,     // 0: reserved
    192,   // 1
    576,   // 2
    1152,  // 3
    2304,  // 4
    4608,  // 5
    0,     // 6: get 8-bit (blocksize-1) from end of header
    0,     // 7: get 16-bit (blocksize-1) from end of header
    256,   // 8
    512,   // 9
    1024,  // 10
    2048,  // 11
    4096,  // 12
    8192,  // 13
    16384, // 14
    32768, // 15
];

/// Bits per sample lookup table for frame header
pub const BITS_PER_SAMPLE_TABLE: [u8; 8] = [
    0,  // 0: get from STREAMINFO
    8,  // 1
    12, // 2
    0,  // 3: reserved
    16, // 4
    20, // 5
    24, // 6
    32, // 7 (FLAC 1.4.0+)
];

/// Encode bits per sample to frame header code
pub const fn bits_per_sample_to_code(bits: u8) -> Option<u8> {
    match bits {
        8 => Some(1),
        12 => Some(2),
        16 => Some(4),
        20 => Some(5),
        24 => Some(6),
        32 => Some(7),
        _ => None,
    }
}

/// Encode sample rate to frame header code, returns (code, extra_bytes)
/// extra_bytes: 0 = none, 1 = 8-bit kHz, 2 = 16-bit Hz, 3 = 16-bit 10Hz
pub const fn sample_rate_to_code(rate: u32) -> (u8, u8) {
    match rate {
        88200 => (1, 0),
        176400 => (2, 0),
        192000 => (3, 0),
        8000 => (4, 0),
        16000 => (5, 0),
        22050 => (6, 0),
        24000 => (7, 0),
        32000 => (8, 0),
        44100 => (9, 0),
        48000 => (10, 0),
        96000 => (11, 0),
        _ => {
            // Check if expressible as kHz (8-bit)
            if rate.is_multiple_of(1000) && rate / 1000 <= 255 {
                (12, 1)
            }
            // Check if expressible as Hz (16-bit)
            else if rate <= 65535 {
                (13, 2)
            }
            // Express as 10Hz units (16-bit)
            else if rate.is_multiple_of(10) && rate / 10 <= 65535 {
                (14, 3)
            }
            // Fallback to STREAMINFO
            else {
                (0, 0)
            }
        }
    }
}

/// Encode block size to frame header code, returns (code, extra_bytes)
pub const fn block_size_to_code(size: u32) -> (u8, u8) {
    match size {
        192 => (1, 0),
        576 => (2, 0),
        1152 => (3, 0),
        2304 => (4, 0),
        4608 => (5, 0),
        256 => (8, 0),
        512 => (9, 0),
        1024 => (10, 0),
        2048 => (11, 0),
        4096 => (12, 0),
        8192 => (13, 0),
        16384 => (14, 0),
        32768 => (15, 0),
        _ => {
            // Use 8-bit if fits
            if size <= 256 { (6, 1) } else { (7, 2) }
        }
    }
}
