// bench_flac.c
// Build: gcc -O3 -march=native -DNDEBUG bench_flac.c -o bench_flac -lFLAC -lm
// Run:   ./bench_flac
// Output: CSV to flac_bench.csv (same format as sndfile_bench.csv)
//
// Note: requires libFLAC (package `libflac-dev` on Debian/Ubuntu, `flac` on Arch).
// Check availability: pkg-config --exists flac && echo yes

#include <FLAC/stream_decoder.h>
#include <FLAC/stream_encoder.h>

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#if defined(_WIN32)
#error "This file uses POSIX stat/time; adapt for Windows if needed."
#endif

static const int SAMPLE_RATES[] = {44100, 96000};
static const int CHANNEL_OPTIONS[] = {1, 2, 6};

static const int SIGNAL_DURATION_MS = 250;
static const char *ASSET_DIR = "target/bench_assets_flac_c";

// Defaults roughly aligned with your Criterion config.
static double g_warmup_s = 3.0;
static double g_measure_s = 8.0;
static int g_samples = 50;
static FILE *g_out = NULL;

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void black_box_ptr(const void *p)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(p) : "memory");
#else
    static volatile const void *sink;
    sink = p;
#endif
}

static inline void black_box_u64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(x) : "memory");
#else
    static volatile uint64_t sink;
    sink = x;
#endif
}

static void die(const char *msg)
{
    fprintf(stderr, "fatal: %s\n", msg);
    exit(1);
}

static void die_errno(const char *msg)
{
    fprintf(stderr, "fatal: %s (errno=%d: %s)\n", msg, errno, strerror(errno));
    exit(1);
}

static uint64_t file_size_bytes(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0)
        die_errno("stat failed");
    return (uint64_t)st.st_size;
}

static void ensure_dir(const char *path)
{
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof(tmp))
        die("asset dir path too long");

    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; i++)
    {
        if (tmp[i] == '/')
        {
            tmp[i] = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
                die_errno("mkdir failed");
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        die_errno("mkdir failed");
}

static int frames_for(int sample_rate)
{
    return (sample_rate * SIGNAL_DURATION_MS) / 1000;
}

static inline double frac(double x) { return x - floor(x); }
static inline double square_wave(double phase) { return (phase < 0.5) ? 1.0 : -1.0; }
static inline double saw_wave(double phase) { return 2.0 * phase - 1.0; }
static inline double chirp_phase(double t, double f0, double f1, double T)
{
    double k = (f1 - f0) / T;
    return 2.0 * M_PI * (f0 * t + 0.5 * k * t * t);
}

static void generate_interleaved_f64(double *out, int frames, int channels, int sample_rate)
{
    double T = (double)SIGNAL_DURATION_MS / 1000.0;

    for (int f = 0; f < frames; f++)
    {
        double t = (double)f / (double)sample_rate;

        for (int ch = 0; ch < channels; ch++)
        {
            double base_freq = 110.0 + 55.0 * (double)ch;
            double amp = 0.35 + 0.1 * (double)(ch % 4);

            double x = 0.0;
            switch (ch % 5)
            {
            case 0:
                x = sin(2.0 * M_PI * base_freq * t);
                break;
            case 1:
                x = cos(2.0 * M_PI * (base_freq * 1.5) * t);
                amp *= 0.9;
                break;
            case 2:
                x = square_wave(frac(base_freq * 0.75 * t));
                amp *= 0.8;
                break;
            case 3:
                x = saw_wave(frac(base_freq * 1.2 * t));
                amp *= 0.7;
                break;
            default:
                x = sin(chirp_phase(t, base_freq, base_freq * 3.0, T));
                amp *= 0.85;
                break;
            }

            out[(size_t)f * (size_t)channels + (size_t)ch] = x * amp;
        }
    }
}

typedef enum
{
    FMT_I16,
    FMT_I24,
    FMT_I32,
} SampleFmt;

static const char *fmt_label(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16: return "i16";
    case FMT_I24: return "i24";
    case FMT_I32: return "i32";
    default: return "unknown";
    }
}

static int bits_for_fmt(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16: return 16;
    case FMT_I24: return 24;
    // libFLAC supports up to 32-bit only in subset variations; 24-bit is the
    // most broadly supported depth.  We cap FMT_I32 at 24-bit on the wire for
    // parity with the sndfile bench and with our Rust encoder.
    case FMT_I32: return 24;
    default: return 16;
    }
}

static int bytes_per_sample_payload(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16: return 2;
    case FMT_I24: return 3;
    case FMT_I32: return 4;
    default: return 2;
    }
}

static void convert_to_i32_buffer(const double *in, FLAC__int32 *out, size_t n, SampleFmt f)
{
    double scale = 0.0;
    long max_v = 0, min_v = 0;
    switch (f)
    {
    case FMT_I16:
        scale = 32767.0; max_v = 32767; min_v = -32768; break;
    case FMT_I24:
    case FMT_I32:
        // encode all of these as 24-bit-in-int32
        scale = 8388607.0; max_v = 8388607; min_v = -8388608; break;
    }

    for (size_t i = 0; i < n; i++)
    {
        double x = in[i];
        if (x > 1.0) x = 1.0;
        if (x < -1.0) x = -1.0;
        long v = lround(x * scale);
        if (v > max_v) v = max_v;
        if (v < min_v) v = min_v;
        out[i] = (FLAC__int32)v;
    }
}

static void asset_path(char *dst, size_t cap, SampleFmt fmt, int sample_rate, int channels)
{
    snprintf(dst, cap, "%s/%s_%dhz_%dch.flac", ASSET_DIR, fmt_label(fmt), sample_rate, channels);
}

// --- Memory-backed write buffer ----------------------------------------------

typedef struct
{
    uint8_t *data;
    size_t size;
    size_t pos;
    size_t cap;
} MemBuf;

static void mem_grow(MemBuf *m, size_t want)
{
    if (want <= m->cap) return;
    size_t newcap = m->cap ? m->cap : 4096;
    while (newcap < want) newcap *= 2;
    uint8_t *p = (uint8_t *)realloc(m->data, newcap);
    if (!p) die("realloc failed");
    m->data = p;
    m->cap = newcap;
}

// --- Encoder (writes to MemBuf) ----------------------------------------------

static FLAC__StreamEncoderWriteStatus enc_write_cb(
    const FLAC__StreamEncoder *encoder,
    const FLAC__byte buffer[],
    size_t bytes,
    uint32_t samples,
    uint32_t current_frame,
    void *client_data)
{
    (void)encoder; (void)samples; (void)current_frame;
    MemBuf *m = (MemBuf *)client_data;
    size_t end = m->pos + bytes;
    mem_grow(m, end);
    memcpy(m->data + m->pos, buffer, bytes);
    m->pos = end;
    if (m->pos > m->size) m->size = m->pos;
    return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
}

static FLAC__StreamEncoderSeekStatus enc_seek_cb(
    const FLAC__StreamEncoder *encoder,
    FLAC__uint64 absolute_byte_offset,
    void *client_data)
{
    (void)encoder;
    MemBuf *m = (MemBuf *)client_data;
    if (absolute_byte_offset > m->size) return FLAC__STREAM_ENCODER_SEEK_STATUS_ERROR;
    m->pos = (size_t)absolute_byte_offset;
    return FLAC__STREAM_ENCODER_SEEK_STATUS_OK;
}

static FLAC__StreamEncoderTellStatus enc_tell_cb(
    const FLAC__StreamEncoder *encoder,
    FLAC__uint64 *absolute_byte_offset,
    void *client_data)
{
    (void)encoder;
    MemBuf *m = (MemBuf *)client_data;
    *absolute_byte_offset = (FLAC__uint64)m->pos;
    return FLAC__STREAM_ENCODER_TELL_STATUS_OK;
}

// Encode an int32 interleaved buffer to a MemBuf. Returns bytes written.
static size_t flac_encode_to_membuf(
    const FLAC__int32 *interleaved, int frames, int channels, int sample_rate,
    int bits_per_sample, MemBuf *out)
{
    FLAC__StreamEncoder *enc = FLAC__stream_encoder_new();
    if (!enc) die("FLAC__stream_encoder_new failed");

    FLAC__stream_encoder_set_channels(enc, (uint32_t)channels);
    FLAC__stream_encoder_set_bits_per_sample(enc, (uint32_t)bits_per_sample);
    FLAC__stream_encoder_set_sample_rate(enc, (uint32_t)sample_rate);
    FLAC__stream_encoder_set_compression_level(enc, 5);
    FLAC__stream_encoder_set_total_samples_estimate(enc, (FLAC__uint64)frames);

    FLAC__StreamEncoderInitStatus st = FLAC__stream_encoder_init_stream(
        enc, enc_write_cb, enc_seek_cb, enc_tell_cb, NULL, out);
    if (st != FLAC__STREAM_ENCODER_INIT_STATUS_OK)
    {
        fprintf(stderr, "FLAC encoder init failed: %d\n", st);
        exit(1);
    }

    if (!FLAC__stream_encoder_process_interleaved(enc, interleaved, (uint32_t)frames))
    {
        fprintf(stderr, "FLAC process failed: %s\n",
                FLAC__stream_encoder_get_resolved_state_string(enc));
        exit(1);
    }
    FLAC__stream_encoder_finish(enc);
    FLAC__stream_encoder_delete(enc);

    return out->size;
}

static void write_flac_asset(SampleFmt fmt, int sample_rate, int channels)
{
    ensure_dir(ASSET_DIR);

    int frames = frames_for(sample_rate);
    size_t n = (size_t)frames * (size_t)channels;

    double *sig = (double *)malloc(n * sizeof(double));
    if (!sig) die("malloc sig failed");
    generate_interleaved_f64(sig, frames, channels, sample_rate);

    FLAC__int32 *buf = (FLAC__int32 *)malloc(n * sizeof(FLAC__int32));
    if (!buf) die("malloc i32 buf failed");
    convert_to_i32_buffer(sig, buf, n, fmt);

    MemBuf mb = {0};
    flac_encode_to_membuf(buf, frames, channels, sample_rate, bits_for_fmt(fmt), &mb);

    char path[512];
    asset_path(path, sizeof(path), fmt, sample_rate, channels);
    FILE *fp = fopen(path, "wb");
    if (!fp) die_errno("fopen(asset) failed");
    if (fwrite(mb.data, 1, mb.size, fp) != mb.size) die("short write of asset");
    fclose(fp);

    free(mb.data);
    free(buf);
    free(sig);
}

static void ensure_assets_exist(void)
{
    for (size_t si = 0; si < sizeof(SAMPLE_RATES)/sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS)/sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];
            for (SampleFmt fmt = FMT_I16; fmt <= FMT_I32; fmt++)
            {
                char path[512];
                asset_path(path, sizeof(path), fmt, sr, ch);
                struct stat st;
                if (stat(path, &st) != 0)
                {
                    write_flac_asset(fmt, sr, ch);
                }
            }
        }
    }
}

// --- Decoder -----------------------------------------------------------------

typedef struct
{
    FLAC__int32 *out;
    size_t cap;
    size_t pos;          // in samples (interleaved)
    int channels;
    int frames_total;
} DecCtx;

static FLAC__StreamDecoderWriteStatus dec_write_cb(
    const FLAC__StreamDecoder *decoder,
    const FLAC__Frame *frame,
    const FLAC__int32 * const buffer[],
    void *client_data)
{
    (void)decoder;
    DecCtx *ctx = (DecCtx *)client_data;
    uint32_t bs = frame->header.blocksize;
    uint32_t ch = frame->header.channels;
    size_t need = ctx->pos + (size_t)bs * (size_t)ch;
    if (need > ctx->cap)
    {
        size_t newcap = ctx->cap ? ctx->cap : 4096;
        while (newcap < need) newcap *= 2;
        FLAC__int32 *p = (FLAC__int32 *)realloc(ctx->out, newcap * sizeof(FLAC__int32));
        if (!p) die("realloc decoder buf failed");
        ctx->out = p;
        ctx->cap = newcap;
    }
    for (uint32_t i = 0; i < bs; i++)
    {
        for (uint32_t c = 0; c < ch; c++)
        {
            ctx->out[ctx->pos++] = buffer[c][i];
        }
    }
    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

static void dec_meta_cb(
    const FLAC__StreamDecoder *decoder,
    const FLAC__StreamMetadata *metadata,
    void *client_data)
{
    (void)decoder;
    DecCtx *ctx = (DecCtx *)client_data;
    if (metadata->type == FLAC__METADATA_TYPE_STREAMINFO)
    {
        ctx->channels = (int)metadata->data.stream_info.channels;
        ctx->frames_total = (int)metadata->data.stream_info.total_samples;
    }
}

static void dec_error_cb(
    const FLAC__StreamDecoder *decoder,
    FLAC__StreamDecoderErrorStatus status,
    void *client_data)
{
    (void)decoder; (void)client_data;
    fprintf(stderr, "FLAC decode error: %d\n", status);
}

// --- Timing harness (copied shape from bench_sndfile.c) ----------------------

typedef struct
{
    double mean_ns;
    double stdev_ns;
    double mean_MBps;
} BenchStats;

static BenchStats bench_case(
    const char *group,
    const char *bench_id,
    const char *case_label,
    uint64_t throughput_bytes,
    void (*fn)(void *ctx),
    void *ctx)
{
    uint64_t t0 = now_ns();
    while ((now_ns() - t0) < (uint64_t)(g_warmup_s * 1e9))
    {
        fn(ctx);
    }

    double *samples = (double *)calloc((size_t)g_samples, sizeof(double));
    if (!samples) die("calloc samples failed");

    int collected = 0;
    uint64_t start = now_ns();
    uint64_t end_deadline = start + (uint64_t)(g_measure_s * 1e9);

    while (now_ns() < end_deadline && collected < g_samples)
    {
        uint64_t a = now_ns();
        fn(ctx);
        uint64_t b = now_ns();
        samples[collected++] = (double)(b - a);
    }

    if (collected == 0) die("collected 0 samples");

    double sum = 0.0;
    for (int i = 0; i < collected; i++) sum += samples[i];
    double mean = sum / (double)collected;

    double var = 0.0;
    for (int i = 0; i < collected; i++)
    {
        double d = samples[i] - mean;
        var += d * d;
    }
    var /= (double)collected;
    double stdev = sqrt(var);

    double mean_s = mean / 1e9;
    double mbps = (mean_s > 0.0) ? ((double)throughput_bytes / (1024.0 * 1024.0)) / mean_s : 0.0;

    fprintf(g_out, "%s,%s,%s,%d,%.2f,%.2f,%" PRIu64 ",%.2f\n",
            group, bench_id, case_label, collected, mean, stdev, throughput_bytes, mbps);

    free(samples);

    BenchStats s = {mean, stdev, mbps};
    return s;
}

// --- Read benchmark ----------------------------------------------------------

typedef struct
{
    char path[512];
    SampleFmt fmt;
    int frames;
    int channels;
    uint64_t file_bytes;
} ReadCtx;

static void do_read_once(void *vctx)
{
    ReadCtx *rctx = (ReadCtx *)vctx;

    FLAC__StreamDecoder *dec = FLAC__stream_decoder_new();
    if (!dec) die("FLAC__stream_decoder_new failed");

    DecCtx ctx = {0};
    ctx.channels = rctx->channels;
    ctx.frames_total = rctx->frames;
    ctx.cap = (size_t)rctx->frames * (size_t)rctx->channels;
    ctx.out = (FLAC__int32 *)malloc(ctx.cap * sizeof(FLAC__int32));
    if (!ctx.out) die("malloc dec buf failed");
    ctx.pos = 0;

    FLAC__StreamDecoderInitStatus st = FLAC__stream_decoder_init_file(
        dec, rctx->path, dec_write_cb, dec_meta_cb, dec_error_cb, &ctx);
    if (st != FLAC__STREAM_DECODER_INIT_STATUS_OK)
    {
        fprintf(stderr, "decoder init failed: %d for %s\n", st, rctx->path);
        exit(1);
    }

    if (!FLAC__stream_decoder_process_until_end_of_stream(dec))
    {
        fprintf(stderr, "decode failed for %s\n", rctx->path);
        exit(1);
    }
    FLAC__stream_decoder_finish(dec);
    FLAC__stream_decoder_delete(dec);

    black_box_ptr(ctx.out);
    black_box_u64((uint64_t)ctx.pos);
    free(ctx.out);
}

// --- Write benchmark ---------------------------------------------------------

typedef struct
{
    SampleFmt fmt;
    int sample_rate;
    int frames;
    int channels;
    uint64_t payload_bytes;
    FLAC__int32 *payload; // interleaved, already scaled to the target bit depth
} WriteCtx;

static void do_write_once(void *vctx)
{
    WriteCtx *ctx = (WriteCtx *)vctx;
    MemBuf mb = {0};
    mb.cap = (size_t)ctx->payload_bytes + 1024;
    mb.data = (uint8_t *)malloc(mb.cap);
    if (!mb.data) die("malloc membuf failed");

    size_t wrote = flac_encode_to_membuf(
        ctx->payload, ctx->frames, ctx->channels, ctx->sample_rate,
        bits_for_fmt(ctx->fmt), &mb);

    black_box_u64((uint64_t)wrote);
    black_box_u64((uint64_t)mb.size);
    free(mb.data);
}

static WriteCtx make_write_ctx(SampleFmt fmt, int sample_rate, int channels)
{
    WriteCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.fmt = fmt;
    ctx.sample_rate = sample_rate;
    ctx.channels = channels;
    ctx.frames = frames_for(sample_rate);

    uint64_t frames_u = (uint64_t)ctx.frames;
    uint64_t ch_u = (uint64_t)channels;
    ctx.payload_bytes = frames_u * ch_u * (uint64_t)bytes_per_sample_payload(fmt);

    size_t n = (size_t)ctx.frames * (size_t)channels;
    double *sig = (double *)malloc(n * sizeof(double));
    if (!sig) die("malloc write sig failed");
    generate_interleaved_f64(sig, ctx.frames, channels, sample_rate);

    FLAC__int32 *buf = (FLAC__int32 *)malloc(n * sizeof(FLAC__int32));
    if (!buf) die("malloc write i32 buf failed");
    convert_to_i32_buffer(sig, buf, n, fmt);
    ctx.payload = buf;

    free(sig);
    return ctx;
}

static void free_write_ctx(WriteCtx *ctx)
{
    free(ctx->payload);
    ctx->payload = NULL;
}

// --- CLI / main --------------------------------------------------------------

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
        {
            g_warmup_s = atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--measure") && i + 1 < argc)
        {
            g_measure_s = atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc)
        {
            g_samples = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--help"))
        {
            fprintf(stderr,
                    "Usage: %s [--warmup S] [--measure S] [--samples N]\n"
                    "Defaults: warmup=%.1f measure=%.1f samples=%d\n",
                    argv[0], g_warmup_s, g_measure_s, g_samples);
            exit(0);
        }
        else
        {
            fprintf(stderr, "Unknown arg: %s (try --help)\n", argv[i]);
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    ensure_assets_exist();
    g_out = fopen("flac_bench.csv", "w");
    if (!g_out) die_errno("failed to open output file");
    fprintf(g_out,
            "group,bench_id,case_label,samples,mean_ns,stdev_ns,throughput_bytes,MBps\n");

    // flac_read group
    for (size_t si = 0; si < sizeof(SAMPLE_RATES)/sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS)/sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];
            int frames = frames_for(sr);

            char case_label[64];
            snprintf(case_label, sizeof(case_label), "%dhz_%dch", sr, ch);

            for (SampleFmt fmt = FMT_I16; fmt <= FMT_I32; fmt++)
            {
                ReadCtx r = {0};
                r.fmt = fmt;
                r.frames = frames;
                r.channels = ch;

                asset_path(r.path, sizeof(r.path), fmt, sr, ch);
                r.file_bytes = file_size_bytes(r.path);

                char bench_id[64];
                snprintf(bench_id, sizeof(bench_id), "libflac-%s", fmt_label(fmt));

                bench_case("flac_read", bench_id, case_label, r.file_bytes, do_read_once, &r);
            }
        }
    }

    // flac_write group
    for (size_t si = 0; si < sizeof(SAMPLE_RATES)/sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS)/sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];

            char case_label[64];
            snprintf(case_label, sizeof(case_label), "%dhz_%dch", sr, ch);

            for (SampleFmt fmt = FMT_I16; fmt <= FMT_I32; fmt++)
            {
                WriteCtx w = make_write_ctx(fmt, sr, ch);

                char bench_id[64];
                snprintf(bench_id, sizeof(bench_id), "libflac-%s", fmt_label(fmt));
                bench_case("flac_write", bench_id, case_label, w.payload_bytes, do_write_once, &w);
                free_write_ctx(&w);
            }
        }
    }

    fclose(g_out);
    return 0;
}
