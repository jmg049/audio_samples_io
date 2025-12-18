// bench_sndfile.c
// Build: gcc -O3 -march=native -DNDEBUG bench_sndfile.c -o bench_sndfile $(pkg-config --cflags --libs sndfile) -lm
// Run:   ./bench_sndfile
// Output: CSV to stdout.

#include <sndfile.h>

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
static const char *ASSET_DIR = "target/bench_assets_sndfile";

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
    // Fallback: volatile sink.
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
    // 250ms => sr * 0.25
    // Ensure integer frames.
    return (sample_rate * SIGNAL_DURATION_MS) / 1000;
}

static inline double frac(double x)
{
    return x - floor(x);
}

static inline double square_wave(double phase)
{
    return (phase < 0.5) ? 1.0 : -1.0;
}

static inline double saw_wave(double phase)
{
    // Range [-1, 1]
    return 2.0 * phase - 1.0;
}

static inline double chirp_phase(double t, double f0, double f1, double T)
{
    // Linear chirp: instantaneous freq f(t) = f0 + (f1-f0)*(t/T)
    // phase(t) = 2pi*(f0*t + 0.5*(f1-f0)*t^2/T)
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
            {
                double ph = 2.0 * M_PI * base_freq * t;
                x = sin(ph);
            }
            break;
            case 1:
            {
                double ph = 2.0 * M_PI * (base_freq * 1.5) * t;
                x = cos(ph);
                amp *= 0.9;
            }
            break;
            case 2:
            {
                double ph = frac(base_freq * 0.75 * t);
                x = square_wave(ph);
                amp *= 0.8;
            }
            break;
            case 3:
            {
                double ph = frac(base_freq * 1.2 * t);
                x = saw_wave(ph);
                amp *= 0.7;
            }
            break;
            default:
            {
                double ph = chirp_phase(t, base_freq, base_freq * 3.0, T);
                x = sin(ph);
                amp *= 0.85;
            }
            break;
            }

            out[(size_t)f * (size_t)channels + (size_t)ch] = x * amp;
        }
    }
}

typedef enum
{
    FMT_I16,
    FMT_I24, // stored in int32_t container
    FMT_I32,
    FMT_F32,
    FMT_F64,
} SampleFmt;

static const char *fmt_label(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16:
        return "i16";
    case FMT_I24:
        return "i24";
    case FMT_I32:
        return "i32";
    case FMT_F32:
        return "f32";
    case FMT_F64:
        return "f64";
    default:
        return "unknown";
    }
}

static int bytes_per_sample(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16:
        return 2;
    case FMT_I24:
        return 3; // on-disk payload matches 24-bit
    case FMT_I32:
        return 4;
    case FMT_F32:
        return 4;
    case FMT_F64:
        return 8;
    default:
        return 0;
    }
}

static int wav_subtype(SampleFmt f)
{
    switch (f)
    {
    case FMT_I16:
        return SF_FORMAT_PCM_16;
    case FMT_I24:
        return SF_FORMAT_PCM_24;
    case FMT_I32:
        return SF_FORMAT_PCM_32;
    case FMT_F32:
        return SF_FORMAT_FLOAT;
    case FMT_F64:
        return SF_FORMAT_DOUBLE;
    default:
        return SF_FORMAT_PCM_16;
    }
}

static void convert_to_i16(const double *in, int16_t *out, size_t n)
{
    const double scale = 32767.0;
    for (size_t i = 0; i < n; i++)
    {
        double x = in[i];
        if (x > 1.0)
            x = 1.0;
        if (x < -1.0)
            x = -1.0;
        long v = lround(x * scale);
        if (v > 32767)
            v = 32767;
        if (v < -32768)
            v = -32768;
        out[i] = (int16_t)v;
    }
}

static void convert_to_i24_in_i32(const double *in, int32_t *out, size_t n)
{
    // Store 24-bit audio in a 32-bit container (libsndfile common pattern).
    const double scale = 8388607.0; // 2^23 - 1
    for (size_t i = 0; i < n; i++)
    {
        double x = in[i];
        if (x > 1.0)
            x = 1.0;
        if (x < -1.0)
            x = -1.0;
        long v = lround(x * scale);
        if (v > 8388607)
            v = 8388607;
        if (v < -8388608)
            v = -8388608;
        out[i] = (int32_t)v;
    }
}

static void convert_to_i32(const double *in, int32_t *out, size_t n)
{
    const double scale = 2147483647.0; // 2^31 - 1
    for (size_t i = 0; i < n; i++)
    {
        double x = in[i];
        if (x > 1.0)
            x = 1.0;
        if (x < -1.0)
            x = -1.0;
        long long v = llround(x * scale);
        if (v > 2147483647LL)
            v = 2147483647LL;
        if (v < -2147483648LL)
            v = -2147483648LL;
        out[i] = (int32_t)v;
    }
}

static void convert_to_f32(const double *in, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = (float)in[i];
}

static void convert_to_f64(const double *in, double *out, size_t n)
{
    memcpy(out, in, n * sizeof(double));
}

static void asset_path(char *dst, size_t cap, SampleFmt fmt, int sample_rate, int channels)
{
    snprintf(dst, cap, "%s/%s_%dhz_%dch.wav", ASSET_DIR, fmt_label(fmt), sample_rate, channels);
}

static void write_wav_asset(SampleFmt fmt, int sample_rate, int channels)
{
    ensure_dir(ASSET_DIR);

    int frames = frames_for(sample_rate);
    size_t n = (size_t)frames * (size_t)channels;

    double *sig = (double *)malloc(n * sizeof(double));
    if (!sig)
        die("malloc sig failed");
    generate_interleaved_f64(sig, frames, channels, sample_rate);

    SF_INFO info;
    memset(&info, 0, sizeof(info));
    info.samplerate = sample_rate;
    info.channels = channels;
    info.format = SF_FORMAT_WAV | wav_subtype(fmt);

    char path[512];
    asset_path(path, sizeof(path), fmt, sample_rate, channels);

    SNDFILE *sf = sf_open(path, SFM_WRITE, &info);
    if (!sf)
    {
        fprintf(stderr, "sf_open(write) failed for %s: %s\n", path, sf_strerror(NULL));
        exit(1);
    }

    sf_count_t wrote = 0;
    if (fmt == FMT_I16)
    {
        int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
        if (!buf)
            die("malloc i16 buf failed");
        convert_to_i16(sig, buf, n);
        wrote = sf_writef_short(sf, (const short *)buf, frames);
        free(buf);
    }
    else if (fmt == FMT_I24)
    {
        int32_t *buf = (int32_t *)malloc(n * sizeof(int32_t));
        if (!buf)
            die("malloc i24 buf failed");
        convert_to_i24_in_i32(sig, buf, n);
        wrote = sf_writef_int(sf, (const int *)buf, frames);
        free(buf);
    }
    else if (fmt == FMT_I32)
    {
        int32_t *buf = (int32_t *)malloc(n * sizeof(int32_t));
        if (!buf)
            die("malloc i32 buf failed");
        convert_to_i32(sig, buf, n);
        wrote = sf_writef_int(sf, (const int *)buf, frames);
        free(buf);
    }
    else if (fmt == FMT_F32)
    {
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf)
            die("malloc f32 buf failed");
        convert_to_f32(sig, buf, n);
        wrote = sf_writef_float(sf, (const float *)buf, frames);
        free(buf);
    }
    else if (fmt == FMT_F64)
    {
        double *buf = (double *)malloc(n * sizeof(double));
        if (!buf)
            die("malloc f64 buf failed");
        convert_to_f64(sig, buf, n);
        wrote = sf_writef_double(sf, (const double *)buf, frames);
        free(buf);
    }

    if (wrote != frames)
    {
        fprintf(stderr, "asset write short for %s: wrote=%" PRId64 " frames=%d\n",
                path, (int64_t)wrote, frames);
        exit(1);
    }

    sf_close(sf);
    free(sig);
}

static void ensure_assets_exist(void)
{
    for (size_t si = 0; si < sizeof(SAMPLE_RATES) / sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS) / sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];

            for (SampleFmt fmt = FMT_I16; fmt <= FMT_F64; fmt++)
            {
                char path[512];
                asset_path(path, sizeof(path), fmt, sr, ch);
                struct stat st;
                if (stat(path, &st) != 0)
                {
                    write_wav_asset(fmt, sr, ch);
                }
            }
        }
    }
}

typedef struct
{
    uint8_t *data;
    sf_count_t size;
    sf_count_t pos;
    sf_count_t cap;
} MemBuf;

static sf_count_t vio_get_filelen(void *user_data)
{
    MemBuf *m = (MemBuf *)user_data;
    return m->size;
}

static sf_count_t vio_seek(sf_count_t offset, int whence, void *user_data)
{
    MemBuf *m = (MemBuf *)user_data;
    sf_count_t base = 0;

    if (whence == SEEK_SET)
        base = 0;
    else if (whence == SEEK_CUR)
        base = m->pos;
    else if (whence == SEEK_END)
        base = m->size;
    else
        return -1;

    sf_count_t next = base + offset;
    if (next < 0)
        return -1;
    m->pos = next;
    return m->pos;
}

static sf_count_t vio_read(void *ptr, sf_count_t count, void *user_data)
{
    MemBuf *m = (MemBuf *)user_data;
    sf_count_t remain = m->size - m->pos;
    if (remain <= 0)
        return 0;
    if (count > remain)
        count = remain;
    memcpy(ptr, m->data + m->pos, (size_t)count);
    m->pos += count;
    return count;
}

static void mem_grow(MemBuf *m, sf_count_t want)
{
    if (want <= m->cap)
        return;
    sf_count_t newcap = m->cap ? m->cap : 4096;
    while (newcap < want)
        newcap *= 2;
    uint8_t *p = (uint8_t *)realloc(m->data, (size_t)newcap);
    if (!p)
        die("realloc failed");
    m->data = p;
    m->cap = newcap;
}

static sf_count_t vio_write(const void *ptr, sf_count_t count, void *user_data)
{
    MemBuf *m = (MemBuf *)user_data;
    sf_count_t end = m->pos + count;
    mem_grow(m, end);
    memcpy(m->data + m->pos, ptr, (size_t)count);
    m->pos = end;
    if (m->pos > m->size)
        m->size = m->pos;
    return count;
}

static sf_count_t vio_tell(void *user_data)
{
    MemBuf *m = (MemBuf *)user_data;
    return m->pos;
}

static SF_VIRTUAL_IO make_vio(void)
{
    SF_VIRTUAL_IO vio;
    vio.get_filelen = vio_get_filelen;
    vio.seek = vio_seek;
    vio.read = vio_read;
    vio.write = vio_write;
    vio.tell = vio_tell;
    return vio;
}

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
    // Warmup
    uint64_t t0 = now_ns();
    while ((now_ns() - t0) < (uint64_t)(g_warmup_s * 1e9))
    {
        fn(ctx);
    }

    // Measurement: collect up to g_samples, but keep going until g_measure_s elapsed
    double *samples = (double *)calloc((size_t)g_samples, sizeof(double));
    if (!samples)
        die("calloc samples failed");

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

    // If time was short, still compute with what we have
    if (collected == 0)
        die("collected 0 samples");

    double sum = 0.0;
    for (int i = 0; i < collected; i++)
        sum += samples[i];
    double mean = sum / (double)collected;

    double var = 0.0;
    for (int i = 0; i < collected; i++)
    {
        double d = samples[i] - mean;
        var += d * d;
    }
    var /= (double)collected;
    double stdev = sqrt(var);

    // Throughput (MB/s): bytes per iteration / seconds per iteration
    double mean_s = mean / 1e9;
    double mbps = (mean_s > 0.0) ? ((double)throughput_bytes / (1024.0 * 1024.0)) / mean_s : 0.0;

    // CSV line
    // group,bench_id,case_label,samples,mean_ns,stdev_ns,throughput_bytes,MBps
    fprintf(g_out, "%s,%s,%s,%d,%.2f,%.2f,%" PRIu64 ",%.2f\n",
            group, bench_id, case_label, collected, mean, stdev, throughput_bytes, mbps);

    free(samples);

    BenchStats s;
    s.mean_ns = mean;
    s.stdev_ns = stdev;
    s.mean_MBps = mbps;
    return s;
}

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
    ReadCtx *ctx = (ReadCtx *)vctx;

    SF_INFO info;
    memset(&info, 0, sizeof(info));
    SNDFILE *sf = sf_open(ctx->path, SFM_READ, &info);
    if (!sf)
    {
        fprintf(stderr, "sf_open(read) failed for %s: %s\n", ctx->path, sf_strerror(NULL));
        exit(1);
    }

    int frames = ctx->frames;
    int channels = ctx->channels;
    size_t n = (size_t)frames * (size_t)channels;

    sf_count_t got = 0;
    if (ctx->fmt == FMT_I16)
    {
        int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
        if (!buf)
            die("malloc read i16 failed");
        got = sf_readf_short(sf, (short *)buf, frames);
        black_box_ptr(buf);
        free(buf);
    }
    else if (ctx->fmt == FMT_I24 || ctx->fmt == FMT_I32)
    {
        int32_t *buf = (int32_t *)malloc(n * sizeof(int32_t));
        if (!buf)
            die("malloc read i32 failed");
        got = sf_readf_int(sf, (int *)buf, frames);
        black_box_ptr(buf);
        free(buf);
    }
    else if (ctx->fmt == FMT_F32)
    {
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf)
            die("malloc read f32 failed");
        got = sf_readf_float(sf, buf, frames);
        black_box_ptr(buf);
        free(buf);
    }
    else if (ctx->fmt == FMT_F64)
    {
        double *buf = (double *)malloc(n * sizeof(double));
        if (!buf)
            die("malloc read f64 failed");
        got = sf_readf_double(sf, buf, frames);
        black_box_ptr(buf);
        free(buf);
    }

    black_box_u64((uint64_t)got);
    sf_close(sf);
}

typedef struct
{
    SampleFmt fmt;
    int sample_rate;
    int frames;
    int channels;
    uint64_t payload_bytes;
    void *payload;
} WriteCtx;

static void do_write_once(void *vctx)
{
    WriteCtx *ctx = (WriteCtx *)vctx;

    SF_INFO info;
    memset(&info, 0, sizeof(info));
    info.samplerate = ctx->sample_rate;
    info.channels = ctx->channels;
    info.format = SF_FORMAT_WAV | wav_subtype(ctx->fmt);

    MemBuf m;
    memset(&m, 0, sizeof(m));
    m.cap = (sf_count_t)ctx->payload_bytes + 1024;
    m.data = (uint8_t *)malloc((size_t)m.cap);
    if (!m.data)
        die("malloc membuf failed");
    m.size = 0;
    m.pos = 0;

    SF_VIRTUAL_IO vio = make_vio();
    SNDFILE *sf = sf_open_virtual(&vio, SFM_WRITE, &info, &m);
    if (!sf)
    {
        fprintf(stderr, "sf_open_virtual(write) failed: %s\n", sf_strerror(NULL));
        exit(1);
    }

    sf_count_t wrote = 0;
    if (ctx->fmt == FMT_I16)
    {
        wrote = sf_writef_short(sf, (const short *)ctx->payload, ctx->frames);
    }
    else if (ctx->fmt == FMT_I24 || ctx->fmt == FMT_I32)
    {
        wrote = sf_writef_int(sf, (const int *)ctx->payload, ctx->frames);
    }
    else if (ctx->fmt == FMT_F32)
    {
        wrote = sf_writef_float(sf, (const float *)ctx->payload, ctx->frames);
    }
    else if (ctx->fmt == FMT_F64)
    {
        wrote = sf_writef_double(sf, (const double *)ctx->payload, ctx->frames);
    }

    sf_close(sf);

    black_box_u64((uint64_t)wrote);
    black_box_u64((uint64_t)m.size);
    free(m.data);
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
    ctx.payload_bytes = frames_u * ch_u * (uint64_t)bytes_per_sample(fmt);

    size_t n = (size_t)ctx.frames * (size_t)channels;

    double *sig = (double *)malloc(n * sizeof(double));
    if (!sig)
        die("malloc write sig failed");
    generate_interleaved_f64(sig, ctx.frames, channels, sample_rate);

    if (fmt == FMT_I16)
    {
        int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
        if (!buf)
            die("malloc write i16 buf failed");
        convert_to_i16(sig, buf, n);
        ctx.payload = buf;
    }
    else if (fmt == FMT_I24)
    {
        int32_t *buf = (int32_t *)malloc(n * sizeof(int32_t));
        if (!buf)
            die("malloc write i24 buf failed");
        convert_to_i24_in_i32(sig, buf, n);
        ctx.payload = buf;
    }
    else if (fmt == FMT_I32)
    {
        int32_t *buf = (int32_t *)malloc(n * sizeof(int32_t));
        if (!buf)
            die("malloc write i32 buf failed");
        convert_to_i32(sig, buf, n);
        ctx.payload = buf;
    }
    else if (fmt == FMT_F32)
    {
        float *buf = (float *)malloc(n * sizeof(float));
        if (!buf)
            die("malloc write f32 buf failed");
        convert_to_f32(sig, buf, n);
        ctx.payload = buf;
    }
    else if (fmt == FMT_F64)
    {
        double *buf = (double *)malloc(n * sizeof(double));
        if (!buf)
            die("malloc write f64 buf failed");
        convert_to_f64(sig, buf, n);
        ctx.payload = buf;
    }

    free(sig);
    return ctx;
}

static void free_write_ctx(WriteCtx *ctx)
{
    free(ctx->payload);
    ctx->payload = NULL;
}

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
    g_out = fopen("sndfile_bench.csv", "w");
    if (!g_out)
        die_errno("failed to open output file");
    fprintf(g_out,
            "group,bench_id,case_label,samples,mean_ns,stdev_ns,throughput_bytes,MBps\n");

    // wav_read group
    for (size_t si = 0; si < sizeof(SAMPLE_RATES) / sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS) / sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];
            int frames = frames_for(sr);

            char case_label[64];
            snprintf(case_label, sizeof(case_label), "%dhz_%dch", sr, ch);

            for (SampleFmt fmt = FMT_I16; fmt <= FMT_F64; fmt++)
            {
                ReadCtx r;
                memset(&r, 0, sizeof(r));
                r.fmt = fmt;
                r.frames = frames;
                r.channels = ch;

                asset_path(r.path, sizeof(r.path), fmt, sr, ch);
                r.file_bytes = file_size_bytes(r.path);

                char bench_id[64];
                snprintf(bench_id, sizeof(bench_id), "sndfile-%s", fmt_label(fmt));

                bench_case("wav_read", bench_id, case_label, r.file_bytes, do_read_once, &r);
            }
        }
    }

    for (size_t si = 0; si < sizeof(SAMPLE_RATES) / sizeof(SAMPLE_RATES[0]); si++)
    {
        for (size_t ci = 0; ci < sizeof(CHANNEL_OPTIONS) / sizeof(CHANNEL_OPTIONS[0]); ci++)
        {
            int sr = SAMPLE_RATES[si];
            int ch = CHANNEL_OPTIONS[ci];

            char case_label[64];
            snprintf(case_label, sizeof(case_label), "%dhz_%dch", sr, ch);

            for (SampleFmt fmt = FMT_I16; fmt <= FMT_F64; fmt++)
            {
                WriteCtx w = make_write_ctx(fmt, sr, ch);

                char bench_id[64];
                snprintf(bench_id, sizeof(bench_id), "sndfile-%s", fmt_label(fmt));
                bench_case("wav_write", bench_id, case_label, w.payload_bytes, do_write_once, &w);
                free_write_ctx(&w);
            }
        }
    }
    fclose(g_out);
    return 0;
}
