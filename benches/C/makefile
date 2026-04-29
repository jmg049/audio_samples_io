CC=gcc
CFLAGS=-O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic

SNDFILE_LDLIBS=$(shell pkg-config --libs sndfile) -lm
SNDFILE_CPPFLAGS=$(shell pkg-config --cflags sndfile)

FLAC_LDLIBS=$(shell pkg-config --libs flac 2>/dev/null) -lm
FLAC_CPPFLAGS=$(shell pkg-config --cflags flac 2>/dev/null)

all: bench_sndfile bench_flac

bench_sndfile: bench_sndfile.c
	$(CC) $(CFLAGS) $(SNDFILE_CPPFLAGS) $< -o $@ $(SNDFILE_LDLIBS)

bench_flac: bench_flac.c
	$(CC) $(CFLAGS) $(FLAC_CPPFLAGS) $< -o $@ $(FLAC_LDLIBS)

clean:
	rm -f bench_sndfile bench_flac
