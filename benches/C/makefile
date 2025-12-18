CC=gcc
CFLAGS=-O3 -march=native -DNDEBUG -Wall -Wextra -Wpedantic
LDLIBS=$(shell pkg-config --libs sndfile) -lm
CPPFLAGS=$(shell pkg-config --cflags sndfile)

all: bench_sndfile

bench_sndfile: bench_sndfile.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f bench_sndfile
