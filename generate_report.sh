#!/bin/bash

# Audio I/O vs Hound Benchmark Report Generator
set -e

echo "🔍 Audio I/O vs Hound Benchmark Report Generator"
echo "================================================"

# Check if criterion results exist
if [ ! -d "target/criterion" ]; then
    echo "No criterion results found."
    echo "Please run benchmarks first:"
    echo "   RUSTFLAGS=\"-Ctarget-cpu=native\" cargo bench"
    exit 1
fi

# Generate the report
echo "Generating benchmark report..."
cargo run --bin benchmark_reporter --features benchmark_reporting --release

# Check if report was generated
if [ -f "benchmark_report.md" ]; then
    echo "Report generated successfully!"
    echo "Report saved to: benchmark_report.md"
    echo ""
    echo "Quick Summary:"
    grep -A 5 "audio_samples_io wins" benchmark_report.md || echo "No summary found"
    echo ""
    echo "To view the full report:"
    echo "   b/cat benchmark_report.md"
    echo "   or open in your favorite markdown viewer"
else
    echo "Failed to generate report"
    exit 1
fi