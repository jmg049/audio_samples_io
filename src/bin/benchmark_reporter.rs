#![allow(unused)]

#[cfg(not(feature = "benchmark_reporting"))]
fn main() {
    eprintln!("Error: benchmark_reporter requires the 'benchmark_reporting' feature.");
    eprintln!("Enable it with: cargo run --bin benchmark_reporter --features benchmark_reporting");
    std::process::exit(1);
}

#[cfg(feature = "benchmark_reporting")]
mod benchmark_report {
    use serde::Deserialize;
    use sysinfo::{Disks, System};
    use walkdir::WalkDir;

    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    #[derive(Debug, Deserialize)]
    struct Estimates {
        mean: StatEstimate,
        median: StatEstimate,
        std_dev: StatEstimate,
    }

    #[derive(Debug, Deserialize)]
    struct StatEstimate {
        point_estimate: f64,
        confidence_interval: ConfidenceInterval,
        standard_error: f64,
    }

    #[derive(Debug, Deserialize)]
    struct ConfidenceInterval {
        confidence_level: f64,
        lower_bound: f64,
        upper_bound: f64,
    }

    #[derive(Debug, Deserialize)]
    struct BenchmarkInfo {
        group_id: String,
        function_id: String,
        value_str: String,
        throughput: Option<Throughput>,
        full_id: String,
        title: String,
    }

    #[derive(Debug, Deserialize)]
    struct Throughput {
        #[serde(rename = "Elements")]
        elements: Option<u64>,
    }

    #[derive(Debug, Clone)]
    struct BenchmarkResult {
        library: String,
        operation: String, // read or write
        format: String,
        duration: String,
        sample_rate: Option<u32>,
        channels: Option<u32>,
        mean_time_ns: f64,
        std_dev_ns: f64,
        throughput_samples_per_sec: Option<f64>,
        sample_count: Option<u64>,
        confidence_interval_low: f64,
        confidence_interval_high: f64,
    }

    // Parse duration strings like "44100hz_2ch" into components. Returns
    // (sample_rate_opt, channels_opt, normalized_label).
    fn parse_duration(duration: &str) -> (Option<u32>, Option<u32>, String) {
        let s = duration.to_lowercase();

        // Try to extract sample rate (digits before "hz")
        let mut sample_rate: Option<u32> = None;
        if let Some(pos) = s.find("hz") {
            let prefix = &s[..pos];
            // find trailing digits in prefix
            let digits: String = prefix
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            if !digits.is_empty() {
                if let Ok(sr) = digits.parse::<u32>() {
                    sample_rate = Some(sr);
                }
            }
        }

        // Try to extract channels (digits before "ch")
        let mut channels: Option<u32> = None;
        if let Some(pos) = s.find("ch") {
            // scan backwards for digits before pos
            let mut i = pos;
            let bytes = s.as_bytes();
            // move left to find start of digits
            let mut start = None;
            while i > 0 {
                i -= 1;
                if bytes[i].is_ascii_digit() {
                    start = Some(i);
                } else if start.is_some() {
                    break;
                }
            }
            if let Some(start_idx) = start {
                // collect digits from start_idx up to pos
                let dig = &s[start_idx..pos];
                if let Ok(ch) = dig.parse::<u32>() {
                    channels = Some(ch);
                }
            }
        }

        // Build a normalized label
        let label = if let (Some(sr), Some(ch)) = (sample_rate, channels) {
            format!("{} Hz, {} ch", sr, ch)
        } else if let Some(sr) = sample_rate {
            format!("{} Hz", sr)
        } else {
            duration.to_string()
        };

        (sample_rate, channels, label)
    }

    fn parse_benchmark_name(group_id: &str) -> (String, String, String) {
        // Handle audio_io vs hound parsing differently
        if let Some(rest) = group_id.strip_prefix("audio-") {
            // Remove "audio_io_"
            let parts: Vec<&str> = rest.split('_').collect();

            if parts.len() >= 2 {
                let operation = parts[0].to_string(); // read or write
                let mut format = parts[1..].join("_"); // everything after operation

                // Normalize format names for comparison
                if format == "i24_only" {
                    format = "i24".to_string();
                }

                ("audio_io".to_string(), operation, format)
            } else {
                (
                    group_id.to_string(),
                    "unknown".to_string(),
                    "unknown".to_string(),
                )
            }
        } else if let Some(rest) = group_id.strip_prefix("hound-") {
            // Remove "hound_"
            let parts: Vec<&str> = rest.split('_').collect();

            if parts.len() >= 2 {
                let operation = parts[0].to_string(); // read or write
                let mut format = parts[1..].join("_"); // everything after operation

                // Normalize format names for comparison
                if format == "f64_unsupported" {
                    format = "f64".to_string();
                }

                ("hound".to_string(), operation, format)
            } else {
                (
                    group_id.to_string(),
                    "unknown".to_string(),
                    "unknown".to_string(),
                )
            }
        } else {
            (
                group_id.to_string(),
                "unknown".to_string(),
                "unknown".to_string(),
            )
        }
    }

    // Parse both the Criterion `group_id` (e.g. "wav_read") and the
    // `function_id` (e.g. "audio-I24") from benchmark.json metadata into
    // (library, operation, format).
    fn parse_group_and_function(group_id: &str, function_id: &str) -> (String, String, String) {
        let operation = if group_id.to_lowercase().contains("read") {
            "read".to_string()
        } else if group_id.to_lowercase().contains("write") {
            "write".to_string()
        } else {
            "unknown".to_string()
        };

        let fid = function_id.to_string();
        let fid_lc = fid.to_lowercase();

        let (library, mut format) = if fid_lc.starts_with("audio-") || fid_lc.starts_with("audio_")
        {
            let parts: Vec<&str> = fid.split(|c| c == '-' || c == '_').collect();
            let fmt = parts
                .get(1)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            ("audio_io".to_string(), fmt)
        } else if fid_lc.starts_with("hound-") || fid_lc.starts_with("hound_") {
            let parts: Vec<&str> = fid.split(|c| c == '-' || c == '_').collect();
            let fmt = parts
                .get(1)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            ("hound".to_string(), fmt)
        } else {
            let parts: Vec<&str> = fid.split(|c: char| !c.is_alphanumeric()).collect();
            if parts.len() >= 2 {
                let lib = parts[0].to_string();
                let fmt = parts[1].to_string();
                (lib, fmt)
            } else {
                (fid_lc.clone(), "unknown".to_string())
            }
        };

        if format.eq_ignore_ascii_case("i24_only") {
            format = "i24".to_string();
        }
        if format.eq_ignore_ascii_case("f64_unsupported") {
            format = "f64".to_string();
        }

        (library, operation, format.to_lowercase())
    }

    fn scan_criterion_results() -> Result<Vec<BenchmarkResult>, Box<dyn std::error::Error>> {
        let criterion_dir = Path::new("target/criterion");

        if !criterion_dir.exists() {
            return Err(
                "Criterion results directory not found. Please run benchmarks first.".into(),
            );
        }

        let mut results = Vec::new();

        // Walk the criterion directory and locate all `benchmark.json` files.
        for entry in WalkDir::new(criterion_dir)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            if entry.file_name() != "benchmark.json" {
                continue;
            }

            let bench_path = entry.path();

            // Only consider benchmark.json files that live under a `new` directory
            // (Criterion writes the latest results into a `new/` folder).
            let parent_name = bench_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");

            if parent_name != "new" {
                continue;
            }

            let benchmark_content = fs::read_to_string(bench_path)?;
            let benchmark_info: BenchmarkInfo = serde_json::from_str(&benchmark_content)?;

            // Skip unsupported placeholders
            if benchmark_info.group_id.contains("unsupported")
                || benchmark_info.function_id.contains("unsupported")
            {
                continue;
            }

            // Filter to audio/hound function ids or wav_read/wav_write group ids
            if !benchmark_info.function_id.to_lowercase().contains("audio")
                && !benchmark_info.function_id.to_lowercase().contains("hound")
                && !benchmark_info.group_id.to_lowercase().contains("wav_read")
                && !benchmark_info.group_id.to_lowercase().contains("wav_write")
            {
                continue;
            }

            let (library, operation, format) =
                parse_group_and_function(&benchmark_info.group_id, &benchmark_info.function_id);

            let duration = benchmark_info.value_str.clone();

            // Log the resolved mapping for debugging and verification
            println!(
                "Parsed benchmark: group='{}' function='{}' => library='{}' operation='{}' format='{}' duration='{}' path={}",
                benchmark_info.group_id,
                benchmark_info.function_id,
                library,
                operation,
                format,
                duration,
                bench_path.display()
            );

            // Parent of benchmark.json should be the `new` directory
            let new_path = bench_path.parent().unwrap_or(criterion_dir);

            if let Ok(result) =
                load_benchmark_result(new_path, &library, &operation, &format, &duration)
            {
                results.push(result);
            } else {
                eprintln!(
                    "Warning: failed to load result for {}",
                    bench_path.display()
                );
            }
        }

        Ok(results)
    }

    fn scan_benchmark_group(
        group_path: &Path,
        group_id: &str,
        results: &mut Vec<BenchmarkResult>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (library, operation, format) = parse_benchmark_name(group_id);

        // Look for function subdirectories
        for entry in WalkDir::new(group_path) {
            let entry = entry?;
            let func_path = entry.path();
            println!("Checking function path: {:?}", func_path);
            if func_path.is_dir() {
                scan_function_benchmarks(&func_path, &library, &operation, &format, results)?;
            }
        }

        Ok(())
    }

    fn scan_function_benchmarks(
        func_path: &Path,
        library: &str,
        operation: &str,
        format: &str,
        results: &mut Vec<BenchmarkResult>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Look for duration subdirectories (short, medium, long)
        for entry in fs::read_dir(func_path)? {
            let entry = entry?;
            let duration_path = entry.path();

            if duration_path.is_dir() {
                let duration = duration_path.file_name().unwrap().to_string_lossy();
                let new_path = duration_path.join("new");

                if new_path.exists()
                    && let Ok(result) =
                        load_benchmark_result(&new_path, library, operation, format, &duration)
                {
                    results.push(result);
                }
            }
        }

        Ok(())
    }

    fn load_benchmark_result(
        new_path: &Path,
        library: &str,
        operation: &str,
        format: &str,
        duration: &str,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let estimates_path = new_path.join("estimates.json");
        let benchmark_path = new_path.join("benchmark.json");

        let estimates_content = fs::read_to_string(&estimates_path)?;
        let benchmark_content = fs::read_to_string(&benchmark_path)?;

        let estimates: Estimates = serde_json::from_str(&estimates_content)?;
        let benchmark_info: BenchmarkInfo = serde_json::from_str(&benchmark_content)?;

        let sample_count = benchmark_info.throughput.as_ref().and_then(|t| t.elements);
        let throughput_samples_per_sec = sample_count
            .map(|count| count as f64 / (estimates.mean.point_estimate / 1_000_000_000.0));

        // Normalize duration into structured fields and a readable label
        let (sample_rate, channels, normalized_label) = parse_duration(duration);

        Ok(BenchmarkResult {
            library: library.to_string(),
            operation: operation.to_string(),
            format: format.to_string(),
            duration: normalized_label,
            sample_rate,
            channels,
            mean_time_ns: estimates.mean.point_estimate,
            std_dev_ns: estimates.std_dev.point_estimate,
            throughput_samples_per_sec,
            sample_count,
            confidence_interval_low: estimates.mean.confidence_interval.lower_bound,
            confidence_interval_high: estimates.mean.confidence_interval.upper_bound,
        })
    }

    fn format_time(ns: f64) -> String {
        if ns >= 1_000_000_000.0 {
            format!("{:.2} s", ns / 1_000_000_000.0)
        } else if ns >= 1_000_000.0 {
            format!("{:.2} ms", ns / 1_000_000.0)
        } else if ns >= 1_000.0 {
            format!("{:.2} μs", ns / 1_000.0)
        } else {
            format!("{:.2} ns", ns)
        }
    }

    fn format_throughput(throughput: f64) -> String {
        if throughput >= 1_000_000.0 {
            format!("{:.2} M samples/s", throughput / 1_000_000.0)
        } else if throughput >= 1_000.0 {
            format!("{:.2} K samples/s", throughput / 1_000.0)
        } else {
            format!("{:.2} samples/s", throughput)
        }
    }

    fn calculate_performance_ratio(audio_io_time: f64, hound_time: f64) -> f64 {
        hound_time / audio_io_time
    }

    fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: f64 = 1024.0;

        if bytes == 0 {
            return "0 B".to_string();
        }

        let bytes_f = bytes as f64;
        let unit_index = (bytes_f.log(THRESHOLD) as usize).min(UNITS.len() - 1);
        let value = bytes_f / THRESHOLD.powi(unit_index as i32);

        format!("{:.1} {}", value, UNITS[unit_index])
    }

    fn generate_system_info() -> String {
        let mut system = System::new_all();
        system.refresh_all();

        let mut info = String::new();

        info.push_str("\n## System Environment\n\n");

        // CPU Information
        info.push_str("### CPU Information\n\n");
        if let Some(cpu) = system.cpus().first() {
            info.push_str(&format!("- **CPU Model**: {}\n", cpu.brand()));
            info.push_str(&format!(
                "- **Core Count**: {} physical cores\n",
                System::physical_core_count().unwrap_or(0)
            ));
            info.push_str(&format!(
                "- **Thread Count**: {} logical cores\n",
                system.cpus().len()
            ));

            // CPU frequency (average across cores)
            let avg_freq: u64 = system.cpus().iter().map(|cpu| cpu.frequency()).sum::<u64>()
                / system.cpus().len() as u64;
            info.push_str(&format!(
                "- **Base Frequency**: {:.1} GHz\n",
                avg_freq as f64 / 1000.0
            ));

            // Architecture detection
            let arch = std::env::consts::ARCH;
            info.push_str(&format!("- **Architecture**: {}\n", arch));
        }

        // Memory Information
        info.push_str("\n### Memory Information\n\n");
        let total_memory = system.total_memory();
        let available_memory = system.available_memory();
        let used_memory = system.used_memory();

        info.push_str(&format!(
            "- **Total RAM**: {}\n",
            format_bytes(total_memory)
        ));
        info.push_str(&format!(
            "- **Available RAM**: {}\n",
            format_bytes(available_memory)
        ));
        info.push_str(&format!(
            "- **Used RAM**: {} ({:.1}%)\n",
            format_bytes(used_memory),
            (used_memory as f64 / total_memory as f64) * 100.0
        ));

        let total_swap = system.total_swap();
        if total_swap > 0 {
            info.push_str(&format!("- **Total Swap**: {}\n", format_bytes(total_swap)));
        }

        // Operating System Information
        info.push_str("\n### Operating System\n\n");
        info.push_str(&format!(
            "- **OS Name**: {}\n",
            System::name().unwrap_or_else(|| "Unknown".to_string())
        ));
        info.push_str(&format!(
            "- **OS Version**: {}\n",
            System::os_version().unwrap_or_else(|| "Unknown".to_string())
        ));
        info.push_str(&format!(
            "- **Kernel Version**: {}\n",
            System::kernel_version().unwrap_or_else(|| "Unknown".to_string())
        ));

        // Disk Information (for the current disk)
        info.push_str("\n### Storage Information\n\n");
        let current_dir = std::env::current_dir().unwrap_or_default();

        let mut disks = Disks::new_with_refreshed_list();
        for disk in &disks {
            let mount_point = disk.mount_point();
            if current_dir.starts_with(mount_point) {
                info.push_str(&format!("- **Disk Type**: {:?}\n", disk.kind()));
                info.push_str(&format!("- **File System**: {:?}\n", disk.file_system()));
                break;
            }
        }

        // Load Average (Linux/macOS)
        let load_avg = System::load_average();
        if load_avg.one > 0.0 || load_avg.five > 0.0 || load_avg.fifteen > 0.0 {
            info.push_str("\n### System Load\n\n");
            info.push_str(&format!(
                "- **Load Average (1/5/15 min)**: {:.2} / {:.2} / {:.2}\n",
                load_avg.one, load_avg.five, load_avg.fifteen
            ));
        }

        // Compilation target information
        info.push_str("\n### Compilation Target\n\n");
        info.push_str(&format!(
            "- **Target Triple**: {}\n",
            std::env::consts::ARCH
        ));
        info.push_str(&format!("- **Target OS**: {}\n", std::env::consts::OS));
        info.push_str(&format!(
            "- **Target Family**: {}\n",
            std::env::consts::FAMILY
        ));

        // Rust compilation info
        if let Ok(rustc_version) = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            && let Ok(version_str) = std::str::from_utf8(&rustc_version.stdout)
        {
            info.push_str(&format!("- **Rust Compiler**: {}\n", version_str.trim()));
        }

        // Check for CPU features
        info.push_str("\n### Performance Features\n\n");

        if cfg!(target_feature = "sse4.2") {
            info.push_str("- **SSE4.2**: Enabled\n");
        }

        // Check build profile
        if cfg!(debug_assertions) {
            info.push_str("- **Build Mode**: Debug (use --release for benchmarks)\n");
        } else {
            info.push_str("- **Build Mode**: Release\n");
        }

        info
    }

    fn generate_report(results: Vec<BenchmarkResult>) -> String {
        let mut report = String::new();

        report.push_str("# Audio I/O vs Hound Benchmark Report\n\n");
        report.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        // Group results by operation and format
        let mut grouped: BTreeMap<
            String,
            BTreeMap<String, BTreeMap<String, Vec<&BenchmarkResult>>>,
        > = BTreeMap::new();

        for result in &results {
            grouped
                .entry(result.operation.clone())
                .or_default()
                .entry(result.format.clone())
                .or_default()
                .entry(result.duration.clone())
                .or_default()
                .push(result);
        }

        // Summary table
        report.push_str("## Executive Summary\n\n");
        report.push_str("| Operation | Format | Duration | audio_io | hound | Ratio* |\n");
        report.push_str("|-----------|--------|----------|----------|-------|--------|\n");

        for (operation, formats) in &grouped {
            for (format, durations) in formats {
                for (duration, results_for_duration) in durations {
                    let audio_io = results_for_duration
                        .iter()
                        .find(|r| r.library == "audio_io");
                    let hound = results_for_duration.iter().find(|r| r.library == "hound");

                    if let Some(audio_io_result) = audio_io {
                        let audio_io_time = format_time(audio_io_result.mean_time_ns);

                        if let Some(hound_result) = hound {
                            let hound_time = format_time(hound_result.mean_time_ns);
                            let ratio = calculate_performance_ratio(
                                audio_io_result.mean_time_ns,
                                hound_result.mean_time_ns,
                            );
                            let ratio_str = if ratio > 1.0 {
                                format!("{:.2}x faster", ratio)
                            } else {
                                format!("{:.2}x slower", 1.0 / ratio)
                            };
                            report.push_str(&format!(
                                "| {} | {} | {} | {} | {} | {} |\n",
                                operation, format, duration, audio_io_time, hound_time, ratio_str
                            ));
                        } else {
                            report.push_str(&format!(
                                "| {} | {} | {} | {} | N/A | audio_io only |\n",
                                operation, format, duration, audio_io_time
                            ));
                        }
                    }
                }
            }
        }

        report.push_str("\n*Ratio > 1.0 means audio_io is faster\n\n");

        // Detailed results by operation
        for (operation, formats) in &grouped {
            report.push_str(&format!("## {} Results\n\n", operation.to_uppercase()));

            for (format, durations) in formats {
                report.push_str(&format!("### {} Format\n\n", format.to_uppercase()));

                report.push_str(
                    "| Duration | Library | Mean Time | Throughput | Std Dev | 95% CI |\n",
                );
                report.push_str(
                    "|----------|---------|-----------|------------|---------|--------|\n",
                );

                for (duration, results_for_duration) in durations {
                    for result in results_for_duration {
                        let throughput_str = result
                            .throughput_samples_per_sec
                            .map(format_throughput)
                            .unwrap_or_else(|| "N/A".to_string());

                        let ci_str = format!(
                            "±{:.1}%",
                            (result.confidence_interval_high - result.confidence_interval_low)
                                / result.mean_time_ns
                                * 50.0
                        ); // Rough percentage

                        report.push_str(&format!(
                            "| {} | {} | {} | {} | {} | {} |\n",
                            duration,
                            result.library,
                            format_time(result.mean_time_ns),
                            throughput_str,
                            format_time(result.std_dev_ns),
                            ci_str
                        ));
                    }
                }

                report.push('\n');
            }
        }

        // Performance analysis
        report.push_str("## Performance Analysis\n\n");

        // Calculate averages and best/worst cases
        let mut audio_io_faster_count = 0;
        let mut total_comparisons = 0;
        let mut best_ratio = 1.0;
        let mut worst_ratio = 1.0;

        for formats in grouped.values() {
            for durations in formats.values() {
                for results_for_duration in durations.values() {
                    let audio_io = results_for_duration
                        .iter()
                        .find(|r| r.library == "audio_io");
                    let hound = results_for_duration.iter().find(|r| r.library == "hound");

                    if let (Some(audio_io_result), Some(hound_result)) = (audio_io, hound) {
                        total_comparisons += 1;
                        let ratio = calculate_performance_ratio(
                            audio_io_result.mean_time_ns,
                            hound_result.mean_time_ns,
                        );

                        if ratio > 1.0 {
                            audio_io_faster_count += 1;
                        }

                        if ratio > best_ratio {
                            best_ratio = ratio;
                        }

                        if ratio < worst_ratio {
                            worst_ratio = ratio;
                        }
                    }
                }
            }
        }

        if total_comparisons > 0 {
            let faster_percentage =
                (audio_io_faster_count as f64 / total_comparisons as f64) * 100.0;
            report.push_str(&format!(
                "- **audio_io wins**: {:.1}% of comparable tests ({}/{})\n",
                faster_percentage, audio_io_faster_count, total_comparisons
            ));
            report.push_str(&format!(
                "- **Best performance**: {:.2}x faster than hound\n",
                best_ratio
            ));
            report.push_str(&format!(
                "- **Worst performance**: {:.2}x slower than hound\n",
                1.0 / worst_ratio
            ));
        }

        report.push_str("\n### Key Observations\n\n");
        report.push_str("- **I24 and F64 formats**: Only supported by audio_io\n");
        report.push_str(
            "- **Duration scaling**: Performance characteristics across different file sizes\n",
        );
        report.push_str("- **Format efficiency**: Relative performance by sample format\n");

        // Add system information
        report.push_str(&generate_system_info());

        report.push_str("\n## Test Configuration\n\n");
        report.push_str("- **Rust version**: Latest stable\n");
        report.push_str("- **Optimization**: Release mode with LTO\n");
        report.push_str("- **CPU features**: Native target\n");
        report.push_str("- **Sample rate**: 44.1 kHz\n");
        report.push_str("- **Signal**: 440Hz sine wave at 70% amplitude\n");

        report
    }

    pub fn run_benchmark_reporter() -> Result<(), Box<dyn std::error::Error>> {
        println!("Scanning criterion benchmark results...");

        let results = scan_criterion_results()?;

        if results.is_empty() {
            eprintln!("No benchmark results found. Please run the benchmarks first with:");
            eprintln!("cargo bench --bench audio_io_vs_hound");
            return Ok(());
        }

        println!("Found {} benchmark results", results.len());

        let report = generate_report(results.clone());

        // Write to file
        fs::write("benchmark_report.md", &report)?;
        println!("Report written to benchmark_report.md");

        // Also write CSV exports
        // Detailed results CSV (times in microseconds for readability)
        let mut csv_rows = String::new();
        csv_rows.push_str("library,operation,format,duration,sample_rate,channels,mean_time_us,std_dev_us,throughput_samples_per_sec,sample_count,ci_low_us,ci_high_us\n");
        for r in &results {
            // convert ns -> us
            let mean_us = r.mean_time_ns / 1_000.0;
            let std_us = r.std_dev_ns / 1_000.0;
            let ci_low_us = r.confidence_interval_low / 1_000.0;
            let ci_high_us = r.confidence_interval_high / 1_000.0;

            csv_rows.push_str(&format!(
                "{},{},{},{},{},{},{:.6},{:.6},{},{},{:.6},{:.6}\n",
                r.library,
                r.operation,
                r.format,
                r.duration.replace(',', ""),
                r.sample_rate.map(|v| v.to_string()).unwrap_or_default(),
                r.channels.map(|v| v.to_string()).unwrap_or_default(),
                mean_us,
                std_us,
                r.throughput_samples_per_sec
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_else(|| "".to_string()),
                r.sample_count.map(|v| v.to_string()).unwrap_or_default(),
                ci_low_us,
                ci_high_us
            ));
        }
        fs::write("benchmark_results.csv", csv_rows)?;
        println!("CSV written to benchmark_results.csv");

        // Executive summary CSV (operation,format,duration,audio_io_mean_ns,hound_mean_ns,ratio_str)
        let mut summary_rows = String::new();
        // Executive summary CSV (times in microseconds)
        summary_rows.push_str("operation,format,duration,audio_io_mean_us,hound_mean_us,ratio\n");

        // Group results by operation/format/duration like generate_report
        use std::collections::BTreeMap as Map;
        let mut grouped: Map<String, Map<String, Map<String, Vec<&BenchmarkResult>>>> = Map::new();
        for result in &results {
            grouped
                .entry(result.operation.clone())
                .or_default()
                .entry(result.format.clone())
                .or_default()
                .entry(result.duration.clone())
                .or_default()
                .push(result);
        }

        for (operation, formats) in &grouped {
            for (format, durations) in formats {
                for (duration, results_for_duration) in durations {
                    let audio_io = results_for_duration
                        .iter()
                        .find(|r| r.library == "audio_io");
                    let hound = results_for_duration.iter().find(|r| r.library == "hound");

                    if let Some(audio_io_result) = audio_io {
                        // convert ns -> us for CSV summary
                        let audio_us = audio_io_result.mean_time_ns / 1_000.0;
                        if let Some(hound_result) = hound {
                            let hound_us = hound_result.mean_time_ns / 1_000.0;
                            let ratio = if audio_us > 0.0 {
                                hound_us / audio_us
                            } else {
                                0.0
                            };
                            summary_rows.push_str(&format!(
                                "{},{},{},{:.6},{:.6},{:.6}\n",
                                operation, format, duration, audio_us, hound_us, ratio
                            ));
                        } else {
                            summary_rows.push_str(&format!(
                                "{},{},{},{:.6},,\n",
                                operation, format, duration, audio_us
                            ));
                        }
                    }
                }
            }
        }

        fs::write("benchmark_summary.csv", summary_rows)?;
        println!("CSV written to benchmark_summary.csv");

        // Also print to stdout
        println!("\n{}", report);

        Ok(())
    }
}
#[cfg(feature = "benchmark_reporting")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    benchmark_report::run_benchmark_reporter()
}
