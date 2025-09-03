# Benchmark Script

This repository contains a script to benchmark the performance of the current environment. The script tests CPU, memory, and file I/O performance using `sysbench`, allowing you to compare this environment with others, such as a local server.

## Prerequisites

- `sysbench`: A modular, cross-platform and multi-threaded benchmark tool.

### Installing sysbench

#### On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y sysbench
```

#### On other Linux distributions:

- **CentOS/RHEL/Fedora**: `sudo yum install sysbench` or `sudo dnf install sysbench`
- **macOS**: `brew install sysbench`
- **Windows**: Download from the official sysbench website or use WSL with Ubuntu.

For more installation options, visit the [sysbench GitHub repository](https://github.com/akopytov/sysbench).

## Usage

1. Clone or download this repository.

2. Navigate to the repository directory:
   ```bash
   cd /workspaces/Benchmark-script
   ```

3. Make the script executable:
   ```bash
   chmod +x benchmark.sh
   ```

4. Run the benchmark script:
   ```bash
   ./benchmark.sh
   ```

The script will perform the following tests:
- **CPU Performance**: Calculates prime numbers up to 20,000.
- **Memory Performance**: Tests memory read/write operations with a total size of 10GB.
- **File I/O Performance**: Tests random read/write operations on a 1GB file for 30 seconds.

Results will be displayed in the terminal, including metrics like events per second, total time, and throughput.

## Output Example

```
--- Starting Benchmark ---
--- Running CPU Performance Test ---
sysbench 1.0.20 (using bundled LuaJIT 2.1.0-beta3)

Prime numbers limit: 20000

Initializing worker threads...

Threads started!

CPU speed:
    events per second:   123.45

General statistics:
    total time:                          10.0000s
    total number of events:              1235

Latency (ms):
         min:                                    0.00
         avg:                                    8.10
         max:                                   50.00
         95th percentile:                       15.00
         sum:                                 10000.0

Threads fairness:
    events (avg/stddev):           1235.0000/0.00
    execution time (avg/stddev):   10.0000/0.00

--- Running Memory Performance Test ---
...
```

## Contributing

Feel free to contribute by improving the script or adding more benchmark tests.

## License

This project is open-source. Please check the license file for details.