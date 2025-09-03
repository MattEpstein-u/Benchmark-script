# ML Environment Benchmark Script

This Python script benchmarks a server's performance for common machine learning tasks. It evaluates CPU, memory, disk, and network performance to help you compare different environments.

## Prerequisites

- Python 3.x
- `pip`

## Installation

Install the required Python libraries:

```bash
pip install psutil numpy requests torch
```
*Note: `torch` is used for GPU detection but is not required to run the basic benchmarks.*

## Usage

### Basic Benchmark

To run the script with default settings. This will save results to `benchmark_results.csv`.

```bash
python benchmark.py
```

### Titled Benchmark

You can provide a title for your benchmark run. The title will be used in the CSV filename (e.g., `benchmark_results_my_server.csv`) and as a column in the CSV file.

```bash
python benchmark.py --title my_server
```

### Custom Benchmark

You can customize all benchmark parameters:

```bash
python benchmark.py --title my_server_custom --matrix-size 2000 --mem-size 2 --disk-size 5
```

**Arguments:**
- `--title`: A title for the benchmark run (optional).
- `--matrix-size`: Size of matrices for multiplication tests (default: 1000).
- `--mem-size`: Data size for memory benchmark in GB (default: 1).
- `--disk-size`: File size for disk I/O benchmark in GB (default: 1).

The script will output the results for each test and save them to a CSV file, which you can use to compare the performance of different servers.