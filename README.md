# ML Environment Benchmark

A Python script to benchmark environments for machine learning tasks, focusing on performance metrics relevant to training large models. It outputs results to the console and appends them to a CSV file for easy comparison across different servers.

## Features

- **Comprehensive Benchmarking**: Tests CPU (single and multi-core), GPU, memory bandwidth, disk I/O, and network speed.
- **ML-Focused Metrics**: Measures performance in terms relevant to ML, such as operations per second and GB/s bandwidth.
- **CSV Output**: Appends results to a CSV file, making it easy to compare multiple environments.
- **Customizable**: Use command-line arguments to specify a title for the run and adjust benchmark parameters.

## Prerequisites

- Python 3.x
- Required Python packages: `psutil`, `numpy`, `requests`
- Optional for GPU testing: `torch` (PyTorch)

### Installation

```bash
pip install psutil numpy requests
```

To enable GPU benchmarking, install PyTorch:
```bash
# Follow instructions on the official PyTorch website for your specific CUDA version
# e.g., pip install torch torchvision torchaudio
```

## Usage

Run the script from your terminal. Use the `--title` argument to label your benchmark run (e.g., with the server name).

### Basic Usage

```bash
python benchmark.py --title "MyServer_1"
```

This will run the benchmark with default settings and save the results to `benchmark_results_MyServer_1.csv`. If you run it again with the same title, it will append a new row to the same file.

### Customizing Benchmarks

You can adjust the benchmark parameters for more intensive testing:

```bash
python benchmark.py --title "MyServer_1_Heavy" --matrix-size 2000 --mem-size 4 --disk-size 10
```

- `--title`: A title for the benchmark run.
- `--matrix-size`: Size of matrices for multiplication tests (e.g., 2000 for 2000x2000).
- `--mem-size`: Data size for the memory benchmark in GB.
- `--disk-size`: File size for the disk I/O benchmark in GB.

## Output

The script outputs results to both the console and a CSV file.

### CSV Output

The CSV file (`benchmark_results_{title}.csv`) is designed for easy comparison. The most important performance metrics are placed first, so you can quickly see which server performs best on key tasks.

**Example `benchmark_results.csv`:**

| Title      | GPU_Matrix_Ops_per_Sec | CPU_Matrix_Ops_per_Sec | Memory_Bandwidth_GBs | ... |
|------------|------------------------|------------------------|----------------------|-----|
| Server_A   | 150.5                  | 10.2                   | 25.5                 | ... |
| Server_B   | 250.8                  | 15.5                   | 35.8                 | ... |

Higher numbers are better for all performance metrics.
