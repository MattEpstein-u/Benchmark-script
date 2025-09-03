# ML Environment Benchmark

A simple Python script to benchmark a machine's performance for common machine learning tasks.

## Prerequisites

Install the required Python packages:
```bash
pip install psutil numpy requests torch
```

## Usage

Run the script from your terminal.

**Default Benchmark**
```bash
python benchmark.py --title "MyServerName"
```

**Mini Benchmark (Quick Test)**
For a faster test, use the `--mini` flag, which runs the benchmarks with 1/10th of the default parameters.
```bash
python benchmark.py --title "MyServerName_Mini" --mini
```

## Output

Results are printed to the console and saved to a `benchmark_results_{title}.csv` file. This allows you to easily compare performance across different machines or configurations.

