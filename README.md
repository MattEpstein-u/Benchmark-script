# ML Training Environment Benchmark

This repository contains scripts to benchmark environments for machine learning training on large datasets. The primary Python script (`benchmark.py`) evaluates key performance metrics relevant to ML workloads, allowing you to compare different environments (e.g., local servers, cloud instances) and choose the best one for your training tasks.

## Scripts

- `benchmark.py`: Python-based benchmark tailored for ML training environments

## Prerequisites for Python Script

- Python 3.x
- `psutil`: For system information and monitoring
- `numpy`: For matrix operations simulating ML computations
- `requests`: For network benchmarking
- Optional: `torch` (PyTorch) for GPU detection

### Installing Dependencies

```bash
pip install psutil numpy requests
```

For GPU detection:
```bash
pip install torch
```

## Usage

### Python Benchmark (Recommended for ML)

1. Clone or download this repository.

2. Navigate to the repository directory:
   ```bash
   cd /workspaces/Benchmark-script
   ```

3. Run the Python benchmark:
   ```bash
   python benchmark.py
   ```

The script will perform the following ML-relevant tests:

- **System Information**: CPU cores, memory, disk space
- **GPU Detection**: Checks for CUDA GPUs if PyTorch is installed
- **CPU Benchmark**: Multi-process computation simulating data preprocessing
- **Matrix Multiplication**: Numpy-based matrix ops mimicking ML model computations
- **Memory Benchmark**: Large data allocation and processing (up to 2GB)
- **Disk I/O Benchmark**: File read/write speeds for dataset/model storage
- **Network Benchmark**: Download speed for fetching datasets

## Interpreting Results for ML Training

- **CPU Cores & Speed**: More cores and faster speeds are better for data preprocessing and CPU-based models
- **Memory**: Sufficient RAM is crucial for loading large datasets into memory
- **Disk I/O**: Fast read/write speeds are important for loading datasets and saving model checkpoints
- **GPU**: Presence and number of GPUs significantly impact training speed for GPU-accelerated frameworks
- **Network**: Faster downloads are useful for fetching large datasets or distributed training
- **Matrix Multiplication Speed**: Indicates performance for core ML computations

## Output Example (Python)

```
=== ML Training Environment Benchmark ===
--- System Information ---
System: Linux 5.15.0-119-generic
Processor: x86_64
CPU Cores: 8 (Physical: 4)
Total Memory: 15.60 GB
Disk Space: 100.00 GB total, 50.00 GB free
--------------------------
--- GPU Information ---
CUDA GPUs: 1
  GPU 0: NVIDIA GeForce RTX 3080
--- Running CPU Benchmark ---
Time taken for 10000000 calculations across 4 processes: 2.3456 seconds
--- Running Matrix Multiplication Benchmark ---
Time taken to multiply 1000x1000 matrices: 0.1234 seconds
--- Running Memory Benchmark ---
Time taken to allocate and process 2GB of data: 1.5678 seconds
--- Running Disk I/O Benchmark ---
Write Speed: 500.00 MB/s
Read Speed: 600.00 MB/s
--- Running Network Benchmark ---
Download Speed: 50.00 KB/s
=== Benchmark Complete ===
Use these results to compare environments for ML training on large datasets.
```

## Contributing

Feel free to contribute by improving the benchmarks, adding more ML-specific tests, or supporting additional hardware.

## License

This project is open-source. Please check the license file for details.