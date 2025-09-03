import time
import os
import psutil
import platform
import multiprocessing
import requests
import numpy as np
import argparse
import subprocess

# --- Helper Functions ---

def get_cpu_cache_sizes():
    """Gets CPU cache sizes using lscpu command on Linux."""
    if platform.system() == "Linux":
        try:
            lscpu_output = subprocess.check_output("lscpu", text=True)
            caches = {}
            for line in lscpu_output.split('\n'):
                if "L1d cache" in line:
                    caches['L1d'] = line.split()[-2] + " " + line.split()[-1]
                elif "L1i cache" in line:
                    caches['L1i'] = line.split()[-2] + " " + line.split()[-1]
                elif "L2 cache" in line:
                    caches['L2'] = line.split()[-2] + " " + line.split()[-1]
                elif "L3 cache" in line:
                    caches['L3'] = line.split()[-2] + " " + line.split()[-1]
            return caches
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    return None

def print_section_header(title):
    """Prints a formatted section header."""
    print(f"\n--- {title} ---")

def print_result(metric, value, unit=""):
    """Prints a formatted result."""
    print(f"{metric:<30}: {value} {unit}")

# --- Benchmark Functions ---

def get_system_info():
    """Gathers and returns key system information."""
    print_section_header("System Information")
    print_result("System", f"{platform.system()} {platform.release()}")
    print_result("Processor", platform.processor())
    
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print_result("CPU Cores", f"{logical_cores} (Physical: {physical_cores})")

    cache_sizes = get_cpu_cache_sizes()
    if cache_sizes:
        for cache, size in cache_sizes.items():
            print_result(f"{cache} Cache", size)

    total_mem = psutil.virtual_memory().total / (1024**3)
    print_result("Total Memory", f"{total_mem:.2f}", "GB")
    
    disk = psutil.disk_usage('/')
    print_result("Total Disk Space", f"{disk.total / (1024**3):.2f}", "GB")
    print_result("Free Disk Space", f"{disk.free / (1024**3):.2f}", "GB")

def check_gpu():
    """Checks for GPU availability and details."""
    print_section_header("GPU Information")
    try:
        import torch
        if torch.cuda.is_available():
            print_result("CUDA GPUs Found", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print_result(f"  GPU {i}", torch.cuda.get_device_name(i))
            return True
        else:
            print("No CUDA-enabled GPUs detected by PyTorch.")
            return False
    except ImportError:
        print("PyTorch not installed. GPU check skipped.")
        return False

def cpu_benchmark(n=10000000):
    """Performs a CPU-intensive calculation simulating ML preprocessing."""
    print_section_header("CPU Benchmark (Multi-Core)")
    start_time = time.time()
    
    num_processes = min(multiprocessing.cpu_count(), 8)
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = n // num_processes
    
    tasks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    
    results = pool.starmap(worker_calc, tasks)
    
    pool.close()
    pool.join()
    
    duration = time.time() - start_time
    print_result("Processes", num_processes)
    print_result("Time taken", f"{duration:.4f}", "seconds")
    return duration

def matrix_multiplication_benchmark(size=1000):
    """Benchmarks matrix multiplication on the CPU."""
    print_section_header("CPU Matrix Multiplication")
    start_time = time.time()
    
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    
    duration = time.time() - start_time
    print_result(f"Matrix Size", f"{size}x{size}")
    print_result("Time taken", f"{duration:.4f}", "seconds")
    return duration

def gpu_benchmark(size=1000):
    """Benchmarks matrix multiplication on the GPU."""
    print_section_header("GPU Matrix Multiplication")
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU available, skipping benchmark.")
            return None

        device = torch.device("cuda")
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Warm-up
        for _ in range(3):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        start_time = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        duration = time.time() - start_time

        print_result(f"Matrix Size", f"{size}x{size}")
        print_result("Time taken", f"{duration:.4f}", "seconds")
        return duration

    except ImportError:
        print("PyTorch not installed, skipping GPU benchmark.")
        return None

def memory_benchmark(size_gb=1):
    """Tests memory bandwidth."""
    print_section_header("Memory Bandwidth Benchmark")
    
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if size_gb >= available_mem_gb:
        print(f"Requested size ({size_gb}GB) is too large for available memory ({available_mem_gb:.2f}GB). Skipping.")
        return None

    try:
        data = np.random.rand(int(size_gb * 125_000_000)) # 1GB of floats
        
        start_time = time.time()
        _ = np.sum(data)
        duration = time.time() - start_time
        
        bandwidth = (data.nbytes / (1024**3)) / duration
        
        print_result("Data Size", f"{size_gb}", "GB")
        print_result("Bandwidth", f"{bandwidth:.2f}", "GB/s")
        del data
        return bandwidth
    except MemoryError:
        print(f"Memory allocation failed for {size_gb}GB. Skipping.")
        return None

def disk_io_benchmark(file_size_gb=1):
    """Tests disk write and read speed."""
    print_section_header("Disk I/O Benchmark")
    file_name = "benchmark_temp_file.dat"
    file_size_bytes = int(file_size_gb * (1024**3))
    block_size = 1024 * 1024  # 1MB blocks
    num_blocks = file_size_bytes // block_size

    # Write Test
    write_start_time = time.time()
    with open(file_name, "wb") as f:
        for _ in range(num_blocks):
            f.write(os.urandom(block_size))
    write_duration = time.time() - write_start_time
    write_speed = (file_size_bytes / (1024**2)) / write_duration
    print_result("Write Speed", f"{write_speed:.2f}", "MB/s")

    # Read Test
    read_start_time = time.time()
    with open(file_name, "rb") as f:
        while f.read(block_size):
            pass
    read_duration = time.time() - read_start_time
    read_speed = (file_size_bytes / (1024**2)) / read_duration
    print_result("Read Speed", f"{read_speed:.2f}", "MB/s")

    os.remove(file_name)
    return write_speed, read_speed

def network_benchmark(url="http://speedtest.tele2.net/100MB.zip"):
    """Tests network download speed."""
    print_section_header("Network Benchmark")
    try:
        start_time = time.time()
        response = requests.get(url, timeout=30, stream=True)
        total_size_bytes = int(response.headers.get('content-length', 0))
        
        # Read the content in chunks
        for _ in response.iter_content(chunk_size=1024*1024):
            pass
            
        duration = time.time() - start_time
        speed_mbps = (total_size_bytes * 8) / (duration * 1024 * 1024)
        
        print_result("Download URL", url)
        print_result("Download Speed", f"{speed_mbps:.2f}", "Mbps")
        return speed_mbps
    except Exception as e:
        print(f"Network test failed: {e}")
        return None

def worker_calc(start, end):
    """Worker function for CPU benchmark calculations."""
    result = 0
    for i in range(start, end):
        result += i * i
    return result

def main(args):
    """Main function to run all benchmarks."""
    print("=== ML Training Environment Benchmark ===")
    get_system_info()
    gpu_available = check_gpu()
    
    cpu_benchmark()
    matrix_multiplication_benchmark(size=args.matrix_size)
    if gpu_available:
        gpu_benchmark(size=args.matrix_size)
        
    memory_benchmark(size_gb=args.mem_size)
    disk_io_benchmark(file_size_gb=args.disk_size)
    network_benchmark()
    
    print("\n=== Benchmark Complete ===")
    print("Use these results to compare environments for ML training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark an environment for ML tasks.")
    parser.add_argument("--matrix-size", type=int, default=1000, help="Size of matrices for multiplication benchmarks (NxN).")
    parser.add_argument("--mem-size", type=int, default=1, help="Size of data for memory benchmark in GB.")
    parser.add_argument("--disk-size", type=int, default=1, help="Size of file for disk I/O benchmark in GB.")
    args = parser.parse_args()
    main(args)
