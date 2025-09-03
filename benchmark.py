import time
import os
import psutil
import platform
import multiprocessing
import requests
import numpy as np
import argparse
import subprocess
import csv
from collections import OrderedDict

# --- Helper Functions ---

def get_cpu_info():
    """Gets CPU model and cache sizes using lscpu on Linux."""
    info = {"model": "N/A", "L1d": "N/A", "L1i": "N/A", "L2": "N/A", "L3": "N/A"}
    if platform.system() == "Linux":
        try:
            lscpu_output = subprocess.check_output("lscpu", text=True)
            for line in lscpu_output.split('\n'):
                if "Model name" in line:
                    info["model"] = line.split(":")[-1].strip()
                elif "L1d:" in line:
                    parts = line.split(":")[-1].strip().split()
                    if len(parts) >= 2:
                        info["L1d"] = parts[0] + " " + parts[1]
                elif "L1i:" in line:
                    parts = line.split(":")[-1].strip().split()
                    if len(parts) >= 2:
                        info["L1i"] = parts[0] + " " + parts[1]
                elif "L2:" in line:
                    parts = line.split(":")[-1].strip().split()
                    if len(parts) >= 2:
                        info["L2"] = parts[0] + " " + parts[1]
                elif "L3:" in line:
                    parts = line.split(":")[-1].strip().split()
                    if len(parts) >= 2:
                        info["L3"] = parts[0] + " " + parts[1]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # lscpu not found or failed
    return info

def print_section_header(title):
    """Prints a formatted section header."""
    print(f"\n--- {title} ---")

# --- Benchmark Functions (Return results as a dictionary) ---

def get_system_info():
    """Gathers key system information."""
    print_section_header("System Information")
    results = OrderedDict()
    
    cpu_info = get_cpu_info()
    results['CPU_Model'] = cpu_info['model']
    results['CPU_Physical_Cores'] = psutil.cpu_count(logical=False)
    results['CPU_Logical_Cores'] = psutil.cpu_count(logical=True)
    results['CPU_L1d_Cache'] = cpu_info['L1d']
    results['CPU_L2_Cache'] = cpu_info['L2']
    results['CPU_L3_Cache'] = cpu_info['L3']
    
    total_mem_gb = psutil.virtual_memory().total / (1024**3)
    results['Memory_Total_GB'] = f"{total_mem_gb:.2f}"
    
    disk = psutil.disk_usage('/')
    results['Disk_Total_GB'] = f"{disk.total / (1024**3):.2f}"
    
    return results

def check_gpu():
    """Checks for GPU availability and details."""
    print_section_header("GPU Information")
    results = OrderedDict()
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            results['GPU_Count'] = gpu_count
            for i in range(gpu_count):
                results[f'GPU_{i}_Name'] = torch.cuda.get_device_name(i)
            return results, True
        else:
            results['GPU_Count'] = 0
            results['GPU_Reason'] = 'No CUDA-enabled GPU found'
            return results, False
    except ImportError:
        results['GPU_Count'] = 0
        results['GPU_Reason'] = 'PyTorch not installed'
        return results, False

def cpu_benchmark(n=10000000):
    """Performs a CPU-intensive calculation."""
    print_section_header("CPU Benchmark (Multi-Core)")
    start_time = time.time()
    
    num_processes = min(multiprocessing.cpu_count(), 8)
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = n // num_processes
    tasks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    
    pool.starmap(worker_calc, tasks)
    pool.close()
    pool.join()
    
    duration = time.time() - start_time
    ops_per_sec = n / duration if duration > 0 else 0
    return {'CPU_Multi_Core_Ops_per_Sec': f"{ops_per_sec:.2f}"}

def matrix_multiplication_benchmark(size=1000):
    """Benchmarks matrix multiplication on the CPU."""
    print_section_header("CPU Matrix Multiplication")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    start_time = time.time()
    np.dot(A, B)
    duration = time.time() - start_time
    
    ops_per_sec = 1 / duration if duration > 0 else 0
    return {'CPU_Matrix_Ops_per_Sec': f"{ops_per_sec:.2f}"}

def gpu_benchmark(size=1000):
    """Benchmarks matrix multiplication on the GPU."""
    print_section_header("GPU Matrix Multiplication")
    results = OrderedDict()
    try:
        import torch
        if not torch.cuda.is_available():
            results['GPU_Matrix_Ops_per_Sec'] = 'N/A'
            results['GPU_Benchmark_Reason'] = 'No CUDA-enabled GPU found'
            return results

        device = torch.device("cuda")
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Warm-up
        for _ in range(3):
            torch.matmul(A, B)
        torch.cuda.synchronize()

        start_time = time.time()
        torch.matmul(A, B)
        torch.cuda.synchronize()
        duration = time.time() - start_time

        ops_per_sec = 1 / duration if duration > 0 else 0
        results['GPU_Matrix_Ops_per_Sec'] = f"{ops_per_sec:.2f}"
        return results
    except ImportError:
        results['GPU_Matrix_Ops_per_Sec'] = 'N/A'
        results['GPU_Benchmark_Reason'] = 'PyTorch not installed'
        return results
    except Exception as e:
        results['GPU_Matrix_Ops_per_Sec'] = 'N/A'
        results['GPU_Benchmark_Reason'] = f'Benchmark failed: {e}'
        return results

def memory_benchmark(size_gb=1):
    """Tests memory bandwidth."""
    print_section_header("Memory Bandwidth Benchmark")
    
    available_mem_gb = psutil.virtual_memory().available / (1024**3)
    if size_gb >= available_mem_gb:
        return {'Memory_Bandwidth_GBs': 'Not enough memory'}

    try:
        data = np.random.rand(int(size_gb * 125_000_000))  # 1GB of floats
        
        start_time = time.time()
        np.sum(data)
        duration = time.time() - start_time
        
        bandwidth = (data.nbytes / (1024**3)) / duration if duration > 0 else 0
        del data
        return {'Memory_Bandwidth_GBs': f"{bandwidth:.2f}"}
    except MemoryError:
        return {'Memory_Bandwidth_GBs': 'Memory allocation failed'}

def disk_io_benchmark(file_size_gb=1):
    """Tests disk write and read speed."""
    print_section_header("Disk I/O Benchmark")
    file_name = "benchmark_temp_file.dat"
    file_size_bytes = int(file_size_gb * (1024**3))
    block_size = 1024 * 1024
    num_blocks = file_size_bytes // block_size

    # Write Test
    write_start_time = time.time()
    with open(file_name, "wb") as f:
        for _ in range(num_blocks):
            f.write(os.urandom(block_size))
    write_duration = time.time() - write_start_time
    write_speed = (file_size_bytes / (1024**2)) / write_duration if write_duration > 0 else 0

    # Read Test
    read_start_time = time.time()
    with open(file_name, "rb") as f:
        while f.read(block_size):
            pass
    read_duration = time.time() - read_start_time
    read_speed = (file_size_bytes / (1024**2)) / read_duration if read_duration > 0 else 0

    os.remove(file_name)
    return {'Disk_Write_MBs': f"{write_speed:.2f}", 'Disk_Read_MBs': f"{read_speed:.2f}"}

def network_benchmark(url="http://speedtest.tele2.net/100MB.zip"):
    """Tests network download speed."""
    print_section_header("Network Benchmark")
    results = OrderedDict()
    try:
        start_time = time.time()
        response = requests.get(url, timeout=30, stream=True)
        total_size_bytes = int(response.headers.get('content-length', 0))
        
        for _ in response.iter_content(chunk_size=1024*1024):
            pass
            
        duration = time.time() - start_time
        speed_mbps = (total_size_bytes * 8) / (duration * 1024 * 1024) if duration > 0 else 0
        results['Network_Download_Mbps'] = f"{speed_mbps:.2f}"
        return results
    except Exception as e:
        results['Network_Download_Mbps'] = 'Failed'
        results['Network_Benchmark_Reason'] = f'Download failed: {e}'
        return results

def worker_calc(start, end):
    """Worker function for CPU benchmark calculations."""
    result = 0
    for i in range(start, end):
        result += i * i
    return result

def write_to_csv(results, filename):
    """Writes or appends results to a CSV file, ensuring all headers are included."""
    # Define a comprehensive list of all possible headers in the desired order
    all_headers = [
        'Title', 'CPU_Multi_Core_Ops_per_Sec', 'CPU_Matrix_Ops_per_Sec', 
        'GPU_Matrix_Ops_per_Sec', 'GPU_Benchmark_Reason', 'Memory_Bandwidth_GBs', 
        'Disk_Write_MBs', 'Disk_Read_MBs', 'Network_Download_Mbps', 'Network_Benchmark_Reason',
        'CPU_Model', 'CPU_Physical_Cores', 'CPU_Logical_Cores', 
        'CPU_L1d_Cache', 'CPU_L2_Cache', 'CPU_L3_Cache', 
        'Memory_Total_GB', 'Disk_Total_GB', 'GPU_Count', 'GPU_Reason'
    ]
    # Add GPU names dynamically
    gpu_count = results.get('GPU_Count', 0)
    if gpu_count > 0:
        for i in range(gpu_count):
            all_headers.append(f'GPU_{i}_Name')

    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        # Use the comprehensive header list for the writer
        writer = csv.DictWriter(csvfile, fieldnames=all_headers, extrasaction='ignore')
        
        if not file_exists:
            writer.writeheader()
        
        # Create a full dictionary for the current results to ensure all columns are aligned
        row_data = {header: results.get(header, '') for header in all_headers}
        writer.writerow(row_data)
        
    print(f"\nResults appended to {filename}")

def main(args):
    """Main function to run all benchmarks."""
    print("=== ML Training Environment Benchmark ===")

    # Set benchmark parameters based on whether --mini is used
    if args.mini:
        print("--- Running in MINI mode (1/10th of default parameters) ---")
        matrix_size = args.matrix_size // 10
        mem_size = args.mem_size / 10
        disk_size = args.disk_size / 10
        cpu_ops = args.cpu_ops // 10
    else:
        matrix_size = args.matrix_size
        mem_size = args.mem_size
        disk_size = args.disk_size
        cpu_ops = args.cpu_ops
    
    # --- Run Benchmarks and Collect Results ---
    all_results = OrderedDict()
    all_results['Title'] = args.title if args.title else ''
    
    gpu_info, gpu_available = check_gpu()
    
    # Run performance benchmarks first
    perf_results = OrderedDict()
    if gpu_available:
        perf_results.update(gpu_benchmark(size=matrix_size))
    else:
        # Ensure placeholder is added if GPU is not available
        results = OrderedDict()
        results['GPU_Matrix_Ops_per_Sec'] = 'N/A'
        if 'GPU_Reason' in gpu_info and gpu_info['GPU_Reason']:
             results['GPU_Benchmark_Reason'] = gpu_info['GPU_Reason']
        perf_results.update(results)

    perf_results.update(matrix_multiplication_benchmark(size=matrix_size))
    perf_results.update(memory_benchmark(size_gb=mem_size))
    perf_results.update(cpu_benchmark(n=cpu_ops))
    perf_results.update(disk_io_benchmark(file_size_gb=disk_size))
    perf_results.update(network_benchmark())

    # Combine results with performance metrics first
    all_results.update(perf_results)
    all_results.update(get_system_info())
    all_results.update(gpu_info)

    # --- Print Results to Console ---
    print("\n--- Benchmark Summary ---")
    for key, value in all_results.items():
        print(f"{key:<30}: {value}")

    # --- Write to CSV ---
    csv_filename = f"benchmark_results_{args.title}.csv" if args.title else "benchmark_results.csv"
    write_to_csv(all_results, csv_filename)
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark an environment for ML tasks.")
    parser.add_argument("--title", type=str, help="A title for this benchmark run (e.g., server name).")
    parser.add_argument("--mini", action="store_true", help="Run a mini-benchmark (1/10th of default parameters).")
    parser.add_argument("--matrix-size", type=int, default=2000, help="Size of matrices for multiplication benchmarks (NxN).")
    parser.add_argument("--mem-size", type=float, default=2.0, help="Size of data for memory benchmark in GB.")
    parser.add_argument("--disk-size", type=float, default=2.0, help="Size of file for disk I/O benchmark in GB.")
    parser.add_argument("--cpu-ops", type=int, default=20000000, help="Number of operations for the CPU benchmark.")
    args = parser.parse_args()
    main(args)
