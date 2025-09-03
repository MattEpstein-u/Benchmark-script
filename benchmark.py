import time
import os
import psutil
import platform

def get_system_info():
    """Gathers and returns key system information."""
    print("--- System Information ---")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    total_mem = psutil.virtual_memory().total / (1024**3)
    print(f"Total Memory: {total_mem:.2f} GB")
    print("-" * 26)

def cpu_benchmark(n=20000000):
    """Performs a CPU-intensive calculation."""
    print("--- Running CPU Benchmark ---")
    start_time = time.time()
    for i in range(n):
        _ = i * i
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken to perform {n} calculations: {duration:.4f} seconds")
    return duration

def memory_benchmark(size_mb=512):
    """Tests memory allocation and access speed."""
    print("\n--- Running Memory Benchmark ---")
    start_time = time.time()
    
    # Create a large byte array to simulate memory usage
    data = bytearray(size_mb * 1024 * 1024)
    
    # Write to the memory
    for i in range(len(data)):
        data[i] = i % 256
        
    # Read from the memory
    for i in range(len(data)):
        _ = data[i]
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken to allocate and access {size_mb}MB of memory: {duration:.4f} seconds")
    return duration

def disk_io_benchmark(file_size_mb=256, block_size_kb=128):
    """Tests disk write and read speed."""
    print("\n--- Running Disk I/O Benchmark ---")
    file_name = "benchmark_temp_file.dat"
    file_size = file_size_mb * 1024 * 1024
    block_size = block_size_kb * 1024
    num_blocks = file_size // block_size

    # --- Write Test ---
    write_start_time = time.time()
    with open(file_name, "wb") as f:
        for _ in range(num_blocks):
            f.write(os.urandom(block_size))
    write_end_time = time.time()
    write_duration = write_end_time - write_start_time
    write_speed = file_size_mb / write_duration
    print(f"Write Speed: {write_speed:.2f} MB/s")

    # --- Read Test ---
    read_start_time = time.time()
    with open(file_name, "rb") as f:
        while f.read(block_size):
            pass
    read_end_time = time.time()
    read_duration = read_end_time - read_start_time
    read_speed = file_size_mb / read_duration
    print(f"Read Speed: {read_speed:.2f} MB/s")

    # Clean up the temporary file
    os.remove(file_name)
    
    return write_speed, read_speed

def main():
    """Main function to run all benchmarks."""
    print("=== Starting Python Environment Benchmark ===")
    get_system_info()
    cpu_benchmark()
    memory_benchmark()
    disk_io_benchmark()
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
