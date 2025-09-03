#!/bin/bash

# Benchmark script for the current environment

echo "--- Starting Benchmark ---"

# CPU Performance Test
echo "--- Running CPU Performance Test ---"
sysbench cpu --cpu-max-prime=20000 run

# Memory Performance Test
echo "--- Running Memory Performance Test ---"
sysbench memory --memory-block-size=1K --memory-total-size=10G run

# File I/O Performance Test
echo "--- Running File I/O Performance Test ---"
# Prepare files for the test
sysbench fileio --file-total-size=1G prepare

# Run the file I/O test
sysbench fileio --file-total-size=1G --file-test-mode=rndrw --time=30 --max-requests=0 run

# Clean up the files
sysbench fileio --file-total-size=1G cleanup

echo "--- Benchmark Complete ---"
