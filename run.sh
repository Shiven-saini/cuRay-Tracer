#!/bin/bash

# Run script for cuRay-Tracer

set -e  # Exit on any error

# Check if executable exists
if [ ! -f "build/cuRayTracer" ]; then
    echo "Executable not found. Building project first..."
    ./build.sh
fi

echo "Running cuRay-Tracer..."
cd build
./CudaRayTracer "$@"