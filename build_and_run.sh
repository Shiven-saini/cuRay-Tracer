#!/bin/bash

# Combined build and run script for cuRay-Tracer

set -e  # Exit on any error

echo "=== Building CudaRayTracer ==="
./build.sh

echo ""
echo "=== Running CudaRayTracer ==="
./run.sh "$@"