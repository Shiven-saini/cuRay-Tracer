#!/bin/bash

# Combined build and run script for cuRay-Tracer

set -e  # Exit on any error

echo "=== Building cuRay-Tracer ==="
./build.sh

echo ""
echo "=== Running cuRay-Tracer ==="
./run.sh "$@"