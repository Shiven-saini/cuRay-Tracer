#!/bin/bash

# Build script for cuRay-Tracer

set -e  # Exit on any error

echo "Building CudaRayTracer..."

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
    echo "Created build directory."
fi

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build the project
echo "Compiling..."
make -j$(nproc)

echo "Build complete!"
echo "Executable created: ./build/CudaRayTracer"