#!/bin/bash

# Clean build directory script for cuRay-Tracer

echo "Cleaning build directory..."

# Remove build directory if it exists
if [ -d "build" ]; then
    rm -rf build
    echo "Build directory removed."
else
    echo "Build directory does not exist."
fi

# Create fresh build directory
mkdir build
echo "Fresh build directory created."

echo "Cleanup complete!"