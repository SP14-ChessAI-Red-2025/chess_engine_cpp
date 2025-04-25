#!/usr/bin/env bash
# Exit script on error
set -o errexit

# 1. Install Python dependencies (can sometimes be done after C++ build too)
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 2. Build C++ library
echo "Building C++ library..."
# Ensure build tools are available (cmake, make, g++ are usually present)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  # Or use your specific preset/config
cmake --build build --config Release

echo "Build complete."