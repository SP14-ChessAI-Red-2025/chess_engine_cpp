#!/usr/bin/env bash
set -o errexit # Exit on error

ORT_VERSION="1.18.0" # Specify desired ORT version (check latest or needed version)
ORT_FILENAME="onnxruntime-linux-x64-${ORT_VERSION}"
ORT_TGZ="${ORT_FILENAME}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

echo "Downloading ONNX Runtime C++ library (${ORT_VERSION})..."
# Use curl with -L to follow redirects, -o to specify output filename
curl -L -o "${ORT_TGZ}" "${ORT_URL}"

echo "Extracting ONNX Runtime..."
tar -xzf "${ORT_TGZ}"

# Create a predictable install location and move extracted files
# CMake needs include/ and lib/ directories
mkdir -p ort_install
mv "${ORT_FILENAME}/include" ort_install/
mv "${ORT_FILENAME}/lib" ort_install/

# Clean up downloaded archive and extracted top-level folder
rm -f "${ORT_TGZ}"
rm -rf "${ORT_FILENAME}"
echo "ONNX Runtime extracted to ./ort_install"

# ---> Your existing pip install command can go here or after cmake <---
echo "Installing Python dependencies..."
pip install -r requirements.txt

# ---> Your existing cmake commands follow <---
echo "Building C++ library..."
# Pass the ORT install path to CMake (optional but good practice)
cmake -S . -B build -DNNUE_ENABLED=ON -DCMAKE_BUILD_TYPE=Release -DORT_INSTALL_DIR=$(pwd)/ort_install
make -C build VERBOSE=1 chess_cpp # Keep using make with target for now

echo "Build complete."


echo "Building Cython bindings..."
python src/setup_cython.py build_ext --inplace