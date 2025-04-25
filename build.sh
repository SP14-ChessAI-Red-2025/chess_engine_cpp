#!/usr/bin/env bash
set -o errexit # Exit on error

ORT_VERSION="1.18.0" # Specify desired ORT version (check latest or needed version)
ORT_FILENAME="onnxruntime-linux-x64-${ORT_VERSION}"
ORT_TGZ="${ORT_FILENAME}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}"

# --- Check if ONNX Runtime needs to be installed ---
# Check if either the include or lib directory is missing inside ort_install
if [ ! -d "ort_install/include" ] || [ ! -d "ort_install/lib" ]; then
  echo "ONNX Runtime not found or incomplete in ./ort_install. Downloading and extracting..."

  # --- Download Section ---
  echo "Downloading ONNX Runtime C++ library (${ORT_VERSION})..."
  # Use curl with -L to follow redirects, -o to specify output filename
  curl -L -o "${ORT_TGZ}" "${ORT_URL}"

  # --- Extract Section ---
  echo "Extracting ONNX Runtime..."
  # Extract to the current directory
  tar -xzf "${ORT_TGZ}"

  # --- Installation Section ---
  # Ensure the base install directory exists
  mkdir -p ort_install

  # Remove potentially existing *old* target directories first before moving new ones
  echo "Cleaning potential previous ONNX Runtime installation directories..."
  rm -rf ort_install/include
  rm -rf ort_install/lib

  # Move the newly extracted directories
  echo "Moving new ONNX Runtime files..."
  mv "${ORT_FILENAME}/include" ort_install/
  mv "${ORT_FILENAME}/lib" ort_install/

  # --- Cleanup Section ---
  # Clean up downloaded archive and extracted top-level folder
  rm -f "${ORT_TGZ}"
  rm -rf "${ORT_FILENAME}"
  echo "ONNX Runtime extracted to ./ort_install"

else
  # This block runs if both ort_install/include AND ort_install/lib exist
  echo "ONNX Runtime already found in ./ort_install. Skipping download/extract."
fi
# --- End of ONNX Runtime Installation Chunk ---


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