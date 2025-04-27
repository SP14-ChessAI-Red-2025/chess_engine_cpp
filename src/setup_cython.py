# src/setup_cython.py (Corrected Output Path)
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import platform

# --- Get the directory containing this setup.py script ---
SETUP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SETUP_DIR, '..')) # Assumes setup.py is in src/

# --- Configuration (Paths relative to SETUP_DIR/PROJECT_ROOT) ---
CPP_INCLUDE_DIR = os.path.join(PROJECT_ROOT, 'include')
CPP_BUILD_DIR = os.path.join(PROJECT_ROOT, 'build', 'src')
CPP_LIB_NAME = 'chess_cpp'
CYTHON_SOURCE_FILE = os.path.join(SETUP_DIR, "chess_engine_cython.pyx")

# *** CORRECTED OUTPUT PATH ***
# Point to the 'python/chess_dir' relative to the project root
OUTPUT_PACKAGE_DIR = os.path.join(PROJECT_ROOT, "python", "chess_dir")

# Platform-specific library extension
# ... (rest of platform/path checking remains the same) ...
if platform.system() == "Windows":
    lib_ext = ".dll"
elif platform.system() == "Darwin":
    lib_ext = ".dylib"
else:
    lib_ext = ".so"
cpp_library_path = os.path.join(CPP_BUILD_DIR, f"lib{CPP_LIB_NAME}{lib_ext}")


print("-" * 20)
print(f"CONFIGURATION:")
print(f"  Setup Script Dir:      {SETUP_DIR}")
print(f"  Project Root Dir:      {PROJECT_ROOT}")
print(f"  C++ Include Directory: {CPP_INCLUDE_DIR}")
print(f"  C++ Library Directory: {CPP_BUILD_DIR}")
print(f"  C++ Library Name:      {CPP_LIB_NAME}")
print(f"  Expected Library Path: {cpp_library_path}")
print(f"  Cython Source File:    {CYTHON_SOURCE_FILE}")
print(f"  NumPy Include Dir:     {numpy.get_include()}")
print(f"  Output Package Dir:    {OUTPUT_PACKAGE_DIR}") # Verify this points to python/chess_dir
print("-" * 20)

if not os.path.exists(CYTHON_SOURCE_FILE):
     import sys
     sys.exit(f"Error: Cython source file not found: {CYTHON_SOURCE_FILE}")
if not os.path.exists(cpp_library_path):
     print(f"WARNING: C++ library not found at expected path: {cpp_library_path}")
     print("         Please ensure the C++ code is compiled first (e.g., using CMake in ../build).")

# Define the Cython extension module(s)
extensions = [
    Extension(
        "chess_dir.ai_chess", # Module name remains the same
        [CYTHON_SOURCE_FILE],
        include_dirs=[CPP_INCLUDE_DIR, numpy.get_include()],
        language="c++",
        library_dirs=[CPP_BUILD_DIR],
        libraries=[CPP_LIB_NAME],
        extra_compile_args=["-std=c++20"],
        extra_link_args=[],
    )
]

setup(
    name="Chess Engine Cython Interface",
    version="0.1.2", # Incremented version again
    description="Cython interface for the C++ Chess Engine",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
        annotate=True
    ),
    # Tell setup where the base directory for the 'chess_dir' package is
    # It should be the 'python' directory in the project root.
    package_dir={'': 'python'}, # Root for packages is the 'python' dir
    packages=['chess_dir'],    # The package to install is 'chess_dir' (found under root)
    zip_safe=False,
)

print("\nSetup complete. If successful, the compiled extension should be in:")
print(f"{OUTPUT_PACKAGE_DIR}") # Should now point to python/chess_dir