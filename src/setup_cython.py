from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "cython_feature_extractor", # Name of the module to import in Python
        ["cython_feature_extractor.pyx"], # The Cython source file
        include_dirs=[numpy.get_include()], # Necessary for NumPy C API
        # extra_compile_args=["-O3", "-march=native"],
        # extra_link_args=[],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"} # Use Python 3 syntax
        )
)

# You'll need cython and numpy installed in your environment:
# pip install cython numpy
# or
# conda install cython numpy