from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("SiddonGpuPy",
                             sources=["SiddonGpuPy.pyx"],
                             include_dirs=[numpy.get_include(), 
                                           "include",
                                           "/usr/local/cuda/include",
                                           "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include"],
                             library_dirs = ["lib",
                                            "/usr/local/cuda/lib64",
                                             "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\lib\\x64"],
                             libraries = ["SiddonGpu", "cudart_static"],
                             language = "c++")]
)