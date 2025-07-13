from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "convolution_emulator",
        ["convolution_emulator.cpp"],
        extra_compile_args=["-O3", "-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"],
        extra_link_args=["-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"]
    ),
]

setup(
    name="convolution_emulator",
    ext_modules=ext_modules,
)