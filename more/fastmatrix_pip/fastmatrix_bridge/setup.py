from __future__ import annotations

import sys
from pathlib import Path

from setuptools import Extension, setup

CPP_SOURCE = "src/fastmatrix/_fastmatrix_native.cpp"


def make_extension() -> Extension:
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/std:c++17", "/EHsc"]
        extra_link_args = []
    else:
        extra_compile_args = ["-O3", "-std=c++17", "-pthread"]
        extra_link_args = ["-pthread"]

    return Extension(
        "fastmatrix._fastmatrix_native",
        sources=[CPP_SOURCE],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


setup(ext_modules=[make_extension()])
