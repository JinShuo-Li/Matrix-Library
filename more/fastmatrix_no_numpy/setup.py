from __future__ import annotations

import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None
from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler

SRC = Path("src/fastmatrix/fastmatrix_native.cpp")


def _lib_filename() -> str:
    if sys.platform == "win32":
        return "fastmatrix_native.dll"
    if sys.platform == "darwin":
        return "libfastmatrix_native.dylib"
    return "libfastmatrix_native.so"


def _compile_args() -> tuple[list[str], list[str]]:
    if sys.platform == "win32":
        return ["/O2", "/std:c++17", "/EHsc"], []
    if sys.platform == "darwin":
        return ["-O3", "-std=c++17", "-fPIC", "-pthread"], ["-dynamiclib", "-pthread"]
    return ["-O3", "-std=c++17", "-fPIC", "-pthread"], ["-shared", "-pthread"]


class build_py(_build_py):
    def run(self):
        super().run()
        self.build_native()

    def build_native(self):
        build_ext = self.get_finalized_command("build_ext")
        build_temp = Path(build_ext.build_temp) / "fastmatrix"
        build_temp.mkdir(parents=True, exist_ok=True)

        target_dir = Path(self.build_lib) / "fastmatrix"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _lib_filename()

        compiler = new_compiler()
        customize_compiler(compiler)
        if hasattr(compiler, "compiler_cxx") and compiler.compiler_cxx:
            compiler.set_executable("compiler_so", compiler.compiler_cxx)
            if sys.platform != "win32":
                cxx_linker = list(compiler.compiler_cxx) + ["-shared"]
                compiler.set_executable("linker_so", cxx_linker)
        cargs, largs = _compile_args()
        objects = compiler.compile([str(SRC)], output_dir=str(build_temp), extra_postargs=cargs)
        compiler.link_shared_object(objects, str(target), extra_postargs=largs)
        self.announce(f"built native library: {target}", level=2)


if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            python, abi, plat = super().get_tag()
            return "py3", "none", plat
else:
    bdist_wheel = None

cmdclass = {"build_py": build_py}
if bdist_wheel is not None:
    cmdclass["bdist_wheel"] = bdist_wheel

setup(
    cmdclass=cmdclass,
    package_data={"fastmatrix": ["*.so", "*.dll", "*.dylib"]},
    include_package_data=True,
)
