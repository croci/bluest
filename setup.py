from setuptools import setup
from pybind11.setup_helpers import intree_extensions

ext_modules = intree_extensions(["bluest/cmisc.cpp"])

ext_modules[0].extra_compile_args[:0] = ["-O3", "-m64", "-ftree-vectorize", "-ffast-math", "-march=native"]
ext_modules[0].name = "_cmisc_bluest"

setup(
    name="bluest",
    packages=["bluest"],
    package_dir={"bluest": "bluest"},
    has_ext_modules=lambda: True,
    include_package_data=True,
    zip_safe=False,
    ext_modules = ext_modules,
)
