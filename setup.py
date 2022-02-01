import setuptools
import subprocess
import os
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools.command.build_ext import build_ext
class Build(build_ext):
     """Customized setuptools build command - builds protos on build."""
     def run(self):
         from pybind11 import get_include
         pybind11_dir = get_include()[:-8]
         os.environ["PYBIND11_DIR"] = pybind11_dir
         
         out = subprocess.run(["make"])
         if out.returncode != 0:
             sys.exit(-1)
         build_ext.run(self)

setuptools.setup(
    name="bluest",
    version="1.0.0",
    author="Matteo Croci",
    author_email="matteo.croci@austin.utexas.edu",
    description="BLUE estimator library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/croci/blue/",
    project_urls={
        "Bug Tracker": "https://github.com/croci/blue/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    packages=["bluest"],
    package_dir={"bluest": "bluest"},
    python_requires=">=3.6",
    install_requires=['pybind11'],
    setup_requires=['pybind11'],
    has_ext_modules=lambda: True,
    cmdclass={ 'build_ext': Build,},
    package_data={"bluest":["bluest/cmisc.so"]},
    include_package_data=True,
    zip_safe=False,
)
