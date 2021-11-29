from setuptools import setup, find_packages

import glob, os
dir_path = os.path.dirname(os.path.realpath(__file__))
scripts = [file for file in glob.glob(dir_path + "/*.py")]

setup(
    name="blue",
    version="1.0",
    packages=["blue"],
)
