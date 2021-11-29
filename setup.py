import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    #packages=setuptools.find_packages(where="bluest"),
    python_requires=">=3.7",
)
