from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()

with open("nabu/_version.py", mode="r", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["flowjax", "scipy>=1.10.0", "matplotlib"]

setup(
    name="nabu",
    version=version,
    description=("Smooth inference for reinterpretation studies"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nabu-hep/nabu",
    project_urls={
        "Bug Tracker": "https://github.com/nabu-hep/nabu/issues",
        "Documentation": "https://nabu-hep.readthedocs.io",
        "Repository": "https://github.com/nabu-hep/nabu",
        "Homepage": "https://github.com/nabu-hep/nabu",
        "Download": f"https://github.com/nabu-hep/nabu/archive/refs/tags/v{version}.tar.gz",
    },
    download_url=f"https://github.com/nabu-hep/nabu/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("jack.araz@stonybrook.edu"),
    license="MIT",
    packages=["nabu"],
    scripts=["bin/fit-to-data"],
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
