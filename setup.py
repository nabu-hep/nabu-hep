from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()

with open("src/llhdflow/_version.py", mode="r", encoding="UTF-8") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = ["flowjax==15.1.0", "scipy>=1.10.0", "matplotlib"]

setup(
    name="llhdflow",
    version=version,
    description=("Smooth inference for reinterpretation studies"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpeysideHEP/spey",
    project_urls={
        "Bug Tracker": "https://github.com/SpeysideHEP/spey/issues",
        "Documentation": "https://spey.readthedocs.io",
        "Repository": "https://github.com/SpeysideHEP/spey",
        "Homepage": "https://github.com/SpeysideHEP/spey",
        "Download": f"https://github.com/SpeysideHEP/spey/archive/refs/tags/v{version}.tar.gz",
    },
    download_url=f"https://github.com/SpeysideHEP/spey/archive/refs/tags/v{version}.tar.gz",
    author="Jack Y. Araz",
    author_email=("jack.araz@stonybrook.edu"),
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
