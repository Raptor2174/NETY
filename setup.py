#!/usr/bin/env python3
"""
Setup configuration for NETY 
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nety",
    version="1.0.0",
    author="Raptor_",
    description="NETY",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NETY",
    packages=find_packages(exclude=["tests", "scripts", "documentation", "app", "data", "models"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nety=nety.main:main",
        ],
    },
)
