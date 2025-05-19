#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="chorus-detection",
    version="0.1.0",
    description="A tool for detecting chorus sections in audio files",
    author="Dennis Dang",
    url="https://github.com/dennisvdang/chorus-detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.4.0",
        "librosa>=0.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pydub>=0.24.0",
        "pytube>=11.0.0",
        "scikit-learn>=0.24.0",
        "streamlit>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "chorus-detection=cli.cli_app:run_cli",
            "chorus-detection-web=web.app:run_web_app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
) 