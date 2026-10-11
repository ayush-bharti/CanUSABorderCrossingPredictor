"""Setup script for the border_predictor package."""
from setuptools import setup, find_packages

setup(
    name="border_predictor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.2",
        "flask>=3.0.3",
        "holidays>=0.25",
        "lightgbm>=4.5.0",
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A model to predict border crossing wait times",
    keywords="border, crossing, prediction, lightgbm, time series",
    url="https://github.com/yourusername/CanUSABorderCrossingPredictor",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
