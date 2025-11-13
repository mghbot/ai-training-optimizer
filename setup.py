"""
Setup script for ACMDS optimizer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="acmds-optimizer",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Adaptive Curriculum with Memory-Driven Difficulty Scaling for PyTorch training on RTX 3090",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/acmds-optimizer",
    py_modules=["acmds_optimizer", "train"],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "acmds-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="pytorch deep-learning gpu-optimization curriculum-learning memory-management rtx-3090",
)
