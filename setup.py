"""
Setup configuration for HalluField package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="hallufield",
    version="0.1.0",
    author="HalluField Team",
    author_email="your.email@example.com",
    description="Detecting LLM Hallucinations via Field-Theoretic Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hallufield",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hallufield-generate=hallufield.core.generate:main",
            "hallufield-compute=hallufield.core.compute:main",
        ],
    },
    include_package_data=True,
    keywords="llm hallucination detection nlp deep-learning transformers",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/hallufield/issues",
        "Source": "https://github.com/yourusername/hallufield",
        "Paper": "https://arxiv.org/abs/2509.10753",
    },
)
