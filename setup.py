"""
ZEEPT - Advanced LLM Fine-tuning and Evaluation Platform
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="zeept",
    version="1.0.0",
    author="ZEEPT Team",
    author_email="contact@zeept.ai",
    description="Advanced LLM Fine-tuning and Evaluation Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zeept",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
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
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "sphinx>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zeept-train=train:main",
            "zeept-app=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="nlp llm transformer fine-tuning evaluation pytorch",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/zeept/issues",
        "Source": "https://github.com/yourusername/zeept",
        "Documentation": "https://zeept.readthedocs.io",
    },
)
