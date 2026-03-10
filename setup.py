"""Package setup for RAG Email Reply Generator."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="email-reply-generator",
    version="1.0.0",
    author="RAG Email Team",
    description="Production-ready RAG-based Email Reply Generator using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhinav23bce9835-dev/email-reply-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications :: Email",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "email-reply-ingest=scripts.ingest_data:main",
            "email-reply-evaluate=scripts.run_evaluation:main",
        ],
    },
)
