from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="libxsl",
    version="0.1.0",
    author="Bowen Yuan",
    author_email="yuanbwhust@gmail.com",
    description="A ML package for applying LLMs to text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryaninhust/neo",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "transformers>=4.0.0",
        "numpy",
        "tqdm",
        "pyyaml"
    ],
    dependency_links=[
        "git+https://github.com/ryaninhust/pyxclib.git#egg=pyxclib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)

