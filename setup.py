from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        "fire",
        "graphviz",
        "jsbeautifier",
        "jsonlines",
        "pyjsparser",
        "tqdm",
        "requests",
        "regex",
        "loguru",
        "pyarrow",

        # Data
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",

        # PyTorch
        "apex @ git+https://github.com/NVIDIA/apex.git#egg=apex",  # apex does not encode dependency on torch
        "pytorch-lightning",
        "torch",
        "torchtext",
        "wandb",

        # NLP dependencies
        "sentencepiece",
        "sacremoses",
        "transformers>=3.2.0",
        "tokenizers",
        "datasets",
    ],
    extras_require={"test": ["pytest"]}
)
