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
        "matplotlib",
        "numpy",
        "pandas",
        "pyjsparser",
        "pytorch-lightning",
        "seaborn",
        "sentencepiece",
        "torch",
        "torchtext",
        "tqdm",
        "requests",
        "regex",
        "sacremoses",
        "wandb",
        "loguru",
        "transformers>=3.1.0",
        "tokenizers",
        "pyarrow"
    ],
    extras_require={"test": ["pytest"]}
)
