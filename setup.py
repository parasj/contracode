from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        # "apex @ git+https://github.com/NVIDIA/apex.git#egg=apex"  # apex does not encode dependency on torch
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
        "transformers",
        "requests",
        "regex",
        "sacremoses",
        "wandb",
        "loguru"
    ],
    extras_require={"test": ["pytest"]}
)
