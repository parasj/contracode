from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        "boto3",
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
        "allennlp",
        "openai",
    ],
    extras_require={"test": ["pytest"]}
)
