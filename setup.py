from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        "fire",
        "graphviz",
        "jsonlines",
        "matplotlib",
        "numpy",
        "pandas",
        "pyjsparser",
        "seaborn",
        "sentencepiece",
        "torch",
        "torchtext",
        "torchtext",
        "tqdm",
        "transformers",
        "wandb"
    ],
    extras_require={"test": ["pytest"]}
)
