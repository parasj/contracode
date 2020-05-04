from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
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
    ],
    extras_require={"test": ["pytest"]}
)
