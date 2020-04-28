from setuptools import setup

setup(
    name='representjs',
    version='1.0',
    packages=["representjs"],
    python_requires=">=3.7",
    install_requires=[
        "transformers",
        "torch",
        "torchtext",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyjsparser",
        "graphviz"
    ],
    extras_require={"test": ["pytest"]}
)
