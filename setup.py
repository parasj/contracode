from setuptools import setup, find_packages

setup(name='representjs',
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
          "seaborn"
      ])