# Unsupervised representation learning of Javascript methods

## Installation
Dependencies: Python 3.7, NodeJS, NPM
```bash
$ bash download_data.sh
$ npm install
$ pip install -e "."
```

## Example transformations / invariant coding
* Method extraction (always runs first)
* **Subsampling inputs (analagous to cropping an image), 1D or 2D**
* **Variable renaming to a unique identifier**
* **Clojure compiler transformations for DCE, variable declaration hoisting, etc.**
* const <-> var conversion
* **noop insertion, e.g. add a variable declaration, `console.log(<RANDOMSTRING>)` or a comment**
* type masking
* **ast subtree masking**
* local line reordering
* remove comments, `console.log`, etc.
* convert for-loops to while-loops
* for-loop unrolling
* prepack.io precomputation
* function declared through variable <-> direct function declaration
* ~Coffeescript compiler / decaffienate~
* Add closures around subtrees in AST [Example](https://repl.it/repls/BlushingGoldenrodBrain)

## Todos
* Transformations
* Dataloader
