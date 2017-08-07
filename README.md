# ANALOGY
The repository contains C++ implementation of the following paper

[Analogical Inference for Multi-Relational Embeddings](https://arxiv.org/abs/1705.02426).

Hanxiao Liu, Yuexin Wu, Yiming Yang.

International Conference on Machine Learning, ICML 2017.


## Basic Usage
For training
```
make && ./main -model_path analogy.model
```
For prediction
```
./main -prediction 1 -model_path analogy.model
```
