# ANALOGY
The repository provides lightweight C++ implementations for the following papers

* [Analogical Inference for Multi-Relational Embeddings](https://arxiv.org/abs/1705.02426). Hanxiao Liu, Yuexin Wu and Yiming Yang. ICML 2017.

* [Complex Embeddings for Simple Link Prediction](http://proceedings.mlr.press/v48/trouillon16.pdf). Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard. ICML 2016.

* [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf). Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao and Li Deng. ICLR 2015.

## Basic Usage
Training
```
make && ./main -algorithm Analogy -model_path output.model
```
Prediction
```
./main -algorithm Analogy -model_path output.model -prediction 1
```
The program runs with 32 threads by default. For more options, please refer to `main.cc`.
## Contributors
Please cite the following if you use the code for publication
```
@article{liu2017analogical,
  title={Analogical Inference for Multi-Relational Embeddings},
  author={Liu, Hanxiao and Wu, Yuexin and Yang, Yiming},
  journal={arXiv preprint arXiv:1705.02426},
  year={2017}
}
```
