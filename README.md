# S-GWL
This package includes the implementation of my NeurIPS2019 work **"Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching"** [https://arxiv.org/pdf/1905.07645.pdf]

The examples include:
* Partition a single graph (i.e., community detection)
* Match two or more graphs (i.e., network alignment)

The package is developed on Ubuntu 18.04. 
Specifically, the baseline "metis" has been compiled as a library "libmetis.so" in the "baselines/metis-5.1.0" folder. The source code can be found at http://glaros.dtc.umn.edu/gkhome/metis/metis/download.

# Dependencies
* matplotlib
* metis (a wrapper of metis)  https://pypi.org/project/metis/
* networkx
* numpy
* pandas
* python-louvain (community) https://github.com/taynaud/python-louvain
* sklearn

Note that after install the wrapper "metis", you need to set the path of **libmetis.so** manually in **metis.py**.

# Baselines
Graph partitioning
* Metis
* Louvain
* FastGreedy
* Fluid

Graph matching
* HubAlign [https://academic.oup.com/bioinformatics/article/30/17/i438/200169]
* NETAL [https://academic.oup.com/bioinformatics/article/29/13/1654/185807]

We download these two packages from https://ttic.uchicago.edu/~hashemifar/. 
Please read the "readme" file in their folders to run these two methods.

# Citations
@article{xu2019scalable,
  title={Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching},
  author={Xu, Hongteng and Luo, Dixin and Carin, Lawrence},
  journal={arXiv preprint arXiv:1905.07645},
  year={2019}
}

