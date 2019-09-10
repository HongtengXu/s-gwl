# S-GWL
This package includes the implementation of my NeurIPS2019 work **Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching** [https://arxiv.org/pdf/1905.07645.pdf]

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

Note that after install the wrapper "metis", you need to set the path of **libmetis.so** mantually in **metis.py**.

# Baselines
Graph partitioning
* Metis
* Louvain
* FastGreedy
* Fluid

Graph matching
* HubAlign
* NETAL
