"""
ompare our Gromov-Wasserstein graph partitioning method (GWL) with the state-of-the-art methods

baseline 1: metis
  George Karypis and Vipin Kumar.
  A fast and high quality multilevel scheme for partitioning irregular graphs.
  International Conference on Parallel Processing, pp. 113-122, 1995.

baseline 2: Modularity-based communities (Clauset-Newman-Moore greedy modularity maximization)
  Clauset, A., Newman, M. E., & Moore, C.
  “Finding community structure in very large networks.”
  Physical Review E 70(6), 2004.

baseline 3: Louvain Community Detection
  Blondel, Vincent D; Guillaume, Jean-Loup; Lambiotte, Renaud; Lefebvre, Etienne (9 October 2008).
  "Fast unfolding of communities in large networks".
  Journal of Statistical Mechanics: Theory and Experiment. 2008 (10): P10008. arXiv:0803.0476

baseline 4: Fluid Communities algorithm.
  Parés F., Garcia-Gasulla D. et al.
  “Fluid Communities: A Competitive and Highly Scalable Community Detection Algorithm”.
  [https://arxiv.org/pdf/1703.09307.pdf].
"""
import methods.GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import numpy as np
import pickle
import time
import metis
import community
import warnings

from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from sklearn import metrics

warnings.filterwarnings("ignore")

methods = ['metis', 'Max-Modularity', 'Louvain', 'Fluid', 'GWL']
num_nodes = 1991
num_partitions = 12
num_methods = len(methods)
mutual_info = np.zeros((num_methods,))
runtime = np.zeros((num_methods,))

# load data
with open('data/India_database.pkl', 'rb') as f:
    database = pickle.load(f)
G = nx.Graph()
for i in range(num_nodes):
    G.add_node(i)
for edge in database['edges']:
    G.add_edge(edge[0], edge[1])
# add noise
for j in range(int(0.5 * len(database['edges']))):
    x1 = int(num_nodes * np.random.rand())
    x2 = int(num_nodes * np.random.rand())
    if database['label'][x1] != database['label'][x2]:
        G.add_edge(x1, x2)

# baseline 1: metis
time_s = time.time()
_, parts = metis.part_graph(G.to_undirected(), nparts=num_partitions)
runtime[0] = time.time() - time_s
mutual_info[0] = metrics.adjusted_mutual_info_score(database['label'], np.asarray(parts))
print('{}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[0], runtime[0], mutual_info[0]))

# baseline 2: FastGreedy Clauset-Newman-Moore greedy modularity maximization
time_s = time.time()
list_nodes = list(greedy_modularity_communities(G))
est_idx = np.zeros((num_nodes,))
for i in range(len(list_nodes)):
    for idx in list_nodes[i]:
        est_idx[idx] = i
runtime[1] = time.time() - time_s
mutual_info[1] = metrics.adjusted_mutual_info_score(database['label'], est_idx)
print('{}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[1], runtime[1], mutual_info[1]))

# baseline 3: Louvain Community Detection
time_s = time.time()
partition = community.best_partition(G)
est_idx = np.zeros((num_nodes,))
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    for idx in list_nodes:
        est_idx[idx] = com
runtime[2] = time.time() - time_s
mutual_info[2] = metrics.adjusted_mutual_info_score(database['label'], est_idx)
print('{}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[2], runtime[2], mutual_info[2]))


# baseline 4: Fluid Communities algorithm.
time_s = time.time()
comp = asyn_fluidc(G.to_undirected(), k=num_partitions)
list_nodes = [frozenset(c) for c in comp]
est_idx = np.zeros((num_nodes,))
for i in range(len(list_nodes)):
    for idx in list_nodes[i]:
        est_idx[idx] = i
runtime[3] = time.time() - time_s
mutual_info[3] = metrics.adjusted_mutual_info_score(database['label'], est_idx)
print('{}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[3], runtime[3], mutual_info[3]))


# proposed method: scalable Gromov-Wasserstein learning
ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 5e-5,
           'outer_iteration': 200,
           # outer, inner iteration, error bound of optimal transport
           'iter_bound': 1e-30,
           'inner_iteration': 1,
           'sk_bound': 1e-30,
           'node_prior': 0,
           'max_iter': 200,  # iteration and error bound for calcuating barycenter
           'cost_bound': 1e-16,
           'update_p': False,  # optional updates of source distribution
           'lr': 0,
           'alpha': 0}

cost = database['cost']
p_s = database['prob'] + 5e-1
p_s /= np.sum(p_s)
p_t = GwGt.estimate_target_distribution({0: p_s}, dim_t=num_partitions)

time_s = time.time()
sub_costs, sub_probs, sub_idx2nodes, trans = GwGt.graph_partition(cost,
                                                                  p_s,
                                                                  p_t,
                                                                  database['idx2node'],
                                                                  ot_dict)
est_idx = np.argmax(trans, axis=1)
runtime[4] = time.time() - time_s
mutual_info[4] = metrics.adjusted_mutual_info_score(database['label'], est_idx)
print('{}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[4], runtime[4], mutual_info[4]))


with open('results/cmp_single_graph_partition_India.pkl', 'wb') as f:
    pickle.dump([methods, mutual_info, runtime], f)





