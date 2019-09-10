"""
Compare our Gromov-Wasserstein graph partitioning method with the state-of-the-art methods
In this case, we assume that the number of partitions is UNKNOWN for various methods.

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
import methods.DataIO as DataIO
import methods.GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import numpy as np
import time
import metis
import community
import pickle
import warnings

from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.asyn_fluid import asyn_fluidc
from sklearn import metrics

warnings.filterwarnings("ignore")

num_trials = 10
num_nodes = 4000
clique_size = 200
p_in = 0.2
ps_out = [0.03, 0.06, 0.09, 0.12, 0.15]


methods = ['metis', 'FastGreedy', 'Louvain', 'Fluid', 'S-GWL']
num_methods = len(methods)
mutual_info = np.zeros((num_methods, len(ps_out), num_trials))
runtime = np.zeros((num_methods, len(ps_out), num_trials))

for tn in range(num_trials):
    for pn in range(len(ps_out)):
        print('Trial {}, noise level {:.3f}'.format(tn + 1, ps_out[pn] / p_in))
        # generate synthetic graph
        G = nx.gaussian_random_partition_graph(n=num_nodes, s=clique_size, v=5,
                                               p_in=p_in, p_out=ps_out[pn], directed=True)
        p_s, cost_s, idx2node = DataIO.extract_graph_info(G)
        p_s = (p_s + 1) ** 0.01
        p_s /= np.sum(p_s)
        gt = np.zeros((num_nodes,))
        for i in range(len(G.nodes)):
            gt[i] = G.nodes[i]['block']
        num_partitions = int(np.max(gt) + 1)
        print('The number of partitions = {}'.format(num_partitions))

        # baseline 3: Louvain Community Detection
        time_s = time.time()
        partition = community.best_partition(G.to_undirected())
        est_idx = np.zeros((num_nodes,))
        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys()
                          if partition[nodes] == com]
            for idx in list_nodes:
                est_idx[idx] = com
        est_number = int(np.max(est_idx) + 1)
        print('The estimated number of partitions = {}'.format(est_number))
        mutual_info[2, pn, tn] = metrics.adjusted_mutual_info_score(gt, est_idx)
        runtime[2, pn, tn] = time.time() - time_s
        print('-- {}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[2],
                                                                            runtime[2, pn, tn],
                                                                            mutual_info[2, pn, tn]))

        # baseline 1: metis
        time_s = time.time()
        _, parts = metis.part_graph(G, nparts=est_number)
        est_idx = np.asarray(parts)
        mutual_info[0, pn, tn] = metrics.adjusted_mutual_info_score(gt, est_idx)
        runtime[0, pn, tn] = time.time() - time_s
        print('-- {}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[0],
                                                                            runtime[0, pn, tn],
                                                                            mutual_info[0, pn, tn]))

        # baseline 2: FastGreedy, Clauset-Newman-Moore greedy modularity maximization
        time_s = time.time()
        list_nodes = list(greedy_modularity_communities(G.to_undirected()))
        est_idx = np.zeros((num_nodes,))
        for i in range(len(list_nodes)):
            for idx in list_nodes[i]:
                est_idx[idx] = i
        mutual_info[1, pn, tn] = metrics.adjusted_mutual_info_score(gt, est_idx)
        runtime[1, pn, tn] = time.time() - time_s
        print('-- {}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[1],
                                                                            runtime[1, pn, tn],
                                                                            mutual_info[1, pn, tn]))

        # baseline 4: Fluid Communities algorithm.
        time_s = time.time()
        comp = asyn_fluidc(G.to_undirected(), k=est_number)
        list_nodes = [frozenset(c) for c in comp]
        est_idx = np.zeros((num_nodes,))
        for i in range(len(list_nodes)):
            for idx in list_nodes[i]:
                est_idx[idx] = i
        mutual_info[3, pn, tn] = metrics.adjusted_mutual_info_score(gt, est_idx)
        runtime[3, pn, tn] = time.time() - time_s
        print('-- {}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[3],
                                                                            runtime[3, pn, tn],
                                                                            mutual_info[3, pn, tn]))

        ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                   'ot_method': 'proximal',
                   'beta': 0.15,
                   'outer_iteration': 2 * num_nodes,  # outer, inner iterations and error bound of optimal transport
                   'iter_bound': 1e-30,
                   'inner_iteration': 5,
                   'sk_bound': 1e-30,
                   'node_prior': 0.0001,
                   'max_iter': 1,  # iteration and error bound for calcuating barycenter
                   'cost_bound': 1e-16,
                   'update_p': False,  # optional updates of source distribution
                   'lr': 0,
                   'alpha': 0}

        time_s = time.time()
        sub_costs, sub_probs, sub_idx2nodes = GwGt.recursive_graph_partition(0.5 * (cost_s + cost_s.T),
                                                                             p_s,
                                                                             idx2node,
                                                                             ot_dict,
                                                                             max_node_num=300)
        est_idx = np.zeros((num_nodes,))
        for n_cluster in range(len(sub_idx2nodes)):
            for key in sub_idx2nodes[n_cluster].keys():
                idx = sub_idx2nodes[n_cluster][key]
                est_idx[idx] = n_cluster

        mutual_info[4, pn, tn] = metrics.adjusted_mutual_info_score(gt, est_idx)
        runtime[4, pn, tn] = time.time() - time_s
        print('-- {}: runtime={:.4f}sec, mutual information={:.4f}.'.format(methods[4],
                                                                            runtime[4, pn, tn],
                                                                            mutual_info[4, pn, tn]))

with open('results/cmp_single_graph_partition_SyntheticGraph.pkl', 'wb') as f:
    pickle.dump([methods, mutual_info, runtime], f)
