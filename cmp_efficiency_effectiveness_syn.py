"""
This file achieves the comparisons among the r-gwl methods with different configurations.

Three kinds of graphs are considered:
1) BA graph
2) Bipartite graph
3) Community graph
The size of nodes is from 100 to 5000.

The task is matching a graph and its noisy version.
The noise is 5% additional edges
"""
import methods.DataIO as DataIO
import methods.EvaluationMeasure as Eval
import methods.GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import numpy as np
import pickle
import time

# parameters for synthetic graphs
num_trials = 10
num_nodes = [500, 1000, 2000, 3000, 4000, 5000]
graph_types = ['barabasi', 'community']
methods = ['gwl', 's-gwl-3', 's-gwl-2', 's-gwl-1']
noise = 0.01
clique_size = 200
p_in = 0.2
p_out = 0.02

# parameters for r-gwl
cluster_num = [2, 4, 8]
partition_level = [3, 2, 1]

nc = np.zeros((len(num_nodes), 4, len(graph_types), num_trials))
runtime = np.zeros((len(num_nodes), 4, len(graph_types), num_trials))

for tn in range(num_trials):
    for nn in range(len(num_nodes)):
        for gn in range(len(graph_types)):

            # generate synthetic graph
            if gn == 0:
                G_src = nx.powerlaw_cluster_graph(n=num_nodes[nn], m=int(clique_size * p_in),
                                                  p=p_out * clique_size / num_nodes[nn])
                # int(np.log(num_nodes[nn]) + 1))
            else:
                G_src = nx.gaussian_random_partition_graph(n=num_nodes[nn], s=clique_size, v=5,
                                                           p_in=p_in,
                                                           p_out=p_out,
                                                           directed=True)

            G_dst = DataIO.add_noisy_edges(G_src, noise)
            # G_src = G_src.to_undirected()
            # G_dst = G_dst.to_undirected()
            print('Trial {}, #nodes {}, graph type: {}'.format(tn + 1, num_nodes[nn], graph_types[gn]))
            print('#edges: {}, {}'.format(len(G_src.edges), len(G_dst.edges)))

            # weights = np.random.rand(num_nodes[nn], num_nodes[nn]) + 1
            p_s, cost_s, idx2node_s = DataIO.extract_graph_info(G_src, weights=None)
            p_s /= np.sum(p_s)
            p_t, cost_t, idx2node_t = DataIO.extract_graph_info(G_dst, weights=None)
            p_t /= np.sum(p_t)

            for mn in range(4):
                ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                           'ot_method': 'proximal',
                           'beta': 0.2,
                           'outer_iteration': num_nodes[nn],  # outer, inner iteration, error bound of optimal transport
                           'iter_bound': 1e-10,
                           'inner_iteration': 2,
                           'sk_bound': 1e-10,
                           'node_prior': 10,
                           'max_iter': 5,  # iteration and error bound for calcuating barycenter
                           'cost_bound': 1e-16,
                           'update_p': False,  # optional updates of source distribution
                           'lr': 0,
                           'alpha': 0}
                # print(ot_dict['outer_iteration'])
                time_s = time.time()
                if mn == 0:
                    pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
                        cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict)
                else:
                    pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
                        cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict,
                        weights=None, predefine_barycenter=False, cluster_num=cluster_num[mn - 1],
                        partition_level=partition_level[mn - 1], max_node_num=0)

                runtime[nn, mn, gn, tn] = time.time() - time_s
                nc[nn, mn, gn, tn] = Eval.calculate_node_correctness(pairs_name, num_correspondence=num_nodes[nn])
                print('method: {}, duration {:.4f}second, nc {:.4f}.'.format(
                    methods[mn], runtime[nn, mn, gn, tn], nc[nn, mn, gn, tn]))

with open('results/cmp_efficiency_effectiveness.pkl', 'wb') as f:
    pickle.dump([nc, runtime], f)
