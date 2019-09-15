"""
Data preparation for PPI dataset

database = {'costs': a list of adjacency matrices of different graphs,
            'probs': a list of distributions of nodes in different graphs,
            'idx2nodes': a list of dictionaries mapping index to node name,
            'correspondence': None or a list of correspondence set}
"""

import methods.EvaluationMeasure as Eval
import methods.GromovWassersteinGraphToolkit as GwGt
import pickle
import time
import warnings

warnings.filterwarnings("ignore")


with open('data/PPI_syn_database.pkl', 'rb') as f:
    database = pickle.load(f)

num_iter = 2000
ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 0.025,
           'outer_iteration': num_iter,
           # outer, inner iteration, error bound of optimal transport
           'iter_bound': 1e-30,
           'inner_iteration': 2,
           'sk_bound': 1e-30,
           'node_prior': 1e3,
           'max_iter': 4,  # iteration and error bound for calcuating barycenter
           'cost_bound': 1e-26,
           'update_p': False,  # optional updates of source distribution
           'lr': 0,
           'alpha': 0}

for i in range(5):
    cost_s = database['costs'][0]
    cost_t = database['costs'][i+1]
    p_s = database['probs'][0]
    p_t = database['probs'][i+1]
    idx2node_s = database['idx2nodes'][0]
    idx2node_t = database['idx2nodes'][i+1]
    num_nodes = min([len(idx2node_s), len(idx2node_t)])

    time_s = time.time()
    ot_dict['outer_iteration'] = num_iter
    pairs_idx, pairs_name, pairs_confidence = GwGt.direct_graph_matching(
        0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t, idx2node_s, idx2node_t, ot_dict)
    runtime = time.time() - time_s
    nc = Eval.calculate_node_correctness(pairs_name, num_correspondence=num_nodes)
    print('method: gwl, duration {:.4f}s, nc {:.4f}.'.format(runtime, nc))
    with open('results/gwl_ppi_syn_{}.pkl'.format(i + 1), 'wb') as f:
        pickle.dump([nc, runtime], f)

    time_s = time.time()
    ot_dict['outer_iteration'] = num_iter
    pairs_idx, pairs_name, pairs_confidence = GwGt.recursive_direct_graph_matching(
        0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t, idx2node_s, idx2node_t, ot_dict,
        weights=None, predefine_barycenter=False, cluster_num=2,
        partition_level=3, max_node_num=0)
    runtime = time.time() - time_s
    nc = Eval.calculate_node_correctness(pairs_name, num_correspondence=num_nodes)
    print('method: s-gwl, duration {:.4f}s, nc {:.4f}.'.format(runtime, nc))
    with open('results/sgwl_ppi_syn_{}.pkl'.format(i + 1), 'wb') as f:
        pickle.dump([nc, runtime], f)




