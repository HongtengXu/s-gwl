"""
Data preparation for PPI dataset

database = {'costs': a list of adjacency matrices of different graphs,
            'probs': a list of distributions of nodes in different graphs,
            'idx2nodes': a list of dictionaries mapping index to node name,
            'correspondence': None or a list of correspondence set}
"""
import methods.GromovWassersteinGraphToolkit as GwGt
import pickle
import time
import warnings

warnings.filterwarnings("ignore")


with open('data/PPI_syn_database.pkl', 'rb') as f:
    database = pickle.load(f)

num_iters = 4000
ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 0.025,
           'outer_iteration': num_iters,
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

for N in [2, 3, 4, 5]:
    pairs_name = {}
    print('match {} graphs'.format(N+1))
    for i in range(N):
        cost_s = database['costs'][0]
        cost_t = database['costs'][i+1]
        p_s = database['probs'][0]
        p_t = database['probs'][i+1]
        idx2node_s = database['idx2nodes'][0]
        idx2node_t = database['idx2nodes'][i+1]
        num_nodes = min([len(idx2node_s), len(idx2node_t)])

        time_s = time.time()
        ot_dict['outer_iteration'] = num_iters
        pairs_idx, pairs_name[i], pairs_confidence = GwGt.recursive_direct_graph_matching(
           cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict,
           weights=None, predefine_barycenter=False, cluster_num=2,
           partition_level=3, max_node_num=0)
        # pairs_idx, pairs_name[i], pairs_confidence = GwGt.direct_graph_matching(
        #     cost_s, cost_t, p_s, p_t, idx2node_s, idx2node_t, ot_dict)
        runtime = time.time() - time_s
        print('-- G{} -> G{}, time = {:.2f}sec'.format(0, i+1, runtime))

    pairs = []
    for i in range(1004):
        tmp = []
        for j in range(N):
            if j == 0:
                for pair in pairs_name[j]:
                    # print(pair)
                    if pair[0] == database['idx2nodes'][0][i]:
                        tmp = pair
            else:
                for pair in pairs_name[j]:
                    # print(pair)
                    if pair[0] == database['idx2nodes'][0][i]:
                        tmp.append(pair[1])
        if len(tmp) == N+1:
            pairs.append(tmp)

    nca = 0
    nc1 = 0
    for sets in pairs:
        tmp = 0
        for n in range(N):
            if sets[n] == sets[n+1]:
                tmp += 1
        if tmp == N:
            nca += 1

        tmp = 0
        for n in range(N):
            for m in range(n+1, N+1):
                if sets[n] == sets[m]:
                    tmp = 1
                    break
        if tmp == 1:
            nc1 += 1

    print('NC@1={:.4f}, NC@All={:.4f}'.format(nc1/1004*100, nca/1004*100))

