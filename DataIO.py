"""
This script contains the data I/O operations for Graph data
"""
import copy
import networkx as nx
import numpy as np
import pandas
import random
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, List, Tuple


def load_txt_community_file(edge_path: str, label_path: str, flag: str = ' ') -> Dict:
    """
    Load edge list in .txt file and community label in .txt file
    Args:
        edge_path: the path of an edge list
        label_path: the path of community labels
        flag: the segment flag between src and dst

    Returns:
        database = {'cost': an adjacency matrix of a graph,
                    'prob': a distribution of nodes in a graph,
                    'idx2node': a dictionary mapping index to node name,
                    'label': community index}
    """
    with open(edge_path) as f:
        edges_str = f.readlines()
    f.close()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    edges_str = [x.strip() for x in edges_str]

    edges = []
    node2idx = {}
    index = 0
    for edge in edges_str:
        idx = edge.find(flag)
        if idx > -1:
            src = edge[:idx]
            dst = edge[(idx + len(flag)):]
            if src not in node2idx.keys():
                node2idx[src] = index
                index += 1
            if dst not in node2idx.keys():
                node2idx[dst] = index
                index += 1
            edges.append([node2idx[src], node2idx[dst]])

    idx2node = {}
    for name in node2idx.keys():
        index = node2idx[name]
        idx2node[index] = name

    # build adjacency matrix and node distribution
    num_nodes = len(node2idx)
    prob = np.zeros((num_nodes, 1))
    cost = lil_matrix((num_nodes, num_nodes))
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        cost[src, dst] += 1
        prob[src, 0] += 1
        prob[dst, 0] += 1
    cost = csr_matrix(cost)
    prob /= np.sum(prob)

    with open(label_path) as f:
        labels_str = f.readlines()
    f.close()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    labels_str = [x.strip() for x in labels_str]
    labels = np.zeros((num_nodes, ))
    for label in labels_str:
        idx = label.find(flag)
        if idx > -1:
            node = label[:idx]
            index = node2idx[node]
            community = int(label[(idx + len(flag)):])
            labels[index] = community

    database = {'cost': cost,
                'prob': prob,
                'idx2node': idx2node,
                'label': labels,
                'edges': edges}
    return database


def load_multilayer_edge_file(file_path: str, tags: List) -> Dict:
    """
    Load edge list stored in .csv file
    The file should be one edge per line as follows,
    src1, net1, dst1
    src2, net2, dst2
    ...
    srcN, netN, dstN

    Args:
        file_path: the path of an edge list file.
        tags: a list of column tags in csv files

    Returns:
        database = {'costs': a list of adjacency matrices of different graphs,
                    'probs': a list of distributions of nodes in different graphs,
                    'idx2nodes': a list of dictionaries mapping index to node name,
                    'correspondence': None or a list of correspondence set}
    """
    pd_lib = pandas.read_csv(file_path)

    num_edges = 0
    index = 0
    node2idx = {}
    for i, row in pd_lib.iterrows():
        num_edges += 1
        src = row[tags[0]]
        dst = row[tags[2]]
        if src not in node2idx.keys():
            node2idx[src] = index
            index += 1
        if dst not in node2idx.keys():
            node2idx[dst] = index
            index += 1
    print(num_edges)
    print(len(node2idx))

    # build graph2idx
    graph2idx = {}
    index = 0
    for i, row in pd_lib.iterrows():
        net = row[tags[1]]
        if net not in graph2idx.keys():
            graph2idx[net] = index
            index += 1

    # build node2idxs and idx2nodes and edge lists
    num_graphs = len(graph2idx)
    node2idxs = [{} for _ in range(num_graphs)]
    indices = [0 for _ in range(num_graphs)]
    edges = [[] for _ in range(num_graphs)]
    for i, row in pd_lib.iterrows():
        net = row[tags[1]]
        src = row[tags[0]]
        dst = row[tags[2]]
        net_idx = graph2idx[net]
        if src not in node2idxs[net_idx].keys():
            node2idxs[net_idx][src] = indices[net_idx]
            indices[net_idx] += 1
        if dst not in node2idxs[net_idx].keys():
            node2idxs[net_idx][dst] = indices[net_idx]
            indices[net_idx] += 1
        edges[net_idx].append([node2idxs[net_idx][src], node2idxs[net_idx][dst]])

    costs = []
    probs = []
    idx2nodes = []
    for i in range(len(edges)):
        idx2node = {}
        for name in node2idxs[i].keys():
            idx = node2idxs[i][name]
            idx2node[name] = idx
        idx2nodes.append(idx2node)
        num_nodes = len(idx2node)
        print(num_nodes)
        prob = np.zeros((num_nodes, 1))
        cost = lil_matrix((num_nodes, num_nodes))
        for edge in edges[i]:
            src = edge[0]
            dst = edge[1]
            cost[src, dst] += 1
            prob[src, 0] += 1
            prob[dst, 0] += 1
        cost = csr_matrix(cost)
        prob /= np.sum(prob)
        costs.append(cost)
        probs.append(prob)

    database = {'costs': costs,
                'probs': probs,
                'idx2nodes': idx2nodes}

    return database


def load_layer_edge_file(file_path: str, tags: List, net_idx: int) -> Tuple[csr_matrix, np.ndarray, Dict, Dict]:
    """
    Load edge list stored in .csv file
    The file should be one edge per line as follows,
    src1, net1, dst1
    src2, net2, dst2
    ...
    srcN, netN, dstN

    Args:
        file_path: the path of an edge list file.
        tags: a list of column tags in csv files
        net_idx: the index of network

    Returns:
        database = {'costs': a list of adjacency matrices of different graphs,
                    'probs': a list of distributions of nodes in different graphs,
                    'idx2nodes': a list of dictionaries mapping index to node name,
                    'correspondence': None or a list of correspondence set}
    """
    pd_lib = pandas.read_csv(file_path)

    # build node2idxs and idx2nodes and edge lists
    node2idx = {}
    index = 0
    edges = []
    for i, row in pd_lib.iterrows():
        net = row[tags[1]]
        src = row[tags[0]]
        dst = row[tags[2]]
        if net == net_idx:
            if src not in node2idx.keys():
                node2idx[src] = index
                index += 1
            if dst not in node2idx.keys():
                node2idx[dst] = index
                index += 1
            edges.append([node2idx[src], node2idx[dst]])

    idx2node = {}
    for name in node2idx.keys():
        idx = node2idx[name]
        idx2node[idx] = name

    num_nodes = len(idx2node)
    prob = np.zeros((num_nodes, 1))
    cost = lil_matrix((num_nodes, num_nodes))
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        cost[src, dst] += 1
        prob[src, 0] += 1
        prob[dst, 0] += 1
    cost = csr_matrix(cost)
    prob /= np.sum(prob)

    # with open('{}.tab'.format(net_idx), 'a') as f:
    #     for edge in edges:
    #         src = 'n' + str(idx2node[edge[0]])
    #         dst = 'n' + str(idx2node[edge[1]])
    #         f.write('{}\t{}\n'.format(src, dst))
    # f.close()
    return cost, prob, idx2node, node2idx


def load_txt_edge_file(file_path: str, flag: str = '\t') -> Tuple[csr_matrix, np.ndarray, Dict, Dict]:
    """
    Load edge list stored in .tab/.txt/other text-format file
    The file should be one edge per line as follows,
    src1 dst1
    src2 dst2
    ...
    srcN dstN

    Args:
        file_path: the path of an edge list file.
        flag: the string used to segment src and dst

    Returns:
        database = {'node2gt': a list of correspondence between each observed graph and the ground truth,
                    'correspondence': a (num_node, num_graph) array storing all correspondences across graphs
                    'nums': a list of #nodes in each graph,
                    'realE': a list of real edges in each graph,
                    'obsE': a list of observed edges in each graph}
    """
    # build edge list, node2idx and idx2node maps
    with open(file_path) as f:
        edges_str = f.readlines()
    f.close()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    edges_str = [x.strip() for x in edges_str]
    edges = []
    node2idx = {}
    index = 0
    for edge in edges_str:
        idx = edge.find(flag)
        if idx > -1:
            src = edge[:idx]
            dst = edge[(idx+len(flag)):]
            if src not in node2idx.keys():
                node2idx[src] = index
                index += 1
            if dst not in node2idx.keys():
                node2idx[dst] = index
                index += 1
            edges.append([node2idx[src], node2idx[dst]])

    idx2node = {}
    for name in node2idx.keys():
        index = node2idx[name]
        idx2node[index] = name

    # build adjacency matrix and node distribution
    num_nodes = len(node2idx)
    prob = np.zeros((num_nodes, 1))
    cost = lil_matrix((num_nodes, num_nodes))
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        cost[src, dst] += 1
        prob[src, 0] += 1
        prob[dst, 0] += 1
    cost = csr_matrix(cost)
    prob /= np.sum(prob)

    return cost, prob, idx2node, node2idx


def csv2tab_edge_files(path_list: List):
    for n in range(len(path_list)):
        pd_lib = pandas.read_csv(path_list[n])

        with open('graph_{}.tab'.format(n), 'a') as f:
            for i, row in pd_lib.iterrows():
                source = row['Source']
                target = row['Target']
                f.write('{}\t{}\n'.format(source, target))
        f.close()


def extract_graph_info(graph: nx.Graph, weights: np.ndarray = None) -> Tuple[np.ndarray, csr_matrix, Dict]:
    """
    Plot adjacency matrix of a graph as a pdf file
    Args:
        graph: the graph instance generated via networkx
        weights: the weights of edge

    Returns:
        probs: the distribution of nodes
        adj: adjacency matrix
        idx2node: a dictionary {key: idx, value: node name}
    """
    idx2node = {}
    for i in range(len(graph.nodes)):
        idx2node[i] = i

    probs = np.zeros((len(graph.nodes), 1))
    adj = lil_matrix((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        src = edge[0]
        dst = edge[1]
        if weights is None:
            adj[src, dst] += 1
            probs[src, 0] += 1
            probs[dst, 0] += 1
        else:
            adj[src, dst] += weights[src, dst]
            probs[src, 0] += weights[src, dst]
            probs[dst, 0] += weights[src, dst]

    return probs, csr_matrix(adj), idx2node


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    """
        Add noisy (random) nodes in a graph
        Args:
            graph: the graph instance generated via networkx
            noisy_level: the percentage of noisy nodes compared with original edges

        Returns:
            graph_noisy: the noisy graph
        """
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def add_noisy_edges(graph: nx.graph, noisy_level: float) -> nx.graph:
    """
    Add noisy (random) edges in a graph
    Args:
        graph: the graph instance generated via networkx
        noisy_level: the percentage of noisy edges compared with original edges

    Returns:
        graph_noisy: the noisy graph
    """
    nodes = list(graph.nodes)
    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges)
    graph_noisy = copy.deepcopy(graph)
    if num_noisy_edges > 0:
        i = 0
        while i < num_noisy_edges:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if (src, dst) not in graph_noisy.edges:
                graph_noisy.add_edge(src, dst)
                i += 1
    return graph_noisy
