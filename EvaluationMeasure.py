"""
This file contains evaluation and visualization functions for graph analysis method
"""
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List


def tab2pairs(filename, flag='\t'):
    pairs = []

    with open(filename) as f:
        contents = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    contents = [x.strip() for x in contents]
    for content in contents:
        idx = content.find(flag)
        if idx > -1:
            src = content[:idx]
            dst = content[(idx+len(flag)):]
            pairs.append([src, dst])
    return pairs


def calculate_edge_correctness(pairs: List, cost_s, cost_t) -> float:
    edge_correctness = 0
    num_edges = 0
    for r in range(cost_s.shape[0]):
        for c in range(cost_s.shape[0]):
            if cost_s[r, c] > 0:
                num_edges += 1
                r1 = -1
                c1 = -1
                for pair in pairs:
                    if pair[0] == r:
                        r1 = pair[1]
                        break

                for pair in pairs:
                    if pair[0] == c:
                        c1 = pair[1]
                        break

                if r1 > -1 and c1 > -1:
                    if cost_t[r1, c1] > 0 or cost_t[c1, r1] > 0:
                        edge_correctness += 1

    edge_correctness /= num_edges
    return edge_correctness


def calculate_node_correctness(pairs: List, num_correspondence: int) -> float:
    """
    Calculate node correctness given estimated correspondences
    Args:
        pairs: a list of pairs of nodes
        num_correspondence: the real number of correspondences

    Returns:
        node_correctness: the percentage of correctly-matched nodes
    """
    node_correctness = 0
    for pair in pairs:
        if pair[0] == pair[1]:
            node_correctness += 1
    node_correctness /= num_correspondence
    return node_correctness


def heatmap(data, row_labels, col_labels, ax=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Args:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize='xx-small')
    ax.set_yticklabels(row_labels, fontsize='xx-small')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-75, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Args:
        im         : The AxesImage to be labeled.
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=9)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def plot_adjacency_matrix(graph: nx.Graph, graph_name: str):
    """
    Plot adjacency matrix of a graph as a pdf file
    Args:
        graph: the graph instance generated via networkx
        graph_name: the name of the graph
    """
    adj = np.zeros((len(graph.nodes), len(graph.nodes)))
    for edge in graph.edges:
        src = edge[0]
        dst = edge[1]
        adj[src, dst] += 1
    plt.imshow(adj)
    plt.savefig('{}.pdf'.format(graph_name))
    plt.close('all')
