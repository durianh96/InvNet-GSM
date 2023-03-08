import numpy as np
from utils.graph_algorithms import find_preds_of_node, find_topo_sort


def get_cascading_normal_stats(edges: list, edge_qty: dict, ext_mean_of_node: dict, ext_std_of_node: dict):
    """
    For each node, we find the mean and variance of the node's value, given the values of its
    predecessors
    
    :param edges: list of tuples, each tuple is an edge
    :type edges: list
    :param edge_qty: a dictionary of the form {(i, j): qty} where i and j are nodes and qty is the
    quantity of flow from i to j
    :type edge_qty: dict
    :param ext_mean_of_node: the mean of the external node
    :type ext_mean_of_node: dict
    :param ext_std_of_node: the standard deviation of the external nodes
    :type ext_std_of_node: dict
    :return: The cascading mean and standard deviation of each node.
    """
    nodes = set([node for tu in edges for node in tu])
    preds_of_node = find_preds_of_node(edges)

    cascading_var_of_node = {node: ext_std_of_node.get(node, 0.) ** 2 for node in nodes}
    cascading_mean_of_node = {node: ext_mean_of_node.get(node, 0.) for node in nodes}

    reverse_edges = [(j, i) for i, j in edges]
    reverse_topo_sort = find_topo_sort(reverse_edges)

    for node in reverse_topo_sort:
        if len(preds_of_node[node]) > 0:
            for pred in preds_of_node[node]:
                cascading_var_of_node[pred] += (edge_qty[pred, node] ** 2) * cascading_var_of_node[node]
                cascading_mean_of_node[pred] += edge_qty[pred, node] * cascading_mean_of_node[node]

    cascading_std_of_node = {node: np.power(v, 1 / 2) for node, v in cascading_var_of_node.items()}
    return cascading_mean_of_node, cascading_std_of_node
