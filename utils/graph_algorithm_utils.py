from matplotlib.pyplot import connect
from collections import Counter
from itertools import chain
import random
import re
import sys

sys.setrecursionlimit(1000000)


def find_preds_of_node(edge_list: list):
    nodes = set([node for tu in edge_list for node in tu])
    preds_of_node = {node: set() for node in nodes}
    for pred, succ in edge_list:
        preds_of_node[succ].add(pred)
    return preds_of_node


def find_succs_of_node(edge_list: list):
    nodes = set([node for tu in edge_list for node in tu])
    succs_of_node = {node: set() for node in nodes}
    for pred, succ in edge_list:
        succs_of_node[pred].add(succ)
    return succs_of_node


def find_adjs_of_node(edge_list: list):
    nodes = set([node for tu in edge_list for node in tu])
    adjs_of_node = {node: set() for node in nodes}
    for i, j in edge_list:
        adjs_of_node[i].add(j)
        adjs_of_node[j].add(i)
    return adjs_of_node


def find_topo_sort(edge_list: list):
    topo_sort = []
    nodes = set([node for tu in edge_list for node in tu])
    succs_of_node = find_succs_of_node(edge_list)
    visited = set()

    def _dfs(node):
        if node not in visited:
            visited.add(node)
            if len(succs_of_node[node]) > 0:
                for succ in succs_of_node[node]:
                    _dfs(succ)
            topo_sort.append(node)

    for node in nodes:
        _dfs(node)
    return topo_sort[::-1]


def cal_cum_lt(edge_list: list, node_lt: dict):
    roots = list(set([i for i, _ in edge_list]) - set([j for _, j in edge_list]))
    ts = find_topo_sort(edge_list)
    preds_of_node = find_preds_of_node(edge_list)
    preds_of_node.update({j: {'start'} for j in roots})

    cum_lt = {j: -float('inf') for j in ts}
    cum_lt['start'] = 0.0
    for node in ts:
        for pred in preds_of_node[node]:
            tmp = cum_lt[pred] + node_lt[node]
            if cum_lt[node] < tmp:
                cum_lt[node] = tmp
    cum_lt.pop('start')
    return cum_lt


def find_connected(adj_dict, j):
    visited = set()

    def _dfs(node):
        if node not in visited:
            visited.add(node)
            for adj in adj_dict[node]:
                _dfs(adj)

    _dfs(j)

    return visited


def find_weakly_connected_components(all_nodes, edge_list):
    nodes = set([i for i, _ in edge_list]) | set([j for _, j in edge_list])
    visited = set()
    components = []

    single_nodes = all_nodes - nodes
    if bool(single_nodes):
        for k in single_nodes:
            single_node = {k}
            components.append((single_node, []))

    adj_dict = find_adjs_of_node(edge_list)

    for j in nodes:
        if j not in visited:
            connected_nodes = find_connected(adj_dict, j)
            visited = visited | connected_nodes
            sub_edge_list = [(i, j) for i, j in edge_list if ((i in connected_nodes) and (j in connected_nodes))]
            components.append((connected_nodes, sub_edge_list))

    return components


def find_mst_prim(edge_list, edge_weight_dict):
    nodes = set([i for i, _ in edge_list]) | set([j for _, j in edge_list])
    adj_dict = find_adjs_of_node(edge_list)
    visited = set()
    tree = []
    init_node = nodes.pop()
    visited.add(init_node)

    while bool(nodes):
        valid_edge_dict = {}
        for j in visited:
            adj_nodes = adj_dict[j]
            valid_adj_nodes = adj_nodes & nodes
            for k in valid_adj_nodes:
                edge = (j, k) if int(re.findall(r'\d+', j)[0]) < int(re.findall(r'\d+', k)[0]) else (k, j)
                valid_edge_dict[edge] = edge_weight_dict[edge]
        min_edge = min(valid_edge_dict, key=lambda x: valid_edge_dict[x])
        tree.append(min_edge)
        visited.add(min_edge[0])
        visited.add(min_edge[1])
        nodes.discard(min_edge[0])
        nodes.discard(min_edge[1])

    return tree


def generate_random_tree(nodes_num):
    prufer_seq = [random.choice(range(nodes_num)) for i in range(nodes_num - 2)]
    degree_dict = Counter(chain(prufer_seq, range(nodes_num)))
    tree = []
    while bool(prufer_seq):
        j = next(v for v in range(nodes_num) if degree_dict[v] == 1)
        k = prufer_seq[0]
        del (prufer_seq[0])
        edge = (j, k) if j < k else (k, j)
        tree.append(edge)
        degree_dict[j] -= 1
        degree_dict[k] -= 1
    remain = set(v for v in range(nodes_num) if degree_dict[v] == 1)
    j, k = remain
    edge = (j, k) if j < k else (k, j)
    tree.append(edge)

    return tree


def generate_random_graph(nodes_num, edges_num):
    nodes = [i for i in range(nodes_num)]
    max_edges_num = nodes_num * (nodes_num - 1) / 2
    if edges_num > max_edges_num:
        raise ValueError
    edges_count = 0
    edges_list = []
    adj_dict = {node: set() for node in nodes}
    i = 0
    while edges_count < edges_num:
        j = random.choice(nodes)
        k = random.choice(nodes)
        edge = (j, k) if j < k else (k, j)
        validate_set = (adj_dict[k], j) if len(adj_dict[k]) < len(adj_dict[j]) else (adj_dict[j], k)
        has_edge_flag = True if validate_set[1] in validate_set[0] else False
        if j == k or has_edge_flag:
            continue
        else:
            edges_list.append(edge)
            edges_count += 1
            adj_dict[j].add(k)
            adj_dict[k].add(j)
    return edges_list
