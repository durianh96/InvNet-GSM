import random
from collections import Counter
from itertools import chain
from typing import Optional


def edges_generating_by_type(nodes_num: int,
                             graph_type='GENERAL',
                             edges_num: Optional[int] = None):
    node_list = ['N' + str(i).zfill(6) for i in range(nodes_num)]
    if graph_type == 'SERIAL':
        edge_list = [(node_list[i], node_list[i + 1]) for i in range(nodes_num - 1)]
    elif graph_type == 'ASSEMBLY':
        unassigned_nodes = set(node_list)
        edge_list = []
        for _ in range(nodes_num - 1):
            u = unassigned_nodes.pop()
            v = random.choice(list(unassigned_nodes))
            edge_list.append((u, v))
    elif graph_type == 'DISTRIBUTION':
        unassigned_nodes = set(node_list)
        edge_list = []
        for _ in range(nodes_num - 1):
            v = unassigned_nodes.pop()
            u = random.choice(list(unassigned_nodes))
            edge_list.append((u, v))
    elif graph_type == 'GENERAL':
        random_tree = tree_generating(nodes_num)
        edge_list = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in random_tree]
        if nodes_num < 10000:
            edge_choices = [(node_list[i], v) for i in range(nodes_num - 1) for v in node_list[i + 1:]]
        else:
            random_G = general_edges_generating(nodes_num, edges_num * np.power(nodes_num, 1 / 4))
            edge_choices = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in random_G]
        if edges_num > len(edge_choices):
            raise ValueError
        select_edge_choices = list(set(edge_choices) - set(edge_list))
        select_edge_list = random.sample(select_edge_choices, k=edges_num - nodes_num + 1)
        edge_list = edge_list + select_edge_list
    else:
        raise AttributeError('wrong graph type')
    return edge_list


def tree_generating(nodes_num):
    prufer_seq = [random.choice(range(nodes_num)) for i in range(nodes_num - 2)]
    degree_dict = Counter(chain(prufer_seq, range(nodes_num)))
    edge_list = []
    while bool(prufer_seq):
        j = next(v for v in range(nodes_num) if degree_dict[v] == 1)
        k = prufer_seq[0]
        del (prufer_seq[0])
        edge = (j, k) if j < k else (k, j)
        edge_list.append(edge)
        degree_dict[j] -= 1
        degree_dict[k] -= 1
    remain = set(v for v in range(nodes_num) if degree_dict[v] == 1)
    j, k = remain
    edge = (j, k) if j < k else (k, j)
    edge_list.append(edge)

    return edge_list


def general_edges_generating(nodes_num, edges_num):
    nodes = [i for i in range(nodes_num)]
    max_edges_num = nodes_num * (nodes_num - 1) / 2
    if edges_num > max_edges_num:
        raise ValueError
    edges_count = 0
    edge_list = []
    adj_dict = {node: set() for node in nodes}
    while edges_count < edges_num:
        j = random.choice(nodes)
        k = random.choice(nodes)
        edge = (j, k) if j < k else (k, j)
        validate_set = (adj_dict[k], j) if len(adj_dict[k]) < len(adj_dict[j]) else (adj_dict[j], k)
        has_edge_flag = True if validate_set[1] in validate_set[0] else False
        if j == k or has_edge_flag:
            continue
        else:
            edge_list.append(edge)
            edges_count += 1
            adj_dict[j].add(k)
            adj_dict[k].add(j)
    return edge_list


