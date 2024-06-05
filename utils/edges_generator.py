import random
import numpy as np
from collections import Counter
from itertools import chain
from typing import Optional


def edges_generating_by_shape(nodes_num: int, max_depth: int, roots_num: int, sinks_num: int,
                              edges_num: Optional[int] = None, skip_edges_num: int = 0):

    if max_depth < 2:
        raise ValueError('Max depth of graph must be at least 2')
    if nodes_num < roots_num + sinks_num + max_depth - 2:
        raise ValueError('Num of nodes is too small')

    # sampling nodes
    num_layer_nodes = [roots_num]
    for i in range(max_depth - 1):
        if len(num_layer_nodes) + 1 == max_depth:
            num_layer_nodes.append(sinks_num)
        elif len(num_layer_nodes) + 2 == max_depth:
            num_layer_nodes.append(nodes_num - (sum(num_layer_nodes) + sinks_num))
        else:
            # every layer at least have one node
            lb = 1
            ub = nodes_num - (sum(num_layer_nodes) + sinks_num) - (max_depth - len(num_layer_nodes) - 2)
            num_layer_nodes.append(random.randint(lb, ub))

    nodes = [i for i in range(nodes_num)]
    layer_nodes = [nodes[sum(num_layer_nodes[: i]): sum(num_layer_nodes[: i + 1])] for i in range(max_depth)]

    # sampling edges
    num_edge_lb = [max(num_layer_nodes[i], num_layer_nodes[i + 1]) for i in range(max_depth - 1)]
    num_edge_ub = [num_layer_nodes[i] * num_layer_nodes[i + 1] for i in range(max_depth - 1)]
    num_layer_edges = [lb for lb in num_edge_lb]

    if edges_num is None:
        # edges_num = random.randint(sum(num_edge_lb) + skip_edges_num, sum(num_edge_ub) + skip_edges_num)
        edges_num = skip_edges_num + sum(num_layer_edges)
    else:
        if edges_num < sum(num_edge_lb) + skip_edges_num:
            raise ValueError('Num of edges is to small')

        if edges_num > sum(num_edge_ub) + skip_edges_num:
            raise ValueError('Num of edges is to large')

    # find a feasible solution
    # num_layer_edges = [lb for lb in num_edge_lb]
    num_to_assign = edges_num - skip_edges_num - sum(num_layer_edges)
    layer_choices = [i for i in range(max_depth - 1)]
    while num_to_assign > 0:
        i = random.choice(layer_choices)
        if num_layer_edges[i] < num_edge_ub[i]:
            num_layer_edges[i] += 1
            num_to_assign -= 1
        else:
            layer_choices.remove(i)

    edges = set()
    for i in range(max_depth - 1):
        pred_nodes = layer_nodes[i]
        succ_nodes = layer_nodes[i + 1]
        if len(pred_nodes) >= len(succ_nodes):
            first = random.sample(pred_nodes, len(succ_nodes))
            layer_edges = set(list(zip(first, succ_nodes)))
            second = set(pred_nodes) - set(first)
            layer_edges = layer_edges | set(list(zip(second, random.choices(succ_nodes, k=len(second)))))

        else:
            first = random.sample(succ_nodes, len(pred_nodes))
            layer_edges = set(list(zip(pred_nodes, first)))
            second = set(succ_nodes) - set(first)
            layer_edges = layer_edges | set(list(zip(random.choices(pred_nodes, k=len(second)), second)))

        if len(layer_edges) < num_layer_edges[i]:
            layer_choices = [(pred, succ) for pred in pred_nodes for succ in succ_nodes
                             if (pred, succ) not in layer_edges]
            layer_edges = layer_edges | set(random.sample(layer_choices, num_layer_edges[i] - len(layer_edges)))
        edges = edges | layer_edges

    if skip_edges_num > 0:
        if max_depth < 3:
            raise ValueError('Max depth of graph must be at least 3 to generate jumping edges')
        jump_choices = []
        for i in range(max_depth - 2):
            jump_choices.extend([(i, j) for j in range(i + 2, max_depth)])

        skip_edges = set()
        for index in range(skip_edges_num):
            while True:
                je = random.choice(jump_choices)
                pred = random.choice(layer_nodes[je[0]])
                succ = random.choice(layer_nodes[je[1]])
                if index == 0 or (pred, succ) not in skip_edges:
                    break
            skip_edges.add((pred, succ))
            edges.add((pred, succ))
    edge_list = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in edges]
    return edge_list


def edges_generating_by_type(nodes_num: int,
                             graph_type='MIXED',
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
    # add mixed tree
    elif graph_type == 'MIXED':
        tree_list = tree_generating(nodes_num)
        edge_list = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in tree_list]
    elif graph_type == 'GENERAL':
        mixed_tree = tree_generating(nodes_num)
        edge_list = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in mixed_tree]
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