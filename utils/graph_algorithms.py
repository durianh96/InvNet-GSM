import sys

sys.setrecursionlimit(1000000)


def find_preds_of_node(edges: list):
    nodes = set([node for tu in edges for node in tu])
    preds_of_node = {node: set() for node in nodes}
    for pred, succ in edges:
        preds_of_node[succ].add(pred)
    return preds_of_node


def find_succs_of_node(edges: list):
    nodes = set([node for tu in edges for node in tu])
    succs_of_node = {node: set() for node in nodes}
    for pred, succ in edges:
        succs_of_node[pred].add(succ)
    return succs_of_node


def find_adjs_of_node(edges: list):
    nodes = set([node for tu in edges for node in tu])
    adjs_of_node = {node: set() for node in nodes}
    for i, j in edges:
        adjs_of_node[i].add(j)
        adjs_of_node[j].add(i)
    return adjs_of_node


def find_descendants(succs_of_node, i: str):
    """
    It takes a node and returns the set of all nodes that are descendants of that node, and the set of
    all edges that connect those descendants
    
    :param succs_of_node: a dictionary of the form {node: [list of successors]}
    :param i: the node we're looking at
    :type i: str
    :return: The descendants of the node i and the edges between the descendants of the node i.
    """
    descendants = set()
    sub_des_edges = []

    def _dfs(node):
        if node not in descendants:
            descendants.add(node)
            if len(succs_of_node[node]) > 0:
                for succ in succs_of_node[node]:
                    sub_des_edges.append((node, succ))
                    _dfs(succ)

    _dfs(i)

    return descendants, sub_des_edges


def find_ancestors(preds_of_node, j: str):
    ancestors = set()
    sub_ans_edges = []

    def _dfs(node):
        if node not in ancestors:
            ancestors.add(node)
            if len(preds_of_node[node]) > 0:
                for pred in preds_of_node[node]:
                    sub_ans_edges.append((pred, node))
                    _dfs(pred)

    _dfs(j)

    return ancestors, sub_ans_edges


def find_topo_sort(edges: list):
    topo_sort = []
    nodes = set([node for tu in edges for node in tu])
    succs_of_node = find_succs_of_node(edges)
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


def cal_all_degree_of_node(edges: list):
    nodes = set([node for tu in edges for node in tu])
    in_degree_of_node = {node: 0 for node in nodes}
    out_degree_of_node = {node: 0 for node in nodes}
    degree_of_node = {node: 0 for node in nodes}
    for pred, succ in edges:
        in_degree_of_node[succ] += 1
        out_degree_of_node[pred] += 1
        degree_of_node[succ] += 1
        degree_of_node[pred] += 1
    return in_degree_of_node, out_degree_of_node, degree_of_node


# def cal_cum_lt_of_node(edges: list, node_lt: dict, edge_lt: dict):
#     roots = list(set([i for i, _ in edges]) - set([j for _, j in edges]))
#     ts = find_topo_sort(edges)
#     preds_of_node = find_preds_of_node(edges)
#     preds_of_node.update({j: {'start'} for j in roots})
#
#     edge_lt.update({('start', j): 0 for j in roots})
#     cum_lt_of_node = {j: -float('inf') for j in ts}
#     longest_pred = {j: set() for j in ts}
#     cum_lt_of_node['start'] = 0.0
#     for node in ts:
#         for pred in preds_of_node[node]:
#             tmp = cum_lt_of_node[pred] + node_lt[node] + edge_lt[pred, node]
#             if cum_lt_of_node[node] < tmp:
#                 longest_pred[node] = {pred}
#                 cum_lt_of_node[node] = tmp
#             elif cum_lt_of_node[node] == tmp:
#                 longest_pred[node].add(pred)
#     cum_lt_of_node.pop('start')
#     return cum_lt_of_node, longest_pred


# def cal_lt_of_node(edges: list, process_lt_of_node: dict, lt_of_edge: dict):
#     nodes = set([node for tu in edges for node in tu])
#     preds_of_node = find_preds_of_node(edges)
#     lt_of_node = {node: process_lt for node, process_lt in process_lt_of_node.items()}
#     for node in nodes:
#         if len(preds_of_node[node]) > 0:
#             lt_of_node[node] += max([lt_of_edge[(pred, node)] for pred in preds_of_node[node]])
#     return lt_of_node


def cal_cum_lt_of_node(edges: list, lt_of_node: dict):
    roots = list(set([i for i, _ in edges]) - set([j for _, j in edges]))
    ts = find_topo_sort(edges)
    preds_of_node = find_preds_of_node(edges)
    preds_of_node.update({j: {'start'} for j in roots})

    cum_lt_of_node = {j: -float('inf') for j in ts}
    cum_lt_of_node['start'] = 0.0
    longest_pred_of_node = {j: set() for j in ts}

    # Iterating through the tree structure and printing the text of each node.
    for node in ts:
        for pred in preds_of_node[node]:
            tmp = cum_lt_of_node[pred] + lt_of_node[node]
            if cum_lt_of_node[node] < tmp:
                longest_pred_of_node[node] = {pred}
                cum_lt_of_node[node] = tmp
            elif cum_lt_of_node[node] == tmp:
                longest_pred_of_node[node].add(pred)
    cum_lt_of_node.pop('start')
    return cum_lt_of_node, longest_pred_of_node


def cal_net_qty(edges, edge_qty):
    net_qty = {}
    preds_of_node = find_preds_of_node(edges)
    sinks = set([j for _, j in edges]) - set([i for i, _ in edges])
    for n_id in sinks:
        n_ancestors, sub_edges = find_ancestors(preds_of_node, n_id)
        sub_succs_of_node = find_succs_of_node(sub_edges)

        n_net_qty = {an: 0 for an in n_ancestors}
        n_net_qty.update({n_id: 1})

        sub_topo_sort = find_topo_sort(sub_edges)
        sub_topo_sort = sub_topo_sort[::-1]
        for i in sub_topo_sort[1:]:
            n_net_qty[i] = sum([edge_qty[i, succ] * n_net_qty[succ] for succ in sub_succs_of_node[i]])

        net_qty.update({(an, n_id): q for an, q in n_net_qty.items()})
    return net_qty


def find_connected(adjs_of_node: dict, j: str):
    visited = set()

    def _dfs(node):
        if node not in visited:
            visited.add(node)
            for adj in adjs_of_node[node]:
                _dfs(adj)

    _dfs(j)

    return visited


def find_all_paths(succ_of_node: dict, u: str, v: str, path=None):
    if path is None:
        path = []
    path = path + [u]
    if u == v:
        return [path]

    paths = []
    for node in succ_of_node[u]:
        if node not in path:
            newpaths = find_all_paths(succ_of_node, node, v, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def find_weakly_connected_components(all_nodes, edges):
    """
    It finds the weakly connected components of a graph by first finding the strongly connected
    components, then finding the connected components of the complement graph
    
    :param all_nodes: a set of all nodes in the graph
    :param edges: a list of tuples, each tuple is an edge, e.g. [(1, 2), (2, 3), (3, 1)]
    :return: A list of tuples. Each tuple contains a set of nodes and a list of edges.
    """
    nodes = set([i for i, _ in edges]) | set([j for _, j in edges])
    visited = set()
    components = []

    single_nodes = all_nodes - nodes
    if bool(single_nodes):
        for k in single_nodes:
            single_node = {k}
            components.append((single_node, []))

    adjs_of_node = find_adjs_of_node(edges)

    for j in nodes:
        if j not in visited:
            connected_nodes = find_connected(adjs_of_node, j)
            visited = visited | connected_nodes
            sub_edges = [(i, j) for i, j in edges if ((i in connected_nodes) and (j in connected_nodes))]
            components.append((connected_nodes, sub_edges))

    return components


def is_fully_connected(nodes, edges):
    components = find_weakly_connected_components(nodes, edges)
    return len(components)


def is_tree(nodes, edges):
    return is_fully_connected(nodes, edges) and (len(edges) == len(nodes) - 1)


def two_path_corr(m_path, n_path):
    """
    It finds the longest common subsequence of two lists
    
    :param m_path: the path of the first image
    :param n_path: the path of the node you want to find the correlation of
    :return: the path that is common to both the paths.
    """
    start_i = 0
    end_i = 0
    for start_i, mi in enumerate(m_path):
        for start_j, nj in enumerate(n_path):
            if mi == nj:
                break
        else:
            continue
        break

    for end_i, mi in enumerate(m_path[::-1]):
        for end_j, nj in enumerate(n_path[::-1]):
            if mi == nj:
                break
        else:
            continue
        break
    corr_path = m_path[start_i: -end_i]

    return corr_path
