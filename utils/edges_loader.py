import pandas as pd

from utils.graph_algorithms import find_topo_sort


def edges_loading_from_willems(data_id):
    data_dir = 'data/willems_2008/' + str(data_id).zfill(2) + '.csv'
    chain_df = pd.read_csv(data_dir, header=1)

    edge_columns = ['/arcs/arc/@from', '/arcs/arc/@to']
    origin_edge_list = chain_df[edge_columns].dropna().values.tolist()
    topo_sort = find_topo_sort(origin_edge_list)
    node_list = ['N' + str(index).zfill(3) for index in range(len(topo_sort))]
    map_dict = dict(zip(topo_sort, node_list))
    edges = [(map_dict[u], map_dict[v]) for [u, v] in origin_edge_list]
    return edges


def edges_loading_from_huang(avg_deg):
    nodes_num = 4000
    edges_num = round(nodes_num * avg_deg / 2)
    data_dir = 'data/huang_2022/edges_' + str(edges_num) + '.csv'
    edge_df = pd.read_csv(data_dir)
    edges = [(u, v) for u, v in edge_df.values]
    return edges
