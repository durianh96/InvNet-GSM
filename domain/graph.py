from typing import Optional
from utils.graph_algorithm_utils import *


class DiGraph:
    def __init__(self, all_nodes: set, edge_list: list, graph_type: Optional[str] = None):
        self.all_nodes = all_nodes
        self.edge_list = edge_list
        if graph_type is None:
            if self.is_tree():
                self.graph_type = 'tree'
            else:
                self.graph_type = 'general'
        else:
            self.graph_type = graph_type
        self.pred_dict = find_preds_of_node(edge_list)
        self.succ_dict = find_succs_of_node(edge_list)
        self.supply_nodes = list(set([i for i, _ in edge_list]) - set([j for _, j in edge_list]))
        self.demand_nodes = list(set([j for _, j in edge_list]) - set([i for i, _ in edge_list]))

    def is_fully_connected(self):
        components = find_weakly_connected_components(self.all_nodes, self.edge_list)
        return len(components) == 1

    def is_tree(self):
        return self.is_fully_connected() and (len(self.edge_list) == len(self.all_nodes) - 1)

    def to_undirected(self):
        un_di_graph = UnDiGraph(self.all_nodes, self.edge_list)
        return un_di_graph

    def decompose_graph(self, to_remove_edges=None):
        if to_remove_edges is None:
            to_remove_edges = []

        new_edge_list = [e for e in self.edge_list if e not in to_remove_edges]
        components = find_weakly_connected_components(self.all_nodes, new_edge_list)

        if len(components) == 1:
            print('This graph can not be decomposed')
            return [self]
        else:
            print('This graph can be decomposed')
            sub_graph_list = []
            for component in components:
                connected_nodes = component[0]
                sub_edge_list = component[1]
                sub_graph = DiGraph(connected_nodes, sub_edge_list)
                sub_graph_list.append(sub_graph)
            return sub_graph_list


class UnDiGraph:
    def __init__(self, all_nodes: set, edge_list: list):
        self.all_nodes = all_nodes
        self.edge_list = edge_list
        if len(self.edge_list):
            self.adj_dict = find_adjs_of_node(edge_list)
        else:
            self.adj_dict = {node: {} for node in self.all_nodes}
        
        self.degree_dict = {j: len(self.adj_dict[j]) for j in all_nodes}

    def is_fully_connected(self):
        visited = set()

        def _dfs(node):
            if node not in visited:
                visited.add(node)
                for adj in self.adj_dict[node]:
                    _dfs(adj)

        _dfs(list(self.all_nodes)[0])

        return len(visited) == len(self.all_nodes)

    def remove_nodes(self, rm_nodes: list):
        new_all_nodes = self.all_nodes - set(rm_nodes)
        new_edge_list = [(i, j) for i, j in self.edge_list if ((i in new_all_nodes) and (j in new_all_nodes))]
        new_un_di_graph = UnDiGraph(new_all_nodes, new_edge_list)
        return new_un_di_graph
