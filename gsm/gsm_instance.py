from typing import Optional
from utils.graph_algorithms import find_preds_of_node, find_succs_of_node, find_weakly_connected_components, \
    is_tree


class GSMInstance:
    def __init__(self, instance_id: str,
                 nodes: set,
                 edges: list,
                 lt_of_node: dict,
                 edge_qty: dict,
                 hc_of_node: dict,
                 sla_of_node: dict,
                 cum_lt_of_node: dict,
                 mean_of_node: Optional[dict] = None,
                 std_of_node: Optional[dict] = None,
                 demand_bound_pool: Optional[dict] = None):
        self._instance_id = instance_id
        self._nodes = nodes
        self._edges = edges
        if is_tree(nodes, edges):
            self._graph_type = 'TREE'
        else:
            self._graph_type = 'GENERAL'

        self._preds_of_node = find_preds_of_node(self._edges)
        self._succs_of_node = find_succs_of_node(self._edges)
        self._roots = list(set([i for i, _ in edges]) - set([j for _, j in edges]))
        self._sinks = list(set([j for _, j in edges]) - set([i for i, _ in edges]))

        self._lt_of_node = lt_of_node
        self._edge_qty = edge_qty
        self._hc_of_node = hc_of_node
        self._sla_of_node = sla_of_node
        self._cum_lt_of_node = cum_lt_of_node
        self._mean_of_node = mean_of_node
        self._std_of_node = std_of_node
        self._demand_bound_pool = demand_bound_pool

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def graph_type(self):
        return self._graph_type

    @property
    def preds_of_node(self):
        return self._preds_of_node

    @property
    def succs_of_node(self):
        return self._succs_of_node

    @property
    def roots(self):
        return self._roots

    @property
    def sinks(self):
        return self._sinks

    @property
    def lt_of_node(self):
        return self._lt_of_node

    @property
    def edge_qty(self):
        return self._edge_qty

    @property
    def hc_of_node(self):
        return self._hc_of_node

    @property
    def sla_of_node(self):
        return self._sla_of_node

    @property
    def cum_lt_of_node(self):
        return self._cum_lt_of_node

    @property
    def mean_of_node(self):
        return self._mean_of_node

    @property
    def std_of_node(self):
        return self._std_of_node

    @property
    def demand_bound_pool(self):
        return self._demand_bound_pool

    def update_mean_of_node(self, new_mean_of_node: dict):
        self._mean_of_node = new_mean_of_node

    def update_std_of_node(self, new_std_of_node: dict):
        self._std_of_node = new_std_of_node

    def update_demand_bound_pool(self, new_demand_bound_pool: dict):
        self._demand_bound_pool = new_demand_bound_pool

    def get_sub_instances(self, local_sol_info):
        """
        It decomposes the instance into sub-instances.
        
        :param local_sol_info: a dictionary containing the following keys:
        :return: A list of sub-instances.
        """
        to_remove_edges = []
        s_ub_dict = {}
        si_lb_dict = {}
        for node in local_sol_info['completely_fix_nodes']:
            if len(self.succs_of_node[node]) > 0:
                for v in self.succs_of_node[node]:
                    to_remove_edges.append((node, v))
                    if v not in si_lb_dict.keys():
                        si_lb_dict[v] = local_sol_info['completely_fix_S'][node]
                    else:
                        si_lb_dict[v] = max(si_lb_dict[v], local_sol_info['completely_fix_S'][node])
            if len(self.preds_of_node[node]) > 0:
                for u in self.preds_of_node[node]:
                    to_remove_edges.append((u, node))
                    if u not in s_ub_dict.keys():
                        s_ub_dict[u] = local_sol_info['completely_fix_SI'][node]
                    else:
                        s_ub_dict[u] = min(s_ub_dict[u], local_sol_info['completely_fix_SI'][node])

        sub_instance_list = self.decompose_instance(to_remove_edges)
        return sub_instance_list, s_ub_dict, si_lb_dict

    def decompose_instance(self, to_remove_edges=None):
        """
        It decomposes the instance into sub-instances.
        
        Args:
          to_remove_edges: a list of edges to be removed from the graph.
        
        Returns:
          A list of sub-instances.
        """
        if to_remove_edges is None:
            to_remove_edges = []

        new_edges = [e for e in self._edges if e not in to_remove_edges]

        components = find_weakly_connected_components(self._nodes, new_edges)

        sub_instance_list = []
        if len(components) == 1:
            print('This graph can not be decomposed')
        else:
            print('This graph can be decomposed')
            for component in components:
                sub_nodes = component[0]
                sub_edges = component[1]
                sub_instance = self.get_sub_instance(sub_nodes, sub_edges)
                sub_instance_list.append(sub_instance)
        return sub_instance_list

    def get_sub_instance(self, sub_nodes, sub_edges):
        sub_lt_of_node = {n_id: self._lt_of_node[n_id] for n_id in sub_nodes}
        sub_edge_qty = {e_id: self._edge_qty[e_id] for e_id in sub_edges}
        sub_hc_of_node = {n_id: self._hc_of_node[n_id] for n_id in sub_nodes}
        sub_sla_of_node = {n_id: self._sla_of_node[n_id] for n_id in sub_nodes if n_id in self._sla_of_node.keys()}
        sub_cum_lt_of_node = {n_id: self._cum_lt_of_node[n_id] for n_id in sub_nodes}

        if self._mean_of_node is not None:
            sub_mean_of_node = {n_id: self._mean_of_node[n_id] for n_id in sub_nodes}
        else:
            sub_mean_of_node = None

        if self._std_of_node is not None:
            sub_std_of_node = {n_id: self._std_of_node[n_id] for n_id in sub_nodes}
        else:
            sub_std_of_node = None

        if self._demand_bound_pool is not None:
            sub_demand_bound_pool = {n_id: self._demand_bound_pool[n_id] for n_id in sub_nodes}
        else:
            sub_demand_bound_pool = None

        sub_gsm_instance = GSMInstance(
            instance_id='sub_' + self._instance_id,
            nodes=sub_nodes,
            edges=sub_edges,
            lt_of_node=sub_lt_of_node,
            edge_qty=sub_edge_qty,
            hc_of_node=sub_hc_of_node,
            sla_of_node=sub_sla_of_node,
            cum_lt_of_node=sub_cum_lt_of_node,
            mean_of_node=sub_mean_of_node,
            std_of_node=sub_std_of_node,
            demand_bound_pool=sub_demand_bound_pool
        )
        return sub_gsm_instance
