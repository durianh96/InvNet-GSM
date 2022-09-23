from typing import Optional
from domain.graph import DiGraph
import numpy as np
from autograd import grad
import autograd.numpy as gnp
from scipy.stats import norm
from utils.graph_algorithm_utils import *


class GSMInstance:
    def __init__(self, instance_id: str,
                 all_nodes: set,
                 edge_list: list,
                 lt_dict: dict,
                 qty_dict: dict,
                 hc_dict: dict,
                 sla_dict: dict,
                 mu_dict: dict,
                 sigma_dict: dict,
                 tau=0.95,
                 pooling_factor=2,
                 cum_lt_dict: Optional[dict] = None,
                 network_mu_dict: Optional[dict] = None,
                 volatility_constant_dict: Optional[dict] = None,
                 db_func: Optional[dict] = None,
                 vb_func: Optional[dict] = None,
                 grad_vb_func: Optional[dict] = None
                 ):
        self.graph = DiGraph(all_nodes, edge_list)
        self.instance_id = instance_id
        self.all_nodes = all_nodes
        self.edge_list = edge_list
        self.lt_dict = lt_dict
        self.qty_dict = qty_dict
        self.hc_dict = hc_dict
        self.sla_dict = sla_dict
        self.mu_dict = mu_dict
        self.sigma_dict = sigma_dict

        self.tau = tau
        self.pooling_factor = pooling_factor

        if cum_lt_dict is None:
            self.update_info()
        else:
            self.cum_lt_dict = cum_lt_dict
            self.network_mu_dict = network_mu_dict
            self.volatility_constant_dict = volatility_constant_dict
            self.db_func = db_func
            self.vb_func = vb_func
            self.grad_vb_func = grad_vb_func

    def update_info(self):
        self.cum_lt_dict = cal_cum_lt(self.edge_list, self.lt_dict)
        self.set_n_bound_func()

    def set_n_bound_func(self):
        network_mu_dict, volatility_constant_dict = self.cal_n_bound_para()
        self.network_mu_dict = network_mu_dict
        self.volatility_constant_dict = volatility_constant_dict
        self.db_func = {}
        self.vb_func = {}
        self.grad_vb_func = {}

        def _vb_f(vc):
            return lambda x: vc * gnp.sqrt(x)

        def _db_f(vc, net_mu):
            return lambda x: vc * gnp.sqrt(x) + net_mu * x

        for node in self.all_nodes:
            self.vb_func[node] = _vb_f(volatility_constant_dict[node])
            self.db_func[node] = _db_f(volatility_constant_dict[node], network_mu_dict[node])
            self.grad_vb_func[node] = grad(_vb_f(volatility_constant_dict[node]))

    def cal_n_bound_para(self):
        ksigma_dict = {node: 0. for node in self.all_nodes}
        ksigma_dict.update({node: norm.ppf(self.tau) * self.sigma_dict.get(node, 0.) for node in self.all_nodes})

        network_mu_dict = {node: 0. for node in self.all_nodes}
        network_mu_dict.update({node: self.mu_dict.get(node, 0.) for node in self.all_nodes})

        reverse_edges = [(j, i) for i, j in self.graph.edge_list]
        reverse_topo_sort = find_topo_sort(reverse_edges)

        constant_dict = {node: v ** self.pooling_factor for node, v in ksigma_dict.items()}
        for node in reverse_topo_sort:
            if len(self.graph.pred_dict[node]) > 0:
                for pred in self.graph.pred_dict[node]:
                    constant_dict[pred] += (self.qty_dict[pred, node] ** self.pooling_factor) * constant_dict[node]
                    network_mu_dict[pred] += self.qty_dict[pred, node] * network_mu_dict[node]

        volatility_constant_dict = {node: np.power(v, 1 / self.pooling_factor) for node, v in constant_dict.items()}
        return network_mu_dict, volatility_constant_dict

    def deal_completely_fix(self, nodes_info):
        to_remove_edges = []
        s_ub_dict = {}
        si_lb_dict = {}
        for node in nodes_info['completely_fix_nodes']:
            if len(self.graph.succ_dict[node]) > 0:
                for v in self.graph.succ_dict[node]:
                    to_remove_edges.append((node, v))
                    if v not in si_lb_dict.keys():
                        si_lb_dict[v] = nodes_info['completely_fix_S'][node]
                    else:
                        si_lb_dict[v] = max(si_lb_dict[v], nodes_info['completely_fix_S'][node])
            if len(self.graph.pred_dict[node]) > 0:
                for u in self.graph.pred_dict[node]:
                    to_remove_edges.append((u, node))
                    if u not in s_ub_dict.keys():
                        s_ub_dict[u] = nodes_info['completely_fix_SI'][node]
                    else:
                        s_ub_dict[u] = min(s_ub_dict[u], nodes_info['completely_fix_SI'][node])

        sub_graph_list = self.graph.decompose_graph(to_remove_edges)
        sub_gsm_instance_list = []
        if len(sub_graph_list) > 1:
            for sub_graph in sub_graph_list:
                sub_lt_dict = {node: self.lt_dict[node] for node in sub_graph.all_nodes}
                sub_qty_dict = {(u, v): self.qty_dict[u, v] for u, v in sub_graph.edge_list}
                sub_hc_dict = {node: self.hc_dict[node] for node in sub_graph.all_nodes}
                sub_sla_dict = {node: self.sla_dict[node] for node in sub_graph.all_nodes if
                                node in self.sla_dict.keys()}
                sub_mu_dict = {node: self.mu_dict[node] for node in sub_graph.all_nodes if node in self.mu_dict.keys()}
                sub_sigma_dict = {node: self.sigma_dict[node] for node in sub_graph.all_nodes
                                  if node in self.sigma_dict.keys()}
                sub_cum_lt_dict = {node: self.cum_lt_dict[node] for node in sub_graph.all_nodes}
                sub_db_func = {node: self.db_func[node] for node in sub_graph.all_nodes}
                sub_vb_func = {node: self.vb_func[node] for node in sub_graph.all_nodes}
                sub_grad_vb_func = {node: self.grad_vb_func[node] for node in sub_graph.all_nodes}
                sub_gsm_instance = GSMInstance(
                    instance_id='sub_' + self.instance_id,
                    all_nodes=sub_graph.all_nodes,
                    edge_list=sub_graph.edge_list,
                    lt_dict=sub_lt_dict,
                    qty_dict=sub_qty_dict,
                    hc_dict=sub_hc_dict,
                    sla_dict=sub_sla_dict,
                    mu_dict=sub_mu_dict,
                    sigma_dict=sub_sigma_dict,
                    tau=self.tau,
                    pooling_factor=self.pooling_factor,
                    cum_lt_dict=sub_cum_lt_dict,
                    db_func=sub_db_func,
                    vb_func=sub_vb_func,
                    grad_vb_func=sub_grad_vb_func
                )
                sub_gsm_instance_list.append(sub_gsm_instance)
        return sub_gsm_instance_list, s_ub_dict, si_lb_dict
