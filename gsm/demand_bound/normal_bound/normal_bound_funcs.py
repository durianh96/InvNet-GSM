import numpy as np
from scipy.stats import norm
from autograd import grad
import autograd.numpy as gnp
from utils.graph_algorithms import find_preds_of_node, find_topo_sort


def mean_func(mean):
    return lambda x: mean * x


def cascading_normal_db_func(cascading_mean, volatility_constant):
    return lambda x: volatility_constant * gnp.sqrt(x) + cascading_mean * x


def cascading_normal_vb_func(volatility_constant):
    return lambda x: volatility_constant * gnp.sqrt(x)


def cascading_normal_vb_gradient_func(volatility_constant):
    return grad(cascading_normal_vb_func(volatility_constant))


def net_normal_db_func(net_mean, net_std, tau):
    return lambda x: norm.ppf(tau) * net_std * gnp.sqrt(x) + net_mean * x


def net_normal_vb_func(net_std, tau):
    return lambda x: norm.ppf(tau) * net_std * gnp.sqrt(x)


def net_normal_vb_gradient_func(net_std, tau):
    return grad(net_normal_vb_func(net_std, tau))


def get_cascading_normal_bound_para(edges: list, edge_qty: dict, ext_mean_of_node: dict, ext_std_of_node: dict,
                                    tau: float, pooling_factor: int):
    nodes = set([node for tu in edges for node in tu])
    preds_of_node = find_preds_of_node(edges)

    kstd_dict = {node: norm.ppf(tau) * ext_std_of_node.get(node, 0.) for node in nodes}
    cascading_mean_of_node = {node: ext_mean_of_node.get(node, 0.) for node in nodes}

    reverse_edges = [(j, i) for i, j in edges]
    reverse_topo_sort = find_topo_sort(reverse_edges)

    constant_dict = {node: v ** pooling_factor for node, v in kstd_dict.items()}
    for node in reverse_topo_sort:
        if len(preds_of_node[node]) > 0:
            for pred in preds_of_node[node]:
                constant_dict[pred] += (edge_qty[pred, node] ** pooling_factor) * constant_dict[node]
                cascading_mean_of_node[pred] += edge_qty[pred, node] * cascading_mean_of_node[node]

    volatility_constant_of_node = {node: np.power(v, 1 / pooling_factor) for node, v in constant_dict.items()}
    return cascading_mean_of_node, volatility_constant_of_node
