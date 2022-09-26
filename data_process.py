from typing import Optional
import numpy as np
import pandas as pd
import pickle
import os
from utils.graph_algorithm_utils import *
from domain.graph import DiGraph
from domain.gsm import GSMInstance
from domain.policy import Policy


def generate_graph(nodes_num: int,
                   edges_num: Optional[int] = None,
                   graph_type='general'):
    node_list = ['N' + str(i).zfill(6) for i in range(nodes_num)]
    if graph_type == 'serial':
        edge_list = [(node_list[i], node_list[i + 1]) for i in range(nodes_num - 1)]
    elif graph_type == 'assembly':
        unassigned_nodes = set(node_list)
        edge_list = []
        for _ in range(nodes_num - 1):
            u = unassigned_nodes.pop()
            v = random.choice(list(unassigned_nodes))
            edge_list.append((u, v))
    elif graph_type == 'distribution':
        unassigned_nodes = set(node_list)
        edge_list = []
        for _ in range(nodes_num - 1):
            v = unassigned_nodes.pop()
            u = random.choice(list(unassigned_nodes))
            edge_list.append((u, v))
    elif graph_type == 'general':
        random_tree = generate_random_tree(nodes_num)
        edge_list = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in random_tree]
        if nodes_num < 10000:
            edge_choices = [(node_list[i], v) for i in range(nodes_num - 1) for v in node_list[i + 1:]]
        else:
            random_G = generate_random_graph(nodes_num, edges_num * np.power(nodes_num, 1 / 4))
            edge_choices = [('N' + str(u).zfill(6), 'N' + str(v).zfill(6)) for (u, v) in random_G]
        if edges_num > len(edge_choices):
            raise ValueError
        select_edge_choices = list(set(edge_choices) - set(edge_list))
        select_edge_list = random.sample(select_edge_choices, k=edges_num - nodes_num + 1)
        edge_list = edge_list + select_edge_list
    else:
        raise AttributeError('wrong graph type')
    graph = DiGraph(all_nodes=set(node_list), edge_list=edge_list, graph_type=graph_type)
    return graph


def generate_gsm_instance(graph,
                          instance_id,
                          qty_lb=1,
                          qty_ub=3,
                          lt_lb=1,
                          lt_ub=10,
                          hc_lb=0,
                          hc_ub=1,
                          sla_lt_lb=0,
                          sla_lt_ub=10,
                          mu_lb=0,
                          mu_ub=100,
                          sigma_lb=0,
                          sigma_ub=10):
    # random quantity
    qty_dict = {}
    for u, v in graph.edge_list:
        qty_dict[(u, v)] = np.random.uniform(qty_lb, qty_ub)

    # random lead time
    lt_dict = {node: np.random.randint(lt_lb, lt_ub) for node in graph.all_nodes}

    # random holding cost (must larger than assemblies)
    topo_sort = cal_cum_lt(graph.edge_list, lt_dict)
    preds_of_node = find_preds_of_node(graph.edge_list)
    hc_dict = {}
    for node in topo_sort:
        if len(preds_of_node[node]) > 0:
            hc_dict[node] = sum([hc_dict[pred] * qty_dict[pred, node] for pred in preds_of_node[node]]) \
                            + np.random.uniform(hc_lb, hc_ub)
        else:
            hc_dict[node] = np.random.uniform(hc_lb, hc_ub)

    # random sla
    sla_dict = {node: np.random.randint(sla_lt_lb, sla_lt_ub) + lt_dict[node] for node in graph.demand_nodes}

    # random demand parameters
    mu_dict = {node: np.random.uniform(mu_lb, mu_ub) for node in graph.demand_nodes}

    sigma_dict = {node: np.random.uniform(sigma_lb, sigma_ub) for node in graph.demand_nodes}

    gsm_instance = GSMInstance(instance_id=instance_id, all_nodes=graph.all_nodes, edge_list=graph.edge_list,
                               lt_dict=lt_dict, qty_dict=qty_dict,
                               hc_dict=hc_dict, sla_dict=sla_dict, mu_dict=mu_dict, sigma_dict=sigma_dict)
    return gsm_instance


def write_instance_to_csv(gsm_instance, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    edge_qty_list = [(u, v, qty) for (u, v), qty in
                     gsm_instance.qty_dict.items()]
    edge_df = pd.DataFrame(edge_qty_list, columns=['pred', 'succ', 'quantity'])

    sla_dict = {node: np.nan for node in gsm_instance.all_nodes}
    sla_dict.update(gsm_instance.sla_dict)
    mu_dict = {node: np.nan for node in gsm_instance.all_nodes}
    mu_dict.update(gsm_instance.mu_dict)
    sigma_dict = {node: np.nan for node in gsm_instance.all_nodes}
    sigma_dict.update(gsm_instance.sigma_dict)

    instance_info = {'instance_id': gsm_instance.instance_id, 'tau': gsm_instance.tau,
                     'pooling_factor': gsm_instance.pooling_factor}
    instance_info_df = pd.DataFrame.from_dict(instance_info, orient='index').reset_index()
    instance_info_df.columns = ['instance_info', 'value']

    nodes_info = [gsm_instance.lt_dict, gsm_instance.hc_dict, sla_dict, mu_dict, sigma_dict]
    node_df = pd.DataFrame(nodes_info).T.reset_index()
    node_df.columns = ['node_id', 'lt', 'hc', 'sla', 'mu', 'sigma']

    edge_df.to_csv(data_dir + 'edge.csv', index=False)
    node_df.to_csv(data_dir + 'node.csv', index=False)
    instance_info_df.to_csv(data_dir + 'instance_info.csv', index=False)


def load_instance_from_csv(data_dir):
    edge_df = pd.read_csv(data_dir + 'edge.csv')
    node_df = pd.read_csv(data_dir + 'node.csv')
    instance_info_df = pd.read_csv(data_dir + 'instance_info.csv')

    edge_list = [(u, v) for u, v, _ in edge_df.values]
    qty_dict = {(u, v): qty for u, v, qty in edge_df.values}

    all_nodes = set(node_df['node_id'])
    lt_dict = dict(zip(node_df['node_id'], node_df['lt']))
    hc_dict = dict(zip(node_df['node_id'], node_df['hc']))
    sla_dict = {node: sla for node, sla in node_df[['node_id', 'sla']].dropna().values.tolist()}
    mu_dict = {node: mu for node, mu in node_df[['node_id', 'mu']].dropna().values.tolist()}
    sigma_dict = {node: sigma for node, sigma in node_df[['node_id', 'sigma']].dropna().values.tolist()}

    instance_info = dict(zip(instance_info_df['instance_info'], instance_info_df['value']))

    gsm_instance = GSMInstance(instance_id=instance_info['instance_id'], all_nodes=all_nodes, edge_list=edge_list,
                               lt_dict=lt_dict, qty_dict=qty_dict,
                               hc_dict=hc_dict, sla_dict=sla_dict,
                               mu_dict=mu_dict, sigma_dict=sigma_dict,
                               tau=float(instance_info['tau']),
                               pooling_factor=int(instance_info['pooling_factor']))
    return gsm_instance


def write_policy_to_csv(policy, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    sol_list = [(node, policy.sol_S[node], policy.sol_SI[node], policy.sol_CT[node]) for node in policy.all_nodes]
    sol_df = pd.DataFrame(sol_list, columns=['node_id', 'S', 'SI', 'CT'])

    policy_info = [policy.base_stock, policy.safety_stock]
    policy_df = pd.DataFrame(policy_info).T.reset_index()
    policy_df.columns = ['node_id', 'base_stock', 'safety_stock']

    sol_df.to_csv(data_dir + 'sol.csv', index=False)
    policy_df.to_csv(data_dir + 'policy.csv', index=False)


def load_policy_from_csv(data_dir):
    sol_df = pd.read_csv(data_dir + 'sol.csv')
    policy_df = pd.read_csv(data_dir + 'policy.csv')

    sol = {'S': dict(zip(sol_df['node_id'], sol_df['S'])), 'SI': dict(zip(sol_df['node_id'], sol_df['SI'])),
           'CT': dict(zip(sol_df['node_id'], sol_df['CT']))}

    all_nodes = set(sol_df['node_id'])
    base_stock = dict(zip(policy_df['node_id'], policy_df['base_stock']))
    safety_stock = dict(zip(policy_df['node_id'], policy_df['safety_stock']))

    policy = Policy(all_nodes)
    policy.update_sol(sol)
    policy.update_base_stock(base_stock)
    policy.update_safety_stock(safety_stock)
    return policy
