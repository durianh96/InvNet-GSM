import numpy as np
from gsm.generator.default_gsm_paras import QTY_LB, QTY_UB, LT_LB, LT_UB, HC_LB, HC_UB, SLA_LT_LB, SLA_LT_UB
from utils.graph_algorithms import find_topo_sort, find_preds_of_node, cal_cum_lt_of_node
from gsm.gsm_instance import GSMInstance


def properties_generating(instance_id,
                          edges,
                          qty_lb=QTY_LB,
                          qty_ub=QTY_UB,
                          lt_lb=LT_LB,
                          lt_ub=LT_UB,
                          hc_lb=HC_LB,
                          hc_ub=HC_UB,
                          sla_lt_lb=SLA_LT_LB,
                          sla_lt_ub=SLA_LT_UB):
    nodes = set([node for tu in edges for node in tu])
    sinks = list(set([j for _, j in edges]) - set([i for i, _ in edges]))

    # random lt
    lt_of_node = {n_id: np.random.randint(lt_lb, lt_ub) for n_id in nodes}

    # random edge qty
    edge_qty = {}
    for u, v in edges:
        edge_qty[(u, v)] = np.random.uniform(qty_lb, qty_ub)

    # random holding cost (must larger than assemblies)
    topo_sort = find_topo_sort(edges)
    preds_of_node = find_preds_of_node(edges)
    hc_of_node = {}
    for n_id in topo_sort:
        if len(preds_of_node[n_id]) > 0:
            hc_of_node[n_id] = sum([hc_of_node[pred] * edge_qty[pred, n_id] for pred in preds_of_node[n_id]]) \
                               + np.random.uniform(hc_lb, hc_ub)
        else:
            hc_of_node[n_id] = np.random.uniform(hc_lb, hc_ub)

    # random sla
    sla_of_node = {n_id: np.random.randint(sla_lt_lb, sla_lt_ub) + lt_of_node[n_id]
                   for n_id in sinks}

    cum_lt_of_node, _ = cal_cum_lt_of_node(edges, lt_of_node)

    gsm_instance = GSMInstance(
        instance_id=instance_id,
        nodes=nodes,
        edges=edges,
        lt_of_node=lt_of_node,
        edge_qty=edge_qty,
        hc_of_node=hc_of_node,
        sla_of_node=sla_of_node,
        cum_lt_of_node=cum_lt_of_node
    )
    return gsm_instance
