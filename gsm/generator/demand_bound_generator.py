from gsm.demand_bound.demand_bound_collector import DemandBoundCollector
from gsm.generator.default_gsm_paras import DEMAND_MEAN_LB, DEMAND_MEAN_UB, DEMAND_STD_LB, DEMAND_STD_UB, \
    TAU, POOLING_FACTOR
from gsm.demand_bound.cascading_stats import get_cascading_normal_stats
from gsm.gsm_instance import GSMInstance
import numpy as np


def cascading_normal_bound_generating_from_paras(gsm_instance: GSMInstance,
                                                 demand_mean_lb=DEMAND_MEAN_LB,
                                                 demand_mean_ub=DEMAND_MEAN_UB,
                                                 demand_std_lb=DEMAND_STD_LB,
                                                 demand_std_ub=DEMAND_STD_UB,
                                                 tau=TAU,
                                                 pooling_factor=POOLING_FACTOR):
    ext_mean_of_node = {n_id: np.random.uniform(demand_mean_lb, demand_mean_ub) for n_id in gsm_instance.sinks}
    ext_std_of_node = {n_id: np.random.uniform(demand_std_lb, demand_std_ub) for n_id in gsm_instance.sinks}

    mean_of_node, std_of_node = get_cascading_normal_stats(edges=gsm_instance.edges,
                                                           edge_qty=gsm_instance.edge_qty,
                                                           ext_mean_of_node=ext_mean_of_node,
                                                           ext_std_of_node=ext_std_of_node)

    db_collector = DemandBoundCollector(gsm_instance)
    db_collector.set_cascading_normal_bound_given_paras(ext_mean_of_node, ext_std_of_node, tau, pooling_factor)
    demand_bound_pool = db_collector.demand_bound_pool

    return demand_bound_pool, mean_of_node, std_of_node
