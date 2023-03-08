from gsm.generator.demand_bound_generator import cascading_normal_bound_generating_from_paras
from gsm.generator.properties_generator import properties_generating
from gsm.generator.default_gsm_paras import *


def gsm_instance_generating_given_paras_ranges(instance_id,
                                               edges,
                                               qty_lb=QTY_LB,
                                               qty_ub=QTY_UB,
                                               lt_lb=LT_LB,
                                               lt_ub=LT_UB,
                                               hc_lb=HC_LB,
                                               hc_ub=HC_UB,
                                               sla_lt_lb=SLA_LT_LB,
                                               sla_lt_ub=SLA_LT_UB,
                                               demand_mean_lb=DEMAND_MEAN_LB,
                                               demand_mean_ub=DEMAND_MEAN_UB,
                                               demand_std_lb=DEMAND_STD_LB,
                                               demand_std_ub=DEMAND_STD_UB,
                                               tau=TAU,
                                               pooling_factor=POOLING_FACTOR):
    gsm_instance = properties_generating(instance_id, edges, qty_lb, qty_ub, lt_lb, lt_ub, hc_lb, hc_ub,
                                         sla_lt_lb, sla_lt_ub)
    demand_bound_pool, mean_of_node, std_of_node = cascading_normal_bound_generating_from_paras(
        gsm_instance, demand_mean_lb, demand_mean_ub, demand_std_lb, demand_std_ub, tau, pooling_factor
    )

    gsm_instance.update_mean_of_node(mean_of_node)
    gsm_instance.update_std_of_node(std_of_node)
    gsm_instance.update_demand_bound_pool(demand_bound_pool)

    return gsm_instance
