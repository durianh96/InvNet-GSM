from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import LOCAL_SOL_NUM_ID, STABILITY_TYPE, STABILITY_THRESHOLD, \
    BOUND_VALUE_TYPE
from gsm.gsm_instance import GSMInstance
from typing import Optional


class IterativeDecomposition(object):
    def __init__(self, gsm_instance: GSMInstance,
                 local_sol_num,
                 stability_type,
                 stability_threshold,
                 bound_value_type):
        self.gsm_instance = gsm_instance
        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.sinks = gsm_instance.sinks

        self.succs_of_node = gsm_instance.succs_of_node
        self.preds_of_node = gsm_instance.preds_of_node

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.cum_lt_of_node = gsm_instance.cum_lt_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.local_sol_num = local_sol_num
        self.stability_type = stability_type
        self.stability_threshold = stability_threshold
        self.bound_value_type = bound_value_type

        self.to_run_pool = []
        self.solved_pool = []

        self.sol = {'S': {}, 'SI': {}, 'CT': {}}
        
        self.need_solver = None

    def get_policy(self):
        pass

    def get_db_value_of_node(self, n_id, ct_value):
        if self.bound_value_type == 'APPROX':
            db_value = self.demand_bound_pool[n_id].get_db_value(ct_value)
        elif self.bound_value_type == 'FUNC':
            db_value = self.demand_bound_pool[n_id].db_func(ct_value)
        else:
            raise AttributeError('wrong bound value type')
        return db_value

    def get_vb_value_of_node(self, n_id, ct_value):
        if self.bound_value_type == 'APPROX':
            vb_value = self.demand_bound_pool[n_id].get_vb_value(ct_value)
        elif self.bound_value_type == 'FUNC':
            vb_value = self.demand_bound_pool[n_id].vb_func(ct_value)
        else:
            raise AttributeError('wrong bound value type')
        return vb_value

    def get_vb_gradient_value_of_node(self, n_id, ct_value):
        if self.bound_value_type == 'APPROX':
            pwl_vb_paras = self.demand_bound_pool[n_id].get_pwl_vb_paras(ct_value)
            vb_gradient_value = pwl_vb_paras['gradient']
        elif self.bound_value_type == 'FUNC':
            vb_gradient_value = self.demand_bound_pool[n_id].vb_gradient_func(ct_value)
        else:
            raise AttributeError('wrong bound value type')
        return vb_gradient_value


class LocalApproach(object):
    def __init__(self, gsm_instance: GSMInstance,
                 local_sol_num,
                 stability_type,
                 stability_threshold,
                 bound_value_type,
                 input_s_ub_dict: Optional[dict] = None,
                 input_si_lb_dict: Optional[dict] = None):
        self.gsm_instance = gsm_instance
        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.sinks = gsm_instance.sinks

        self.succs_of_node = gsm_instance.succs_of_node
        self.preds_of_node = gsm_instance.preds_of_node

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.cum_lt_of_node = gsm_instance.cum_lt_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.local_sol_num = local_sol_num
        self.stability_type = stability_type
        self.stability_threshold = stability_threshold
        self.bound_value_type = bound_value_type

        if input_si_lb_dict is None:
            input_si_lb_dict = {}
        if input_s_ub_dict is None:
            input_s_ub_dict = {}
        self.input_s_ub_dict = input_s_ub_dict
        self.input_si_lb_dict = input_si_lb_dict
        self.sol = {'S': {}, 'SI': {}, 'CT': {}}
        self.gsm_sol_set = GSMSolutionSet(nodes=self.nodes, lt_of_node=self.lt_of_node,
                                          cum_lt_of_node=self.cum_lt_of_node, sla_of_node=self.sla_of_node)
        self.status = 'TO_RUN'

    def run(self):
        pass

    def get_sub_pool(self):
        pass
