import datetime
from typing import Optional
from copy import copy
import pandas as pd
import os
from algorithm.pwl import PieceWiseLinear
from algorithm.hgna import HeuristicGeneralNetworksAlgorithm
from algorithm.ds import DynamicSloping
from algorithm.imip import IterativeMIP
from algorithm.simple_slp import SimpleSLP
from algorithm.if_slp import IterativeFixingSLP
from algorithm.id_slp import IterativeDecompositionSLP
from default_paras import SOLVER

_approach_map = {
    'PWL': PieceWiseLinear,
    'HGNA': HeuristicGeneralNetworksAlgorithm,
    'DS': DynamicSloping,
    'IMIP': IterativeMIP,
    'Simple-SLP': SimpleSLP,
    'IF-SLP': IterativeFixingSLP,
    'ID-SLP': IterativeDecompositionSLP
}


class Task:
    def __init__(self, task_id, gsm_instance, approach_name, approach_paras: Optional[dict] = None):
        self.task_id = task_id
        self.gsm_instance = gsm_instance
        self.approach_name = approach_name
        if approach_paras is None:
            self.approach_paras = {}
        else:
            self.approach_paras = approach_paras
        self.task_status = 'UNSOLVED'  # 'SOLVED'/ 'UNSOLVED'
        self.policy = None
        self.task_time = None
        self.task_info = None

    def run(self, solver=SOLVER):
        kwargs = copy(self.approach_paras)
        kwargs.update({'gsm_instance': self.gsm_instance})
        approach = _approach_map[self.approach_name](**kwargs)
        time0 = datetime.datetime.now()
        if approach.need_solver:
            self.policy = approach.get_policy(solver)
        else:
            self.policy = approach.get_policy()
        self.task_time = datetime.datetime.now() - time0
        self.approach_paras.update(approach.get_approach_paras())
        self.task_status = 'SOLVED'
        self.gen_task_info()

    def gen_task_info(self):
        task_info = {
            'task_id': self.task_id,
            'instance_id': self.gsm_instance.instance_id,
            'graph_type': self.gsm_instance.graph.graph_type,
            'nodes_num': len(self.gsm_instance.all_nodes),
            'edges_num': len(self.gsm_instance.edge_list),
            'tau': self.gsm_instance.tau,
            'pooling_factor': self.gsm_instance.pooling_factor,
            'approach_name': self.approach_name,
            'task_time': self.task_time,
            'ss_cost': self.policy.ss_cost
        }
        task_info.update(self.approach_paras)
        self.task_info = task_info

    def write_to_csv(self, data_dir):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        sol_list = [(node, self.policy.sol_S[node], self.policy.sol_SI[node], self.policy.sol_CT[node])
                    for node in self.policy.all_nodes]
        sol_df = pd.DataFrame(sol_list, columns=['node_id', 'S', 'SI', 'CT'])

        policy_info = [self.policy.base_stock, self.policy.safety_stock]
        policy_df = pd.DataFrame(policy_info).T.reset_index()
        policy_df.columns = ['node_id', 'base_stock', 'safety_stock']

        task_info_df = pd.DataFrame.from_dict(self.task_info, orient='index').reset_index()
        task_info_df.columns = ['task_info', 'value']

        sol_df.to_csv(data_dir + str(self.task_id) + '_sol.csv', index=False)
        policy_df.to_csv(data_dir + str(self.task_id) + '_policy.csv', index=False)
        task_info_df.to_csv(data_dir + str(self.task_id) + '_task_info.csv', index=False)
