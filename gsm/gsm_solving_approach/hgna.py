import math
from gsm.gsm_solving_approach.dp import DynamicProgramming
from utils.system_utils import *
from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import MAX_ITER_NUM_HGNA, BOUND_VALUE_TYPE, SYSTEM_TIME_UNIT
from gsm.gsm_instance import GSMInstance
from utils.graph_algorithms import *


class HeuristicGeneralNetworksAlgorithm:
    def __init__(self, gsm_instance: GSMInstance,
                 time_unit=SYSTEM_TIME_UNIT,
                 max_iter_num=MAX_ITER_NUM_HGNA,
                 bound_value_type=BOUND_VALUE_TYPE):
        self.gsm_instance = gsm_instance

        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.sinks = gsm_instance.sinks

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.std_of_node = gsm_instance.std_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.preds_of_node = gsm_instance.preds_of_node
        self.succs_of_node = gsm_instance.succs_of_node

        self.time_unit = time_unit
        self.max_iter_num = max_iter_num
        self.bound_value_type = bound_value_type

        self.iter_round = 0
        self.visited_violate_edges = []
        self.violate_sol_list = []
        self.edge_cost = None
        self.mst_gsm_instance = None

        self.sol_results = []
        self.init_sol = self.get_init_sol()
        self.cur_best_sol = self.init_sol

        self.need_solver = False

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

    def get_init_sol(self):
        init_sol = {'S': {node: 0 for node in self.nodes}, 'SI': {node: 0 for node in self.nodes},
                    'CT': {}}
        for node in self.sinks:
            val = min(self.lt_of_node[node], self.sla_of_node[node])
            init_sol['S'].update({node: val})
        init_sol['CT'] = {node: init_sol['SI'][node] + self.lt_of_node[node] - init_sol['S'][node]
                          for node in self.nodes}
        init_sol['obj_value'] = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, init_sol['CT'][node]) for node in self.nodes])
        self.sol_results.append(init_sol)
        return init_sol

    def get_cur_best_sol(self):
        cost_list = [sol['obj_value'] for sol in self.sol_results]
        best_i = np.argmin(cost_list)
        self.cur_best_sol = self.sol_results[best_i]
        return self.cur_best_sol

    def select_sptree(self):
        # add cost on edge
        self.edge_cost = {(pred, succ): (self.hc_of_node[pred] + self.hc_of_node[succ]) * self.std_of_node[succ]
                          for pred, succ in self.edges}
        # select tree with minimal cost
        mst_edges = find_mst_prim(edges=self.edges, edge_weight_dict=self.edge_cost)
        sub_nodes = set([node for tu in mst_edges for node in tu])
        self.mst_gsm_instance = self.gsm_instance.get_sub_instance(sub_nodes, mst_edges)

    def run_sptree_dp(self, violate_dict):
        input_s_ub_dict = violate_dict['ub']
        input_si_lb_dict = violate_dict['lb']
        dp_algo = DynamicProgramming(self.mst_gsm_instance, input_s_ub_dict, input_si_lb_dict)
        dp_index_dict = {'S_index': dp_algo.S_index, 'SI_index': dp_algo.SI_index}
        dp_policy = dp_algo.get_policy()
        dp_sol = {'S': dp_policy.S_of_node,
                  'SI': dp_policy.SI_of_node,
                  'CT': dp_policy.CT_of_node,
                  'obj_value': dp_policy.ss_cost}
        return dp_index_dict, dp_sol

    def get_feasible_sol(self, dp_sol, index_dict):
        feasible_sol = {'S': dp_sol['S'].copy(), 'SI': dp_sol['SI'].copy(), 'CT': {}}

        for node, pred_list in self.preds_of_node.items():
            if len(pred_list):
                S_hat = max([feasible_sol['S'][pred_node] for pred_node in pred_list])
                SI_hat = min(S_hat, np.max(index_dict['SI_index'][node]))
                feasible_sol['SI'].update({node: SI_hat})

        feasible_sol['CT'] = {node: feasible_sol['SI'][node] + self.lt_of_node[node] - feasible_sol['S'][node]
                              for node in self.nodes}

        violate_flag, _ = self.check_global_constraints(feasible_sol)
        if violate_flag:
            return self.init_sol
        else:
            feasible_sol['obj_value'] = sum([
                self.hc_of_node[node] * self.get_vb_value_of_node(node, feasible_sol['CT'][node])
                for node in self.nodes])
            return feasible_sol

    def check_global_constraints(self, dp_sol):
        # for (j,i): S_j <= SI_i
        violate_flag = False
        violate_edge_cost_dict = {}
        violate_edge = ()
        for node, succ_list in self.succs_of_node.items():
            if len(succ_list):
                for _, succ_node in enumerate(succ_list):
                    if dp_sol['S'][node] > dp_sol['SI'][succ_node]:
                        violate_flag = True
                        violate_edge_cost_dict[(node, succ_node)] = self.edge_cost[(node, succ_node)]
        if violate_flag:
            violate_edge = min(violate_edge_cost_dict, key=lambda x: violate_edge_cost_dict[x])

        return violate_flag, violate_edge

    def call_routine(self, violate_dict):
        self.get_cur_best_sol()
        self.iter_round += 1
        if self.iter_round > self.max_iter_num:
            return self.cur_best_sol
        else:
            dp_index_dict, dp_sol = self.run_sptree_dp(violate_dict)

        violate_flag, violate_edge = self.check_global_constraints(dp_sol)

        if violate_flag:
            feasible_sol = self.get_feasible_sol(dp_sol, dp_index_dict)
            pred = violate_edge[0]
            succ = violate_edge[1]
            S_bar = dp_sol['SI'][succ] + math.floor((dp_sol['S'][pred] - dp_sol['SI'][succ]) * 1.0 / 2)
            self.sol_results.append(feasible_sol)
            # upperbound
            violate_ub_dict = {'ub': violate_dict['ub'].copy(), 'lb': {}}
            violate_ub_dict['ub'].update({pred: S_bar})
            violate_ub_dict['lb'] = violate_dict['lb'].copy()
            ub_sol = self.call_routine(violate_ub_dict)

            # lower-bound
            if violate_edge in self.visited_violate_edges:
                lb_sol = self.init_sol
            else:
                violate_lb_dict = {'ub': violate_dict['ub'].copy(), 'lb': violate_dict['lb'].copy()}
                succ_list = self.succs_of_node[pred]
                if len(succ_list):
                    for node_k in succ_list:
                        violate_lb_dict['lb'].update({node_k: S_bar})
                lb_sol = self.call_routine(violate_lb_dict)
                self.visited_violate_edges.append(violate_edge)

            sol_list = [feasible_sol, ub_sol, lb_sol]
            cost_list = [sol['obj_value'] for sol in sol_list]
            opt_sol = sol_list[np.argmin(cost_list)]
            self.sol_results.append(opt_sol)
            return opt_sol
        else:
            return dp_sol

    @timer
    def get_policy(self):
        if self.gsm_instance.graph_type == 'TREE':
            dp_algo = DynamicProgramming(self.gsm_instance)
            gsm_sol = dp_algo.get_policy()
        else:
            self.select_sptree()
            default_violate_dict = {'lb': {}, 'ub': {}}
            sol = self.call_routine(default_violate_dict)

            gsm_sol = GSMSolution(nodes=self.nodes)
            gsm_sol.update_sol(sol)
            oul_of_node = {node: self.get_db_value_of_node(node, gsm_sol.CT_of_node[node]) for node in self.nodes}
            ss_of_node = {node: self.get_vb_value_of_node(node, gsm_sol.CT_of_node[node]) for node in self.nodes}

            ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, 'HGNA')

            gsm_sol.update_oul(oul_of_node)
            gsm_sol.update_ss(ss_of_node)
            gsm_sol.update_ss_cost(ss_cost)
        return gsm_sol

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit,
                 'max_iter_num': self.max_iter_num,
                 'bound_value_type': self.bound_value_type}
        return paras


def find_mst_prim(edges, edge_weight_dict):
    nodes = set([i for i, _ in edges]) | set([j for _, j in edges])
    adjs_of_node = find_adjs_of_node(edges)
    preds_of_node = find_preds_of_node(edges)
    visited = set()
    tree = []
    init_node = nodes.pop()
    visited.add(init_node)

    while bool(nodes):
        valid_edge_dict = {}
        for j in visited:
            adj_nodes = adjs_of_node[j]
            valid_adj_nodes = adj_nodes & nodes
            for k in valid_adj_nodes:
                if k in preds_of_node[j]:
                    edge = (k, j)
                else:
                    edge = (j, k)
                valid_edge_dict[edge] = edge_weight_dict[edge]
        min_edge = min(valid_edge_dict, key=lambda x: valid_edge_dict[x])
        tree.append(min_edge)
        visited.add(min_edge[0])
        visited.add(min_edge[1])
        nodes.discard(min_edge[0])
        nodes.discard(min_edge[1])

    return tree
