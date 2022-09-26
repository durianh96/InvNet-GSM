import math
from algorithm.dp import DynamicProgramming
from utils.utils import *
from utils.gsm_utils import *
from data_process import *
from domain.policy import Policy
from default_paras import TIME_UNIT, MAX_ITER_NUM


class HeuristicGeneralNetworksAlgorithm:
    def __init__(self, gsm_instance, time_unit=TIME_UNIT, max_iter_num=MAX_ITER_NUM):
        self.gsm_instance = gsm_instance
        self.graph = gsm_instance.graph

        self.all_nodes = gsm_instance.all_nodes
        self.demand_nodes = self.graph.demand_nodes
        self.edge_list = self.graph.edge_list
        self.lt_dict = gsm_instance.lt_dict
        self.qty_dict = gsm_instance.qty_dict
        self.cum_lt_dict = gsm_instance.cum_lt_dict
        self.hc_dict = gsm_instance.hc_dict
        self.sla_dict = gsm_instance.sla_dict
        self.mu_dict = gsm_instance.mu_dict
        self.sigma_dict = gsm_instance.sigma_dict

        self.pooling_factor = gsm_instance.pooling_factor
        self.pred_dict = self.graph.pred_dict
        self.succ_dict = self.graph.succ_dict

        self.vb_func = gsm_instance.vb_func
        self.db_func = gsm_instance.db_func

        self.time_unit = time_unit
        self.max_iter_num = max_iter_num

        self.iter_round = 0
        self.visited_violate_edges = []
        self.violate_sol_list = []

        self.initial_bounds()
        self.cur_opt_dict = {self.init_ss_cost: self.init_sol}

        self.need_solver = False

    def rank_edge(self):
        # add cost on each edge
        self.dev_dict = self.get_node_deviation()
        self.edge_cost_dict = {}
        for pred, succ in self.edge_list:
            cost = (self.hc_dict[pred] + self.hc_dict[succ]) \
                   * self.dev_dict[succ]
            self.edge_cost_dict.update({(pred, succ): cost})

    def get_node_deviation(self):
        reverse_edges = [(j, i) for i, j in self.edge_list]
        reverse_topo_sort = find_topo_sort(reverse_edges)
        constant_dict = {node: 0. for node in self.all_nodes}
        constant_dict.update({node: s ** self.pooling_factor for node, s in self.sigma_dict.items()})

        for node in reverse_topo_sort:
            if len(self.pred_dict[node]) > 0:
                for pred in self.pred_dict[node]:
                    constant_dict[pred] += (self.qty_dict[pred, node] ** self.pooling_factor) * constant_dict[node]

        dev_dict = {node: np.power(v, 1 / self.pooling_factor) for node, v in constant_dict.items()}

        return dev_dict

    def select_sptree(self):
        # select tree with minimal cost
        self.rank_edge()
        # add cost on edge
        mst_edge_list = find_mst_prim(edge_list=self.edge_list, edge_weight_dict=self.edge_cost_dict)
        self.mst_gsm_instance = GSMInstance(instance_id='sub_' + self.gsm_instance.instance_id,
                                            edge_list=mst_edge_list, all_nodes=self.all_nodes, lt_dict=self.lt_dict,
                                            qty_dict=self.qty_dict, hc_dict=self.hc_dict, sla_dict=self.sla_dict,
                                            mu_dict=self.mu_dict, sigma_dict=self.sigma_dict)
        self.mst_gsm_instance.db_func = self.gsm_instance.db_func
        self.mst_gsm_instance.vb_func = self.gsm_instance.vb_func
        self.mst_gsm_instance.grad_vb_func = self.gsm_instance.grad_vb_func

    def initial_bounds(self):
        self.init_sol = {'S': {node: 0 for node in self.all_nodes}, 'SI': {node: 0 for node in self.all_nodes},
                         'CT': {}}
        for node in self.demand_nodes:
            val = min(self.lt_dict[node], self.sla_dict[node])
            self.init_sol['S'].update({node: val})
        self.init_sol['CT'] = {node: self.init_sol['SI'][node] + self.lt_dict[node] -
                                     self.init_sol['S'][node] for node in self.all_nodes}
        init_bs_dict = {node: self.db_func[node](self.init_sol['CT'][node]) for node in self.all_nodes}
        init_ss_dict = {node: self.vb_func[node](self.init_sol['CT'][node]) for node in self.all_nodes}
        self.init_ss_cost = cal_cost(self.hc_dict, init_ss_dict, 'Initialization')

    def run_sptree_dp(self, violate_dict):
        input_s_ub_dict = violate_dict['ub']
        input_si_lb_dict = violate_dict['lb']
        dp_algo = DynamicProgramming(self.mst_gsm_instance, input_s_ub_dict, input_si_lb_dict)
        dp_index_dict = {'S_index': dp_algo.S_index, 'SI_index': dp_algo.SI_index}
        dp_policy = dp_algo.get_policy()
        dp_sol = {'S': dp_policy.sol_S,
                  'SI': dp_policy.sol_SI,
                  'CT': dp_policy.sol_CT}
        dp_ss_cost = dp_policy.ss_cost
        return dp_index_dict, dp_sol, dp_ss_cost

    def get_feasible_sol(self, dp_sol, index_dict):
        feasible_sol = {'S': dp_sol['S'].copy(), 'SI': dp_sol['SI'].copy(), 'CT': {}}

        for node, pred_list in self.pred_dict.items():
            if len(pred_list):
                S_hat = max([feasible_sol['S'][pred_node] for pred_node in pred_list])
                SI_hat = min(S_hat, np.max(index_dict['SI_index'][node]))
                feasible_sol['SI'].update({node: SI_hat})

        feasible_sol['CT'] = {node: feasible_sol['SI'][node] + self.lt_dict[node] -
                                    feasible_sol['S'][node] for node in self.all_nodes}

        violate_flag, _ = self.check_global_constraints(feasible_sol)
        if violate_flag:
            return self.init_sol, self.init_ss_cost
        else:
            feasible_ss_dict = {node: self.vb_func[node](feasible_sol['CT'][node]) for node in self.all_nodes}
            feasible_ss_cost = cal_cost(self.hc_dict, feasible_ss_dict, 'Feasible')
            return feasible_sol, feasible_ss_cost

    def check_global_constraints(self, dp_sol):
        # for(j,i): S_j <= SI_i
        violate_flag = False
        violate_edge_cost_dict = {}
        violate_edge = ()
        for node, succ_list in self.succ_dict.items():
            if len(succ_list):
                for _, succ_node in enumerate(succ_list):
                    if dp_sol['S'][node] > dp_sol['SI'][succ_node]:
                        violate_flag = True
                        violate_edge_cost_dict[(node, succ_node)] = self.edge_cost_dict[(node, succ_node)]
        if violate_flag:
            violate_edge = min(violate_edge_cost_dict, key=lambda x: violate_edge_cost_dict[x])

        return violate_flag, violate_edge

    def call_routine(self, violate_dict):
        self.cur_opt_ss_cost = min(list(self.cur_opt_dict.keys()))
        self.cur_opt_sol = self.cur_opt_dict[self.cur_opt_ss_cost]
        self.iter_round += 1
        if self.iter_round > self.max_iter_num:
            return self.cur_opt_sol, self.cur_opt_ss_cost
        else:
            dp_index_dict, dp_sol, dp_ss_cost = self.run_sptree_dp(violate_dict)

        violate_flag, violate_edge = self.check_global_constraints(dp_sol)

        if violate_flag:
            feasible_sol, feasible_ss_cost = self.get_feasible_sol(dp_sol, dp_index_dict)
            pred = violate_edge[0]
            succ = violate_edge[1]
            S_bar = dp_sol['SI'][succ] + math.floor((dp_sol['S'][pred] - dp_sol['SI'][succ]) * 1.0 / 2)

            self.cur_opt_dict[feasible_ss_cost] = feasible_sol

            # upperbound
            violate_ub_dict = {'ub': violate_dict['ub'].copy(), 'lb': {}}
            violate_ub_dict['ub'].update({pred: S_bar})
            violate_ub_dict['lb'] = violate_dict['lb'].copy()
            ub_sol, ub_ss_cost = self.call_routine(violate_ub_dict)

            # lowerbound
            if violate_edge in self.visited_violate_edges:
                lb_sol, lb_ss_cost = self.init_sol, self.init_ss_cost
            else:
                violate_lb_dict = {'ub': violate_dict['ub'].copy(), 'lb': violate_dict['lb'].copy()}
                succ_list = self.succ_dict[pred]
                if len(succ_list):
                    for node_k in succ_list:
                        violate_lb_dict['lb'].update({node_k: S_bar})
                lb_sol, lb_ss_cost = self.call_routine(violate_lb_dict)
                self.visited_violate_edges.append(violate_edge)

            sol_list = [feasible_sol, ub_sol, lb_sol]
            cost_list = [feasible_ss_cost, ub_ss_cost, lb_ss_cost]
            opt_ss_cost = min(cost_list)
            opt_index = cost_list.index(min(cost_list))
            opt_sol = sol_list[opt_index]
            self.cur_opt_dict[opt_ss_cost] = opt_sol
            return opt_sol, opt_ss_cost

        else:
            return dp_sol, dp_ss_cost

    @timer
    def get_policy(self):
        if self.graph.is_tree():
            dp_algo = DynamicProgramming(self.gsm_instance)
            dp_policy = dp_algo.get_policy()
            sol = {'S': dp_policy.sol_S,
                   'SI': dp_policy.sol_SI,
                   'CT': dp_policy.sol_CT}
            bs_dict = dp_policy.base_stock
            ss_dict = dp_policy.safety_stock
            ss_cost = dp_policy.ss_cost
        else:
            self.select_sptree()
            default_violate_dict = {'lb': {}, 'ub': {}}
            sol, _ = self.call_routine(default_violate_dict)
            bs_dict = {node: self.db_func[node](sol['CT'][node]) for node in self.all_nodes}
            ss_dict = {node: self.vb_func[node](sol['CT'][node]) for node in self.all_nodes}
            ss_cost = cal_cost(self.hc_dict, ss_dict, 'HGNA')

        policy = Policy(self.all_nodes)
        policy.update_sol(sol)
        policy.update_base_stock(bs_dict)
        policy.update_safety_stock(ss_dict)
        policy.update_ss_cost(ss_cost)
        return policy

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit, 'max_iter_num': self.max_iter_num}
        return paras
