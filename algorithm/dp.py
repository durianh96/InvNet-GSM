from collections import defaultdict
from utils.gsm_utils import *
from utils.utils import *
from domain.policy import Policy
from algorithm.default_paras import *


class DynamicProgramming:
    def __init__(self, gsm_instance, input_s_ub_dict=None, input_si_lb_dict=None, time_unit=TIME_UNIT):
        self.gsm_instance = gsm_instance
        self.graph = gsm_instance.graph
        if not self.graph.is_tree():
            raise AttributeError('Graph is not tree')
        if input_si_lb_dict is None:
            input_si_lb_dict = {}
        if input_s_ub_dict is None:
            input_s_ub_dict = {}

        self.all_nodes = gsm_instance.all_nodes
        self.demand_nodes = self.graph.demand_nodes
        self.edge_list = self.graph.edge_list
        self.lt_dict = gsm_instance.lt_dict
        self.cum_lt_dict = gsm_instance.cum_lt_dict
        self.hc_dict = gsm_instance.hc_dict
        self.sla_dict = gsm_instance.sla_dict

        self.pred_dict = self.graph.pred_dict
        self.succ_dict = self.graph.succ_dict
        self.time_unit = time_unit

        self.vb_func = gsm_instance.vb_func
        self.db_func = gsm_instance.db_func

        self.s_ub_dict = {node: min(self.sla_dict.get(node, 9999), input_s_ub_dict.get(node, 9999),
                                    self.cum_lt_dict[node]) for node in self.all_nodes}

        self.si_lb_dict = {node: input_si_lb_dict.get(node, 0) for node in self.all_nodes}
        self.si_ub_dict = {node: self.cum_lt_dict[node] - self.lt_dict[node] for node in self.all_nodes}

        self.S_index = {}
        self.SI_index = {}
        for node in self.all_nodes:
            if self.s_ub_dict[node] >= self.time_unit:
                self.S_index[node] = np.arange(0., self.s_ub_dict[node] + self.time_unit, self.time_unit)
            else:
                self.S_index[node] = [0., self.s_ub_dict[node]]
            if self.si_ub_dict[node] - self.si_lb_dict[node] >= self.time_unit:
                self.SI_index[node] = np.arange(self.si_lb_dict[node], self.si_ub_dict[node] + self.time_unit,
                                                self.time_unit)
            elif self.si_ub_dict[node] - self.si_lb_dict[node] >= 0:
                self.SI_index[node] = [self.si_lb_dict[node], self.si_ub_dict[node]]
            else:
                raise ValueError('dp bound error')

        self.sorted_list = self.sort()
        self.parent_dict = self.get_parent_dict()

        self.to_eva_f_list, self.to_eva_g_list, self.sub_pred_dict, self.sub_succ_dict = self.classify_node()

        self.f_cost = {(node, S): -float('inf') for node in self.to_eva_f_list for S in self.S_index[node]}
        self.f_argmin = {(node, S): -float('inf') for node in self.to_eva_f_list for S in self.S_index[node]}

        self.g_cost = {(node, SI): -float('inf') for node in self.to_eva_g_list for SI in self.SI_index[node]}
        self.g_argmin = {(node, SI): -float('inf') for node in self.to_eva_g_list for SI in self.SI_index[node]}
        self.cost_record = defaultdict(dict)

        self.ss_ct_dict = {(node, ct): self.vb_func[node](ct) for node in self.all_nodes
                           for ct in np.arange(0., self.cum_lt_dict[node] + self.time_unit, self.time_unit)}

        self.on_hand_cost = {(node, ct): self.hc_dict[node] * self.ss_ct_dict[node, ct] for node in self.all_nodes
                             for ct in np.arange(0., self.cum_lt_dict[node] + self.time_unit, self.time_unit)}

    @timer
    def get_policy(self):
        for node in self.sorted_list[:-1]:
            if node in self.to_eva_f_list:
                for S in self.S_index[node]:
                    self.evaluate_f_node(node, S)
            if node in self.to_eva_g_list:
                for SI in self.SI_index[node]:
                    self.evaluate_g_node(node, SI)

        end_node = self.sorted_list[-1]
        for SI in self.SI_index[end_node]:
            self.evaluate_g_node(end_node, SI)

        opt_sol = {'S': {}, 'SI': {}, 'CT': {}}

        end_g_dict = {si: self.g_cost[end_node, si] for si in self.SI_index[end_node]}
        end_node_SI = min(end_g_dict, key=end_g_dict.get)
        end_node_S = self.g_argmin[end_node, end_node_SI]

        opt_sol['SI'][end_node] = end_node_SI
        opt_sol['S'][end_node] = end_node_S

        for node in self.sorted_list[-2::-1]:
            parent_node = self.parent_dict[node]
            if node in self.sub_succ_dict[parent_node]:
                node_SI = opt_sol['S'][parent_node]
                node_S = self.g_argmin[node, node_SI]
            elif node in self.sub_pred_dict[parent_node]:
                node_S = min(opt_sol['SI'][parent_node], max(self.S_index[node]))
                node_SI = self.f_argmin[node, node_S]
            else:
                raise Exception
            opt_sol['S'][node] = node_S
            opt_sol['SI'][node] = node_SI

        opt_sol['CT'] = {node: opt_sol['SI'][node] + self.lt_dict[node] - opt_sol['S'][node]
                         for node in self.sorted_list}
        bs_dict = {node: self.db_func[node](ct) for node, ct in opt_sol['CT'].items()}
        ss_dict = {node: self.vb_func[node](ct) for node, ct in opt_sol['CT'].items()}
        cost = cal_cost(self.hc_dict, ss_dict, method='DP')

        policy = Policy(self.all_nodes)
        policy.update_sol(opt_sol)
        policy.update_base_stock(bs_dict)
        policy.update_safety_stock(ss_dict)
        policy.update_ss_cost(cost)
        return policy

    def evaluate_f_node(self, node, S):
        SI_lb = max(self.si_lb_dict[node], S - self.lt_dict[node])
        SI_ub = self.cum_lt_dict[node] - self.lt_dict[node]
        if SI_ub - SI_lb > self.time_unit:
            to_test_SI = np.arange(SI_lb, SI_ub + self.time_unit, self.time_unit)
        elif SI_ub - SI_lb >= 0:
            to_test_SI = [SI_lb, SI_ub]
        else:
            raise ValueError('dp bound error')

        for SI in to_test_SI:
            CT = SI + self.lt_dict[node] - S
            self.cost_record[node][S, SI] = self.on_hand_cost[node, CT]
            if len(self.sub_pred_dict[node]) > 0:
                for pred in self.sub_pred_dict[node]:
                    self.cost_record[node][S, SI] += min([self.f_cost[pred, s] for s in self.S_index[pred] if s <= SI])
            if len(self.sub_succ_dict[node]) > 0:
                for succ in self.sub_succ_dict[node]:
                    self.cost_record[node][S, SI] += min(
                        [self.g_cost[succ, si] for si in self.SI_index[succ] if si >= S])
        cost_SI_dict = {si: self.cost_record[node][S, si] for si in to_test_SI}
        best_SI = min(cost_SI_dict, key=cost_SI_dict.get)
        self.f_cost[node, S] = cost_SI_dict[best_SI]
        self.f_argmin[node, S] = best_SI

    def evaluate_g_node(self, node, SI):
        S_ub = min(self.s_ub_dict[node], SI + self.lt_dict[node])
        if S_ub > self.time_unit:
            to_test_S = np.arange(0., S_ub + self.time_unit, self.time_unit)
        else:
            to_test_S = [0., S_ub]

        for S in to_test_S:
            CT = SI + self.lt_dict[node] - S
            self.cost_record[node][S, SI] = self.on_hand_cost[node, CT]
            if len(self.sub_pred_dict[node]) > 0:
                for pred in self.sub_pred_dict[node]:
                    self.cost_record[node][S, SI] += min([self.f_cost[pred, s] for s in self.S_index[pred] if s <= SI])
            if len(self.sub_succ_dict[node]) > 0:
                for succ in self.sub_succ_dict[node]:
                    self.cost_record[node][S, SI] += min(
                        [self.g_cost[succ, si] for si in self.SI_index[succ] if si >= S])
        cost_S_dict = {s: self.cost_record[node][s, SI] for s in to_test_S}
        best_S = min(cost_S_dict, key=cost_S_dict.get)
        self.g_cost[node, SI] = cost_S_dict[best_S]
        self.g_argmin[node, SI] = best_S

    def sort(self):
        un_di_graph = self.graph.to_undirected()
        nodes_num = len(un_di_graph.all_nodes)
        labeled_list = []
        while len(labeled_list) < nodes_num:
            border_nodes = [node for node, degree in un_di_graph.degree_dict.items() if degree <= 1]
            labeled_list.extend(border_nodes)
            un_di_graph = un_di_graph.remove_nodes(border_nodes)
        return labeled_list

    def get_parent_dict(self):
        un_di_graph = self.graph.to_undirected()
        labeled_dict = {node: i for i, node in enumerate(self.sorted_list)}
        parent_dict = {}
        for node in self.sorted_list:
            c = 0
            for neighbor in un_di_graph.adj_dict[node]:
                if labeled_dict[neighbor] > labeled_dict[node]:
                    c += 1
                    parent_dict[node] = neighbor
            if c > 1:
                raise Exception('wrong label')
        return parent_dict

    def classify_node(self):
        un_di_graph = self.graph.to_undirected()
        labeled_dict = {node: i for i, node in enumerate(self.sorted_list)}
        to_eva_fk_list = []
        to_eva_gk_list = []
        for node in self.sorted_list:
            for neighbor in un_di_graph.adj_dict[node]:
                if labeled_dict[neighbor] > labeled_dict[node]:
                    if neighbor in self.succ_dict[node]:
                        to_eva_fk_list.append(node)
                    elif neighbor in self.pred_dict[node]:
                        to_eva_gk_list.append(node)
                    else:
                        raise Exception('wrong')
        sub_pred_dict = {node: [p for p in self.pred_dict[node] if labeled_dict[p] < labeled_dict[node]]
                         for node in self.sorted_list}
        sub_succ_dict = {node: [s for s in self.succ_dict[node] if labeled_dict[s] < labeled_dict[node]]
                         for node in self.sorted_list}
        return to_eva_fk_list, to_eva_gk_list, sub_pred_dict, sub_succ_dict

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit}
        return paras
