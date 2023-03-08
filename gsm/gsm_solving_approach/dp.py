from collections import defaultdict
from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import BOUND_VALUE_TYPE, SYSTEM_TIME_UNIT
from gsm.gsm_instance import GSMInstance
from utils.graph_algorithms import find_adjs_of_node


class DynamicProgramming:
    def __init__(self, gsm_instance: GSMInstance,
                 input_s_ub_dict=None,
                 input_si_lb_dict=None,
                 time_unit=SYSTEM_TIME_UNIT,
                 bound_value_type=BOUND_VALUE_TYPE):
        if gsm_instance.graph_type != 'TREE':
            raise AttributeError('Graph is not tree')
        self.gsm_instance = gsm_instance
        if input_si_lb_dict is None:
            input_si_lb_dict = {}
        if input_s_ub_dict is None:
            input_s_ub_dict = {}

        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.sinks = gsm_instance.sinks

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.cum_lt_of_node = gsm_instance.cum_lt_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.preds_of_node = gsm_instance.preds_of_node
        self.succs_of_node = gsm_instance.succs_of_node

        self.time_unit = time_unit
        self.bound_value_type = bound_value_type

        self.s_ub_dict = {node: min(self.sla_of_node.get(node, 9999), input_s_ub_dict.get(node, 9999),
                                    self.cum_lt_of_node[node]) for node in self.nodes}

        self.si_lb_dict = {node: input_si_lb_dict.get(node, 0) for node in self.nodes}
        self.si_ub_dict = {node: self.cum_lt_of_node[node] - self.lt_of_node[node] for node in self.nodes}

        self.S_index = {}
        self.SI_index = {}
        for node in self.nodes:
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

        self.ss_ct_dict = {(node, ct): self.get_vb_value_of_node(node, ct) for node in self.nodes
                           for ct in np.arange(0., self.cum_lt_of_node[node] + self.time_unit, self.time_unit)}

        self.on_hand_cost = {(node, ct): self.hc_of_node[node] * self.ss_ct_dict[node, ct] for node in self.nodes
                             for ct in np.arange(0., self.cum_lt_of_node[node] + self.time_unit, self.time_unit)}

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

    # @timer
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
                node_SI = max(opt_sol['S'][parent_node], min(self.SI_index[node]))
                node_S = self.g_argmin[node, node_SI]
            elif node in self.sub_pred_dict[parent_node]:
                node_S = min(opt_sol['SI'][parent_node], max(self.S_index[node]))
                node_SI = self.f_argmin[node, node_S]
            else:
                raise Exception
            opt_sol['S'][node] = node_S
            opt_sol['SI'][node] = node_SI

        opt_sol['CT'] = {node: opt_sol['SI'][node] + self.lt_of_node[node] - opt_sol['S'][node]
                         for node in self.sorted_list}
        opt_sol['obj_value'] = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, opt_sol['CT'][node]) for node in self.nodes])

        dp_sol = GSMSolution(nodes=self.nodes)
        dp_sol.update_sol(opt_sol)
        oul_of_node = {node: self.get_db_value_of_node(node, dp_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, dp_sol.CT_of_node[node]) for node in self.nodes}
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method='DP', w=False)

        dp_sol.update_oul(oul_of_node)
        dp_sol.update_ss(ss_of_node)
        dp_sol.update_ss_cost(ss_cost)
        return dp_sol

    def evaluate_f_node(self, node, S):
        # The above code is calculating the f_cost and f_argmin for each node and S.
        SI_lb = max(self.si_lb_dict[node], S - self.lt_of_node[node])
        SI_ub = self.cum_lt_of_node[node] - self.lt_of_node[node]
        if SI_ub - SI_lb > self.time_unit:
            to_test_SI = np.arange(SI_lb, SI_ub + self.time_unit, self.time_unit)
        elif SI_ub - SI_lb >= 0:
            to_test_SI = [SI_lb, SI_ub]
        else:
            raise ValueError('dp bound error')

        for SI in to_test_SI:
            CT = SI + self.lt_of_node[node] - S
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
        S_ub = min(self.s_ub_dict[node], SI + self.lt_of_node[node])
        if S_ub > self.time_unit:
            to_test_S = np.arange(0., S_ub + self.time_unit, self.time_unit)
        else:
            to_test_S = [0., S_ub]

        for S in to_test_S:
            CT = SI + self.lt_of_node[node] - S
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
        un_di_graph = UnDiGraph(self.nodes, self.edges)
        nodes_num = len(un_di_graph.nodes)
        labeled_list = []
        while len(labeled_list) < nodes_num:
            border_nodes = [node for node, degree in un_di_graph.degree_of_node.items() if degree <= 1]
            labeled_list.extend(border_nodes)
            un_di_graph = un_di_graph.remove_nodes(border_nodes)
        return labeled_list

    def get_parent_dict(self):
        un_di_graph = UnDiGraph(self.nodes, self.edges)
        labeled_dict = {node: i for i, node in enumerate(self.sorted_list)}
        parent_dict = {}
        for node in self.sorted_list:
            c = 0
            for neighbor in un_di_graph.adjs_of_node[node]:
                if labeled_dict[neighbor] > labeled_dict[node]:
                    c += 1
                    parent_dict[node] = neighbor
            if c > 1:
                raise Exception('wrong label')
        return parent_dict

    def classify_node(self):
        """
        For each node in the sorted list, we find all its neighbors that are labeled with a larger label. 
        If the neighbor is in the successor list of the node, we add the node to the to_eva_fk_list. 
        If the neighbor is in the predecessor list of the node, we add the node to the to_eva_gk_list.
        :return: The return value is a tuple of four lists.
        """
        un_di_graph = UnDiGraph(self.nodes, self.edges)
        labeled_dict = {node: i for i, node in enumerate(self.sorted_list)}
        to_eva_fk_list = []
        to_eva_gk_list = []
        for node in self.sorted_list:
            for neighbor in un_di_graph.adjs_of_node[node]:
                if labeled_dict[neighbor] > labeled_dict[node]:
                    if neighbor in self.succs_of_node[node]:
                        to_eva_fk_list.append(node)
                    elif neighbor in self.preds_of_node[node]:
                        to_eva_gk_list.append(node)
                    else:
                        raise Exception('wrong')
        sub_pred_dict = {node: [p for p in self.preds_of_node[node] if labeled_dict[p] < labeled_dict[node]]
                         for node in self.sorted_list}
        sub_succ_dict = {node: [s for s in self.succs_of_node[node] if labeled_dict[s] < labeled_dict[node]]
                         for node in self.sorted_list}
        return to_eva_fk_list, to_eva_gk_list, sub_pred_dict, sub_succ_dict

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit, 'bound_value_type': self.bound_value_type}
        return paras


class UnDiGraph:
    def __init__(self, nodes: set, edges: list):
        self.nodes = nodes
        self.edges = edges
        if len(self.edges):
            self.adjs_of_node = find_adjs_of_node(edges)
        else:
            self.adjs_of_node = {node: {} for node in self.nodes}

        self.degree_of_node = {j: len(self.adjs_of_node[j]) for j in nodes}

    def is_fully_connected(self):
        visited = set()

        def _dfs(node):
            if node not in visited:
                visited.add(node)
                for adj in self.adjs_of_node[node]:
                    _dfs(adj)

        _dfs(list(self.nodes)[0])

        return len(visited) == len(self.nodes)

    def remove_nodes(self, rm_nodes: list):
        new_all_nodes = self.nodes - set(rm_nodes)
        new_edge_list = [(i, j) for i, j in self.edges if ((i in new_all_nodes) and (j in new_all_nodes))]
        new_un_di_graph = UnDiGraph(new_all_nodes, new_edge_list)
        return new_un_di_graph

    def cycle_detect_from_source(self, source):
        visited = set()

        def _dfs(u, parent=0):
            visited.add(u)
            for v in self.adjs_of_node[u]:
                if v not in visited:
                    if _dfs(v, u):
                        return True
                elif v != parent:
                    return True
            return False

        return _dfs(source)
