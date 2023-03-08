import math
import random
from copy import deepcopy
from utils.system_utils import *
from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import BOUND_VALUE_TYPE
from gsm.gsm_instance import GSMInstance
from utils.graph_algorithms import *


class SimulatedAnnealing:
    def __init__(self, gsm_instance: GSMInstance,
                 bound_value_type=BOUND_VALUE_TYPE):
        self.gsm_instance = gsm_instance

        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.roots = gsm_instance.roots
        self.sinks = gsm_instance.sinks

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.preds_of_node = gsm_instance.preds_of_node
        self.succs_of_node = gsm_instance.succs_of_node

        self.bound_value_type = bound_value_type

        self.get_all_path_of_node()
        self.get_init_sol()

        self.need_solver = False

    def get_all_path_of_node(self):
        self.paths = {node: [] for node in self.nodes}
        for node in self.nodes:
            for root_node in self.roots:
                self.paths[node].extend(find_all_paths(succ_of_node=self.succs_of_node,
                                                       u=root_node,
                                                       v=node))

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

    def initial_time_value_of_node(self):
        self.L_value = {node: 0 for node in self.nodes}
        self.M_value = {node: 0 for node in self.nodes}
        self.CT_value = {node: 0 for node in self.nodes}
        self.Z_value = {node: 0 for node in self.nodes}

        reverse_edges = [(j, i) for i, j in self.edges]
        self.reverse_topo_sort = find_topo_sort(reverse_edges)
        self.topo_sort = find_topo_sort(self.edges)

        self.L_value.update({node: self.lt_of_node[node] for node in self.roots})
        self.M_value.update({node: self.sla_of_node[node] for node in self.sinks})

    def cal_time_value_step(self, input_Z):
        step_L = deepcopy(self.L_value)
        step_M = deepcopy(self.M_value)
        step_CT = deepcopy(self.CT_value)
        for node in self.topo_sort:
            if node in self.roots:
                continue
            else:
                step_L.update({node: self.lt_of_node[node] + max(
                    [step_L[j] * (1 - input_Z[j]) for j in self.preds_of_node[node]])})

        step_CT.update({node: input_Z[node] * max(step_L[node] - step_M[node], 0) for node in self.sinks})

        for node in self.reverse_topo_sort:
            if node in self.sinks:
                continue
            else:
                step_M.update(
                    {node: min([step_CT[j] - self.lt_of_node[j] + step_M[j] for j in self.succs_of_node[node]])})
                step_CT.update({node: input_Z[node] * max(step_L[node] - step_M[node], 0)})

        step_ss_cost = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_CT[node]) for node in self.nodes])

        sol = {'Z': input_Z, 'CT': step_CT, 'obj_val': step_ss_cost}
        return sol

    def get_init_sol(self):
        self.initial_time_value_of_node()
        Z1_dict = {node: 1 for node in self.nodes}
        Z2_dict = {node: 0 for node in self.nodes}
        Z2_dict.update({node: 1 for node in self.roots + self.sinks})
        Z3_dict = {node: 0 for node in self.nodes}
        Z3_dict.update({node: 1 for node in self.nodes if self.succs_of_node[node] == set()})

        Z1_ss_cost = self.cal_time_value_step(input_Z=Z1_dict)['obj_val']
        Z2_ss_cost = self.cal_time_value_step(input_Z=Z2_dict)['obj_val']
        Z3_ss_cost = self.cal_time_value_step(input_Z=Z3_dict)['obj_val']

        Z_dict = [Z1_dict, Z2_dict, Z3_dict]
        costs = [Z1_ss_cost, Z2_ss_cost, Z3_ss_cost]

        opt_index = costs.index(min(costs))
        self.init_Z = Z_dict[opt_index]
        self.init_CT = self.cal_time_value_step(input_Z=self.init_Z)['CT']
        init_ss_of_node = {node: self.get_vb_value_of_node(node, self.init_CT[node]) for node in self.nodes}
        init_char = 'SA_Initial'
        self.init_ss_cost = cal_ss_cost(self.hc_of_node, init_ss_of_node, method=init_char)

    def check_path_constraints(self, input_CT):
        check_flag = True
        for node in self.nodes:
            node_paths = self.paths[node]
            if len(node_paths):
                constr1_lhs = max([sum([self.lt_of_node[j] - input_CT[j] for j in path]) for path in node_paths])
                constr1_rhs = 0
                if constr1_lhs < constr1_rhs:
                    check_flag = False
                    return check_flag
                if node in self.sinks:
                    for path in node_paths:
                        constr2_lhs = sum([input_CT[j] for j in path])
                        constr2_rhs = sum([self.lt_of_node[j] for j in path]) - sum(
                            [self.sla_of_node[i] for i in path if i in self.sinks])
                        if constr2_lhs < constr2_rhs:
                            check_flag = False
                            return check_flag
        return check_flag

    class SimulatedAnnealing:
        def __init__(self):
            self.sigma = 0.8
            self.init_sample = 100
            self.iter_round = 0
            self.non_improve_counter = 0
            self.cooling_parm = 0
            self.cooling_scale = 0.9
            self.worse_permit_parm = 0
            self.max_stay_counter = 100
            self.violate_sol = []

        def get_neighbors_of_sol(self, sol):
            sol_index = list(sol.keys())
            temp_sols = []
            for i in range(self.init_sample):
                flip_index = random.choice(sol_index)
                ngbr_sol = deepcopy(sol)
                ngbr_sol.update({flip_index: 1 - sol[flip_index]})
                temp_sols.append(ngbr_sol)
            return temp_sols

        def cool_down(self):
            self.iter_round += 1
            self.cooling_parm = self.cooling_parm * self.cooling_scale

        def local_search(self, sol):
            sol_index = list(sol.keys())
            flip_prob = 1 / len(sol_index)
            flip_sol = deepcopy(sol)
            flip_sol.update(
                {node: np.random.choice([1 - sol[node], sol[node]], p=[flip_prob, 1 - flip_prob]) for node in
                 sol_index})
            while (flip_sol in self.violate_sol) or flip_sol == sol:
                flip_sol.update(
                    {node: np.random.choice([1 - sol[node], sol[node]], p=[flip_prob, 1 - flip_prob]) for node in
                     sol_index})
            return flip_sol

        def update_worse_permit_parm(self, diff):
            self.worse_permit_parm = math.exp(- diff / self.cooling_parm) if (diff > 0) & (self.cooling_parm > 0) else 0

    def run_SimulatedAnnealing(self):
        SA = self.SimulatedAnnealing()
        ngbr_of_init_Z = SA.get_neighbors_of_sol(sol=self.init_Z)
        ngbr_sols = [self.cal_time_value_step(input_Z=Z) for Z in ngbr_of_init_Z]
        for sol in ngbr_sols:
            if self.check_path_constraints(input_CT=sol['CT']) == False:
                SA.init_sample -= 1
                sol['obj_val'] = 0
        init_diff = sum([(sol['obj_val'] - self.init_ss_cost) for sol in ngbr_sols]) / SA.init_sample
        SA.cooling_parm = - init_diff / math.log(SA.sigma)

        opt_Z = self.init_Z
        cur_Z = deepcopy(opt_Z)
        opt_cost = self.init_ss_cost
        cur_cost = opt_cost
        while (SA.non_improve_counter <= SA.max_stay_counter):
            new_Z = SA.local_search(sol=cur_Z)
            new_sol = self.cal_time_value_step(input_Z=new_Z)
            if self.check_path_constraints(input_CT=new_sol['CT']) == False:
                SA.violate_sol.append(new_Z)
                continue
            new_cost = new_sol['obj_val']
            delta = new_cost - cur_cost
            SA.cool_down()
            SA.non_improve_counter = SA.non_improve_counter + 1 if new_cost >= opt_cost else 0
            SA.update_worse_permit_parm(diff=delta)
            if (delta < 0) or (SA.worse_permit_parm > random.random()):
                cur_Z = deepcopy(new_Z)
                cur_cost = new_cost
                if cur_cost < opt_cost:
                    opt_Z = deepcopy(cur_Z)
                    opt_cost = cur_cost
        return opt_Z

    @timer
    def get_policy(self):
        opt_Z = self.run_SimulatedAnnealing()
        opt_sol = self.cal_time_value_step(input_Z=opt_Z)
        self.CT_value.update(opt_sol['CT'])
        sol = {'S': {}, 'SI': {}, 'CT': self.CT_value}
        mh_sol = GSMSolution(nodes=self.nodes)
        mh_sol.update_sol(sol)

        oul_of_node = {node: self.get_db_value_of_node(node, mh_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, mh_sol.CT_of_node[node]) for node in self.nodes}
        method = 'SA'
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

        mh_sol.update_oul(oul_of_node)
        mh_sol.update_ss(ss_of_node)
        mh_sol.update_ss_cost(ss_cost)
        return mh_sol

    def get_approach_paras(self):
        paras = {'bound_value_type': self.bound_value_type}
        return paras
