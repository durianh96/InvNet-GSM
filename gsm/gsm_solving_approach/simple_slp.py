import multiprocess as mp
from gsm.gsm_solving_approach.base_slp import *
from gsm.gsm_solving_approach.solving_default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, LOCAL_SOL_NUM


class SimpleSLP(BaseSLP):
    def __init__(self, gsm_instance: GSMInstance,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 local_sol_num=LOCAL_SOL_NUM,
                 bound_value_type=BOUND_VALUE_TYPE):
        super().__init__(gsm_instance, termination_parm, opt_gap, max_iter_num, bound_value_type)
        self.nodes = gsm_instance.nodes
        self.local_sol_num = local_sol_num

        self.need_solver = True

    @timer
    def get_policy(self, solver):
        for i in range(self.local_sol_num):
            sol = self.get_one_random_sol(solver)
            self.gsm_sol_set.add_one_sol(sol)
        simple_slp_sol = self.gsm_sol_set.get_best_local_sol()

        oul_of_node = {node: self.get_db_value_of_node(node, simple_slp_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, simple_slp_sol.CT_of_node[node]) for node in self.nodes}
        method = 'Simple-SLP_' + solver
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

        simple_slp_sol.update_oul(oul_of_node)
        simple_slp_sol.update_ss(ss_of_node)
        simple_slp_sol.update_ss_cost(ss_cost)
        return simple_slp_sol

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm, 'opt_gap': self.opt_gap, 'max_iter_num': self.max_iter_num,
                 'local_sol_num': self.local_sol_num, 'bound_value_type': self.bound_value_type}
        return paras


class ParallelSimpleSLP(BaseSLP):
    def __init__(self, gsm_instance: GSMInstance,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 local_sol_num=LOCAL_SOL_NUM,
                 bound_value_type=BOUND_VALUE_TYPE):
        super().__init__(gsm_instance, termination_parm, opt_gap, max_iter_num, bound_value_type)
        self.nodes = gsm_instance.nodes
        self.local_sol_num = local_sol_num

    @timer
    def get_policy(self, solver):
        m = BaseSLP(self.gsm_instance, self.graph, self.termination_parm, self.opt_gap, self.max_iter_num)
        pool = mp.Pool(processes=8)
        sol_results = pool.map(m.get_one_random_sol, [solver] * self.local_sol_num)
        pool.close()
        pool.join()
        for sol in sol_results:
            self.gsm_sol_set.add_one_sol(sol)
        simple_slp_sol = self.gsm_sol_set.get_best_local_sol()

        oul_of_node = {node: self.get_db_value_of_node(node, simple_slp_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, simple_slp_sol.CT_of_node[node]) for node in self.nodes}
        method = 'Simple-SLP_' + solver
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

        simple_slp_sol.update_oul(oul_of_node)
        simple_slp_sol.update_ss(ss_of_node)
        simple_slp_sol.update_ss_cost(ss_cost)
        return simple_slp_sol

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm, 'opt_gap': self.opt_gap, 'max_iter_num': self.max_iter_num,
                 'local_sol_num': self.local_sol_num, 'bound_value_type': self.bound_value_type}
        return paras
