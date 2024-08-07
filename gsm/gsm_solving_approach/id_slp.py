from gsm.gsm_solving_approach.dp import DynamicProgramming
from gsm.gsm_solving_approach.base_slp import BaseSLP
from gsm.gsm_solving_approach.id import *
from utils.system_utils import *
from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, LOCAL_SOL_NUM_ID, \
    STABILITY_TYPE, STABILITY_THRESHOLD, SOLVER, CPU_PROCESS_TYPE_ID, TREE_BASE_ALGO_ID, MAX_DECOMPOSE_NUM_ID
from gsm.gsm_instance import GSMInstance

class IterativeDecompositionSLP(IterativeDecomposition):
    def __init__(self, gsm_instance: GSMInstance,
                 local_sol_num=LOCAL_SOL_NUM_ID,
                 stability_type=STABILITY_TYPE,
                 stability_threshold=STABILITY_THRESHOLD,
                 bound_value_type=BOUND_VALUE_TYPE,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 solver=SOLVER,
                 cpu_process_type=CPU_PROCESS_TYPE_ID,
                 tree_base_algo=TREE_BASE_ALGO_ID,
                 max_decompose_num=MAX_DECOMPOSE_NUM_ID):
        super().__init__(gsm_instance, local_sol_num, stability_type, stability_threshold, bound_value_type)
        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num
        self.solver = solver
        self.cpu_process_type = cpu_process_type
        self.tree_base_algo = tree_base_algo
        self.max_decompose_num = max_decompose_num
        self.need_solver = True

    @timer
    def get_policy(self, solver):
        # initinalize upper bound
        init_s_ub_dict = {node: min(self.gsm_instance.sla_of_node.get(node, 9999), self.gsm_instance.cum_lt_of_node[node])
                        for node in self.gsm_instance.nodes}
        start_slp_model = LocalSLP(
            gsm_instance=self.gsm_instance,
            local_sol_num=self.local_sol_num,
            stability_type=self.stability_type,
            stability_threshold=self.stability_threshold,
            bound_value_type=self.bound_value_type,
            termination_parm=self.termination_parm,
            opt_gap=self.opt_gap,
            max_iter_num=self.max_iter_num,
            solver=solver,
            decompose_round=1,
            input_s_ub_dict=init_s_ub_dict
        )
        self.to_run_pool.append(start_slp_model)

        while len(self.to_run_pool) > 0:
            single_slp_model = self.to_run_pool.pop(0)
            single_slp_model.run(self.tree_base_algo, self.cpu_process_type, self.max_decompose_num)
            self.solved_pool.append(single_slp_model)
            self.sol['S'].update(single_slp_model.sol['S'])
            self.sol['SI'].update(single_slp_model.sol['SI'])
            self.sol['CT'].update(single_slp_model.sol['CT'])
            if single_slp_model.status == 'SPLIT':
                self.to_run_pool.extend(single_slp_model.sub_pool)

        error_sol = check_solution_feasibility(self.gsm_instance, self.sol)
        if len(error_sol) > 0:
            logger.error(error_sol)
        
        id_slp_sol = GSMSolution(nodes=self.nodes)
        id_slp_sol.update_sol(self.sol)

        oul_of_node = {node: self.get_db_value_of_node(node, id_slp_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, id_slp_sol.CT_of_node[node]) for node in self.nodes}
        method = 'ID-SLP_' + solver + '_' + self.stability_type
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

        id_slp_sol.update_oul(oul_of_node)
        id_slp_sol.update_ss(ss_of_node)
        id_slp_sol.update_ss_cost(ss_cost)
        return id_slp_sol

    def get_approach_paras(self):
        paras = {'local_sol_num': self.local_sol_num,
                 'stability_type': self.stability_type,
                 'stability_threshold': self.stability_threshold,
                 'bound_value_type': self.bound_value_type,
                 'termination_parm': self.termination_parm,
                 'opt_gap': self.opt_gap,
                 'max_iter_num': self.max_iter_num,
                 'solver': self.solver
                 }
        return paras


class LocalSLP(LocalApproach):
    def __init__(self, gsm_instance: GSMInstance,
                 local_sol_num,
                 stability_type,
                 stability_threshold,
                 bound_value_type,
                 termination_parm,
                 opt_gap,
                 max_iter_num,
                 solver,
                 decompose_round,
                 input_s_ub_dict: Optional[dict] = None, input_si_lb_dict: Optional[dict] = None):
        super().__init__(gsm_instance, local_sol_num, stability_type, stability_threshold, bound_value_type,
                         input_s_ub_dict, input_si_lb_dict)

        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num
        self.solver = solver
        self.decompose_round = decompose_round

        self.simple_sol = None
        self.local_sol_info = None

        self.sub_pool = []

    def run(self, tree_base_algo, cpu_process_type, max_decompose_num, process_num=None):
        if self.stability_type == 'kl' or 'cn':
            self.gsm_sol_set.init_beta_para()
        slp_problem = BaseSLP(
            gsm_instance=self.gsm_instance,
            termination_parm=self.termination_parm,
            opt_gap=self.opt_gap,
            max_iter_num=self.max_iter_num,
            bound_value_type=self.bound_value_type,
            input_s_ub_dict=self.input_s_ub_dict,
            input_si_lb_dict=self.input_si_lb_dict
        )

        if cpu_process_type == 'PARALLEL':
            # parallel processing
            import multiprocessing as mp
            mp.set_start_method('fork', force=True)
            manager = mp.Manager()
            shared_gsm_sol_pool = manager.list()
            process_pool = []
            process_num = min(int(mp.cpu_count() * 0.5), self.local_sol_num) if process_num is None else process_num
            parallel_sol_num = np.ceil(self.local_sol_num / process_num).astype(int)
            for _ in range(process_num):
                process = mp.Process(target=slp_problem.get_multiple_random_sol, 
                                    args=('GRB', parallel_sol_num, shared_gsm_sol_pool, ))
                process_pool.append(process)
                process.start()
            for process in process_pool:
                process.join(timeout = None)
            for sol in shared_gsm_sol_pool:
                self.gsm_sol_set.add_one_sol(sol)
        else:
            for _ in range(self.local_sol_num):
                once_slp_sol = slp_problem.get_one_random_sol(self.solver)
                if self.stability_type == 'cv':
                    self.gsm_sol_set.add_one_sol(once_slp_sol)
                else:
                    self.gsm_sol_set.add_one_sol(once_slp_sol, update_beta=True)

        self.simple_sol = self.gsm_sol_set.get_best_local_sol()
        if self.decompose_round <= max_decompose_num:
            self.local_sol_info = self.gsm_sol_set.get_local_sol_info(self.stability_type, self.stability_threshold)
            self.get_sub_pool(tree_base_algo)
        else:
            self.status = 'SOLVED'
            self.sol['S'].update(self.simple_sol.S_of_node)
            self.sol['SI'].update(self.simple_sol.SI_of_node)
            self.sol['CT'].update(self.simple_sol.CT_of_node)

    def get_sub_pool(self, tree_base_algo):
        # update global bound
        sub_gsm_instance_list, cur_s_ub_dict, cur_si_lb_dict = self.gsm_instance.get_sub_instances(self.local_sol_info)
        s_ub_dict = {node: min(cur_s_ub_dict.get(node, 999), self.input_s_ub_dict.get(node, 999))
                    for node in set(cur_s_ub_dict.keys()) | set(self.input_s_ub_dict.keys())}
        si_lb_dict = {node: max(cur_si_lb_dict.get(node, 0), self.input_si_lb_dict.get(node, 0))
                    for node in set(cur_si_lb_dict.keys()) | set(self.input_si_lb_dict.keys())}
        if len(sub_gsm_instance_list) > 1:
            self.status = 'SPLIT'
            # update stable nodes
            stable_nodes = self.local_sol_info['completely_fix_nodes']
            for sub_gsm_instance in sub_gsm_instance_list:
                sub_s_ub = {node: s_ub_dict[node] for node in sub_gsm_instance.nodes if node in s_ub_dict.keys()}
                sub_si_lb = {node: si_lb_dict[node] for node in sub_gsm_instance.nodes if node in si_lb_dict.keys()}
                if len(sub_gsm_instance.nodes) == 1:
                    single_node = list(sub_gsm_instance.nodes)[0]
                    if single_node in stable_nodes:
                        self.sol['S'][single_node] = self.local_sol_info['completely_fix_S'][single_node]
                        self.sol['SI'][single_node] = self.local_sol_info['completely_fix_SI'][single_node]
                        self.sol['CT'][single_node] = self.local_sol_info['completely_fix_CT'][single_node]
                    else:
                        self.sol['S'][single_node] = sub_s_ub.get(single_node, 999)
                        self.sol['SI'][single_node] = sub_si_lb.get(single_node, 0)
                        self.sol['CT'][single_node] = max(0, self.sol['SI'][single_node]
                                                          + sub_gsm_instance.lt_of_node[single_node]
                                                          - self.sol['S'][single_node])
                elif (sub_gsm_instance.graph_type == 'TREE') & (tree_base_algo == 'DP'):
                    sub_dp_model = DynamicProgramming(
                        gsm_instance=sub_gsm_instance,
                        bound_value_type=self.bound_value_type,
                        input_s_ub_dict=sub_s_ub,
                        input_si_lb_dict=sub_si_lb
                    )
                    sub_policy = sub_dp_model.get_policy()
                    self.sol['S'].update(sub_policy.S_of_node)
                    self.sol['SI'].update(sub_policy.SI_of_node)
                    self.sol['CT'].update(sub_policy.CT_of_node)
                else:
                    sub_slp_model = LocalSLP(
                        gsm_instance=sub_gsm_instance,
                        local_sol_num=self.local_sol_num,
                        stability_type=self.stability_type,
                        stability_threshold=self.stability_threshold,
                        bound_value_type=self.bound_value_type,
                        termination_parm=self.termination_parm,
                        opt_gap=self.opt_gap,
                        max_iter_num=self.max_iter_num,
                        solver=self.solver,
                        decompose_round=self.decompose_round+1,
                        input_s_ub_dict=sub_s_ub,
                        input_si_lb_dict=sub_si_lb
                    )
                    self.sub_pool.append(sub_slp_model)
        else:
            self.status = 'SOLVED'
            self.sol['S'].update(self.simple_sol.S_of_node)
            self.sol['SI'].update(self.simple_sol.SI_of_node)
            self.sol['CT'].update(self.simple_sol.CT_of_node)

