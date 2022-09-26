from approach.dp import DynamicProgramming
from approach.base_slp import *
from utils.copt_pyomo import *
from utils.gsm_utils import *
from utils.utils import *
from domain.policy import Policy
from default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, LOCAL_SOL_NUM_ID, STABILITY_THRESHOLD


class IterativeDecompositionSLP(object):
    def __init__(self, gsm_instance, termination_parm=TERMINATION_PARM, opt_gap=OPT_GAP, max_iter_num=MAX_ITER_NUM,
                 local_sol_num=LOCAL_SOL_NUM_ID, stability_threshold=STABILITY_THRESHOLD):
        self.all_nodes = gsm_instance.all_nodes
        self.gsm_instance = gsm_instance
        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num
        self.local_sol_num = local_sol_num
        self.stability_threshold = stability_threshold

        self.to_run_slp_pool = []
        self.solved_slp_pool = []

        self.sol = {'S': {}, 'SI': {}, 'CT': {}}

        self.need_solver = True

    @timer
    def get_policy(self, solver):
        start_slp_model = SingleSLP(
            gsm_instance=self.gsm_instance,
            status='TO_RUN_SLP',
            termination_parm=self.termination_parm,
            opt_gap=self.opt_gap,
            max_iter_num=self.max_iter_num,
            local_sol_num=self.local_sol_num
        )
        self.to_run_slp_pool.append(start_slp_model)

        while len(self.to_run_slp_pool) > 0:
            single_slp_model = self.to_run_slp_pool.pop(0)
            single_slp_model.run_slp(solver, self.stability_threshold)
            self.solved_slp_pool.append(single_slp_model)
            self.sol['S'].update(single_slp_model.sol['S'])
            self.sol['SI'].update(single_slp_model.sol['SI'])
            self.sol['CT'].update(single_slp_model.sol['CT'])
            if single_slp_model.status == 'SPLIT':
                self.to_run_slp_pool.extend(single_slp_model.sub_slp_pool)

        error_sol = check_solution_feasibility(self.gsm_instance, self.sol)
        if len(error_sol) > 0:
            logger.error(error_sol)

        bs_dict = {node: self.gsm_instance.db_func[node](self.sol['CT'][node]) for node in self.gsm_instance.all_nodes}
        ss_dict = {node: self.gsm_instance.vb_func[node](self.sol['CT'][node]) for node in self.gsm_instance.all_nodes}
        method = 'ID-SLP_' + solver
        cost = cal_cost(self.gsm_instance.hc_dict, ss_dict, method=method)

        policy = Policy(self.all_nodes)
        policy.update_sol(self.sol)
        policy.update_base_stock(bs_dict)
        policy.update_safety_stock(ss_dict)
        policy.update_ss_cost(cost)
        return policy

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm, 'opt_gap': self.opt_gap, 'max_iter_num': self.max_iter_num,
                 'local_sol_num': self.local_sol_num, 'stability_threshold': self.stability_threshold}
        return paras


class SingleSLP(BaseSLP):
    def __init__(self, gsm_instance, status, termination_parm=TERMINATION_PARM, opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM, local_sol_num=LOCAL_SOL_NUM_ID,
                 input_s_ub_dict=None, input_si_lb_dict=None):
        super().__init__(gsm_instance, termination_parm, opt_gap, max_iter_num, input_s_ub_dict, input_si_lb_dict)
        self.status = status  # 'TO_RUN_SLP' / 'SPLIT' / 'TO_RUN_DP' /'SOLVED'
        self.local_sol_num = local_sol_num

        self.results = []
        self.slp_sol = None
        self.sol = {'S': {}, 'SI': {}, 'CT': {}}
        self.slp_nodes_info = None

        self.sub_slp_pool = []

    def run_slp(self, solver, stability_threshold):
        for _ in range(self.local_sol_num):
            init_CT = {j: float(random.randint(1, 150)) for j in self.all_nodes}
            self.run_one_instance(init_CT, solver)

        self.slp_sol = self.output_results()

        self.slp_nodes_info = self.get_nodes_info(stability_threshold)
        self.get_sub_pool()

    def get_sub_pool(self):
        sub_gsm_instance_list, s_ub_dict, si_lb_dict = self.gsm_instance.deal_completely_fix(
            self.slp_nodes_info)
        if len(sub_gsm_instance_list) > 1:
            self.status = 'SPLIT'
            for sub_gsm_instance in sub_gsm_instance_list:
                sub_s_ub = {node: s_ub_dict[node] for node in sub_gsm_instance.all_nodes if node in s_ub_dict.keys()}
                sub_si_lb = {node: si_lb_dict[node] for node in sub_gsm_instance.all_nodes if node in si_lb_dict.keys()}
                if len(sub_gsm_instance.all_nodes) == 1:
                    single_node = list(sub_gsm_instance.all_nodes)[0]
                    if single_node in self.slp_nodes_info['completely_fix_nodes']:
                        self.sol['S'][single_node] = self.slp_nodes_info['completely_fix_S'][single_node]
                        self.sol['SI'][single_node] = self.slp_nodes_info['completely_fix_SI'][single_node]
                        self.sol['CT'][single_node] = self.slp_nodes_info['completely_fix_CT'][single_node]
                    else:
                        self.sol['S'][single_node] = sub_s_ub.get(single_node, 999)
                        self.sol['SI'][single_node] = sub_si_lb.get(single_node, 0)
                        self.sol['CT'][single_node] = max(0, self.sol['SI'][single_node]
                                                          + sub_gsm_instance.lt_dict[single_node]
                                                          - self.sol['S'][single_node])
                elif sub_gsm_instance.graph.is_tree():
                    sub_dp_model = DynamicProgramming(
                        gsm_instance=sub_gsm_instance,
                        input_s_ub_dict=sub_s_ub,
                        input_si_lb_dict=sub_si_lb
                    )
                    sub_policy = sub_dp_model.get_policy()
                    self.sol['S'].update(sub_policy.sol_S)
                    self.sol['SI'].update(sub_policy.sol_SI)
                    self.sol['CT'].update(sub_policy.sol_CT)
                else:
                    sub_slp_model = SingleSLP(
                        gsm_instance=sub_gsm_instance,
                        status='TO_RUN_SLP',
                        termination_parm=self.termination_parm,
                        opt_gap=self.opt_gap,
                        max_iter_num=self.max_iter_num,
                        local_sol_num=self.local_sol_num,
                        input_s_ub_dict=sub_s_ub,
                        input_si_lb_dict=sub_si_lb
                    )
                    self.sub_slp_pool.append(sub_slp_model)
        else:
            self.status = 'SOLVED'
            self.sol['S'].update(self.slp_sol['S'])
            self.sol['SI'].update(self.slp_sol['SI'])
            self.sol['CT'].update(self.slp_sol['CT'])
