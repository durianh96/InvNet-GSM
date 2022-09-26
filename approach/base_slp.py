import copy
import random

from default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM
from utils.copt_pyomo import *
from utils.gsm_utils import *
from utils.utils import *


class BaseSLP:
    def __init__(self, gsm_instance, termination_parm=TERMINATION_PARM, opt_gap=OPT_GAP, max_iter_num=MAX_ITER_NUM,
                 input_s_ub_dict=None, input_si_lb_dict=None):
        self.gsm_instance = gsm_instance
        self.graph = gsm_instance.graph
        self.all_nodes = gsm_instance.all_nodes
        self.demand_nodes = self.graph.demand_nodes
        self.edge_list = self.graph.edge_list
        self.lt_dict = gsm_instance.lt_dict
        self.cum_lt_dict = gsm_instance.cum_lt_dict
        self.hc_dict = gsm_instance.hc_dict
        self.sla_dict = gsm_instance.sla_dict

        self.vb_func = gsm_instance.vb_func
        self.db_func = gsm_instance.db_func
        self.grad_vb_func = gsm_instance.grad_vb_func

        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num

        if input_s_ub_dict is None:
            self.s_ub_dict = {}
        else:
            self.s_ub_dict = input_s_ub_dict
        if input_si_lb_dict is None:
            self.si_lb_dict = {}
        else:
            self.si_lb_dict = input_si_lb_dict

        self.results = []
        self.best_sol = None

    def get_one_instance_policy(self, solver):
        init_CT = {j: float(random.randint(1, 150)) for j in self.all_nodes}
        sol = self.run_one_instance(init_CT, solver)
        return sol

    def cal_para(self, CT):
        A = {node: self.hc_dict[node] * self.grad_vb_func[node](max(ctk, EPS)) for node, ctk in CT.items()}
        B = {node: self.hc_dict[node] * self.vb_func[node](max(ctk, EPS)) - A[node] * ctk for node, ctk in CT.items()}
        obj_para = {'A': A, 'B': B}
        return obj_para

    def run_one_instance(self, init_CT, solver):
        CT_step = copy.copy(init_CT)
        obj_value = [0]
        for i in range(self.max_iter_num):
            step_obj_para = self.cal_para(CT_step)
            if solver == 'GRB':
                step_sol = self.slp_step_grb(step_obj_para)
            elif solver == 'COPT':
                step_sol = self.slp_step_copt(step_obj_para)
            elif solver == 'PYO_COPT':
                step_sol = self.slp_step_pyomo(step_obj_para, pyo_solver='COPT')
            elif solver == 'PYO_GRB':
                step_sol = self.slp_step_pyomo(step_obj_para, pyo_solver='GRB')
            elif solver == 'PYO_CBC':
                step_sol = self.slp_step_pyomo(step_obj_para, pyo_solver='CBC')
            else:
                raise AttributeError('undefined solver')
            obj_value.append(step_sol['obj_value'])
            CT_step = step_sol['CT']
            if (i > 0) and (abs(obj_value[i - 1] - obj_value[i]) <= self.termination_parm):
                error_sol = check_solution_feasibility(self.gsm_instance, step_sol)
                if len(error_sol) > 0:
                    logger.error(error_sol)
                    return
                else:
                    self.results.append(step_sol)
                    return step_sol

    def slp_step_grb(self, obj_para):
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model('slp_step')
        # adding variables
        S = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0)
        SI = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0)
        CT = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in self.all_nodes))
        # sla
        m.addConstrs((S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes if j in self.sla_dict.keys()))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list))

        # s upper bound
        if len(self.s_ub_dict) > 0:
            m.addConstrs((S[j] <= ub for j, ub in self.s_ub_dict.items()))

        # si lower bound
        if len(self.si_lb_dict) > 0:
            m.addConstrs((SI[j] >= lb for j, lb in self.si_lb_dict.items()))

        m.setObjective(gp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in self.all_nodes), GRB.MINIMIZE)
        m.Params.MIPGap = self.opt_gap
        m.Params.LogToConsole = 0
        m.optimize()

        if m.status == GRB.OPTIMAL:
            step_sol = {'S': {node: float(S[node].x) for node in self.all_nodes},
                        'SI': {node: float(SI[node].x) for node in self.all_nodes},
                        'CT': {node: float(CT[node].x) for node in self.all_nodes}}
            step_sol['obj_value'] = sum(
                [self.hc_dict[node] * self.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])
            return step_sol
        elif m.status == GRB.INFEASIBLE:
            raise Exception('Infeasible model')
        elif m.status == GRB.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == GRB.TIME_LIMIT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def slp_step_copt(self, obj_para):
        import coptpy as cp
        from coptpy import COPT
        env = cp.Envr()
        m = env.createModel('slp_step')
        # adding variables
        S = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0)
        SI = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0)
        CT = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in self.all_nodes))
        # sla
        m.addConstrs((S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes if j in self.sla_dict.keys()))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list))

        # s upper bound
        if len(self.s_ub_dict) > 0:
            m.addConstrs((S[j] <= ub for j, ub in self.s_ub_dict.items()))

        # si lower bound
        if len(self.si_lb_dict) > 0:
            m.addConstrs((SI[j] >= lb for j, lb in self.si_lb_dict.items()))

        m.setObjective(cp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in self.all_nodes), COPT.MINIMIZE)
        m.setParam(COPT.Param.RelGap, self.opt_gap)
        m.setParam(COPT.Param.Logging, False)
        m.setParam(COPT.Param.LogToConsole, False)
        m.solve()

        if m.status == COPT.OPTIMAL:
            step_sol = {'S': {node: float(S[node].x) for node in self.all_nodes},
                        'SI': {node: float(SI[node].x) for node in self.all_nodes},
                        'CT': {node: float(CT[node].x) for node in self.all_nodes}}
            step_sol['obj_value'] = sum(
                [self.hc_dict[node] * self.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])
            return step_sol
        elif m.status == COPT.INFEASIBLE:
            raise Exception('Infeasible model')
        elif m.status == COPT.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == COPT.TIMEOUT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def slp_step_pyomo(self, obj_para, pyo_solver):
        import pyomo.environ as pyo
        import pyomo.opt as pyopt

        m = pyo.ConcreteModel('slp_step')
        # adding variables
        m.S = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)
        m.SI = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)
        m.CT = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in self.all_nodes:
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_dict[j] - m.S[j])

        # sla
        for j in set(self.demand_nodes) & set(self.sla_dict.keys()):
            m.constrs.add(m.S[j] <= int(self.sla_dict[j]))

        # si >= s
        for pred, succ in self.edge_list:
            m.constrs.add(m.SI[succ] - m.S[pred] >= 0)

        # s upper bound
        if len(self.s_ub_dict) > 0:
            for j, ub in self.s_ub_dict.items():
                m.constrs.add(m.S[j] <= ub)
        # si lower bound
        if len(self.si_lb_dict) > 0:
            for j, lb in self.si_lb_dict.items():
                m.constrs.add(m.SI[j] >= lb)

        m.Cost = pyo.Objective(
            expr=sum([obj_para['A'][j] * m.CT[j] + obj_para['B'][j] for j in self.all_nodes]),
            sense=pyo.minimize
        )
        if pyo_solver == 'COPT':
            solver = pyopt.SolverFactory('copt_direct')
            solver.options['RelGap'] = self.opt_gap
        elif pyo_solver == 'GRB':
            solver = pyopt.SolverFactory('gurobi_direct')
            solver.options['MIPGap'] = self.opt_gap
        elif pyo_solver == 'CBC':
            solver = pyopt.SolverFactory('cbc')
            solver.options['ratio'] = self.opt_gap
        else:
            raise AttributeError

        solver.solve(m, tee=False)

        step_sol = {'S': {node: float(m.S[node].value) for node in self.all_nodes},
                    'SI': {node: float(m.SI[node].value) for node in self.all_nodes},
                    'CT': {node: float(m.CT[node].value) for node in self.all_nodes}}
        step_sol['obj_value'] = sum(
            [self.hc_dict[node] * self.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])
        return step_sol

    def output_results(self, input_results=None):
        if input_results:
            self.results = input_results
        cost_list = [sol['obj_value'] for sol in self.results]
        best_i = np.argmin(cost_list)
        best_sol = self.results[best_i]
        error_sol = check_solution_feasibility(self.gsm_instance, best_sol)
        if len(error_sol) > 0:
            logger.error(error_sol)
        return best_sol

    def get_nodes_info(self, stability_threshold):
        node_S_results = {node: [r['S'][node] for r in self.results] for node in self.all_nodes}
        node_S_mean = {node: np.mean(node_S_results[node]) for node in self.all_nodes}

        node_SI_results = {node: [r['SI'][node] for r in self.results] for node in self.all_nodes}
        node_SI_mean = {node: np.mean(node_SI_results[node]) for node in self.all_nodes}

        node_CT_results = {node: [r['CT'][node] for r in self.results] for node in self.all_nodes}
        node_CT_mean = {node: np.mean(node_CT_results[node]) for node in self.all_nodes}

        node_S_std = {node: np.std(node_S_results[node]) for node in self.all_nodes}
        S_stat_dict = {node: node_S_std[node] / (node_S_mean[node] if node_S_mean[node] > 0 else 1)
                       for node in self.all_nodes}
        node_SI_std = {node: np.std(node_SI_results[node]) for node in self.all_nodes}
        SI_stat_dict = {node: node_SI_std[node] / (node_SI_mean[node] if node_SI_mean[node] > 0 else 1)
                        for node in self.all_nodes}
        node_CT_std = {node: np.std(node_CT_results[node]) for node in self.all_nodes}
        CT_stat_dict = {node: node_CT_std[node] / (node_CT_mean[node] if node_CT_mean[node] > 0 else 1)
                        for node in self.all_nodes}

        stationary_S_node = [node for node, v in S_stat_dict.items() if v <= stability_threshold]
        stationary_SI_node = [node for node, v in SI_stat_dict.items() if v <= stability_threshold]
        stationary_CT_node = [node for node, v in CT_stat_dict.items() if v <= stability_threshold]

        fix_S = {}
        fix_SI = {}
        fix_CT = {}
        for node in self.all_nodes:
            if node in stationary_S_node:
                fix_S[node] = float(round(node_S_mean[node], 2))
                if node in stationary_SI_node:
                    fix_SI[node] = float(round(node_SI_mean[node], 2))
                    fix_CT[node] = fix_SI[node] + self.lt_dict[node] - fix_S[node]
                elif node in stationary_CT_node:
                    fix_CT[node] = float(round(node_CT_mean[node], 2))
                    fix_SI[node] = fix_CT[node] + fix_S[node] - self.lt_dict[node]
            elif node in stationary_SI_node:
                fix_SI[node] = float(round(node_SI_mean[node], 2))
                if node in stationary_CT_node:
                    fix_CT[node] = float(round(node_CT_mean[node], 2))
                    fix_S[node] = fix_SI[node] + self.lt_dict[node] - fix_CT[node]

        fix_S_nodes = set(fix_S.keys())
        fix_SI_nodes = set(fix_SI.keys())
        fix_CT_nodes = set(fix_CT.keys())

        completely_fix_nodes = fix_S_nodes & fix_SI_nodes & fix_CT_nodes

        solely_fix_S_nodes = fix_S_nodes - completely_fix_nodes
        free_S_nodes = self.all_nodes - fix_S_nodes
        solely_fix_SI_nodes = fix_SI_nodes - completely_fix_nodes
        free_SI_nodes = self.all_nodes - fix_SI_nodes
        solely_fix_CT_nodes = fix_CT_nodes - completely_fix_nodes
        free_CT_nodes = self.all_nodes - fix_CT_nodes
        completely_free_nodes = self.all_nodes - fix_S_nodes - fix_SI_nodes - fix_CT_nodes
        partially_free_nodes = self.all_nodes - completely_fix_nodes

        completely_fix_S = {j: fix_S[j] for j in completely_fix_nodes}
        completely_fix_SI = {j: fix_SI[j] for j in completely_fix_nodes}
        completely_fix_CT = {j: fix_CT[j] for j in completely_fix_nodes}

        nodes_info = {'fix_S_nodes': fix_S_nodes, 'fix_SI_nodes': fix_SI_nodes, 'fix_CT_nodes': fix_CT_nodes,
                      'completely_fix_nodes': completely_fix_nodes, 'completely_free_nodes': completely_free_nodes,
                      'partially_free_nodes': partially_free_nodes,
                      'solely_fix_S_nodes': solely_fix_S_nodes, 'free_S_nodes': free_S_nodes,
                      'solely_fix_SI_nodes': solely_fix_SI_nodes, 'free_SI_nodes': free_SI_nodes,
                      'solely_fix_CT_nodes': solely_fix_CT_nodes, 'free_CT_nodes': free_CT_nodes,
                      'fix_S': fix_S, 'fix_SI': fix_SI, 'fix_CT': fix_CT,
                      'completely_fix_S': completely_fix_S, 'completely_fix_SI': completely_fix_SI,
                      'completely_fix_CT': completely_fix_CT}
        return nodes_info
