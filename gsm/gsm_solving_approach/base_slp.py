import random
from gsm.gsm_solving_approach.solving_default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, EPS, \
    BOUND_VALUE_TYPE
from gsm.gsm_sol import *
from utils.system_utils import *
from utils.graph_algorithms import *
from gsm.gsm_instance import GSMInstance
from typing import Optional


class BaseSLP:
    def __init__(self, gsm_instance: GSMInstance,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 bound_value_type=BOUND_VALUE_TYPE,
                 input_s_ub_dict: Optional[dict] = None,
                 input_si_lb_dict: Optional[dict] = None):
        self.gsm_instance = gsm_instance
        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.roots = gsm_instance.roots
        self.sinks = gsm_instance.sinks

        self.succs_of_node = gsm_instance.succs_of_node
        self.preds_of_node = gsm_instance.preds_of_node

        self.lt_of_node = gsm_instance.lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.cum_lt_of_node = gsm_instance.cum_lt_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num
        self.bound_value_type = bound_value_type
        
        if input_s_ub_dict is None:
            self.s_ub_dict = {}
        else:
            self.s_ub_dict = input_s_ub_dict
            # update upper bound for sub-instance
            self.sla_of_node.update({node: s_ub for node, s_ub in self.s_ub_dict.items()
                                     if node in self.sinks})        
        if input_si_lb_dict is None:
            self.si_lb_dict = {}
        else:
            self.si_lb_dict = input_si_lb_dict

        self.gsm_sol_set = GSMSolutionSet(nodes=self.nodes, lt_of_node=self.lt_of_node,
                                          cum_lt_of_node=self.cum_lt_of_node, sla_of_node=self.sla_of_node)

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

    def get_vb_gradient_value_of_node(self, n_id, ct_value):
        if self.bound_value_type == 'APPROX':
            pwl_vb_paras = self.demand_bound_pool[n_id].get_pwl_vb_paras(ct_value)
            vb_gradient_value = pwl_vb_paras['gradient']
        elif self.bound_value_type == 'FUNC':
            vb_gradient_value = self.demand_bound_pool[n_id].vb_gradient_func(ct_value)
        else:
            raise AttributeError('wrong bound value type')
        return vb_gradient_value

    def cal_para(self, CT):
        A = {node: self.hc_of_node[node] * self.get_vb_gradient_value_of_node(node, max(ctk, EPS))
             for node, ctk in CT.items()}
        B = {node: self.hc_of_node[node] * self.get_vb_value_of_node(node, max(ctk, EPS)) - A[node] * ctk
             for node, ctk in CT.items()}
        obj_para = {'A': A, 'B': B}
        return obj_para

    def get_one_random_sol(self, solver):
        CT_step = {j: float(random.randint(1, 50)) for j in self.nodes}
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
            elif solver == 'PYO_SCIP':
                step_sol = self.slp_step_pyomo(step_obj_para, pyo_solver='SCIP')
            elif solver == 'PYO_GLPK':
                step_sol = self.slp_step_pyomo(step_obj_para, pyo_solver='GLPK')
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
                    once_slp_sol = GSMSolution(nodes=self.nodes)
                    once_slp_sol.update_sol(step_sol)
                    once_slp_sol.update_ss_cost(step_sol['obj_value'])
                    return once_slp_sol
    
    # parallel implementation
    def get_multiple_random_sol(self, solver, multi_num, gsm_sol_pool):
        for _ in range(multi_num):
            sol = self.get_one_random_sol(solver)
            gsm_sol_pool.append(sol)
    
    def slp_step_grb(self, obj_para):
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model('slp_step')

        # adding variables
        S = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)
        SI = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)
        CT = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes))
        # sla
        m.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks if j in self.sla_of_node.keys()))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges))

        # s upper bound
        if len(self.s_ub_dict) > 0:
            m.addConstrs((S[j] <= ub for j, ub in self.s_ub_dict.items()))

        # si lower bound
        if len(self.si_lb_dict) > 0:
            m.addConstrs((SI[j] >= lb for j, lb in self.si_lb_dict.items()))

        m.setObjective(gp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in self.nodes), GRB.MINIMIZE)
        m.Params.MIPGap = self.opt_gap
        m.Params.LogToConsole = 0
        m.optimize()

        if m.status == GRB.OPTIMAL:
            step_sol = {'S': {node: float(round(S[node].x)) for node in self.nodes},
                        'SI': {node: float(round(SI[node].x)) for node in self.nodes},
                        'CT': {node: float(round(CT[node].x)) for node in self.nodes}}
            step_sol['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])            
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
        S = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0)
        SI = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0)
        CT = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes))
        # sla
        m.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks if j in self.sla_of_node.keys()))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges))

        # s upper bound
        if len(self.s_ub_dict) > 0:
            m.addConstrs((S[j] <= ub for j, ub in self.s_ub_dict.items()))

        # si lower bound
        if len(self.si_lb_dict) > 0:
            m.addConstrs((SI[j] >= lb for j, lb in self.si_lb_dict.items()))

        m.setObjective(cp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in self.nodes), COPT.MINIMIZE)
        m.setParam(COPT.Param.RelGap, self.opt_gap)
        m.setParam(COPT.Param.Logging, False)
        m.setParam(COPT.Param.LogToConsole, False)
        m.solve()

        if m.status == COPT.OPTIMAL:
            step_sol = {'S': {node: float(round(S[node].x)) for node in self.nodes},
                        'SI': {node: float(round(SI[node].x)) for node in self.nodes},
                        'CT': {node: float(round(CT[node].x)) for node in self.nodes}}
            step_sol['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])
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

    def slp_step_pyomo(self, obj_para, pyo_solver='GRB'):
        import pyomo.environ as pyo
        import pyomo.opt as pyopt

        m = pyo.ConcreteModel('slp_step')
        # adding variables
        m.S = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        m.SI = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        m.CT = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in self.nodes:
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_of_node[j] - m.S[j])

        # sla
        for j in set(self.sinks) & set(self.sla_of_node.keys()):
            m.constrs.add(m.S[j] <= int(self.sla_of_node[j]))

        # si >= s
        for pred, succ in self.edges:
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
            expr=sum([obj_para['A'][j] * m.CT[j] + obj_para['B'][j] for j in self.nodes]),
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
        elif pyo_solver == 'GLPK':
            solver = pyopt.SolverFactory('glpk')
            solver.options['TolObj'] = self.opt_gap
        elif pyo_solver == 'SCIP':
            solver = pyopt.SolverFactory('scip')
            solver.options['limits/gap'] = self.opt_gap
            solver.options['limits/time'] = 1e+20
        else:
            raise AttributeError

        solver.solve(m, tee=False)

        step_sol = {'S': {node: float(m.S[node].value) for node in self.nodes},
                    'SI': {node: float(m.SI[node].value) for node in self.nodes},
                    'CT': {node: float(m.CT[node].value) for node in self.nodes}}
        step_sol['obj_value'] = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])
        return step_sol