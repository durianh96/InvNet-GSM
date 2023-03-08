from gsm.gsm_solving_approach.base_slp import *
from gsm.gsm_sol import *
from gsm.gsm_solving_approach.solving_default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, LOCAL_SOL_NUM, \
    STABLE_FINDING_ITER, STABILITY_TYPE, STABILITY_THRESHOLD
import copy


class IterativeFixingSLP(BaseSLP):
    def __init__(self, gsm_instance: GSMInstance,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 bound_value_type=BOUND_VALUE_TYPE,
                 local_sol_num=LOCAL_SOL_NUM,
                 stable_finding_iter=STABLE_FINDING_ITER,
                 stability_type=STABILITY_TYPE,
                 stability_threshold=STABILITY_THRESHOLD,
                 ):
        super().__init__(gsm_instance, termination_parm, opt_gap, max_iter_num, bound_value_type)
        self.local_sol_num = local_sol_num
        self.stable_finding_iter = stable_finding_iter
        self.stability_type = stability_type
        self.stability_threshold = stability_threshold

        self.need_solver = True

    @timer
    def get_policy(self, solver):
        if self.stability_type == 'kl' or 'cn':
            self.gsm_sol_set.init_beta_para()

        local_sol_info = self.gsm_sol_set.get_local_sol_info(self.stability_type, self.stability_threshold)
        for i in range(self.local_sol_num):
            init_CT = {j: float(random.randint(1, 150)) for j in self.nodes}
            init_CT.update(local_sol_info['completely_fix_CT'])
            self.run_once_completely_fix(init_CT, local_sol_info, solver)
            if len(self.gsm_sol_set.gsm_sols) % self.stable_finding_iter == 0:
                local_sol_info = self.gsm_sol_set.get_local_sol_info(self.stability_type, self.stability_threshold)

        if_slp_sol = self.gsm_sol_set.get_best_local_sol()

        oul_of_node = {node: self.get_db_value_of_node(node, if_slp_sol.CT_of_node[node]) for node in self.nodes}
        ss_of_node = {node: self.get_vb_value_of_node(node, if_slp_sol.CT_of_node[node]) for node in self.nodes}
        method = 'IF-SLP_' + solver + '_' + self.stability_type
        ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

        if_slp_sol.update_oul(oul_of_node)
        if_slp_sol.update_ss(ss_of_node)
        if_slp_sol.update_ss_cost(ss_cost)
        return if_slp_sol

    def run_once_completely_fix(self, init_CT, local_sol_info, solver):
        CT_step = copy.copy(init_CT)
        obj_value = [0]
        for i in range(self.max_iter_num):
            step_obj_para = self.cal_para(CT_step)
            if solver == 'GRB':
                step_sol = self.slp_step_completely_fix_grb(step_obj_para, local_sol_info)
            elif solver == 'COPT':
                step_sol = self.slp_step_completely_fix_copt(step_obj_para, local_sol_info)
            elif solver == 'PYO_COPT':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, local_sol_info, pyo_solver='COPT')
            elif solver == 'PYO_GRB':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, local_sol_info, pyo_solver='GRB')
            elif solver == 'PYO_CBC':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, local_sol_info, pyo_solver='CBC')
            elif solver == 'PYO_SCIP':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, local_sol_info, pyo_solver='SCIP')
            elif solver == 'PYO_GLPK':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, local_sol_info, pyo_solver='GLPK')
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
                    once_if_slp_sol = GSMSolution(nodes=self.nodes)
                    once_if_slp_sol.update_sol(step_sol)
                    once_if_slp_sol.update_ss_cost(step_sol['obj_value'])
                    if self.stability_type == 'cv':
                        self.gsm_sol_set.add_one_sol(once_if_slp_sol)
                    else:
                        self.gsm_sol_set.add_one_sol(once_if_slp_sol, update_beta=True)

    def slp_step_completely_fix_grb(self, obj_para, nodes_info):
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model('slp_step_completely_fix')

        S = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)
        SI = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)
        CT = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in nodes_info['partially_free_nodes']))
        # sla
        m.addConstrs(
            (S[j] <= int(self.sla_of_node[j]) for j in self.sinks if j in nodes_info['partially_free_nodes']))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges if
                      (succ in nodes_info['partially_free_nodes']) and (pred in nodes_info['partially_free_nodes'])))

        m.setObjective(gp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in nodes_info['partially_free_nodes']), GRB.MINIMIZE)

        m.Params.MIPGap = self.opt_gap
        m.Params.LogToConsole = 0
        m.optimize()

        if m.status == GRB.OPTIMAL:
            step_sol = {'S': {node: float(round(S[node].x)) for node in nodes_info['partially_free_nodes']},
                        'SI': {node: float(round(SI[node].x)) for node in nodes_info['partially_free_nodes']},
                        'CT': {node: float(round(CT[node].x)) for node in nodes_info['partially_free_nodes']}}
            step_sol['S'].update(nodes_info['completely_fix_S'])
            step_sol['SI'].update(nodes_info['completely_fix_SI'])
            step_sol['CT'].update(nodes_info['completely_fix_CT'])
            step_sol['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])
            return step_sol
        elif m.status == GRB.INFEASIBLE:
            m.computeIIS()
            m.write('wrong.ilp')
            raise Exception('Infeasible model')
        elif m.status == GRB.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == GRB.TIME_LIMIT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def slp_step_completely_fix_copt(self, obj_para, nodes_info):
        import coptpy as cp
        from coptpy import COPT
        env = cp.Envr()
        m = env.createModel('slp_step_completely_fix')

        S = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)
        SI = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)
        CT = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in nodes_info['partially_free_nodes']))
        # sla
        m.addConstrs(
            (S[j] <= int(self.sla_of_node[j]) for j in self.sinks if j in nodes_info['partially_free_nodes']))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges if
                      (succ in nodes_info['partially_free_nodes']) and (pred in nodes_info['partially_free_nodes'])))

        m.setObjective(cp.quicksum(obj_para['A'][node] * CT[node] + obj_para['B'][node]
                                   for node in nodes_info['partially_free_nodes']), COPT.MINIMIZE)
        m.setParam(COPT.Param.RelGap, self.opt_gap)
        m.setParam(COPT.Param.Logging, False)
        m.setParam(COPT.Param.LogToConsole, False)
        m.solve()

        if m.status == COPT.OPTIMAL:
            step_sol = {'S': {node: float(round(S[node].x)) for node in nodes_info['partially_free_nodes']},
                        'SI': {node: float(round(SI[node].x)) for node in nodes_info['partially_free_nodes']},
                        'CT': {node: float(round(CT[node].x)) for node in nodes_info['partially_free_nodes']}}
            step_sol['S'].update(nodes_info['completely_fix_S'])
            step_sol['SI'].update(nodes_info['completely_fix_SI'])
            step_sol['CT'].update(nodes_info['completely_fix_CT'])
            step_sol['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])
            return step_sol
        elif m.status == COPT.INFEASIBLE:
            raise Exception('Infeasible model')
        elif m.status == COPT.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == COPT.TIMEOUT:
            raise Exception('Time out')
        elif m.status == COPT.INF_OR_UNB:
            raise Exception('INF_OR_UNB')
        elif m.status == COPT.NUMERICAL:
            raise Exception('NUMERICAL')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def slp_step_completely_fix_pyomo(self, obj_para, nodes_info, pyo_solver='GRB'):
        import pyomo.environ as pyo
        import pyomo.opt as pyopt
        m = pyo.ConcreteModel('slp_step_completely_fix')
        # adding variables
        m.S = pyo.Var(nodes_info['partially_free_nodes'], domain=pyo.NonNegativeReals)
        m.SI = pyo.Var(nodes_info['partially_free_nodes'], domain=pyo.NonNegativeReals)
        m.CT = pyo.Var(nodes_info['partially_free_nodes'], domain=pyo.NonNegativeReals)

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in nodes_info['partially_free_nodes']:
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_of_node[j] - m.S[j])
        # sla
        for j in self.sinks:
            if j in nodes_info['partially_free_nodes']:
                m.constrs.add(m.S[j] <= int(self.sla_of_node[j]))

        # si >= s
        for pred, succ in self.edges:
            if (succ in nodes_info['partially_free_nodes']) and (
                    pred in nodes_info['partially_free_nodes']):
                m.constrs.add(m.SI[succ] - m.S[pred] >= 0)

        m.Cost = pyo.Objective(
            expr=sum([obj_para['A'][j] * m.CT[j] + obj_para['B'][j] for j in
                      nodes_info['partially_free_nodes']]),
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

        step_sol = {'S': {node: float(m.S[node].value) for node in nodes_info['partially_free_nodes']},
                    'SI': {node: float(m.SI[node].value) for node in nodes_info['partially_free_nodes']},
                    'CT': {node: float(m.CT[node].value) for node in nodes_info['partially_free_nodes']}}
        step_sol['S'].update(nodes_info['completely_fix_S'])
        step_sol['SI'].update(nodes_info['completely_fix_SI'])
        step_sol['CT'].update(nodes_info['completely_fix_CT'])
        step_sol['obj_value'] = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, step_sol['CT'][node]) for node in self.nodes])

        return step_sol

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm,
                 'opt_gap': self.opt_gap,
                 'max_iter_num': self.max_iter_num,
                 'bound_value_type': self.bound_value_type,
                 'local_sol_num': self.local_sol_num,
                 'stable_finding_iter': self.stable_finding_iter,
                 'stability_type': self.stability_type,
                 'stability_threshold': self.stability_threshold}
        return paras
