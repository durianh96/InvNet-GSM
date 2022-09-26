from approach.base_slp import *
from domain.policy import Policy
from default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, LOCAL_SOL_NUM, STABLE_FINDING_ITER, \
    STABILITY_THRESHOLD


class IterativeFixingSLP(BaseSLP):
    def __init__(self, gsm_instance, termination_parm=TERMINATION_PARM, opt_gap=OPT_GAP, max_iter_num=MAX_ITER_NUM,
                 local_sol_num=LOCAL_SOL_NUM, stable_finding_iter=STABLE_FINDING_ITER,
                 stability_threshold=STABILITY_THRESHOLD):
        super().__init__(gsm_instance, termination_parm, opt_gap, max_iter_num)
        self.local_sol_num = local_sol_num
        self.stable_finding_iter = stable_finding_iter
        self.stability_threshold = stability_threshold

        self.need_solver = True

    @timer
    def get_policy(self, solver):
        best_sol = self.get_slp_sol(solver)
        bs_dict = {node: self.db_func[node](best_sol['CT'][node]) for node in self.all_nodes}
        ss_dict = {node: self.vb_func[node](best_sol['CT'][node]) for node in self.all_nodes}
        method = 'IF-SLP_' + solver
        cost = cal_cost(self.hc_dict, ss_dict, method=method)

        policy = Policy(self.all_nodes)
        policy.update_sol(best_sol)
        policy.update_base_stock(bs_dict)
        policy.update_safety_stock(ss_dict)
        policy.update_ss_cost(cost)
        return policy

    def get_slp_sol(self, solver):
        nodes_info = {'fix_S_nodes': set(), 'fix_SI_nodes': set(), 'fix_CT_nodes': set(),
                      'completely_fix_nodes': set(), 'completely_free_nodes': self.all_nodes,
                      'partially_free_nodes': self.all_nodes,
                      'solely_fix_S_nodes': set(), 'free_S_nodes': self.all_nodes,
                      'solely_fix_SI_nodes': set(), 'free_SI_nodes': self.all_nodes,
                      'solely_fix_CT_nodes': set(), 'free_CT_nodes': self.all_nodes,
                      'fix_S': {}, 'fix_SI': {}, 'fix_CT': {},
                      'completely_fix_S': {}, 'completely_fix_SI': {}, 'completely_fix_CT': {}}
        for i in range(self.local_sol_num):
            init_CT = {j: float(random.randint(1, 150)) for j in self.all_nodes}
            init_CT.update(nodes_info['completely_fix_CT'])
            self.run_one_instance_completely_fix(init_CT, nodes_info, solver)
            if len(self.results) > 0:
                if len(self.results) % self.stable_finding_iter == 0:
                    nodes_info = self.get_nodes_info(self.stability_threshold)
        best_sol = self.output_results()
        return best_sol

    def run_one_instance_completely_fix(self, init_CT, nodes_info, solver):
        CT_step = copy.copy(init_CT)
        obj_value = [0]
        for i in range(self.max_iter_num):
            step_obj_para = self.cal_para(CT_step)
            if solver == 'GRB':
                step_sol = self.slp_step_completely_fix_grb(step_obj_para, nodes_info)
            elif solver == 'COPT':
                step_sol = self.slp_step_completely_fix_copt(step_obj_para, nodes_info)
            elif solver == 'PYO_COPT':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, nodes_info, pyo_solver='COPT')
            elif solver == 'PYO_GRB':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, nodes_info, pyo_solver='GRB')
            elif solver == 'PYO_CBC':
                step_sol = self.slp_step_completely_fix_pyomo(step_obj_para, nodes_info, pyo_solver='CBC')
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

    def slp_step_completely_fix_grb(self, obj_para, nodes_info):
        m = gp.Model('slp_step_completely_fix')

        S = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)
        SI = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)
        CT = m.addVars(nodes_info['partially_free_nodes'], vtype=GRB.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in nodes_info['partially_free_nodes']))
        # sla
        m.addConstrs(
            (S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes if j in nodes_info['partially_free_nodes']))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list if
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
                [self.hc_dict[node] * self.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])
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
        env = cp.Envr()
        m = env.createModel('slp_step_completely_fix')

        S = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)
        SI = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)
        CT = m.addVars(nodes_info['partially_free_nodes'], vtype=COPT.CONTINUOUS, lb=0)

        # covering time
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in nodes_info['partially_free_nodes']))
        # sla
        m.addConstrs(
            (S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes if j in nodes_info['partially_free_nodes']))

        # si >= s
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list if
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
                [self.hc_dict[node] * self.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])
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

    def slp_step_completely_fix_pyomo(self, obj_para, nodes_info, pyo_solver):
        m = pyo.ConcreteModel('slp_step_completely_fix')
        # adding variables
        m.S = pyo.Var(nodes_info['completely_free_nodes'], domain=pyo.NonNegativeReals)
        m.SI = pyo.Var(nodes_info['completely_free_nodes'], domain=pyo.NonNegativeReals)
        m.CT = pyo.Var(nodes_info['completely_free_nodes'], domain=pyo.NonNegativeReals)

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in nodes_info['completely_free_nodes']:
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_dict[j] - m.S[j])
        # sla
        for j in self.demand_nodes:
            if j in nodes_info['completely_free_nodes']:
                m.constrs.add(m.S[j] <= int(self.sla_dict[j]))

        # si >= s
        for pred, succ in self.edge_list:
            if (succ in nodes_info['completely_free_nodes']) and (pred in nodes_info['completely_free_nodes']):
                m.constrs.add(m.SI[succ] - m.S[pred] >= 0)
            elif (succ in nodes_info['completely_free_nodes']) and (pred in nodes_info['completely_fix_nodes']):
                m.constrs.add(m.SI[succ] - nodes_info['completely_fix_S'][pred] >= 0)
            elif (succ in nodes_info['completely_fix_nodes']) and (pred in nodes_info['completely_free_nodes']):
                m.constrs.add(nodes_info['completely_fix_SI'][succ] - m.S[pred] >= 0)

        m.Cost = pyo.Objective(
            expr=sum([obj_para['A'][j] * m.CT[j] + obj_para['B'][j] for j in nodes_info['completely_free_nodes']]),
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

        step_sol = {'S': {node: float(m.S[node].value) for node in nodes_info['completely_free_nodes']},
                    'SI': {node: float(m.SI[node].value) for node in nodes_info['completely_free_nodes']},
                    'CT': {node: float(m.CT[node].value) for node in nodes_info['completely_free_nodes']}}
        step_sol['S'].update(nodes_info['completely_fix_S'])
        step_sol['SI'].update(nodes_info['completely_fix_SI'])
        step_sol['CT'].update(nodes_info['completely_fix_CT'])
        step_sol['obj_value'] = sum(
            [self.hc_dict[node] * self.gsm_instance.vb_func[node](step_sol['CT'][node]) for node in self.all_nodes])

        return step_sol

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm, 'opt_gap': self.opt_gap, 'max_iter_num': self.max_iter_num,
                 'local_sol_num': self.local_sol_num, 'stable_finding_iter': self.stable_finding_iter,
                 'stability_threshold': self.stability_threshold}
        return paras
