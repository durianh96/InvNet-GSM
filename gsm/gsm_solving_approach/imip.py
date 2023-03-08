from gsm.gsm_solving_approach.solving_default_paras import TERMINATION_PARM, OPT_GAP, MAX_ITER_NUM, BOUND_VALUE_TYPE
from gsm.gsm_sol import *
from utils.system_utils import *
from gsm.gsm_instance import GSMInstance


class IterativeMIP:
    def __init__(self, gsm_instance: GSMInstance,
                 termination_parm=TERMINATION_PARM,
                 opt_gap=OPT_GAP,
                 max_iter_num=MAX_ITER_NUM,
                 bound_value_type=BOUND_VALUE_TYPE):
        self.gsm_instance = gsm_instance
        self.nodes = gsm_instance.nodes
        self.edges = gsm_instance.edges
        self.sinks = gsm_instance.sinks

        self.lt_of_node = gsm_instance.lt_of_node
        self.cum_lt_of_node = gsm_instance.cum_lt_of_node
        self.hc_of_node = gsm_instance.hc_of_node
        self.sla_of_node = gsm_instance.sla_of_node
        self.demand_bound_pool = gsm_instance.demand_bound_pool

        self.termination_parm = termination_parm
        self.opt_gap = opt_gap
        self.max_iter_num = max_iter_num
        self.bound_value_type = bound_value_type

        self.need_solver = True

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

    @timer
    def get_policy(self, solver):
        CT_k = {j: 1.0 for j in self.nodes}
        a_k = {j: self.get_vb_gradient_value_of_node(j, CT_k[j]) for j in self.nodes}
        f_k = {j: self.get_vb_value_of_node(j, CT_k[j]) - self.get_vb_gradient_value_of_node(j, CT_k[j]) * CT_k[j]
               for j in self.nodes}
        obj_value = []
        for i in range(self.max_iter_num):
            obj_para_k = {'a_k': a_k, 'f_k': f_k}
            if solver == 'GRB':
                sol_k = self.imilp_step_grb(obj_para_k)
            elif solver == 'COPT':
                sol_k = self.imilp_step_copt(obj_para_k)
            elif solver == 'PYO_COPT':
                sol_k = self.imilp_step_pyomo(obj_para_k, pyo_solver='COPT')
            elif solver == 'PYO_GRB':
                sol_k = self.imilp_step_pyomo(obj_para_k, pyo_solver='GRB')
            elif solver == 'PYO_CBC':
                sol_k = self.imilp_step_pyomo(obj_para_k, pyo_solver='CBC')
            else:
                raise AttributeError('undefined solver')
            CT_k = sol_k['CT']
            Y_k = sol_k['Y']
            est_k = {j: a_k[j] * CT_k[j] + f_k[j] * Y_k[j] for j in self.nodes}
            diff_k = {j: self.hc_of_node[j] * abs(est_k[j] - self.get_vb_value_of_node(j, CT_k[j])) for j in self.nodes}
            for j in self.nodes:
                if diff_k[j] == 0:
                    continue
                else:
                    a_k.update({j: self.get_vb_gradient_value_of_node(j, CT_k[j])})
                    f_k.update({j: self.get_vb_value_of_node(j, CT_k[j]) -
                                   self.get_vb_gradient_value_of_node(j, CT_k[j]) * CT_k[j]})
            obj_value.append(sol_k['obj_value'])
            if (i > 0) and (max(diff_k.values()) / len(self.nodes) <= self.termination_parm):
                break
        sol = {'S': sol_k['S'], 'SI': sol_k['SI'], 'CT': sol_k['CT']}
        error_sol = check_solution_feasibility(self.gsm_instance, sol)
        if len(error_sol) > 0:
            logger.error(error_sol)
            raise Exception
        else:
            sol['S'] = {j: round(v, 2) for j, v in sol['S'].items()}
            sol['SI'] = {j: round(v, 2) for j, v in sol['SI'].items()}
            sol['CT'] = {j: round(v, 2) for j, v in sol['CT'].items()}

            imip_sol = GSMSolution(nodes=self.nodes)
            imip_sol.update_sol(sol)

            oul_of_node = {node: self.get_db_value_of_node(node, imip_sol.CT_of_node[node]) for node in self.nodes}
            ss_of_node = {node: self.get_vb_value_of_node(node, imip_sol.CT_of_node[node]) for node in self.nodes}
            method = 'IterativeMIP_' + solver
            ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

            imip_sol.update_oul(oul_of_node)
            imip_sol.update_ss(ss_of_node)
            imip_sol.update_ss_cost(ss_cost)
            return imip_sol

    def imilp_step_grb(self, obj_para):
        import gurobipy as gp
        from gurobipy import GRB
        model = gp.Model('imilp_step')

        # adding variables
        S = model.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)
        SI = model.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)
        CT = model.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0)
        Y = model.addVars(self.nodes, vtype=GRB.BINARY)
        M = 1e8

        # gsm constraint
        model.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes))
        model.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks))

        model.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges))

        model.addConstrs((Y[j] <= M * CT[j] for j in self.nodes))

        model.setObjective(gp.quicksum(
            self.hc_of_node[node] * (obj_para['a_k'][node] * CT[node] + obj_para['f_k'][node] * Y[node])
            for node in self.nodes), GRB.MINIMIZE)
        model.Params.MIPGap = self.opt_gap
        model.Params.LogToConsole = 0
        model.optimize()

        if model.status == GRB.OPTIMAL:
            sol_k = {'S': {node: float(S[node].x) for node in self.nodes},
                     'SI': {node: float(SI[node].x) for node in self.nodes},
                     'CT': {node: float(CT[node].x) for node in self.nodes},
                     'Y': {node: Y[node].x for node in self.nodes}}
            sol_k['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, sol_k['CT'][node]) for node in self.nodes])
            return sol_k
        elif model.status == GRB.INFEASIBLE:
            raise Exception('Infeasible model')
        elif model.status == GRB.UNBOUNDED:
            raise Exception('Unbounded model')
        elif model.status == GRB.TIME_LIMIT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', model.status)
            raise Exception('Solution has not been found')

    def imilp_step_copt(self, obj_para):
        import coptpy as cp
        from coptpy import COPT
        env = cp.Envr()
        model = env.createModel('imilp_step')

        # adding variables
        S = model.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='S')
        SI = model.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='SI')
        CT = model.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='CT')
        Y = model.addVars(self.nodes, vtype=COPT.BINARY, nameprefix='Y')
        M = 1e8

        # gsm constraint
        model.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes),
                         nameprefix='covering_time')
        model.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks), nameprefix='sla')

        model.addConstrs(
            (SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges),
            nameprefix='edge')

        model.addConstrs((Y[j] <= M * CT[j] for j in self.nodes), nameprefix='binary')

        model.setObjective(cp.quicksum(
            self.hc_of_node[node] * (obj_para['a_k'][node] * CT[node] + obj_para['f_k'][node] * Y[node])
            for node in self.nodes), COPT.MINIMIZE)
        model.setParam(COPT.Param.RelGap, self.opt_gap)
        model.setParam(COPT.Param.Logging, False)
        model.setParam(COPT.Param.LogToConsole, False)
        model.solve()

        if model.status == COPT.OPTIMAL:
            sol_k = {'S': {node: float(S[node].x) for node in self.nodes},
                     'SI': {node: float(SI[node].x) for node in self.nodes},
                     'CT': {node: float(CT[node].x) for node in self.nodes},
                     'Y': {node: Y[node].x for node in self.nodes}}
            sol_k['obj_value'] = sum(
                [self.hc_of_node[node] * self.get_vb_value_of_node(node, sol_k['CT'][node]) for node in self.nodes])
            return sol_k
        elif model.status == COPT.INFEASIBLE:
            raise Exception('Infeasible model')
        elif model.status == COPT.UNBOUNDED:
            raise Exception("The problem is unbounded")
        else:
            raise Exception("Solution has not been found within given time limit")

    def imilp_step_pyomo(self, obj_para, pyo_solver):
        import pyomo.environ as pyo
        import pyomo.opt as pyopt
        m = pyo.ConcreteModel('imilp_step')
        # adding variables
        m.S = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        m.SI = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        m.CT = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        m.Y = pyo.Var(self.nodes, domain=pyo.Binary)

        M = 1e8
        # constraints
        m.constrs = pyo.ConstraintList()
        for j in self.nodes:
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_of_node[j] - m.S[j])
            m.constrs.add(m.Y[j] <= M * m.CT[j])

        # sla
        for j in self.sinks:
            m.constrs.add(m.S[j] <= int(self.sla_of_node[j]))

        # si >= s
        for pred, succ in self.edges:
            m.constrs.add(m.SI[succ] - m.S[pred] >= 0)

        m.Cost = pyo.Objective(
            expr=sum([self.hc_of_node[j] * (obj_para['a_k'][j] * m.CT[j] + obj_para['f_k'][j] * m.Y[j])
                      for j in self.nodes]),
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

        sol_k = {'S': {node: float(m.S[node].value) for node in self.nodes},
                 'SI': {node: float(m.SI[node].value) for node in self.nodes},
                 'CT': {node: float(m.CT[node].value) for node in self.nodes},
                 'Y': {node: m.Y[node].value for node in self.nodes}}
        sol_k['obj_value'] = sum(
            [self.hc_of_node[node] * self.get_vb_value_of_node(node, sol_k['CT'][node])for node in self.nodes])
        return sol_k

    def get_approach_paras(self):
        paras = {'termination_parm': self.termination_parm, 'opt_gap': self.opt_gap, 'max_iter_num': self.max_iter_num}
        return paras
