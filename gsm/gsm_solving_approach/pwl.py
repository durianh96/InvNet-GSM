from gsm.gsm_solving_approach.solving_default_paras import OPT_GAP_PWL, TIME_LIMIT, BOUND_VALUE_TYPE, SYSTEM_TIME_UNIT
from gsm.gsm_sol import *
from utils.system_utils import *
from gsm.gsm_instance import GSMInstance


class PieceWiseLinear:
    def __init__(self, gsm_instance: GSMInstance,
                 time_unit=SYSTEM_TIME_UNIT,
                 opt_gap=OPT_GAP_PWL,
                 time_limit=TIME_LIMIT,
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

        self.time_unit = time_unit
        self.opt_gap = opt_gap
        self.time_limit = time_limit
        self.bound_value_type = bound_value_type

        self.ar_dict = None
        self.br_dict = None

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

    def get_pwl_vb_paras_of_node(self, n_id, ct_value):
        if self.bound_value_type == 'APPROX':
            pwl_vb_paras = self.demand_bound_pool[n_id].get_pwl_vb_paras(ct_value)
        elif self.bound_value_type == 'FUNC':
            vb_gradient_value = self.demand_bound_pool[n_id].vb_gradient_func(ct_value)
            vb_intercept_value = self.demand_bound_pool[n_id].vb_func(ct_value) - vb_gradient_value * ct_value
            pwl_vb_paras = {'gradient': vb_gradient_value, 'intercept': vb_intercept_value}
        else:
            raise AttributeError('wrong bound value type')
        return pwl_vb_paras

    @timer
    def get_policy(self, solver):
        self.get_ar_br()
        if solver == 'GRB':
            sol = self.pwl_grb()
        elif solver == 'COPT':
            sol = self.pwl_copt()
        elif solver == 'PYO_COPT':
            sol = self.pwl_pyomo(pyo_solver='COPT')
        elif solver == 'PYO_GRB':
            sol = self.pwl_pyomo(pyo_solver='GRB')
        elif solver == 'PYO_CBC':
            sol = self.pwl_pyomo(pyo_solver='CBC')
        else:
            raise AttributeError
        error_sol = check_solution_feasibility(self.gsm_instance, sol)
        if len(error_sol) > 0:
            logger.error(error_sol)
            raise Exception
        
        else:
            sol['S'] = {j: round(v, 2) for j, v in sol['S'].items()}
            sol['SI'] = {j: round(v, 2) for j, v in sol['SI'].items()}
            sol['CT'] = {j: round(v, 2) for j, v in sol['CT'].items()}

            pwl_sol = GSMSolution(nodes=self.nodes)
            pwl_sol.update_sol(sol)

            oul_of_node = {node: self.get_db_value_of_node(node, pwl_sol.CT_of_node[node]) for node in self.nodes}
            ss_of_node = {node: self.get_vb_value_of_node(node, pwl_sol.CT_of_node[node]) for node in self.nodes}
            method = 'PWL_' + solver
            ss_cost = cal_ss_cost(self.hc_of_node, ss_of_node, method=method)

            pwl_sol.update_oul(oul_of_node)
            pwl_sol.update_ss(ss_of_node)
            pwl_sol.update_ss_cost(ss_cost)
            return pwl_sol

    def get_ar_br(self):
        ar_dict = {}
        br_dict = {}
        for n_id in self.nodes:
            for ct in np.arange(0., self.cum_lt_of_node[n_id] + self.time_unit, self.time_unit):
                pwl_vb_paras = self.get_pwl_vb_paras_of_node(n_id, ct)
                ar_dict[(n_id, ct)] = pwl_vb_paras['gradient']
                br_dict[(n_id, ct)] = pwl_vb_paras['intercept']

        self.ar_dict = ar_dict
        self.br_dict = br_dict

    def pwl_grb(self):
        import gurobipy as gp
        from gurobipy import GRB
        jt_list = list(self.ar_dict.keys())
        jt_dict = {j: list(range(0, int(self.cum_lt_of_node[j]) + 1)) for j in self.nodes}

        m = gp.Model('pwl')
        m.setParam('TimeLimit', self.time_limit)
        # service time
        S = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0, name='S')
        # inbound service time
        SI = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0, name='SI')
        # covering time
        CT = m.addVars(self.nodes, vtype=GRB.CONTINUOUS, lb=0, name='CT')
        # 0-1
        U = m.addVars(jt_list, vtype=GRB.BINARY, name='U')
        # approx
        Z = m.addVars(jt_list, vtype=GRB.CONTINUOUS, lb=0, name='Z')

        # adding objective
        m.setObjective(gp.quicksum(self.hc_of_node[j] * (self.ar_dict[j, t] * Z[j, t] + self.br_dict[j, t] + U[j, t])
                                   for j, t in jt_list), GRB.MINIMIZE)

        # approximating constraint
        m.addConstrs((CT[j] == gp.quicksum(Z[j, t] for t in jt_dict[j]) for j in self.nodes), name='approx_CT')
        m.addConstrs(((t - 1) * U[j, t] <= Z[j, t] for j, t in jt_list), name='time_interval_lhs')
        m.addConstrs((t * U[j, t] >= Z[j, t] for j, t in jt_list), name='time_interval_rhs')
        m.addConstrs((gp.quicksum(U[j, t] for t in jt_dict[j]) == 1 for j in self.nodes), name='choose_one')

        # gsm constraint
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes), name='covering_time')
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges),
                     name='edge')
        m.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks),
                     name='sla')

        # setting parameters
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.opt_gap
        m.Params.Method = 3
        m.Params.LogToConsole = 0
        m.optimize()

        # get optimized solution
        if m.status == GRB.OPTIMAL:
            opt_sol = {'S': {node: float(round(S[node].x)) for node in self.nodes},
                       'SI': {node: float(round(SI[node].x)) for node in self.nodes},
                       'CT': {node: float(round(CT[node].x)) for node in self.nodes}}
            return opt_sol
        elif m.status == GRB.INFEASIBLE:
            raise Exception('Infeasible model')
        elif m.status == GRB.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == GRB.TIME_LIMIT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def pwl_copt(self):
        import coptpy as cp
        from coptpy import COPT
        jt_list = list(self.ar_dict.keys())
        jt_dict = {j: list(range(0, int(self.cum_lt_of_node[j]) + 1)) for j in self.nodes}

        env = cp.Envr()
        m = env.createModel('pwl')

        # service time
        S = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='S')
        # inbound service time
        SI = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='SI')
        # covering time
        CT = m.addVars(self.nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='CT')
        # 0-1
        U = m.addVars(jt_list, vtype=COPT.BINARY, nameprefix='U')
        # approx
        Z = m.addVars(jt_list, vtype=COPT.CONTINUOUS, lb=0, nameprefix='Z')

        # adding objective
        m.setObjective(cp.quicksum(self.hc_of_node[j] * (self.ar_dict[j, t] * Z[j, t] + self.br_dict[j, t] + U[j, t])
                                   for j, t in jt_list), COPT.MINIMIZE)

        # approximating constraint
        m.addConstrs((CT[j] == cp.quicksum(Z[j, t] for t in jt_dict[j]) for j in self.nodes),
                     nameprefix='approx_CT')
        m.addConstrs(((t - 1) * U[j, t] <= Z[j, t] for j, t in jt_list), nameprefix='time_interval_lhs')
        m.addConstrs((t * U[j, t] >= Z[j, t] for j, t in jt_list), nameprefix='time_interval_rhs')
        m.addConstrs((cp.quicksum(U[j, t] for t in jt_dict[j]) == 1 for j in self.nodes), nameprefix='choose_one')

        # gsm constraint
        m.addConstrs((CT[j] == SI[j] + self.lt_of_node[j] - S[j] for j in self.nodes),
                     nameprefix='covering_time')
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edges),
                     nameprefix='edge')
        m.addConstrs((S[j] <= int(self.sla_of_node[j]) for j in self.sinks),
                     nameprefix='sla')

        # setting parameters
        m.setParam(COPT.Param.TimeLimit, self.time_limit)
        m.setParam(COPT.Param.HeurLevel, 3)
        m.setParam(COPT.Param.RelGap, self.opt_gap)
        m.setParam(COPT.Param.Logging, False)
        m.setParam(COPT.Param.LogToConsole, False)
        m.solve()

        # get optimized solution
        if m.status == COPT.OPTIMAL:
            opt_sol = {'S': {node: float(round(S[node].x)) for node in self.nodes},
                       'SI': {node: float(round(SI[node].x)) for node in self.nodes},
                       'CT': {node: float(round(CT[node].x)) for node in self.nodes}}
            return opt_sol
        elif m.status == COPT.INFEASIBLE:
            raise Exception('Infeasible model')
        elif m.status == COPT.UNBOUNDED:
            raise Exception('Unbounded model')
        elif m.status == COPT.TIMEOUT:
            raise Exception('Time out')
        else:
            logger.error('Error status is ', m.status)
            raise Exception('Solution has not been found')

    def pwl_pyomo(self, pyo_solver):
        import pyomo.environ as pyo
        import pyomo.opt as pyopt
        jt_list = list(self.ar_dict.keys())
        jt_dict = {j: list(range(0, int(self.cum_lt_of_node[j]) + 1)) for j in self.nodes}

        m = pyo.ConcreteModel()

        # service time
        m.S = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        # inbound service time
        m.SI = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        # covering time
        m.CT = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        # 0-1
        m.U = pyo.Var(jt_list, domain=pyo.Binary)
        # approx
        m.Z = pyo.Var(jt_list, domain=pyo.NonNegativeReals)

        # adding objective
        m.Cost = pyo.Objective(
            expr=sum([self.hc_of_node[j] * (self.ar_dict[j, t] * m.Z[j, t] + self.br_dict[j, t] + m.U[j, t])
                      for j, t in jt_list]),
            sense=pyo.minimize
        )

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in self.nodes:
            m.constrs.add(m.CT[j] == sum([m.Z[j, t] for t in jt_dict[j]]))
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_of_node[j] - m.S[j])
            m.constrs.add(sum([m.U[j, t] for t in jt_dict[j]]) == 1)

        for j, t in jt_list:
            m.constrs.add((t - 1) * m.U[j, t] <= m.Z[j, t])
            m.constrs.add(t * m.U[j, t] >= m.Z[j, t])

        for pred, succ in self.edges:
            m.constrs.add(m.SI[succ] - m.S[pred] >= 0)

        for j in self.sinks:
            m.constrs.add(m.S[j] <= int(self.sla_of_node[j]))

        if pyo_solver == 'COPT':
            solver = pyopt.SolverFactory('copt_direct')
            solver.options['RelGap'] = self.opt_gap
            solver.options['TimeLimit'] = self.time_limit
        elif pyo_solver == 'GRB':
            solver = pyopt.SolverFactory('gurobi_direct')
            solver.options['MIPGap'] = self.opt_gap
            solver.options['TimeLimit'] = self.time_limit
        elif pyo_solver == 'CBC':
            solver = pyopt.SolverFactory('cbc')
            solver.options['ratio'] = self.opt_gap
            solver.options['sec'] = self.time_limit
        else:
            raise AttributeError

        solver.solve(m, tee=False)
        opt_sol = {'S': {node: round(float(m.S[node].value), 2) for node in self.nodes},
                   'SI': {node: round(float(m.SI[node].value), 2) for node in self.nodes},
                   'CT': {node: round(float(m.CT[node].value), 2) for node in self.nodes}}
        return opt_sol

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit,
                 'opt_gap': self.opt_gap,
                 'time_limit': self.time_limit,
                 'bound_value_type': self.bound_value_type}
        return paras
