from default_paras import TIME_UNIT, OPT_GAP, TIME_LIMIT
from domain.policy import Policy
from utils.gsm_utils import *
from utils.utils import *


class PieceWiseLinear:
    def __init__(self, gsm_instance, time_unit=TIME_UNIT, opt_gap=OPT_GAP, time_limit=TIME_LIMIT):
        self.gsm_instance = gsm_instance
        self.graph = gsm_instance.graph
        self.all_nodes = gsm_instance.all_nodes
        self.demand_nodes = self.graph.demand_nodes
        self.edge_list = self.graph.edge_list
        self.lt_dict = gsm_instance.lt_dict
        self.cum_lt_dict = gsm_instance.cum_lt_dict
        self.hc_dict = gsm_instance.hc_dict
        self.sla_dict = gsm_instance.sla_dict

        self.network_mu_dict = gsm_instance.network_mu_dict
        self.vb_func = gsm_instance.vb_func
        self.db_func = gsm_instance.db_func
        self.grad_vb_func = gsm_instance.grad_vb_func

        self.time_unit = time_unit
        self.opt_gap = opt_gap
        self.time_limit = time_limit

        self.ar_dict = None
        self.br_dict = None

        self.need_solver = True

    @timer
    def get_policy(self, solver):
        self.cal_ar_br()
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

        bs_dict = {node: self.db_func[node](ct) for node, ct in sol['CT'].items()}
        ss_dict = {node: self.vb_func[node](ct) for node, ct in sol['CT'].items()}
        method = 'PWL_' + solver
        cost = cal_cost(self.hc_dict, ss_dict, method=method)

        policy = Policy(self.all_nodes)
        policy.update_sol(sol)
        policy.update_base_stock(bs_dict)
        policy.update_safety_stock(ss_dict)
        policy.update_ss_cost(cost)
        return policy

    def cal_ar_br(self):
        db_ct_dict = {(node, ct): self.db_func[node](ct) for node in self.all_nodes
                      for ct in np.arange(0., self.cum_lt_dict[node] + self.time_unit, self.time_unit)}
        ss_ct_dict = {(node, ct): self.vb_func[node](ct) for node in self.all_nodes
                      for ct in np.arange(0., self.cum_lt_dict[node] + self.time_unit, self.time_unit)}
        db_df = pd.DataFrame.from_dict(db_ct_dict, orient='index')
        ss_df = pd.DataFrame.from_dict(ss_ct_dict, orient='index')
        demand_bound_df = pd.concat([db_df, ss_df], axis=1).reset_index()
        demand_bound_df.columns = ['idx', 'demand_bound', 'ss_qty']
        demand_bound_df['node_id'] = demand_bound_df['idx'].apply(lambda x: x[0])
        demand_bound_df['time'] = demand_bound_df['idx'].apply(lambda x: x[1])
        demand_bound_df = demand_bound_df[['node_id', 'time', 'demand_bound', 'ss_qty']]

        mean_df = pd.DataFrame.from_dict(self.network_mu_dict, orient='index').reset_index().rename(
            columns={'index': 'node_id', 0: 'mean'})
        demand_bound_df = demand_bound_df.merge(mean_df, on='node_id', how='left')

        demand_bound_df['t_diff'] = demand_bound_df['time'].diff()
        demand_bound_df.loc[demand_bound_df['time'] == 0, 't_diff'] = 0

        demand_bound_df['db_diff'] = demand_bound_df['demand_bound'].diff()
        demand_bound_df.loc[demand_bound_df['time'] == 0, 'db_diff'] = 0

        demand_bound_df['ar'] = demand_bound_df['db_diff'] / demand_bound_df['t_diff'] - demand_bound_df['mean']
        demand_bound_df['br'] = demand_bound_df['demand_bound'] - demand_bound_df['mean'] * demand_bound_df['time'] \
                                - demand_bound_df['ar'] * demand_bound_df['time']

        demand_bound_df.loc[demand_bound_df['time'] == 0, 'ar'] = 0
        demand_bound_df.loc[demand_bound_df['time'] == 0, 'br'] = 0

        ar_dict = {(node, time): ar for node, time, ar in demand_bound_df[['node_id', 'time', 'ar']].values}
        br_dict = {(node, time): br for node, time, br in demand_bound_df[['node_id', 'time', 'br']].values}
        self.ar_dict = ar_dict
        self.br_dict = br_dict

    def pwl_grb(self):
        import gurobipy as gp
        from gurobipy import GRB
        jt_list = list(self.ar_dict.keys())
        jt_dict = {j: list(range(0, int(self.cum_lt_dict[j]) + 1)) for j in self.all_nodes}

        m = gp.Model('pwl')

        # service time
        S = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0, name='S')
        # inbound service time
        SI = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0, name='SI')
        # covering time
        CT = m.addVars(self.all_nodes, vtype=GRB.CONTINUOUS, lb=0, name='CT')
        # 0-1
        U = m.addVars(jt_list, vtype=GRB.BINARY, name='U')
        # approx
        Z = m.addVars(jt_list, vtype=GRB.CONTINUOUS, lb=0, name='Z')

        # adding objective
        m.setObjective(gp.quicksum(self.hc_dict[j] * (self.ar_dict[j, t] * Z[j, t] + self.br_dict[j, t] + U[j, t])
                                   for j, t in jt_list), GRB.MINIMIZE)

        # approximating constraint
        m.addConstrs((CT[j] == gp.quicksum(Z[j, t] for t in jt_dict[j]) for j in self.all_nodes), name='approx_CT')
        m.addConstrs(((t - 1) * U[j, t] <= Z[j, t] for j, t in jt_list), name='time_interval_lhs')
        m.addConstrs((t * U[j, t] >= Z[j, t] for j, t in jt_list), name='time_interval_rhs')
        m.addConstrs((gp.quicksum(U[j, t] for t in jt_dict[j]) == 1 for j in self.all_nodes), name='choose_one')

        # gsm constraint
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in self.all_nodes), name='covering_time')
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list),
                     name='edge')
        m.addConstrs((S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes),
                     name='sla')

        # setting parameters
        m.Params.TimeLimit = self.time_limit
        m.Params.MIPGap = self.opt_gap
        m.Params.Method = 3
        m.Params.LogToConsole = 0
        m.optimize()

        # get optimized solution
        if m.status == GRB.OPTIMAL:
            opt_sol = {'S': {node: round(float(S[node].x), 2) for node in self.all_nodes},
                       'SI': {node: round(float(SI[node].x), 2) for node in self.all_nodes},
                       'CT': {node: round(float(CT[node].x), 2) for node in self.all_nodes}}
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
        jt_dict = {j: list(range(0, int(self.cum_lt_dict[j]) + 1)) for j in self.all_nodes}

        env = cp.Envr()
        m = env.createModel('pwl')

        # service time
        S = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='S')
        # inbound service time
        SI = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='SI')
        # covering time
        CT = m.addVars(self.all_nodes, vtype=COPT.CONTINUOUS, lb=0, nameprefix='CT')
        # 0-1
        U = m.addVars(jt_list, vtype=COPT.BINARY, nameprefix='U')
        # approx
        Z = m.addVars(jt_list, vtype=COPT.CONTINUOUS, lb=0, nameprefix='Z')

        # adding objective
        m.setObjective(cp.quicksum(self.hc_dict[j] * (self.ar_dict[j, t] * Z[j, t] + self.br_dict[j, t] + U[j, t])
                                   for j, t in jt_list), COPT.MINIMIZE)

        # approximating constraint
        m.addConstrs((CT[j] == cp.quicksum(Z[j, t] for t in jt_dict[j]) for j in self.all_nodes),
                     nameprefix='approx_CT')
        m.addConstrs(((t - 1) * U[j, t] <= Z[j, t] for j, t in jt_list), nameprefix='time_interval_lhs')
        m.addConstrs((t * U[j, t] >= Z[j, t] for j, t in jt_list), nameprefix='time_interval_rhs')
        m.addConstrs((cp.quicksum(U[j, t] for t in jt_dict[j]) == 1 for j in self.all_nodes), nameprefix='choose_one')

        # gsm constraint
        m.addConstrs((CT[j] == SI[j] + self.lt_dict[j] - S[j] for j in self.all_nodes), nameprefix='covering_time')
        m.addConstrs((SI[succ] - S[pred] >= 0 for (pred, succ) in self.edge_list),
                     nameprefix='edge')
        m.addConstrs((S[j] <= int(self.sla_dict[j]) for j in self.demand_nodes),
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
            opt_sol = {'S': {node: round(float(S[node].x), 2) for node in self.all_nodes},
                       'SI': {node: round(float(SI[node].x), 2) for node in self.all_nodes},
                       'CT': {node: round(float(CT[node].x), 2) for node in self.all_nodes}}
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
        jt_dict = {j: list(range(0, int(self.cum_lt_dict[j]) + 1)) for j in self.all_nodes}

        m = pyo.ConcreteModel()

        # service time
        m.S = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)
        # inbound service time
        m.SI = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)
        # covering time
        m.CT = pyo.Var(self.all_nodes, domain=pyo.NonNegativeReals)
        # 0-1
        m.U = pyo.Var(jt_list, domain=pyo.Binary)
        # approx
        m.Z = pyo.Var(jt_list, domain=pyo.NonNegativeReals)

        # adding objective
        m.Cost = pyo.Objective(
            expr=sum([self.hc_dict[j] * (self.ar_dict[j, t] * m.Z[j, t] + self.br_dict[j, t] + m.U[j, t])
                      for j, t in jt_list]),
            sense=pyo.minimize
        )

        # constraints
        m.constrs = pyo.ConstraintList()
        for j in self.all_nodes:
            m.constrs.add(m.CT[j] == sum([m.Z[j, t] for t in jt_dict[j]]))
            m.constrs.add(m.CT[j] == m.SI[j] + self.lt_dict[j] - m.S[j])
            m.constrs.add(sum([m.U[j, t] for t in jt_dict[j]]) == 1)

        for j, t in jt_list:
            m.constrs.add((t - 1) * m.U[j, t] <= m.Z[j, t])
            m.constrs.add(t * m.U[j, t] >= m.Z[j, t])

        for pred, succ in self.edge_list:
            m.constrs.add(m.SI[succ] - m.S[pred] >= 0)

        for j in self.demand_nodes:
            m.constrs.add(m.S[j] <= int(self.sla_dict[j]))

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
        opt_sol = {'S': {node: round(float(m.S[node].value), 2) for node in self.all_nodes},
                   'SI': {node: round(float(m.SI[node].value), 2) for node in self.all_nodes},
                   'CT': {node: round(float(m.CT[node].value), 2) for node in self.all_nodes}}
        return opt_sol

    def get_approach_paras(self):
        paras = {'time_unit': self.time_unit, 'opt_gap': self.opt_gap, 'time_limit': self.time_limit}
        return paras
