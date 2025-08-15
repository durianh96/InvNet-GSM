import numpy as np
import scipy.special as sc
from utils.system_utils import logger


class GSMSolution:
    def __init__(self, nodes: set):
        self.nodes = nodes
        self.S_of_node = {node: None for node in nodes}
        self.SI_of_node = {node: None for node in nodes}
        self.CT_of_node = {node: None for node in nodes}
        self.oul_of_node = {node: None for node in nodes}
        self.ss_of_node = {node: None for node in nodes}
        self.ss_cost = None

        self.stable_nodes = set()
        self.decompose_round_of_stable_node = {}
    
    def collect_sol_set(self, sol_set):
        self.sol_set = sol_set

    def update_sol(self, sol):
        self.S_of_node.update(sol['S'])
        self.SI_of_node.update(sol['SI'])
        self.CT_of_node.update(sol['CT'])

    def update_oul(self, oul_of_node):
        self.oul_of_node.update(oul_of_node)

    def update_ss(self, ss_of_node):
        self.ss_of_node.update(ss_of_node)

    def update_ss_cost(self, ss_cost):
        self.ss_cost = ss_cost
    
    def update_stable_info(self, decompose_round_of_stable_node):
        self.stable_nodes = set(decompose_round_of_stable_node.keys())
        self.decompose_round_of_stable_node.update(decompose_round_of_stable_node)


class GSMSolutionSet:
    def __init__(self, nodes: set, lt_of_node: dict, cum_lt_of_node: dict, sla_of_node: dict):
        self.nodes = nodes
        self.lt_of_node = lt_of_node
        self.cum_lt_of_node = cum_lt_of_node
        self.sla_of_node = sla_of_node
        self.gsm_sols = []

        # beta parameters
        self.S_beta_para = None
        self.SI_beta_para = None
        self.CT_beta_para = None
        self.S_ub = None
        self.SI_ub = None
        self.CT_ub = None

    def add_one_sol(self, new_sol: GSMSolution, update_beta=False):
        self.gsm_sols.append(new_sol)
        if update_beta and len(self.gsm_sols) > 0:
            self.update_beta_para()

    def get_best_local_sol(self):
        cost_list = [sol.ss_cost for sol in self.gsm_sols]
        best_i = np.argmin(cost_list)
        best_sol = self.gsm_sols[best_i]
        return best_sol

    def get_local_sol_info(self, stability_type='cv', stability_threshold=0):
        if len(self.gsm_sols) == 0:
            local_sol_info = {'fix_S_nodes': set(), 'fix_SI_nodes': set(), 'fix_CT_nodes': set(),
                              'completely_fix_nodes': set(), 'completely_free_nodes': self.nodes,
                              'partially_free_nodes': self.nodes,
                              'solely_fix_S_nodes': set(), 'free_S_nodes': self.nodes,
                              'solely_fix_SI_nodes': set(), 'free_SI_nodes': self.nodes,
                              'solely_fix_CT_nodes': set(), 'free_CT_nodes': self.nodes,
                              'fix_S': {}, 'fix_SI': {}, 'fix_CT': {},
                              'completely_fix_S': {}, 'completely_fix_SI': {}, 'completely_fix_CT': {}}
            return local_sol_info
        S_results = {node: [gsm_sol.S_of_node[node] for gsm_sol in self.gsm_sols] for node in self.nodes}
        S_mean_of_node = {node: np.mean(S_results[node]) for node in self.nodes}

        SI_results = {node: [gsm_sol.SI_of_node[node] for gsm_sol in self.gsm_sols] for node in self.nodes}
        SI_mean_of_node = {node: np.mean(SI_results[node]) for node in self.nodes}

        CT_results = {node: [gsm_sol.CT_of_node[node] for gsm_sol in self.gsm_sols] for node in self.nodes}
        CT_mean_of_node = {node: np.mean(CT_results[node]) for node in self.nodes}

        if stability_type == 'kl':
            S_stat_dict = self.get_last_kl(var_type='S')
            SI_stat_dict = self.get_last_kl(var_type='SI')
            CT_stat_dict = self.get_last_kl(var_type='CT')
        elif stability_type == 'cn':
            S_stat_dict = self.get_last_cn(var_type='S')
            SI_stat_dict = self.get_last_cn(var_type='SI')
            CT_stat_dict = self.get_last_cn(var_type='CT')
        else:
            node_S_std = {node: np.std(S_results[node]) for node in self.nodes}
            S_stat_dict = {node: node_S_std[node] / (S_mean_of_node[node] if S_mean_of_node[node] > 0 else 1)
                           for node in self.nodes}
            node_SI_std = {node: np.std(SI_results[node]) for node in self.nodes}
            SI_stat_dict = {node: node_SI_std[node] / (SI_mean_of_node[node] if SI_mean_of_node[node] > 0 else 1)
                            for node in self.nodes}
            node_CT_std = {node: np.std(CT_results[node]) for node in self.nodes}
            CT_stat_dict = {node: node_CT_std[node] / (CT_mean_of_node[node] if CT_mean_of_node[node] > 0 else 1)
                            for node in self.nodes}

        stationary_S_node = [node for node, v in S_stat_dict.items() if v <= stability_threshold]
        stationary_SI_node = [node for node, v in SI_stat_dict.items() if v <= stability_threshold]
        stationary_CT_node = [node for node, v in CT_stat_dict.items() if v <= stability_threshold]

        fix_S = {}
        fix_SI = {}
        fix_CT = {}
        for node in self.nodes:
            if node in stationary_S_node:
                fix_S[node] = float(round(S_mean_of_node[node]))
                if node in stationary_SI_node:
                    fix_SI[node] = float(round(SI_mean_of_node[node]))
                    fix_CT[node] = max(fix_SI[node] + self.lt_of_node[node] - fix_S[node], 0)
                elif node in stationary_CT_node:
                    fix_CT[node] = float(round(CT_mean_of_node[node]))
                    fix_SI[node] = fix_CT[node] + fix_S[node] - self.lt_of_node[node]
            elif node in stationary_SI_node:
                fix_SI[node] = float(round(SI_mean_of_node[node]))
                if node in stationary_CT_node:
                    fix_CT[node] = float(round(CT_mean_of_node[node]))
                    fix_S[node] = fix_SI[node] + self.lt_of_node[node] - fix_CT[node]

        fix_S_nodes = set(fix_S.keys())
        fix_SI_nodes = set(fix_SI.keys())
        fix_CT_nodes = set(fix_CT.keys())

        completely_fix_nodes = fix_S_nodes & fix_SI_nodes & fix_CT_nodes

        solely_fix_S_nodes = fix_S_nodes - completely_fix_nodes
        free_S_nodes = self.nodes - fix_S_nodes
        solely_fix_SI_nodes = fix_SI_nodes - completely_fix_nodes
        free_SI_nodes = self.nodes - fix_SI_nodes
        solely_fix_CT_nodes = fix_CT_nodes - completely_fix_nodes
        free_CT_nodes = self.nodes - fix_CT_nodes
        completely_free_nodes = self.nodes - fix_S_nodes - fix_SI_nodes - fix_CT_nodes
        partially_free_nodes = self.nodes - completely_fix_nodes

        completely_fix_S = {j: fix_S[j] for j in completely_fix_nodes}
        completely_fix_SI = {j: fix_SI[j] for j in completely_fix_nodes}
        completely_fix_CT = {j: fix_CT[j] for j in completely_fix_nodes}

        local_sol_info = {'fix_S_nodes': fix_S_nodes, 'fix_SI_nodes': fix_SI_nodes, 'fix_CT_nodes': fix_CT_nodes,
                          'completely_fix_nodes': completely_fix_nodes, 'completely_free_nodes': completely_free_nodes,
                          'partially_free_nodes': partially_free_nodes,
                          'solely_fix_S_nodes': solely_fix_S_nodes, 'free_S_nodes': free_S_nodes,
                          'solely_fix_SI_nodes': solely_fix_SI_nodes, 'free_SI_nodes': free_SI_nodes,
                          'solely_fix_CT_nodes': solely_fix_CT_nodes, 'free_CT_nodes': free_CT_nodes,
                          'fix_S': fix_S, 'fix_SI': fix_SI, 'fix_CT': fix_CT,
                          'completely_fix_S': completely_fix_S, 'completely_fix_SI': completely_fix_SI,
                          'completely_fix_CT': completely_fix_CT}
        return local_sol_info

    def init_beta_para(self):
        self.S_beta_para = {'alpha': {node: [1] for node in self.nodes},
                            'beta': {node: [1] for node in self.nodes}}
        self.SI_beta_para = {'alpha': {node: [1] for node in self.nodes},
                             'beta': {node: [1] for node in self.nodes}}
        self.CT_beta_para = {'alpha': {node: [1] for node in self.nodes},
                             'beta': {node: [1] for node in self.nodes}}
        self.S_ub = {node: min(self.sla_of_node.get(node, 9999), self.cum_lt_of_node[node]) for node in self.nodes}
        self.SI_ub = {node: self.cum_lt_of_node[node] - self.lt_of_node[node] for node in self.nodes}
        self.CT_ub = {node: self.cum_lt_of_node[node] for node in self.nodes}

    def update_beta_para(self):
        new_S = self.gsm_sols[-1].S_of_node
        new_SI = self.gsm_sols[-1].SI_of_node
        new_CT = self.gsm_sols[-1].CT_of_node
        for node in self.nodes:
            # update S para
            if self.S_ub[node] > 0:
                new_S_alpha = self.S_beta_para['alpha'][node][-1] + (new_S[node] / self.S_ub[node])
                new_S_beta = self.S_beta_para['beta'][node][-1] + 1 - (new_S[node] / self.S_ub[node])
                self.S_beta_para['alpha'][node].append(new_S_alpha)
                self.S_beta_para['beta'][node].append(new_S_beta)
            else:
                self.S_beta_para['alpha'][node].append(self.S_beta_para['alpha'][node][-1])
                self.S_beta_para['beta'][node].append(self.S_beta_para['beta'][node][-1])
            # update SI para
            if self.SI_ub[node] > 0:
                new_SI_alpha = self.SI_beta_para['alpha'][node][-1] + (new_SI[node] / self.SI_ub[node])
                new_SI_beta = self.SI_beta_para['beta'][node][-1] + 1 - (new_SI[node] / self.SI_ub[node])
                self.SI_beta_para['alpha'][node].append(new_SI_alpha)
                self.SI_beta_para['beta'][node].append(new_SI_beta)
            else:
                self.SI_beta_para['alpha'][node].append(self.SI_beta_para['alpha'][node][-1])
                self.SI_beta_para['beta'][node].append(self.SI_beta_para['beta'][node][-1])
            # update CT para
            if self.CT_ub[node] > 0:
                new_CT_alpha = self.CT_beta_para['alpha'][node][-1] + (new_CT[node] / self.CT_ub[node])
                new_CT_beta = self.CT_beta_para['beta'][node][-1] + 1 - (new_CT[node] / self.CT_ub[node])
                self.CT_beta_para['alpha'][node].append(new_CT_alpha)
                self.CT_beta_para['beta'][node].append(new_CT_beta)
            else:
                self.CT_beta_para['alpha'][node].append(self.CT_beta_para['alpha'][node][-1])
                self.CT_beta_para['beta'][node].append(self.CT_beta_para['beta'][node][-1])

    def get_last_cn(self, var_type):
        if var_type == 'S':
            cn_dict = {node: chernoff_distance(self.S_beta_para['alpha'][node][-2], self.S_beta_para['beta'][node][-2],
                                               self.S_beta_para['alpha'][node][-1], self.S_beta_para['beta'][node][-1])
                       for node in self.nodes}
        elif var_type == 'SI':
            cn_dict = {
                node: chernoff_distance(self.SI_beta_para['alpha'][node][-2], self.SI_beta_para['beta'][node][-2],
                                        self.SI_beta_para['alpha'][node][-1], self.SI_beta_para['beta'][node][-1])
                for node in self.nodes}
        else:
            cn_dict = {
                node: chernoff_distance(self.CT_beta_para['alpha'][node][-2], self.CT_beta_para['beta'][node][-2],
                                        self.CT_beta_para['alpha'][node][-1], self.CT_beta_para['beta'][node][-1])
                for node in self.nodes}
        return cn_dict

    def get_last_kl(self, var_type):
        if var_type == 'S':
            kl_dict = {node: kl_divergence(self.S_beta_para['alpha'][node][-2], self.S_beta_para['beta'][node][-2],
                                           self.S_beta_para['alpha'][node][-1], self.S_beta_para['beta'][node][-1])
                       for node in self.nodes}
        elif var_type == 'SI':
            kl_dict = {node: kl_divergence(self.SI_beta_para['alpha'][node][-2], self.SI_beta_para['beta'][node][-2],
                                           self.SI_beta_para['alpha'][node][-1], self.SI_beta_para['beta'][node][-1])
                       for node in self.nodes}
        else:
            kl_dict = {node: kl_divergence(self.CT_beta_para['alpha'][node][-2], self.CT_beta_para['beta'][node][-2],
                                           self.CT_beta_para['alpha'][node][-1], self.CT_beta_para['beta'][node][-1])
                       for node in self.nodes}
        return kl_dict


def beta_dist_expect(alpha, beta):
    return alpha / (alpha + beta)


def chernoff_distance(alpha1, beta1, alpha2, beta2, lam=0.5):
    cn = np.log(sc.gamma(lam * alpha1 + (1 - lam) * alpha2 + lam * beta1 + (1 - lam) * beta2)) \
         + lam * (np.log(sc.gamma(alpha1)) + np.log(sc.gamma(beta1))) \
         + (1 - lam) * (np.log(sc.gamma(alpha2)) + np.log(sc.gamma(beta2))) \
         - np.log(sc.gamma(lam * alpha1 + (1 - lam) * alpha2)) - np.log(sc.gamma(lam * beta1 + (1 - lam) * beta2)) \
         - lam * np.log(sc.gamma(alpha1 + beta1)) - (1 - lam) * np.log(sc.gamma(alpha2 + beta2))
    return cn


def kl_divergence(alpha1, beta1, alpha2, beta2):
    kl = np.log(sc.beta(alpha2, beta2) / sc.beta(alpha1, beta1)) \
         + (alpha1 - alpha2) * sc.digamma(alpha1) + (beta1 - beta2) * sc.digamma(beta1) \
         + (alpha2 - alpha1 + beta2 - beta1) * sc.digamma(alpha1 + beta1)
    return kl


def cal_ss_cost(hc_of_node, ss_of_node, method='_', w=True):
    cost = sum([hc_of_node.get(node, 0) * ss for node, ss in ss_of_node.items()])
    if w:
        logger.info(method + '_safety stock cost is %.2f' % cost)
    return cost


def check_solution_feasibility(gsm_instance, sol, error_tol=1e-3):
    error_sol = []
    S_dict = sol['S']
    SI_dict = sol['SI']
    CT_dict = sol['CT']
    lt_dict = gsm_instance.lt_of_node
    for node in gsm_instance.nodes:
        # Si, SIi >= 0
        if (SI_dict[node] < -error_tol) or (S_dict[node] < -error_tol) or (CT_dict[node] < -error_tol):
            error_sol.append(['negative_error', 'node:' + str(node), 'S:' + str(S_dict[node]),
                              'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # Si <= sla_i
        if node in gsm_instance.sla_of_node.keys():
            if S_dict[node] - gsm_instance.sla_of_node[node] > error_tol:
                error_sol.append(['sla_error', 'node:' + str(node), 'sla:' + str(gsm_instance.sla_of_node[node]),
                                  'S:' + str(S_dict[node]), 'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # todo: add
        # Si - SIi <= Li for all nodes
        if (abs(CT_dict[node] - (SI_dict[node] + lt_dict[node] - S_dict[node])) > error_tol) \
            & (SI_dict[node] + lt_dict[node] - S_dict[node] > 0) :
            error_sol.append(['ct_error', 'node:' + str(node), 'lt:' + str(lt_dict[node]), 'S:' + str(S_dict[node]),
                              'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # for (j,i): Sj <= SIi
        if len(gsm_instance.succs_of_node[node]) > 0:
            for succ in gsm_instance.succs_of_node[node]:
                if SI_dict[succ] - S_dict[node] < - error_tol:
                    error_sol.append(['link_error', 'pred:' + str(node), 'node_S:' + str(S_dict[node]),
                                      'succ:' + str(succ), 'succ_SI:' + str(SI_dict[succ])])
    
    return error_sol
