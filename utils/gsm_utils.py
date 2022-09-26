from utils.utils import logger


def cal_cost(hc_dict, policy_dict, method='_', w=True):
    cost = sum([hc_dict.get(node, 0) * policy for node, policy in policy_dict.items()])
    if w:
        logger.info(method + '_safety stock cost is %.2f' % cost)
    return cost


def check_solution_feasibility(gsm_instance, sol, error_tol=1e-3):
    error_sol = []
    S_dict = sol['S']
    SI_dict = sol['SI']
    CT_dict = sol['CT']
    lt_dict = gsm_instance.lt_dict
    for node in gsm_instance.all_nodes:
        # Si, SIi >= 0
        if (SI_dict[node] < -error_tol) or (S_dict[node] < -error_tol) or (CT_dict[node] < -error_tol):
            error_sol.append(['negative_error', 'node:' + str(node), 'S:' + str(S_dict[node]),
                              'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # Si <= sla_i
        if node in gsm_instance.sla_dict.keys():
            if S_dict[node] - gsm_instance.sla_dict[node] > error_tol:
                error_sol.append(['sla_error', 'node:' + str(node), 'sla:' + str(gsm_instance.sla_dict[node]),
                                  'S:' + str(S_dict[node]), 'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # Si - SIi <= Li for all nodes
        if abs(CT_dict[node] - (SI_dict[node] + lt_dict[node] - S_dict[node])) > error_tol:
            error_sol.append(['ct_error', 'node:' + str(node), 'lt:' + str(lt_dict[node]), 'S:' + str(S_dict[node]),
                              'SI:' + str(SI_dict[node]), 'CT:' + str(CT_dict[node])])

        # for (j,i): Sj <= SIi
        if len(gsm_instance.graph.succ_dict[node]) > 0:
            for succ in gsm_instance.graph.succ_dict[node]:
                if SI_dict[succ] - S_dict[node] < - error_tol:
                    error_sol.append(['link_error', 'pred:' + str(node), 'node_S:' + str(S_dict[node]),
                                      'succ:' + str(succ), 'succ_SI:' + str(SI_dict[succ])])
        return error_sol
