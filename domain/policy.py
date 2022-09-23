class Policy:
    def __init__(self, all_nodes: set):
        self.all_nodes = all_nodes
        self.sol_S = {node: None for node in all_nodes}
        self.sol_SI = {node: None for node in all_nodes}
        self.sol_CT = {node: None for node in all_nodes}
        self.base_stock = {node: None for node in all_nodes}
        self.safety_stock = {node: None for node in all_nodes}
        self.ss_cost = None

    def update_sol(self, sol):
        self.sol_S.update(sol['S'])
        self.sol_SI.update(sol['SI'])
        self.sol_CT.update(sol['CT'])

    def update_base_stock(self, base_stock):
        self.base_stock.update(base_stock)

    def update_safety_stock(self, safety_stock):
        self.safety_stock.update(safety_stock)

    def update_ss_cost(self, ss_cost):
        self.ss_cost = ss_cost
