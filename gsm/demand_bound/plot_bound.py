import matplotlib.pyplot as plt
import numpy as np
from gsm.demand_bound.node_demand_bound import NodeDemandBound


def plot_pwl(ndb: NodeDemandBound, func_type='vb'):
    x = ndb.grid_points
    if func_type == 'vb':
        y = [round(ndb.value_of_vb[xi], 2) for xi in x]
    elif func_type == 'db':
        y = [round(ndb.value_of_db[xi], 2) for xi in x]
    else:
        raise AttributeError
    plt.plot(x, y)
    plt.show()
