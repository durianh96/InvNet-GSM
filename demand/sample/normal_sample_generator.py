import random
import numpy as np
import pandas as pd
from demand.sample.node_sample import NodeSample
from demand.sample.default_sample_paras import *


def node_normal_samples_generating(n_id: str,
                                   intermittent_factor: float = INTERMITTENT_FACTOR,
                                   order_start_date: str = SAMPLE_START_DATE,
                                   order_end_date: str = SAMPLE_END_DATE,
                                   demand_mean_lb: float = NORMAL_DEMAND_MEAN_LB,
                                   demand_mean_ub: float = NORMAL_DEMAND_MEAN_UB,
                                   demand_std_lb: float = NORMAL_DEMAND_STD_LB,
                                   demand_std_ub: float = NORMAL_DEMAND_MEAN_UB):
    date_index = pd.date_range(start=order_start_date, end=order_end_date)
    date_index = [str(d) for d in date_index]
    num_zero = int(intermittent_factor * len(date_index))
    zero_index = sorted(random.choices(list(range(len(date_index))), k=num_zero))

    mean = np.random.uniform(demand_mean_lb, demand_mean_ub)
    std = np.random.uniform(demand_std_lb, demand_std_ub)
    qty_samples = [max(0., i) for i in np.random.normal(mean, std, len(date_index))]
    for i, date in enumerate(date_index):
        if i in zero_index and i < len(date_index) - 1:
            qty_samples[i + 1] += qty_samples[i]
            qty_samples[i] = 0

    ns = NodeSample(node_id=n_id)
    n_samples = []
    for i, order_date in enumerate(date_index):
        if i not in zero_index:
            n_samples.append((order_date, round(qty_samples[i], 2), n_id))
    ns.add_samples(n_samples)
    return ns
