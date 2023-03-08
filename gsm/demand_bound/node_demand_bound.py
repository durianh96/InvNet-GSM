from math import ceil
from typing import Optional
import numpy as np
from gsm.gsm_solving_approach.solving_default_paras import SYSTEM_TIME_UNIT
from utils.time_grid import find_right_time_grid_index


class NodeDemandBound:
    def __init__(self, node_id: str, cum_lt, grid_unit: Optional[float] = SYSTEM_TIME_UNIT):
        self.node_id = node_id
        self.cum_lt = ceil(cum_lt)

        self.db_func = None
        self.mean_func = None
        self.vb_func = None
        self.vb_gradient_func = None

        self.grid_unit = grid_unit
        self.grid_points = list(np.arange(0., self.cum_lt + grid_unit, grid_unit))
        self.value_of_db = {t: 0. for t in self.grid_points}
        self.value_of_vb = {t: 0. for t in self.grid_points}
        self.pwl_db_paras = None
        self.pwl_vb_paras = None

    def update_grid_unit(self, new_grid_unit):
        self.grid_unit = new_grid_unit
        self.grid_points = list(np.arange(0., self.cum_lt + self.grid_unit, self.grid_unit))

    def set_values_from_func(self):
        self.value_of_db = {t: self.db_func(t) for t in self.grid_points}
        self.update_pwl_db_paras()
        self.value_of_vb = {t: self.vb_func(t) for t in self.grid_points}
        self.update_pwl_vb_paras()

    def set_db_values_from_samples(self, db_samples):
        sample_grid = [t for t, db in db_samples]
        for t in self.grid_points:
            closet_sample_grid = np.argmin(np.abs(np.array(sample_grid) - t))
            self.value_of_db[t] = db_samples[closet_sample_grid][1]
        self.update_pwl_db_paras()

    def set_vb_values_from_samples(self, vb_samples):
        sample_grid = [t for t, vb in vb_samples]
        for t in self.grid_points:
            closet_sample_grid = np.argmin(np.abs(np.array(sample_grid) - t))
            self.value_of_vb[t] = vb_samples[closet_sample_grid][1]
        self.update_pwl_vb_paras()

    def update_pwl_db_paras(self):
        db_paras = {}
        for i, t in enumerate(self.grid_points):
            db_paras[t] = {}
            if i == 0:
                db_paras[t]['gradient'] = 0
                db_paras[t]['intercept'] = 0
            else:
                last_t = self.grid_points[i - 1]
                db_paras[t]['gradient'] = (self.value_of_db[t] - self.value_of_db[last_t]) / self.grid_unit
                db_paras[t]['intercept'] = self.value_of_db[t] - db_paras[t]['gradient'] * t

        self.pwl_db_paras = db_paras

    def update_pwl_vb_paras(self):
        vb_paras = {}
        for i, t in enumerate(self.grid_points):
            vb_paras[t] = {}
            if i == 0:
                vb_paras[t]['gradient'] = 0
                vb_paras[t]['intercept'] = 0
            else:
                last_t = self.grid_points[i - 1]
                vb_paras[t]['gradient'] = (self.value_of_vb[t] - self.value_of_vb[last_t]) / self.grid_unit
                vb_paras[t]['intercept'] = self.value_of_vb[t] - vb_paras[t]['gradient'] * t

        self.pwl_vb_paras = vb_paras

    def get_db_value(self, target_t):
        # index = find_closest_time_grid_index(self.grid_points, target_t)
        index = find_right_time_grid_index(self.grid_points, target_t)
        db_value = self.value_of_db[self.grid_points[index]]
        return db_value

    def get_vb_value(self, target_t):
        # index = find_closest_time_grid_index(self.grid_points, target_t)
        index = find_right_time_grid_index(self.grid_points, target_t)
        vb_value = self.value_of_vb[self.grid_points[index]]
        return vb_value

    def get_pwl_db_paras(self, target_t):
        right_index = find_right_time_grid_index(self.grid_points, target_t)
        t = self.grid_points[right_index]
        return self.pwl_db_paras[t]

    def get_pwl_vb_paras(self, target_t):
        right_index = find_right_time_grid_index(self.grid_points, target_t)
        t = self.grid_points[right_index]
        return self.pwl_vb_paras[t]

    def set_db_func(self, new_db_func):
        self.db_func = new_db_func

    def set_mean_func(self, new_mean_func):
        self.mean_func = new_mean_func

    def set_vb_func(self, new_vb_func):
        self.vb_func = new_vb_func

    def set_vb_gradient_func(self, new_vb_gradient_func):
        self.vb_gradient_func = new_vb_gradient_func
