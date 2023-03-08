from typing import Optional
from gsm.demand_bound.node_demand_bound import NodeDemandBound
from gsm.demand_bound.normal_bound.normal_bound_funcs import net_normal_db_func, mean_func, net_normal_vb_func, \
    net_normal_vb_gradient_func, cascading_normal_db_func, cascading_normal_vb_func, cascading_normal_vb_gradient_func, \
    get_cascading_normal_bound_para
from gsm.gsm_instance import GSMInstance
from gsm.demand_bound.gsm_sample_collector import NodeSampleCollector


class DemandBoundCollector:
    def __init__(self, gsm_instance: GSMInstance, ns_collector: Optional[NodeSampleCollector] = None):
        self.gsm_instance = gsm_instance
        self.ns_collector = ns_collector
        self.demand_bound_pool = {n_id: NodeDemandBound(n_id, self.gsm_instance.cum_lt_of_node.get(n_id))
                                  for n_id in self.gsm_instance.nodes}

    def update_sample_collector(self, new_sample_collector):
        self.ns_collector = new_sample_collector

    def set_net_normal_bound_from_sample(self, tau, agg_para='D', start_date: Optional[str] = None,
                                         end_date: Optional[str] = None):
        net_mean_of_node = self.ns_collector.get_net_mean_of_node(agg_para, start_date, end_date)
        net_std_of_node = self.ns_collector.get_net_std_of_node(agg_para, start_date, end_date)
        for n_id, ndb in self.demand_bound_pool.items():
            ndb.set_db_func(net_normal_db_func(net_mean_of_node[n_id], net_std_of_node[n_id], tau))
            ndb.set_mean_func(mean_func(net_mean_of_node[n_id]))
            ndb.set_vb_func(net_normal_vb_func(net_std_of_node[n_id], tau))
            ndb.set_vb_gradient_func(net_normal_vb_gradient_func(net_std_of_node[n_id], tau))
            ndb.set_values_from_func()

    def set_cascading_normal_bound_from_sample(self, tau, pooling_factor, agg_para='D',
                                               start_date: Optional[str] = None,
                                               end_date: Optional[str] = None):
        ext_mean_of_node = self.ns_collector.get_ext_mean_of_node(agg_para, start_date, end_date)
        ext_std_of_node = self.ns_collector.get_ext_std_of_node(agg_para, start_date, end_date)
        cascading_mean_of_node, volatility_constant_of_node = get_cascading_normal_bound_para(
            edges=self.gsm_instance.edges,
            edge_qty=self.gsm_instance.edge_qty,
            ext_mean_of_node=ext_mean_of_node,
            ext_std_of_node=ext_std_of_node,
            tau=tau,
            pooling_factor=pooling_factor
        )
        for n_id, ndb in self.demand_bound_pool.items():
            ndb.set_db_func(cascading_normal_db_func(cascading_mean_of_node[n_id], volatility_constant_of_node[n_id]))
            ndb.set_mean_func(mean_func(cascading_mean_of_node[n_id]))
            ndb.set_vb_func(cascading_normal_vb_func(volatility_constant_of_node[n_id]))
            ndb.set_vb_gradient_func(cascading_normal_vb_gradient_func(volatility_constant_of_node[n_id]))
            ndb.set_values_from_func()

    def set_cascading_normal_bound_given_paras(self, ext_mean_of_node: dict, ext_std_of_node: dict, tau, pooling_factor):
        cascading_mean_of_node, volatility_constant_of_node = get_cascading_normal_bound_para(
            edges=self.gsm_instance.edges,
            edge_qty=self.gsm_instance.edge_qty,
            ext_mean_of_node=ext_mean_of_node,
            ext_std_of_node=ext_std_of_node,
            tau=tau,
            pooling_factor=pooling_factor
        )
        for n_id, ndb in self.demand_bound_pool.items():
            ndb.set_db_func(cascading_normal_db_func(cascading_mean_of_node[n_id], volatility_constant_of_node[n_id]))
            ndb.set_mean_func(mean_func(cascading_mean_of_node[n_id]))
            ndb.set_vb_func(cascading_normal_vb_func(volatility_constant_of_node[n_id]))
            ndb.set_vb_gradient_func(cascading_normal_vb_gradient_func(volatility_constant_of_node[n_id]))
            ndb.set_values_from_func()

