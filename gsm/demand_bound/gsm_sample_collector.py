from typing import Optional
from gsm.demand_bound.cascading_stats import get_cascading_normal_stats
from demand.sample.node_sample import NodeSample
from gsm.gsm_instance import GSMInstance
from utils.graph_algorithms import find_ancestors, cal_net_qty


class NodeSampleCollector:
    def __init__(self, gsm_instance: GSMInstance, ext_samples_pool: Optional[dict] = None):
        self.gsm_instance = gsm_instance
        if ext_samples_pool is None:
            self.ext_samples_pool = {n_id: NodeSample(n_id) for n_id in self.gsm_instance.sinks}
        else:
            self.ext_samples_pool = ext_samples_pool
        self.net_samples_pool = {n_id: NodeSample(n_id) for n_id in self.gsm_instance.nodes}

    def update_ext_samples_pool(self, new_ext_samples_pool):
        self.ext_samples_pool = new_ext_samples_pool
        self.update_net_samples_pool()

    def update_net_samples_pool(self):
        preds_of_node = self.gsm_instance.preds_of_node
        net_qty = cal_net_qty(self.gsm_instance.edges, self.gsm_instance.edge_qty)
        for n_id in self.gsm_instance.sinks:
            n_ancestors, _ = find_ancestors(preds_of_node, n_id)
            for an in n_ancestors:
                an_samples = [(date, qty * net_qty[an, n_id], n_id)
                              for date, qty, _ in self.ext_samples_pool[n_id].samples]
                self.net_samples_pool[an].add_samples(an_samples)

    def get_ext_mean_of_node(self, agg_para='D', start_date: Optional[str] = None,
                             end_date: Optional[str] = None):
        ext_mean_of_node = {n_id: ns.get_sample_mean(agg_para, start_date, end_date)
                            for n_id, ns in self.ext_samples_pool.items()}
        ext_mean_of_node = {n_id: v for n_id, v in ext_mean_of_node.items() if n_id in self.gsm_instance.sinks}
        return ext_mean_of_node

    def get_ext_std_of_node(self, agg_para='D', start_date: Optional[str] = None,
                            end_date: Optional[str] = None):
        ext_std_of_node = {n_id: ns.get_sample_std(agg_para, start_date, end_date)
                           for n_id, ns in self.ext_samples_pool.items()}
        ext_std_of_node = {n_id: v for n_id, v in ext_std_of_node.items() if n_id in self.gsm_instance.sinks}
        return ext_std_of_node

    def get_net_mean_of_node(self, agg_para='D', start_date: Optional[str] = None,
                             end_date: Optional[str] = None):
        net_mean_of_node = {n_id: ns.get_sample_mean(agg_para, start_date, end_date)
                            for n_id, ns in self.net_samples_pool.items()}
        return net_mean_of_node

    def get_net_std_of_node(self, agg_para='D', start_date: Optional[str] = None,
                            end_date: Optional[str] = None):
        net_std_of_node = {n_id: ns.get_sample_std(agg_para, start_date, end_date)
                           for n_id, ns in self.net_samples_pool.items()}
        return net_std_of_node

    def get_cascading_stats_of_node(self, agg_para='D', start_date: Optional[str] = None,
                                    end_date: Optional[str] = None):
        ext_mean_of_node = self.get_ext_mean_of_node(agg_para, start_date, end_date)
        ext_std_of_node = self.get_ext_std_of_node(agg_para, start_date, end_date)

        cascading_mean_of_node, cascading_std_of_node = get_cascading_normal_stats(
            edges=self.gsm_instance.edges,
            edge_qty=self.gsm_instance.edge_qty,
            ext_mean_of_node=ext_mean_of_node,
            ext_std_of_node=ext_std_of_node,
        )
        return cascading_mean_of_node, cascading_std_of_node


