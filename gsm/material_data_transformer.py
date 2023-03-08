from data_builder.data_template.order_data import OrderData
from gsm.gsm_instance import GSMInstance
from inv_net.graph import MaterialGraph
from demand.sample.node_sample import NodeSample
from gsm.demand_bound.gsm_sample_collector import NodeSampleCollector


def material_data_transforming(instance_id: str, material_graph: MaterialGraph, order_data: OrderData):
    nodes = material_graph.nodes
    edges = material_graph.edges
    lt_of_node = {n_id: n.lt for n_id, n in material_graph.nodes_pool.items()}
    edge_qty = material_graph.edge_qty
    hc_of_node = {n_id: n.holding_cost for n_id, n in material_graph.nodes_pool.items()}
    sla_of_node = {n_id: n.sale_sla for n_id, n in material_graph.nodes_pool.items()
                   if (n_id in material_graph.sinks) and (n.sale_sla is not None)}
    cum_lt_of_node = {n_id: n.cum_lt for n_id, n in material_graph.nodes_pool.items()}
    gsm_instance = GSMInstance(
        instance_id=instance_id,
        nodes=nodes,
        edges=edges,
        lt_of_node=lt_of_node,
        edge_qty=edge_qty,
        hc_of_node=hc_of_node,
        sla_of_node=sla_of_node,
        cum_lt_of_node=cum_lt_of_node
    )

    ext_samples_pool = {n_id: NodeSample(n_id) for n_id, n in material_graph.sink_nodes_pool.items()}
    for n_id, n in material_graph.sink_nodes_pool.items():
        n_samples = order_data.get_material_site_order(material_id=n.material_id, site_id=n.site_id)
        n_samples = [(date, qty, n_id) for date, qty in n_samples]
        if len(n_samples) > 0:
            ext_samples_pool[n_id].add_samples(n_samples)

    ns_collector = NodeSampleCollector(gsm_instance)
    ns_collector.update_ext_samples_pool(ext_samples_pool)

    return gsm_instance, ns_collector
