from domain.task import Task
from data_process import *

nodes_num = 1000
edges_num = 5000
graph_type = 'general'

graph = generate_graph(nodes_num=nodes_num, edges_num=edges_num, graph_type=graph_type)
instance_id = 'INSTANCE_01'
gsm_instance = generate_gsm_instance(graph=graph, instance_id=instance_id)
instance_data_dir = 'data/' + instance_id + '/'
write_instance_to_csv(gsm_instance=gsm_instance, data_dir=instance_data_dir)

# load_instance_from_csv('')

task_id = 'TASK_01'
task = Task(task_id=task_id, gsm_instance=gsm_instance, approach_name='ID-SLP')
task.run(solver='PYO_GRB')
task_data_dir = instance_data_dir + task_id + '/'
task.write_to_csv(data_dir=task_data_dir)
