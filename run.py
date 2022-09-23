from domain.task import Task
from data_process import *

nodes_num = 1000
edges_num = 5000
graph_type = 'general'

graph = generate_graph(nodes_num, edges_num, graph_type)
gsm_instance = generate_gsm_instance(graph, 'INSTANCE_01')
write_instance_to_csv(gsm_instance, 'data/')
# load_instance_from_csv('')

task = Task('TASK_01', gsm_instance, 'ID-SLP')
task.run()
task.write_to_csv('data/')

task = Task('TASK_02', gsm_instance, 'IF-SLP')
task.run()
task.write_to_csv('data/')
