from utils.edges_loader import *
from utils.edges_generator import *
from gsm.generator.gsm_instance_generator import gsm_instance_generating_given_paras_ranges
from gsm.gsm_task import GSMTask


# load real network from willems(2008)
real_network_id = 10
edges = edges_loading_from_willems(real_network_id)


# based on the network, generate the properties of gsm instance
gsm_instance = gsm_instance_generating_given_paras_ranges(instance_id='INSTANCE_01', edges=edges)

# run gsm instance with input approach
task = GSMTask(task_id='TASK_01', gsm_instance=gsm_instance, approach_name='HGNA')
task.run()

task = GSMTask(task_id='TASK_02', gsm_instance=gsm_instance, approach_name='SA')
task.run()

task = GSMTask(task_id='TASK_03', gsm_instance=gsm_instance, approach_name='ID-SLP')
task.run(solver='GRB')

task = GSMTask(task_id='TASK_04', gsm_instance=gsm_instance, approach_name='DS')
task.run(solver='GRB')

task = GSMTask(task_id='TASK_05', gsm_instance=gsm_instance, approach_name='PWL')
task.run(solver='GRB')

task = GSMTask(task_id='TASK_06', gsm_instance=gsm_instance, approach_name='Simple-SLP')
task.run(solver='GRB')

task = GSMTask(task_id='TASK_07', gsm_instance=gsm_instance, approach_name='IMIP')
task.run(solver='GRB')
