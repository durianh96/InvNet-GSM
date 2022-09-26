# Large-scale Inventory Networks Optimizing Based on the Guaranteed Service Model

Optimizing inventory policy on a large-scale inventory network is challenging since it might involve massive nodes and many shared materials.
Two critical issues are: 
- choosing which nodes to place inventory 
- how much to set

The guaranteed service model (GSM) is one of the main approaches to optimizing network inventory policy.

This library provides several approaches to optimizing network inventory policy to solve the guaranteed service model (GSM). 
Users can input GSM instances in the required format and then call approaches to optimize policy. 
Or use our GSM instance generator to generate data for numerical tests.

This library is based on our paper:
- Optimizing Large-scale Inventory Networks: An Iterative Decomposition Approach (link).  

This paper is a pre-print at present and has not yet been peer-reviewed. 

Eight approaches of solving GSM are provided in this library: 

- **Dynamic programming (DP)** from (Graves and Willems 2000).
  This approach is built for tree networks, it takes advantage of the fact that each node in any tree can be labeled with unique indices such that every node except one has at most one adjacent node with an index higher than its own. 
  This approach can find the optimal solution for assembly and distribution problems with tree structure. 
- **Heuristic general networks algorithm (HGNA)** from (Humair and Willems 2011)
  This paper combines the above DP algorithm with a branch-and-bound scheme and provides an exact solution approach called **general networks algorithms (GNA)**. 
  GNA can find optimal solutions on general networks, but it takes a long time to find the solution for large-scale problems (a 2,025-nodes problem takes 577,190.78 seconds to find the optimal solution in their paper). 
  They provide two faster heuristics: **HGNA** and **TGNA**. HGNA is motivated by the structure of the formulation's dual space, whereas TGNA simply terminates the optimization algorithm after a fixed number of iterations.
  We found that HGNA takes a long time to converge on large-scale problems but performs better than TGNA. 
  We add a parameter *max iter num* to terminate the HGNA after a fixed number of iterations like TGNA. 
  Note that HGNA is based on a modified form of the DP algorithm. In fact, when the network is a tree, HGNA runs the above DP algorithm. That is, HGNA finds the optimal solution for the tree structure problem.
- **Piecewise linear approximation (PWL)** from (Magnanti et al. 2006)
  This approach uses piecewise linear functions to approximate the objective function of GSM. Doing that turns the original GSM into a mixed integer programming problem and can be solved with a MIP solver.
- **Dynamic sloping (DS)** and **iterative mixed integer programming (IMIP)** from (Shu and Karimi 2009). The first uses continuous approximation, while the second employs a two-piece linear approximation to approximate the concave objective function. 
- **Simple sequential linear programming (Simple-SLP)** from "(Huang et al. 2022)".
  This approach use sequential linear programming to find several local solutions and return the local solution with the least cost as the solution.
- **Iterative fixing with sequential linear programming (IF-SLP)** from (Huang et al. 2022).
  Similar to the Simple-SLP, in each round search for the local solution, this approach fix the variable values of stable nodes every *stable finding iter* iterations.
- **Iterative decomposition with sequential linear programming (ID-SLP)** from (Huang et al. 2022).
  This approach uses local solutions to decompose the large-scale graph into small sub-graphs iteratively. It combines the fast local solution-finding approach, SLP, with the optimal approach for tree problems (dynamic programming). Numerical results show that this approach performs best especially when the graph size is large and the graph structure is complex. 

We recommend the users to (Graves and Willems 2000), (Eruguz et al. 2016) and (Huang et al. 2022) for more details about the basics of GSM and descriptions of these approaches.

## GSM Instance: generating, saving and loading

One GSM instance contains a graph, all edges' proportions, and all nodes' properties related to GSM, including demand function, holding cost rate, lead time, and service time requirement for demand node.
We break the generation process into two parts: graph generation and properties generation.

We highly recommend generating related data at least once to understand how to prepare their own instance for users who want to import their self-data.

### Instance generating

First, we import our generators:


```python
from data_process import *
```

Second, we generate a graph. Users need to specify the following three parameters:

- *nodes num*: the number of nodes.
- *edges num*: the number of edges (can be empty for serial, assembly, and distribution graph).
- *graph type*: the graph structure, it can be 'serial', 'assembly', 'distribution' and 'general'.

For example, we can generate a general structure graph with 1000 nodes and 5000 edges: 


```python
nodes_num = 1000
edges_num = 5000
graph_type = 'general'
graph = generate_graph(nodes_num=nodes_num, edges_num=edges_num, graph_type=graph_type)
```

Third, we generate related properties of GSM on a given graph. Users need to specify the following parameters:

- Edge proportion range: *qty lb* and *qty ub*.
- Lead time range: *lt lb* and *lt ub*.
- Holding cost rate range: *hc lb* and *hc ub*.
- Service time requirement range parameters: *sla lt lb* and *sla lt ub*.
- Demand mean range: *mu lb* and *mu ub*.
- Demand standard deviation range: *sigma lb* and *sigma ub*.


```python
# generating an instance
instance_id = 'INSTANCE_01'
gsm_instance = generate_gsm_instance(graph=graph, 
                                     instance_id=instance_id, 
                                     qty_lb=1,
                                     qty_ub=3,
                                     lt_lb=1,
                                     lt_ub=10,
                                     hc_lb=0,
                                     hc_ub=1,
                                     sla_lt_lb=0,
                                     sla_lt_ub=10,
                                     mu_lb=0,
                                     mu_ub=100,
                                     sigma_lb=0,
                                     sigma_ub=10)
```

### Instance saving and loading

We provide function 'write_instance_to_csv' and 'load_instance_from_csv' to save and load instances respectively:


```python
# write to csv
instance_data_dir = 'data/' + instance_id + '/'
write_instance_to_csv(gsm_instance=gsm_instance, data_dir=instance_data_dir)
```


```python
# load from csv
load_instance_from_csv(data_dir=instance_data_dir)
```





For users who want to import their own instance data, they need provides three data files:
- 'instance_info.csv'
- 'node.csv'
- 'edge.csv'

'instance_info.csv' has three columns:
- *instance_id*: the unique index of the given instance.
- *tau*: the service level quantile of GSM.
- *pooling_factor*: the pooling factor of normal demand bound functions of GSM.


```python
instance_info_df = pd.read_csv(instance_data_dir + 'instance_info.csv')
instance_info_df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instance_info</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>instance_id</td>
      <td>INSTANCE_01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tau</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pooling_factor</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



'edge.csv' provides the graph edges information, each row is one edge of graph. This file contains three columns:
- *pred*: the predecessor of this edge.
- *succ*: the successor of this edge.
- *quantity*: the proportion indicating how many units of upstream node $i$'s materials are needed for each downstream node $j$.


```python
edge_df = pd.read_csv(instance_data_dir + 'edge.csv')
edge_df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred</th>
      <th>succ</th>
      <th>quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N000000</td>
      <td>N000224</td>
      <td>2.210904</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N000002</td>
      <td>N000758</td>
      <td>1.650182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N000008</td>
      <td>N000337</td>
      <td>2.922625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N000009</td>
      <td>N000039</td>
      <td>1.309693</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N000011</td>
      <td>N000237</td>
      <td>2.592944</td>
    </tr>
  </tbody>
</table>
</div>



'node.csv' provides the properties of nodes in the graph. It contains six columns:
- *node_id*: the unique index of the node.
- *lt*: the lead time of the node.
- *hc*: the holding cost rate of the node.
- *sla*: the service time requirement of the node.
- *mu*: the mean of the node's demand for each period.
- *sigma*: the std of the node's demand for each period.


```python
node_df = pd.read_csv(instance_data_dir + 'node.csv')
node_df.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>lt</th>
      <th>hc</th>
      <th>sla</th>
      <th>mu</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>N000783</td>
      <td>6.0</td>
      <td>2.331568e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>N000422</td>
      <td>3.0</td>
      <td>2.961675e+01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N000658</td>
      <td>3.0</td>
      <td>3.007234e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>N000631</td>
      <td>7.0</td>
      <td>4.409390e+02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N000742</td>
      <td>3.0</td>
      <td>1.660007e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Inventory policy optimizing

To optimize the inventory policy on given GSM instance, users first need to create a task, and specify:
- *task id*: the unique index of the task.
- *gsm_instance*
- *approach name*: the solving approach. 

As mentioned before, we provide eight approaches:
- 'DP'
- 'PWL'
- 'HGNA'
- 'DS'
- 'IMIP'
- 'Simple-SLP'
- 'IF-SLP'
- 'ID-SLP'

The default parameters of these approaches is given in 'default_paras.py'. More details about them can be found in (Huang et al. 2022).

For example, we can create a task to solve the above instance with iterative decomposition approach:


```python
from domain.task import Task
task_id = 'TASK_01'
task = Task(task_id=task_id, gsm_instance=gsm_instance, approach_name='ID-SLP')
```

Users can specify the solver for approaches that involve solving linear or integer programming problems. 
We provide Gurobi (https://www.gurobi.com/) and COPT (https://www.shanshu.ai/copt) choices with their naive Python interface. We also use pyomo 6.4.2 for unified modeling in our library so that the reader can use any solver supported by pyomo.

We provide five choices of solver:
- *GRB* uses the Gurobi interface of Python.
- *COPT* uses the COPT interface of Python.
- *PYO_GRB* uses the pyomo to model and solve with Gurobi.
- *PYO_COPT* uses the pyomo to model and solve with COPT.
- *PYO_CBC* uses the pyomo to model and solve with COIN-CBC.

For more solvers such as Cplex, GLPK and SCIP, users can slightly modify the code of approach to add support.

Here, we use Gurobi to optimize the inventory policy and write files:


```python
task.run(solver='GRB')
task_data_dir = instance_data_dir + task_id + '/'
task.write_to_csv(data_dir=task_data_dir)
```
  


## Reference
- Eruguz AS, Sahin E, Jemai Z, Dallery Y (2016) A comprehensive survey of guaranteed-service models for multi-echelon inventory optimization. International Journal of Production Economics 172:110–125. https://doi.org/10.1016/j.ijpe.2015.11.017
- Graves SC, Willems SP (2000) Optimizing strategic safety stock placement in supply chains. M&SOM 2:68–83. https://doi.org/10.1287/msom.2.1.68.23267
- Huang D, Yu J, Yang C (2022) Optimizing Large-scale Inventory Networks: An Iterative Decomposition Approach.
- Humair S, Willems SP (2011) TECHNICAL NOTE—Optimizing Strategic Safety Stock Placement in General Acyclic Networks. Operations Research 59:781–787. https://doi.org/10.1287/opre.1100.0913
- Magnanti TL, Shen Z-JM, Shu J, et al (2006) Inventory placement in acyclic supply chain networks. Operations Research Letters 34:228–238
- Shu J, Karimi IA (2009) Efficient heuristics for inventory placement in acyclic networks. Computers & Operations Research 36:2899–2904. https://doi.org/10.1016/j.cor.2009.01.001



