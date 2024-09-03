import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import scipy as sp
import time

#Generating a Random Connected Graph
def generate_random_connected_graph(n_nodes, min_degree_m, max_degree_m):
    G = nx.Graph()
    G.add_node(0)

    for i in range(1, n_nodes):
        node_to_connect = random.choice(list(G.nodes))
        G.add_edge(i, node_to_connect)
    
    for node in G.nodes():
        connections = random.randint(min_degree_m, max_degree_m)
        #print("The degree of a node:", node, connections)
        neighbors_nodes = set(G.neighbors(node))
        connections = connections - len(neighbors_nodes)
        potential_nodes = set(range(n_nodes)) - {node} - set(G[node])
        if connections > 0:
            for _ in range(connections):
                if not potential_nodes:
                    break
                new_node = random.choice(list(potential_nodes))
                G.add_edge(node, new_node)
                potential_nodes.remove(new_node)
    return G



#Calculate the number of connected components in the subgraph induced by the vertices of the graph
def number_c1(G, vertices):
    sub_graph = G.subgraph(vertices)
    return nx.number_connected_components(sub_graph)

#Calculate the additional number of coverages needed for the vertices_1
def need_cover(G, vertice, vertices, rand_m):
    neighbors = set(G.neighbors(vertice))
    cover_need = rand_m - len(neighbors & vertices)
    if cover_need < 0:
        cover_need = 0 
    return cover_need

#Calculate the additional number of coverages needed for the vertices_2
def need_cover_all(G, vertices, rand_m):
    total_need = 0
    for node in G.nodes:
        if node not in vertices:
            neighbors = set(G.neighbors(node))
            vertices = set(vertices)
            cover_need = rand_m - len(neighbors & vertices)
            if cover_need > 0:
                total_need += cover_need
    return total_need

#定义potential_function
def potential_function(G, vertices, rand_m):
    number_f = number_c1(G, vertices) + need_cover_all(G, vertices, rand_m)
    return number_f

#Define the neighborhood of vertex that satisfies the lemma conditions
def condition_neighbors(graph, vertex, vertices):
    neighbors_1 = set()
    neighbors_2 = set(graph.neighbors(vertex)) - vertices
    for u in neighbors_2 :
        S_1 = vertices.copy()
        S_1.add(u)
        cover_need = need_cover_all(G, vertices, rand_m) - need_cover_all(G, S_1, rand_m)
        number_c =  number_c1(graph, vertices) - number_c1(graph, S_1)
        if cover_need == 0 and number_c == 0:
            neighbors_1.add(u)
    return neighbors_1  

#Output the point with the smallest weight in the set
def min_weight_node(G, vertices):
    weight_1 = float('inf')
    for node in vertices:
        if G.nodes[node]['weight'] < weight_1:
            min_vertex = node
            weight_1 = G.nodes[node]['weight']
    return min_vertex

#Find the most efficient star structure
def algorithm_2(G, vertices, rand_m):
    max_benefit_star = set()
    max_benefit_rate = 0
    for vertex in G.nodes() - vertices:
        potential_Star = {vertex}
        tem_set =vertices.copy()
        tem_set.add(vertex)
        weight_v1 = G.nodes[vertex]['weight']
        marginal_benefit = - potential_function(G, tem_set, rand_m) + potential_function(G, vertices, rand_m)
        marginal_benefit_rate = marginal_benefit/weight_v1
        if need_cover(G, vertex, vertices, rand_m) == 0: 
            neighbor_set2 = condition_neighbors(G, vertex, vertices) 
            #lenth_set = len(neighbor_set2)
            #if lenth_set > 0:
                #print("Neighbor set satisfying the lemma condition:", neighbor_set2)
            while len(neighbor_set2) != 0:
                u = min_weight_node(G, neighbor_set2)
                neighbor_set2.remove(u)
                weight_v2 = G.nodes[u]['weight']
                tem_set2 = tem_set.copy()
                tem_set2.add(u)
                if number_c1(G, tem_set) - number_c1(G, tem_set2) == 1 and 1/weight_v2 > marginal_benefit_rate:
                    potential_Star.add(u)
                    tem_set.add(u)
                    weight_v1 += weight_v2
                    marginal_benefit += 1  
                    marginal_benefit_rate = marginal_benefit/weight_v1 
        if marginal_benefit_rate > max_benefit_rate:
            max_benefit_star = potential_Star
            max_benefit_rate = marginal_benefit_rate
    return max_benefit_star

#Compute a (1, m) CDS
def algorithm_1(G, rand_m):
    i = 1
    cd_set = set()  
    weight_set = 0
    while number_c1(G, cd_set) + need_cover_all(G, cd_set, rand_m) > 1 :
        eff_star = algorithm_2(G, cd_set, rand_m)
        weight_set = sum(G.nodes[node]['weight'] for node in eff_star)
        lenth_set_1 = len(eff_star)
        if lenth_set_1 > 1:
            print("Top", i, "The optimal star structure:", eff_star, weight_set)
        cd_set.update(eff_star)
        i += 1
    return cd_set

#Calculate an optimal (1, m) CDS
def optimal_algorithm_1(G, rand_m):
    cd_set = set()  
    min_weight = float('inf')
    for i in range(2, n_nodes):
        node_pairs = list(itertools.combinations(G.nodes, i))
        # Enumerate all subsets containing i points
        for pair in node_pairs:
            number_f = potential_function(G, pair, rand_m)
            if number_f == 1:
                weight_1 = sum(G.nodes[node]['weight'] for node in pair)
                if weight_1 < min_weight:
                    min_weight = weight_1
                    pair = set(pair)
                    cd_set = pair
                    print("Connected Dominating Set:", cd_set, min_weight)
    return cd_set


#Number of iterations
repeat_time = 10
number_weight_1 = 0
number_weight_2 = 0
for i in range(repeat_time) :
  
    n_nodes = 30
    min_degree_m = 2
    max_degree_m = 4
    G = generate_random_connected_graph(n_nodes, min_degree_m, max_degree_m)

    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())

    #Assign weights to each vertex in the randomly generated connected graph
    for node in G.nodes() :
        G.nodes[node]['weight'] = random.randint(1, 10000)
        #print(f"Node {node} weight: {G.nodes[node]['weight']}")

    rand_m = 2
    start_time = time.time()
    #Run Algorithm 1 to compute the connected dominating set of graph G
    CD_set = algorithm_1(G, rand_m)

    end_time = time.time()

    run_time = end_time - start_time

    print("Approximate algorithm runtime 1:", run_time) 

    numbor_2 = potential_function(G, CD_set, rand_m)
    min_weight_CDS = sum(G.nodes[node]['weight'] for node in CD_set)
    number_weight_1 += min_weight_CDS
    print("Approximate connected dominating set:", CD_set, min_weight_CDS)     
    print("Is the function value equal to 1:", numbor_2) 


    start_time = time.time()
     
    optimal_set = optimal_algorithm_1(G, rand_m)

    end_time = time.time()
    run_time = end_time - start_time
    print("Optimal algorithm runtime 2:", run_time) 
    numbor_3 = potential_function(G, optimal_set, rand_m)
    min_weight_OPT = sum(G.nodes[node]['weight'] for node in optimal_set)

    number_weight_2 += min_weight_OPT

    print("The minimal connected dominating set:", optimal_set, min_weight_OPT)     
    print("Is the function value equal to 1:", numbor_3) 


    print("Maximum degree:", max_degree)
    print("Minimum degree:", min_degree)

    approximation_rate = min_weight_CDS/min_weight_OPT
    print("The approximation ratio of the approximation algorithm:", approximation_rate)

print("The average approximation ratio of approximation algorithms:", number_weight_1 / repeat_time, number_weight_2 / repeat_time, number_weight_1 / number_weight_2)

