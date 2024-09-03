import networkx as nx
import matplotlib.pyplot as plt
import random
import itertools
import scipy as sp
import numpy as np
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
     #Assign random weights to each node in the graph
    for node in G.nodes() :
        G.nodes[node]['weight'] = random.randint(1, 10000)
        print(f"Node {node} weight: {G.nodes[node]['weight']}")

    return G

#Generate the set of white nodes and establish the inverse mapping.
def terminal_nodes(graph):
    terminals = set()
    mapping = {}
    for node in graph.nodes():
        new_node = f"{node}_1"
        terminals.add(new_node)  # Add new nodes
        mapping[node] = new_node  # Record node mapping relationships
    # Generate the inverse mapping
    inverse_mapping = {v: k for k, v in mapping.items()}
    return terminals, inverse_mapping

#Update the graph's set of black nodes and terminal nodes
def generate_new_graph_1(ori_graph, black_nodes, terminals, vertices):
    graph = ori_graph.copy()

    vertices_1 = vertices.intersection(ori_graph.nodes())
    random_node = random.choice(list(vertices_1))
    vertices_1.remove(random_node)
 
    for node in vertices_1:
        for neighbor in ori_graph.neighbors(node):
            graph.add_edge(random_node, neighbor)
    graph.remove_nodes_from(vertices_1)
    terminals = terminals - vertices
    terminals.add(random_node)
    black_nodes = black_nodes - vertices
    black_nodes.add(random_node)
    graph.nodes[random_node]['weight'] = 0

    return graph, black_nodes, terminals

#The sum of the weights of a subset of nodes in the graph
def sum_weight(graph, nodes):
    weight_sum = 0
    for node in nodes:
        if node in graph.nodes():
            # Retrieve the node's weight attribute and add it to the total weight sum
            weight_sum += graph.nodes[node]['weight']
    return weight_sum

#Compute the shortest path with node weights, excluding the starting node, and output the set of all nodes on the path
def point_weighted_shortest_path(ori_graph, source, target, inverse_mapping):
    graph = ori_graph.copy()

    if not graph.has_node(target):# Check if the node already exists in the graph
        node1 = inverse_mapping[target]
        graph.add_node(target)
        graph.nodes[target]['weight'] = 0
        for neighbor in set(graph.neighbors(node1)) | {node1}:
            graph.add_edge(target, neighbor)
    
    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0  #Excluding the weight of the starting node

    predecessors = {}
    unvisited = set(graph.nodes())
    predecessors[target] = source
    while unvisited:
        current_node = min(unvisited, key=lambda node: distances[node])
        if current_node == target:
            break
        if target in graph.neighbors(current_node):
            weight = graph.nodes[target]['weight'] 
            distances[target] = distances[current_node] + weight 
            predecessors[target] = current_node
            break
        unvisited.remove(current_node)
        for neighbor in graph.neighbors(current_node):
            if neighbor in unvisited:
                weight = G.nodes[neighbor]['weight']  
                new_distance = distances[current_node] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
    shortest_path = [target]
    while target != source:
        target = predecessors[target]
        shortest_path.append(target)
    shortest_path.reverse()
    if len(shortest_path) > 1:
        shortest_distance = distances[shortest_path[-2]] # Compute the shortest path to the second-to-last node
    else :
        shortest_distance = distances[shortest_path[-1]]
    path_nodes = set(shortest_path) 
    return path_nodes, shortest_distance

#Calculate the sum of the node weights for the minimum Steiner tree in the graph containing the points x,y and v，excluding the vertices
def min_steiner_tree(graph, vertices, vertice, inverse_mapping):
    shortest_distance = float('inf')
    shortest_tree = set()
    if len(vertices) == 3:

        for node in graph.nodes() :
            shortest_tree_node = set()
            for node_1 in vertices:
                path_nodes, shortest_distance_1 = point_weighted_shortest_path(graph, node, node_1, inverse_mapping)
                #print("Edge-to-path mapping0:", node, node_1, path_nodes, shortest_distance)
                shortest_tree_node.update(path_nodes)
            
            tree_weight = sum_weight(graph, shortest_tree_node) - graph.nodes[vertice]['weight']#不属于graph的点默认赋值为0
            #print("Edge-to-path mapping00:", shortest_tree_node, tree_weight, shortest_distance)
            if tree_weight < shortest_distance:
                shortest_tree = shortest_tree_node
                shortest_distance = tree_weight
                #print("Edge-to-path mapping000:", shortest_tree, shortest_tree_node)
            
    else :
        if len(vertices) == 2:
            for node in vertices:
                if node != vertice:
                    shortest_tree, shortest_distance = point_weighted_shortest_path(graph, vertice, node, inverse_mapping)
            
            shortest_distance = shortest_distance - graph.nodes[vertice]['weight'] 


        else:
            shortest_tree = vertices
            shortest_distance = 0

    #print("Edge-to-path mapping1:", vertices, vertice, shortest_tree, shortest_distance)

    return shortest_tree, shortest_distance


#Construct a new complete graph
def generate_matching_graph(original_graph, vertices, node, inverse_mapping):
    new_graph = nx.complete_graph(vertices)
    edge_shortest_path = {}

    for (u, v) in new_graph.edges():
        vertices_1 = set()
        vertices_1.add(node)
        vertices_1.add(u)
        vertices_1.add(v)
        shortest_tree, shortest_distance = min_steiner_tree(original_graph, vertices_1, node, inverse_mapping)
        edge_shortest_path[(u, v)] = shortest_tree
        edge_shortest_path[(v, u)] = shortest_tree
        new_graph[u][v]['weight'] = shortest_distance

    return new_graph, edge_shortest_path

#Construct a new graph based on the required number of matchings
def generate_graph_l(graph, number_1):
    new_graph = graph.copy()
    dummy_nodes = ['dummy_' + str(i) for i in range(number_1)]
    new_graph.add_nodes_from(dummy_nodes)
    for dummy_node in dummy_nodes:
        for node in graph.nodes():
            new_graph.add_edge(dummy_node, node)
            new_graph[dummy_node][node]['weight'] = 0
    for (u, v) in new_graph.edges():
        if 'weight' not in new_graph[u][v]:
            print("Edges without weights:", (u, v))
            
    return new_graph, dummy_nodes

# Output the set of nodes corresponding to the paths of edges that have no intersection with the set of dummy nodes
def nodes_without_dummy(matching, dummy_nodes, edge_shortest_path):
    #print("Matching1：", matching)
    new_matching = set(matching)
    path_nodes = set()
    for edge in matching:
        u, v = edge
        if u in dummy_nodes or v in dummy_nodes:
            new_matching.remove(edge)
    #print("Matching2：", new_matching)
    for edge in new_matching:
        path_nodes_1 = edge_shortest_path[edge]
        path_nodes.update(path_nodes_1)

    
    return path_nodes

#Compute the minimum weight perfect matching
def min_weight_matching(graph):
    new_graph = graph.copy()
    sum_weight = 0
    weights_1 = 0
    for (u, v) in new_graph.edges():
        if new_graph[u][v]['weight'] > weights_1:
            weights_1 = new_graph[u][v]['weight']

    weights_1 += 1

    for (u, v) in new_graph.edges():
        # Reset the edge weights
        new_graph[u][v]['weight'] = weights_1 - new_graph[u][v]['weight']
    
    min_matching = nx.max_weight_matching(new_graph, weight='weight', maxcardinality=True)
    for  (u, v) in min_matching:
        sum_weight += graph[u][v]['weight']

    return min_matching, sum_weight



#Compute the minimum 3+branch-spider in the graph centered at node, containing number_1 terminal points
def min_matching(graph, new_complete_graph, node, number_1, inverse_mapping, edge_shortest_path):
    total_weight =  float('inf')
    shortest_spider_3 = set()
    if number_1 % 2 != 0:
        for node_1 in new_complete_graph.nodes():
            shortest_path, shortest_distance = point_weighted_shortest_path(graph, node, node_1, inverse_mapping)
            path_nodes = set(shortest_path)
            G1 = new_complete_graph.copy() 
            G1.remove_node(node_1)
            num_nodes = G1.number_of_nodes()
            number_2 = num_nodes - number_1 + 1
            G1, dummy_nodes = generate_graph_l(G1, number_2)
            min_matching, matching_weight= min_weight_matching(G1)
            if total_weight > matching_weight + graph.nodes[node]['weight'] + shortest_distance:
                total_weight = matching_weight + graph.nodes[node]['weight'] + shortest_distance
                path_nodes_1 = nodes_without_dummy(min_matching, dummy_nodes, edge_shortest_path)
                path_nodes.update(path_nodes_1)
                shortest_spider_3 = path_nodes         

    else :
        G1 = new_complete_graph.copy()
        #print("Test graph G1：", G1.nodes())
        num_nodes = G1.number_of_nodes()
        number_2 = num_nodes - number_1
        G1, dummy_nodes = generate_graph_l(G1, number_2)
        min_matching, matching_weight = min_weight_matching(G1)
        if total_weight > matching_weight + graph.nodes[node]['weight']:
            total_weight = matching_weight + graph.nodes[node]['weight']
            shortest_spider_3 = nodes_without_dummy(min_matching, dummy_nodes, edge_shortest_path)
    
    return shortest_spider_3, total_weight

#Algorithm 2: Identify the most efficient 3+branch-spider
def algorithm_branch_spider_3(graph, terminals, inverse_mapping):
    benefit_branch_spider_3 = set()
    benefit_rate = float('inf')
    number_2 = len(terminals) + 1
    for node in graph.nodes():
        new_complete_graph, edge_shortest_path = generate_matching_graph(graph, terminals, node, inverse_mapping)
        for number_1 in range(3, number_2):
            shortest_spider_3, total_weight = min_matching(graph, new_complete_graph, node, number_1, inverse_mapping, edge_shortest_path)#尽量合并成一个
            #print("test 4：", shortest_spider_3)
            spider_rat = total_weight / number_1
            if spider_rat < benefit_rate:
                benefit_rate = spider_rat
                benefit_branch_spider_3 = shortest_spider_3
    return benefit_branch_spider_3

#Algorithm 1: Compute a Connected Dominating Set
def algorithm_3(new_graph):
    graph = new_graph.copy()
    i = 1
    cd_set = set()  
    black_nodes = set()
    terminals, inverse_mapping = terminal_nodes(graph)
    #print("Test3：", terminals, inverse_mapping)
    while len(terminals) > 2 :
        benefit_branch_spider_3 = algorithm_branch_spider_3(graph, terminals, inverse_mapping)
        print("Top X branch-spiders of type 3:", i, benefit_branch_spider_3)
        i += 1
        spider_nodes = benefit_branch_spider_3.intersection(graph.nodes())
        cd_set.update(spider_nodes)
        graph, black_nodes, terminals = generate_new_graph_1(graph, black_nodes, terminals, benefit_branch_spider_3)
        #print("Test4：", graph.nodes(), black_nodes, terminals, inverse_mapping)
    if len(terminals) == 2 :
        source = random.choice(list(terminals.intersection(black_nodes)))
        terminals.remove(source)
        for target in terminals :
            shortest_path, shortest_distance = point_weighted_shortest_path(graph, source, target, inverse_mapping)
            spider_nodes = shortest_path.intersection(graph.nodes())
        cd_set.update(spider_nodes)

    return cd_set


#Calculate the number of connected components in the subgraph induced by the vertices of the graph
def number_c1(G, vertices):
    sub_graph = G.subgraph(vertices)
    return nx.number_connected_components(sub_graph)

#Calculate the number of additional covers needed for the vertex
def need_cover(G, vertice, vertices, rand_m):
    neighbors = set(G.neighbors(vertice))
    cover_need = rand_m - len(neighbors & vertices)
    if cover_need < 0:
        cover_need = 0 
    return cover_need

#Calculate the total number of additional covers needed
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

#potential_function
def potential_function(G, vertices, rand_m):
    number_f = number_c1(G, vertices) + need_cover_all(G, vertices, rand_m)
    return number_f

#Neighborhood of vertex that satisfies the lemma conditions
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

#Output the vertex with the smallest weight in the set
def min_weight_node(G, vertices):
    weight_1 = float('inf')
    for node in vertices:
        if G.nodes[node]['weight'] < weight_1:
            min_vertex = node
            weight_1 = G.nodes[node]['weight']
    return min_vertex

#Identify the most efficient star structure
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
                #print("Neighborhood that satisfies the lemma conditions:", neighbor_set2)
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

#Approximation Algorithm 1 to compute a (1, m) CDS
def algorithm_1(G, rand_m):
    i = 1
    cd_set = set()  
    weight_set = 0
    while number_c1(G, cd_set) + need_cover_all(G, cd_set, rand_m) > 1 :
        eff_star = algorithm_2(G, cd_set, rand_m)
        weight_set = sum(G.nodes[node]['weight'] for node in eff_star)
        lenth_set_1 = len(eff_star)
        if lenth_set_1 > 1:
            print("TOP", i, "Optimal star structure:", eff_star, weight_set)
        cd_set.update(eff_star)
        i += 1
    return cd_set




#Number of iterations
repeat_time = 3

for i in range(repeat_time) :
    n_nodes = 20
    min_degree_m = 2
    max_degree_m = 10
    G = generate_random_connected_graph(n_nodes, min_degree_m, max_degree_m)


    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())


    rand_m = 1
    start_time = time.time()
    
    CD_set = algorithm_1(G, rand_m)

    end_time = time.time()

     
    run_time = end_time - start_time

    print("Program 1 runtime:", run_time)   

    numbor_2 = potential_function(G, CD_set, rand_m)
    min_weight_CDS = sum(G.nodes[node]['weight'] for node in CD_set)
    print("Approximate connected dominating set:", CD_set, min_weight_CDS)     
    print("Is the function value equal to 1:", numbor_2) 
     
    #Run the optimal Algorithm 1 to compute the connected dominating set of graph
    start_time = time.time()

    optimal_set = algorithm_3(G)
    
    end_time = time.time()

    # Calculate the runtime of the algorithm
    run_time = end_time - start_time

    print("Program 2 runtime:", run_time)    

    numbor_3 = potential_function(G, optimal_set, rand_m)
    min_weight_OPT = sum(G.nodes[node]['weight'] for node in optimal_set)
    
    print("CDS2:", optimal_set, min_weight_OPT)     
    print("Is the function value equal to 1:", numbor_3) 

     
    print("Maximum degree:", max_degree)
    print("Minimum degree:", min_degree)
    # Calculate and output the approximation ratio
    approximation_rate = min_weight_CDS/min_weight_OPT
    print("The approximation ratio of the approximation algorithm:", approximation_rate)

