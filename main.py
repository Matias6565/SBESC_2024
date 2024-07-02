import os
import time
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from network_functions import *
from graph import *
from ga import *
from ilp import *
from config import *
from log import init_logs

random.seed(100)
np.random.seed(100)

# List to save the results
df_disaster_nodes = []

df_ilp_min_distance = []
df_ilp_min_distance_nodes = []
df_ilp_min_nodes = []

df_ga_min_distance = []
df_ga_min_distance_nodes = []
df_ga_min_nodes = []

execution_times_ilp = []
execution_times_ga = []

OUTPUT_DIR = "results"

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

init_logs(OUTPUT_DIR + "/log.txt")

for disaster_node in capacities:
    print(f"===========================Disaster Node {disaster_node}===========================")
    # Create the network graph
    G = create_graph(capacities, links)
    G_ga = G.copy()
    pos = nx.spring_layout(G, seed=42)
    node_labels = get_node_label(G, capacities, used_capacities, initial_usage)
    edge_labels = get_edge_label(G, pos)

    # Plot the original network architecture
    plot_graph(
        G,
        pos,
        "Original Architecture",
        node_labels=node_labels,
        edge_labels=edge_labels,
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_original.pdf",
        show=False,
    )

    # Plot the network architecture with the disaster node
    plot_graph(
        G,
        pos,
        "Architecture With Disaster Node",
        node_labels=node_labels,
        edge_labels=edge_labels,
        disaster_node=disaster_node,
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_desastre.pdf",
        show=False,
    )

    # Create a graph with the lowest distance between the disaster node and the rest.
    G_paths = Graph([Node(i) for i in range(len(capacities))])
    for i, j, _ in links:
        G_paths.add_edge(i, j, dist(pos[i], pos[j]) * 100)
    G_paths.dijkstra(disaster_node)

    print("=======Shortest Paths========")
    for node in G_paths.nodes:
        if node.number != disaster_node:
            path = str(node.number)
            current_node = node.parent

            while current_node is not None:
                path = str(current_node.number) + "-" + path
                current_node = current_node.parent

            print(path, "Total dist:", node.cost)
    print("=============================\n")

    # ILP optimization
    start_time_ilp = time.time()

    model_nodes, y_nodes = ilp_model_minimize_nodes(G, capacities, initial_usage, used_capacities, disaster_node)
    model_dist, y_dist = ilp_model_minimize_distance(
        G, G_paths, capacities, initial_usage, used_capacities, disaster_node
    )

    end_time_ilp = time.time()
    execution_time_ilp = end_time_ilp - start_time_ilp

    ilp_migration_demand_nodes, ilp_remaining_demand_nodes = get_migration_demand(
        G_paths, y_nodes, capacities, used_capacities, disaster_node
    )
    ilp_migration_demand_dist, ilp_remaining_demand_dist = get_migration_demand(
        G_paths, y_dist, capacities, used_capacities, disaster_node
    )

    # ILP results
    print("=========ILP Solution========")
    best_ilp_solution_nodes = model_nodes.objective.value()
    print("ILP best cost for node minimization:", best_ilp_solution_nodes)
    print("ILP best solution for node minimization:", [y_nodes[k].varValue for k in y_nodes])
    print("ILP node minimization demand migration:", ilp_migration_demand_nodes)
    print()
    best_ilp_solution_distance = model_dist.objective.value()
    print("ILP best cost for distance minimization:", best_ilp_solution_distance)
    print("ILP best solution for distance minimization:", [y_dist[k].varValue for k in y_dist])
    print("ILP distance minimization demand migration:", ilp_migration_demand_dist)
    print("=============================")

    ilp_used_capacity, ilp_usage = get_updated_capacity_usage(
        capacities,
        used_capacities,
        initial_usage,
        ilp_migration_demand_nodes,
        ilp_remaining_demand_nodes,
        disaster_node,
    )

    node_labels = get_node_label(G, capacities, ilp_used_capacity, ilp_usage)
    edge_labels = get_edge_label(G, pos)

    plot_graph(
        G,
        pos,
        "ILP Solution for Used Nodes Minimization",
        node_labels=node_labels,
        edge_labels=edge_labels,
        disaster_node=disaster_node,
        migration_nodes=[n[0] for n in ilp_migration_demand_nodes],
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_ilp_nodes.pdf",
        show=False,
    )

    ilp_used_capacity, ilp_usage = get_updated_capacity_usage(
        capacities,
        used_capacities,
        initial_usage,
        ilp_migration_demand_dist,
        ilp_remaining_demand_dist,
        disaster_node,
    )

    node_labels = get_node_label(G, capacities, ilp_used_capacity, ilp_usage)
    edge_labels = get_edge_label(G, pos)

    plot_graph(
        G,
        pos,
        "ILP Solution for Distance Minimization",
        node_labels=node_labels,
        edge_labels=edge_labels,
        disaster_node=disaster_node,
        migration_nodes=[n[0] for n in ilp_migration_demand_dist],
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_ilp_distance.pdf",
        show=False,
    )

    # Genetic Algorithm (GA) optimization
    start_time_ga = time.time()

    ga_nodes = GA(
        G_paths, capacities, initial_usage, disaster_node, len(capacities), dist_weight=0, active_node_weight=1
    )
    ga_dist = GA(
        G_paths, capacities, initial_usage, disaster_node, len(capacities), dist_weight=1, active_node_weight=0
    )

    print("\nGA Nodes Minimization")
    *_, migration_demand_nodes, remaining_demand_nodes = ga_nodes.run()
    print("GA Distance Minimization")
    *_, ga_min_dist, migration_demand_dist, remaining_deman_dist = ga_dist.run()

    end_time_ga = time.time()
    execution_time_ga = end_time_ga - start_time_ga

    # Get GA results
    ga_used_capacity, ga_usage = get_updated_capacity_usage(
        capacities, used_capacities, initial_usage, migration_demand_nodes, remaining_demand_nodes, disaster_node
    )

    node_labels = get_node_label(G_ga, capacities, ga_used_capacity, ga_usage)
    edge_labels = get_edge_label(G_ga, pos)

    plot_graph(
        G_ga,
        pos,
        "GA Solution for Used Nodes Minimization",
        node_labels=node_labels,
        edge_labels=edge_labels,
        disaster_node=disaster_node,
        migration_nodes=[n[0] for n in migration_demand_nodes],
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_ga_nodes.pdf",
        show=False,
    )

    ga_used_capacity, ga_usage = get_updated_capacity_usage(
        capacities, used_capacities, initial_usage, migration_demand_dist, remaining_deman_dist, disaster_node
    )

    node_labels = get_node_label(G_ga, capacities, ga_used_capacity, ga_usage)
    edge_labels = get_edge_label(G_ga, pos)

    plot_graph(
        G_ga,
        pos,
        "GA Solution for Distance Minimization",
        node_labels=node_labels,
        edge_labels=edge_labels,
        disaster_node=disaster_node,
        migration_nodes=[n[0] for n in migration_demand_dist],
        file_name=OUTPUT_DIR + f"/n{disaster_node}_network_ga_dist.pdf",
        show=False,
    )

    print(f"Execution time (ILP): {execution_time_ilp:.4f} sec")
    print(f"Execution time (GA): {execution_time_ga:.4f} sec")
    print(f"=================================================================================\n\n")

    # Results
    df_disaster_nodes.append(disaster_node)

    df_ilp_min_distance.append(best_ilp_solution_distance)
    df_ilp_min_distance_nodes.append(len(ilp_migration_demand_dist))
    df_ilp_min_nodes.append(len(ilp_migration_demand_nodes))

    df_ga_min_distance.append(ga_min_dist)
    df_ga_min_distance_nodes.append(len(migration_demand_dist))
    df_ga_min_nodes.append(len(migration_demand_nodes))

    execution_times_ilp.append(execution_time_ilp)
    execution_times_ga.append(execution_time_ga)

df = pd.DataFrame(
    {
        "Disaster Node": df_disaster_nodes,
        "ILP Distance Minimization": df_ilp_min_distance,
        "ILP Distance Minimization Nodes": df_ilp_min_distance_nodes,
        "ILP Nodes Minimization": df_ilp_min_nodes,
        "GA Distance Minimization": df_ga_min_distance,
        "GA Distance Minimization Nodes": df_ga_min_distance_nodes,
        "GA Nodes Minimization": df_ga_min_nodes,
    }
)

df_dist = df.melt(
    id_vars="Disaster Node",
    value_vars=["ILP Distance Minimization", "GA Distance Minimization"],
    var_name="Solution",
    value_name="Distance Cost",
)

df_dist_node = df.melt(
    id_vars="Disaster Node",
    value_vars=["ILP Distance Minimization Nodes", "GA Distance Minimization Nodes"],
    var_name="Solution",
    value_name="Distance Cost Nodes",
)

df_node = df.melt(
    id_vars="Disaster Node",
    value_vars=["ILP Nodes Minimization", "GA Nodes Minimization"],
    var_name="Solution",
    value_name="Node Cost",
)

sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

# Distance minimization plot
plt.figure(figsize=(16, 8))
ax1 = sns.barplot(x="Disaster Node", y="Distance Cost", hue="Solution", data=df_dist, palette="Set2")
add_value_labels(ax1)
plt.xlabel("Disaster Node")
plt.ylabel("Distance Cost (Km)")
plt.title("Distance Minimization")
plt.legend(loc="lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plt_distance_cost_comparative.pdf"))

# Distance minimization nodes plot
plt.figure(figsize=(16, 8))
ax1 = sns.barplot(x="Disaster Node", y="Distance Cost Nodes", hue="Solution", data=df_dist_node, palette="Set2")
add_value_labels(ax1, is_float=False)
plt.xlabel("Disaster Node")
plt.ylabel("Nodes Used")
plt.title("Distance Minimization Used Nodes")
plt.legend(loc="lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plt_distance_nodes_comparative.pdf"))

# Nodes minimization plot
plt.figure(figsize=(16, 8))
ax1 = sns.barplot(x="Disaster Node", y="Node Cost", hue="Solution", data=df_node, palette="Set2")
add_value_labels(ax1, is_float=False)
plt.xlabel("Disaster Node")
plt.ylabel("Nodes Used")
plt.title("Node Minimization")
plt.legend(loc="lower left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plt_node_cost_comparative.pdf"))


# Execution time plot
df_times = pd.DataFrame(
    {
        "Disaster Node": df_disaster_nodes,
        "ILP Execution Time": execution_times_ilp,
        "GA Execution Time": execution_times_ga,
    }
)

df_times = df_times.melt(id_vars="Disaster Node", var_name="Solution", value_name="Solver Time")

plt.figure(figsize=(16, 8))
ax2 = sns.barplot(x="Disaster Node", y="Solver Time", hue="Solution", data=df_times, palette="Set1")
add_value_labels(ax2)
plt.xlabel("Disaster Node")
plt.ylabel("Solver Time (s)")
plt.title("Execution Time")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plt_execution_time_coparative.pdf"))
