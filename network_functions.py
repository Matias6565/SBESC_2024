import networkx as nx
import math
import matplotlib.pyplot as plt


def plot_graph(
    G,
    pos,
    title,
    node_labels=None,
    edge_labels=None,
    disaster_node=None,
    migration_nodes=None,
    file_name=None,
    show=True,
):
    plt.figure(figsize=(14, 7))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")

    if node_labels:
        node_label_positions = {k: [v[0], v[1] - 0.01] for k, v in pos.items()}  # Fix the vertical position of labels
        nx.draw_networkx_labels(G, node_label_positions, labels=node_labels, font_weight="bold")

    nx.draw_networkx_edges(G, pos, width=2, edge_color="darkgray")

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_weight="bold")

    if disaster_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[disaster_node], node_size=800, node_color="red")

    if migration_nodes is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=migration_nodes, node_size=800, node_color="green")

    plt.title(title, fontsize=16)
    plt.axis("off")

    if file_name:
        plt.savefig(file_name, format="pdf", bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def dist(u, v):
    return math.hypot((u[0] - v[0]), (u[1] - v[1]))


def get_node_label(G, capacities, used_capacities, usage):
    return {
        node: f"{node}\n{capacities[node]}G : [{used_capacities[node]}G, {usage[node]*100:.0f}%]" for node in G.nodes()
    }


def get_edge_label(G, pos):
    return {(u, v): f"{dist(pos[u], pos[v]) * 100:.2f}Km" for u, v in G.edges()}


def create_graph(capacities, enlaces):
    G = nx.Graph()

    for i in capacities:
        G.add_node(i, capacity=capacities[i])

    for u, v, capacity in enlaces:
        G.add_edge(u, v, capacity=capacity)

    return G


def get_updated_capacity_usage(
    capacities, used_capacities, initial_usage, migration_demand, remaining_demand, disaster_node
):
    updated_used_capacity = used_capacities.copy()
    updated_usage = initial_usage.copy()

    updated_used_capacity[disaster_node] = remaining_demand
    updated_usage[disaster_node] = updated_used_capacity[disaster_node] / capacities[disaster_node]

    for node, capacity_migrated in migration_demand:
        updated_used_capacity[node] += capacity_migrated
        updated_usage[node] = updated_used_capacity[node] / capacities[node]

    return updated_used_capacity, updated_usage


def add_value_labels(ax, fontsize=10, is_float=True):
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".3f") if is_float else format(p.get_height(), ".0f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=fontsize,
        )
