import pulp

from graph import *


def ilp_model_minimize_nodes(G, capacities, initial_usage, used_capacities, disaster_node):
    migration_demand = capacities[disaster_node] * initial_usage[disaster_node]

    model = pulp.LpProblem("Recovery_Optimization", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", G.nodes(), cat=pulp.LpBinary)

    # Objective function
    model += pulp.lpSum(y[i] for i in G.nodes() if i != disaster_node), "Objective"
    model += (
        pulp.lpSum((capacities[i] - used_capacities[i]) * y[i] for i in G.nodes() if i != disaster_node)
        >= migration_demand,
        "Migration_Constraint",
    )
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    return model, y


def ilp_model_minimize_distance(G, G_paths, capacities, initial_usage, used_capacities, disaster_node):
    migration_demand = capacities[disaster_node] * initial_usage[disaster_node]

    model = pulp.LpProblem("Recovery_Optimization", pulp.LpMinimize)

    y = pulp.LpVariable.dicts("y", G.nodes(), cat=pulp.LpBinary)

    # Objective function
    model += (
        pulp.lpSum(G_paths.nodes[i].cost * y[i] for i in G.nodes() if i != disaster_node),
        "Objective",
    )
    model += (
        pulp.lpSum((capacities[i] - used_capacities[i]) * y[i] for i in G.nodes() if i != disaster_node)
        >= migration_demand,
        "Migration_Constraint",
    )
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    return model, y


def get_migration_demand(G_paths, y, capacities, used_capacities, disaster_node):
    ilp_active_nodes = [(i, G_paths.nodes[i].cost) for i in range(len(y)) if y[i].varValue == 1]
    ilp_active_nodes.sort(key=lambda x: x[1])

    ilp_migration_demand = []
    remaining_demand = used_capacities[disaster_node]

    for node, _ in ilp_active_nodes:
        capacity_available = capacities[node] - used_capacities[node]

        if remaining_demand <= 0 or capacity_available == 0:
            ilp_migration_demand.append((node, 0))
        else:
            demand_to_assign = min(capacity_available, remaining_demand)
            ilp_migration_demand.append((node, demand_to_assign))
            remaining_demand -= demand_to_assign

    return ilp_migration_demand, remaining_demand
