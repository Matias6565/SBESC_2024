import random
import math


class Individual:
    def __init__(self, genes: list, eval: float, node_migration_demand: list, remaining_demand: float):
        self.genes = genes
        self.eval = eval
        self.probability = None
        self.node_migration_demand = node_migration_demand
        self.remaining_demand = remaining_demand


class GA:
    def __init__(
        self, G_paths, capacities, initial_usage, disaster_node, num_genes, dist_weight=1, active_node_weight=1
    ):
        self.population_size = 20
        self.num_generations = 100
        self.mutation_rate = 0.2
        self.dist_weight = dist_weight
        self.active_node_weight = active_node_weight
        self.penality_weight = 10_000

        self.population = []
        self.best_individual = Individual([], float("inf"), [], float("inf"))

        self.G_paths = G_paths
        self.capacities = capacities
        self.initial_usage = initial_usage
        self.disaster_node = disaster_node
        self.num_genes = num_genes
        self.demand_to_migrate = self.capacities[self.disaster_node] * self.initial_usage[self.disaster_node]

    def run(self):
        self.population = self.initialize_population()
        t_best_individual = self.best_individual

        for _ in range(self.num_generations):
            self.calculate_probabilities(t_best_individual)
            self.selection_and_crossover()
            self.mutation()

            t_best_individual = Individual([], float("inf"), [], float("inf"))
            for individual in self.population:
                if individual.eval < t_best_individual.eval:
                    t_best_individual = individual

            if t_best_individual.eval < self.best_individual.eval:
                self.best_individual = t_best_individual

        genes = self.best_individual.genes
        migration_demand = self.best_individual.node_migration_demand
        remaining_demand = self.best_individual.remaining_demand
        active_nodes = [i for i in range(len(genes)) if genes[i] == 1]
        distance_cost = 0

        for i in active_nodes:
            # distance_cost += self.get_cost(self.G_paths.nodes[i], active_nodes)
            distance_cost += self.G_paths.nodes[i].cost

        print("\n======GA Best solution=======")
        print("Genes:", genes)
        print("Eval:", self.best_individual.eval)
        print("Active nodes:", active_nodes)
        print("Total distance cost:", distance_cost)
        print("Migration demand:", migration_demand)
        print("Remaining demand:", remaining_demand)
        print(f"Weights: n-{self.active_node_weight}   d-{self.dist_weight}")
        print("=============================\n")

        return self.best_individual, active_nodes, distance_cost, migration_demand, remaining_demand

    def initialize_population(self):
        population = []

        for _ in range(self.population_size):
            genes = [random.randint(0, 1) for _ in range(self.num_genes)]
            genes[self.disaster_node] = 0

            individual = Individual(genes, *self.fitness(genes))
            population.append(individual)

            if individual.eval < self.best_individual.eval:
                self.best_individual = individual

        return population

    def calculate_probabilities(self, t_best_individual: Individual):
        def f_scale(individual: Individual):
            return 1 / (1 + individual.eval - t_best_individual.eval)

        total_eval = 0

        for individual in self.population:
            total_eval += f_scale(individual)

        for individual in self.population:
            individual.probability = f_scale(individual) / total_eval

    # def get_cost(self, node: Node, active_nodes: list):
    #     total_cost = node.cost - node.parent.cost
    #     node = node.parent
    #     while True:
    #         if node.number == self.disaster_node or node.number in active_nodes:
    #             return total_cost

    #         total_cost += node.cost - node.parent.cost
    #         node = node.parent

    def fitness(self, genes: list):
        active_nodes = [i for i in range(len(genes)) if genes[i] == 1]
        number_active_nodes = sum(genes)
        distance_cost = 0
        node_dist = []

        for i in active_nodes:
            # distance_cost += self.get_cost(self.G_paths.nodes[i], active_nodes)
            distance_cost += self.G_paths.nodes[i].cost
            node_dist.append((i, self.G_paths.nodes[i].cost))

        node_dist.sort(key=lambda x: x[1])

        penality = 0
        remaining_demand = self.demand_to_migrate
        node_migration_demand = []

        for i in range(len(node_dist)):
            node = node_dist[i][0]
            capacity_available = self.capacities[node] * (1.0 - self.initial_usage[node])

            if remaining_demand <= 0 or capacity_available == 0:
                penality += 1
                node_migration_demand.append((node, 0))
            else:
                demand_to_assign = min(capacity_available, remaining_demand)
                node_migration_demand.append((node, demand_to_assign))
                remaining_demand -= demand_to_assign

        if penality == 0 and remaining_demand != 0:
            penality = remaining_demand**4

        return (
            self.active_node_weight * number_active_nodes
            + self.dist_weight * distance_cost
            + self.penality_weight * penality,
            node_migration_demand,
            remaining_demand,
        )

    def selection_and_crossover(self):
        def roulette_wheel_selection():
            selected = random.uniform(0, 1)
            index = 0
            sum_probability = self.population[index].probability

            while sum_probability < selected:
                index += 1
                sum_probability += self.population[index].probability

            return self.population[index]

        def crossover(parent1: Individual, parent2: Individual):
            crossover_point = random.randint(1, self.num_genes - 1)

            parent1_genes = parent1.genes
            parent2_genes = parent2.genes

            child_genes_1 = parent1_genes[:crossover_point] + parent2_genes[crossover_point:]
            child_genes_1[self.disaster_node] = 0

            child_genes_2 = parent2_genes[:crossover_point] + parent1_genes[crossover_point:]
            child_genes_2[self.disaster_node] = 0

            return [
                Individual(child_genes_1, *self.fitness(child_genes_1)),
                Individual(child_genes_2, *self.fitness(child_genes_2)),
            ]

        new_population = []

        for _ in range(math.ceil(self.population_size / 2)):
            parent1 = roulette_wheel_selection()
            parent2 = roulette_wheel_selection()

            while parent1 == parent2:
                parent2 = roulette_wheel_selection()

            child_1, child_2 = crossover(parent1, parent2)
            new_population.append(child_1)
            new_population.append(child_2)

        if len(new_population) > self.population_size:
            new_population = new_population[:-1]

        self.population = new_population

    def mutation(self):
        for individual in self.population:
            genes = individual.genes

            for i in range(len(genes)):
                if random.uniform(0, 1) < self.mutation_rate:
                    genes[i] = 1 - genes[i]

            genes[self.disaster_node] = 0
            fitness = self.fitness(genes)

            individual.genes = genes
            individual.eval = fitness[0]
            individual.node_migration_demand = fitness[1]
            individual.remaining_demand = fitness[2]
