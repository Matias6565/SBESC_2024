from queue import PriorityQueue


class Node:
    def __init__(self, number: int):
        self.number = number
        self.parent = None
        self.cost = float("inf")
        self.accessed = False
        self.edges = []


class Edge:
    def __init__(self, node: Node, cost: float):
        self.node = node
        self.cost = cost


class Graph:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.nodes.sort(key=lambda x: x.number)

    def add_edge(self, origin, end, cost):
        origin_node = self.nodes[origin]
        end_node = self.nodes[end]
        origin_node.edges.append(Edge(end_node, cost))
        end_node.edges.append(Edge(origin_node, cost))

    def dijkstra(self, origin: int):
        root = self.nodes[origin]
        root.accessed = True
        root.cost = 0

        p_queue = PriorityQueue()
        p_queue.put((root.cost, root))

        while not p_queue.empty():
            _, current = p_queue.get()
            for edge in current.edges:
                if not edge.node.accessed:
                    if current.cost + edge.cost < edge.node.cost:
                        edge.node.cost = current.cost + edge.cost
                        edge.node.parent = current
                        p_queue.put((edge.node.cost, edge.node))
            current.accessed = True
