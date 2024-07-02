# capacities = {0: 50, 1: 50, 2: 20, 3: 20, 4: 50}  # In Gb
# initial_usage = {0: 1.0, 1: 0.5, 2: 0.8, 3: 0.8, 4: 0.5}  # Percentage
# used_capacities = {node: capacities[node] * initial_usage[node] for node in capacities}
# links = [  # In Gbps
#     (0, 1, 10),
#     (0, 2, 10),
#     (0, 3, 10),
#     (1, 3, 100),
#     (2, 3, 10),
#     (2, 4, 10),
#     (3, 4, 100),
#     (4, 0, 10),
#     (4, 1, 100),
# ]

# capacities = {0: 80, 1: 60, 2: 50, 3: 20, 4: 20, 5: 50, 6: 30, 7: 100, 8: 20}
# initial_usage = {0: 1.0, 1: 0.2, 2: 0.5, 3: 0.8, 4: 0.8, 5: 0.5, 6: 0.4, 7: 0.1, 8: 0.9}
# used_capacities = {node: capacities[node] * initial_usage[node] for node in capacities}
# links = [
#     (0, 1, 10),
#     (0, 2, 10),
#     (1, 3, 100),
#     (1, 4, 50),
#     (2, 5, 20),
#     (2, 6, 30),
#     (3, 4, 100),
#     (3, 7, 50),
#     (4, 8, 40),
#     (6, 7, 20),
#     (8, 1, 100),
#     (8, 5, 10),
#     (0, 5, 10),
# ]

capacities = {0: 80, 1: 60, 2: 50, 3: 60, 4: 50, 5: 50, 6: 30, 7: 100, 8: 20}
initial_usage = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.4, 7: 0.4, 8: 0.9}
used_capacities = {node: capacities[node] * initial_usage[node] for node in capacities}
links = [
    (0, 1, 10),
    (0, 2, 10),
    (1, 3, 100),
    (1, 4, 50),
    (2, 5, 20),
    (2, 6, 30),
    (3, 4, 100),
    (3, 7, 50),
    (4, 8, 40),
    (6, 7, 20),
    (8, 1, 100),
    (8, 5, 10),
    (0, 5, 10),
]

# capacities = {
#     0: 200,
#     1: 200,
#     2: 300,
#     3: 300,
#     4: 400,
#     5: 200,
#     6: 200,
#     7: 300,
#     8: 300,
#     9: 400,
#     10: 200,
#     11: 200,
#     12: 300,
#     13: 300,
#     14: 400,
#     15: 200,
#     16: 200,
#     17: 300,
#     18: 300,
#     19: 400,
# }
# initial_usage = {
#     0: 0.7,
#     1: 0.6,
#     2: 0.8,
#     3: 0.5,
#     4: 0.4,
#     5: 0.7,
#     6: 0.6,
#     7: 0.5,
#     8: 0.4,
#     9: 0.8,
#     10: 0.7,
#     11: 0.6,
#     12: 0.8,
#     13: 0.5,
#     14: 0.4,
#     15: 0.7,
#     16: 0.6,
#     17: 0.5,
#     18: 0.4,
#     19: 0.8,
# }
# used_capacities = {node: capacities[node] * initial_usage[node] for node in capacities}
# links = [
#     (0, 1, 100),
#     (0, 2, 100),
#     (0, 3, 100),
#     (1, 3, 200),
#     (1, 4, 100),
#     (2, 3, 200),
#     (2, 5, 100),
#     (3, 4, 200),
#     (3, 6, 100),
#     (4, 7, 200),
#     (5, 6, 100),
#     (5, 8, 200),
#     (6, 7, 200),
#     (6, 9, 100),
#     (7, 0, 100),
#     (8, 1, 200),
#     (8, 9, 200),
#     (9, 4, 200),
#     (10, 11, 100),
#     (10, 12, 100),
#     (10, 13, 100),
#     (11, 13, 200),
#     (11, 14, 100),
#     (12, 13, 200),
#     (12, 15, 100),
#     (13, 14, 200),
#     (13, 16, 100),
#     (14, 17, 200),
#     (15, 16, 100),
#     (15, 18, 200),
#     (16, 17, 200),
#     (16, 19, 100),
#     (17, 10, 100),
#     (18, 11, 200),
#     (18, 19, 200),
#     (19, 14, 200),
#     (9, 10, 100),
# ]
