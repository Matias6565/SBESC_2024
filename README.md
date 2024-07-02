# Network Recovery Optimization

This project implements an optimized network recovery using ILP and GA.

## Dependences

This project uses Python3.

Run `install_dependencies.py` to install the required libs.

## Code Structure
### Graph

- Creates a graph with 9 nodes. Each node has a capacity of 10 to 100.
- Add links between nodes.

### Disaster Node

- Run 9 simulation.
- Each node will be the disaster node in one simulation.

### Optimization

- Run ILP and GA to migrate the demand of the disaster node minimizing the number of nodes used in the migration.
- Run ILP and GA to migrate the demand of the disaster node minimizing the total sum of the distances between the 
disaster node and the nodes used in the migration.

## Licença

This project is under the [MIT LICENSE](LICENSE).
