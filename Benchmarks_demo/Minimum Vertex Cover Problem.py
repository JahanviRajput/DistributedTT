import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc

import sys
sys.path.append('../') 
from Methods import dipts


def func_build():
    """Minimum Vertex Cover Problem."""
    d = 50
    n = 2
    def func(I):    
        # Define the adjacency matrix of the graph
        adjacency_matrix = np.array([[0, 1, 1, 0],
                                     [1, 0, 1, 1],
                                     [1, 1, 0, 1],
                                     [0, 1, 1, 0]])
        
        # Compute the number of vertices selected
        num_vertices_selected = sum(I)
        
        # Compute the number of uncovered edges
        num_uncovered_edges = sum([adjacency_matrix[i, j] for i in range(len(I)) for j in range(i + 1, len(I)) if I[i] == 0 and I[j] == 0])
        
        return num_vertices_selected - num_uncovered_edges

    return d, n, lambda I: np.array([func(i) for i in I])

def demo():
    d, n, f = func_build() # Target function, and array shape
    m = int(5.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = dipts(f, d, n, m, k=1000, nbb = 5, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n | x opt = {i_opt}')


if __name__ == '__main__':
    demo()
