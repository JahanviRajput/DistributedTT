import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import protes_fed_learning


def func_build():
    """Binary knapsack problem."""

    d = 50
    n = 2

    w = [
        80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 59, 32, 22,
        60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22, 50, 30, 45, 30, 60, 50,
        20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]

    p = [
        220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125,
        122, 120, 118, 115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82,
        80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15,
        10, 8, 5, 3, 1]

    C = 1000

    def func(i):
        """Target function: y=f(i); [d] -> float."""
        cost = np.dot(p, i)
        constr = np.dot(w, i)
        return 0 if constr > C else -cost

    return d, n, lambda I: np.array([func(i) for i in I])


def demo():
    """A demonstration for QUBO problem.

    We will solve the binary knapsack problem with fixed weights wi in [5, 20],
    profits pi in [50, 100] (i = 1, 2, . . . , d) and the maximum capacity
    C = 1000. It is from work (Dong et al., 2021) (problem k3; d = 50), where
    anglemodulated bat algorithm (AMBA) was proposed for high-dimensional
    binary optimization problems with engineering application to antenna
    topology optimization. Note that ths problem has known exact solution -3103.

    """
    d, n, f = func_build() # Target function, and array shape
    m = int(5.E+4)         # Number of requests to the objective function

    t = tpc()
    i_opt, y_opt = protes_fed_learning(f, d, n, m, k=1000, nbb=5, log=True)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n | x opt = {i_opt}')


if __name__ == '__main__':
    demo()
