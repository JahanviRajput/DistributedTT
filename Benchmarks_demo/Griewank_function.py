import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc

import sys
sys.path.append('../') 
from Methods import dipts
import numpy as np

def func_build_griewank(d, n):
    """Griewank Function f(X) https://www.sfu.ca/~ssurjano/griewank.html """

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = -10.0          # Lower bound for xi
        b = 10.0           # Upper bound for xi

        X = I / (n - 1) * (b - a) + a
        z= np.sqrt(np.tile(np.arange(1, np.shape(X)[0] + 1).reshape(-1, 1), (1, np.shape(X)[1])))
        # Compute the function value for each sample
        y2=np.prod(np.cos(X)/z)
        result = (np.sum(X**2, axis=1))/4000 - y2 + 1


        return result

    return func

def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    """
    d = 5             # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build_griewank(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = dipts(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n | x opt = {i_opt}')


if __name__ == '__main__':
    demo()
