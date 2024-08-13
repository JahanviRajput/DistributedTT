import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
import os

import numpy as np
from time import perf_counter as tpc
# from submodlib import FacilityLocationFunction
import sys
import pandas as pd
sys.path.append('../')  
from protes import protes


def func_buildfed(d, n):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    df = pd.read_csv("fun_values.csv")
    x = jnp.array(df ["x"])
    y = jnp.array(df ["y"])

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        a = I[:, 0]
        b = I[:, 1]

        # Calculate the distances
        abs_diff = jnp.abs(jnp.outer(a, x) + jnp.outer(b, jnp.ones(len(x))) - y)
        
        # Sum across columns
        X = jnp.sum(abs_diff, axis=1)

        return X

    return func


def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    The result in console should looks like this (note that the exact minimum
    of this function is y = 0 and it is reached at the origin of coordinates):


    """
    d = 2              # Dimension
    n = 11             # Mode size
    m = int(1.E+3)       # Number of requests to the objective function
    
    x = np.random.uniform(0, 10, 100)
    y = (2.5)*x
    data = pd.DataFrame({'x': x, 'y': y})
    # Write the DataFrame to a CSV file
    data.to_csv('fun_values.csv', index=False)

    f = func_buildfed(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = protes(f, d, n, m, log=True, k = 10)
    print(i_opt)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n')


if __name__ == '__main__':
    demo()
