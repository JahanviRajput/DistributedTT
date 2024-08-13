import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])


import numpy as np
from time import perf_counter as tpc


from protes import dipts


def func_build_piston(d, n):
    """Piston multivariable analytic functions"""

    # Parameters
    A = 1.0  # Amplitude
    k = 2.0  # Wave number
    omega = 1.0  # Angular frequency
    phi = 0.0  # Phase constant


    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (omega - phi) + phi 
        y = A * np.sin(k * X - omega  + phi)

        return y

    return func


def demo():
    """A simple demonstration for discretized multivariate analytic function.

    We will find the minimum of an implicitly given "d"-dimensional array
    having "n" elements in each dimension. The array is obtained from the
    discretization of an analytic function.

    The result in console should looks like this (note that the exact minimum
    of this function is y = 0 and it is reached at the origin of coordinates):

    """
    d = 100              # Dimension
    n = 11               # Mode size
    m = int(1.E+4)       # Number of requests to the objective function
    f = func_build_piston(d, n) # Target function, which defines the array elements

    t = tpc()
    i_opt, y_opt = dipts(f, d, n, m, log=True, k = 100)
    print(f'\nRESULT | y opt = {y_opt:-11.4e} | time = {tpc()-t:-10.4f}\n\n | x opt = {i_opt}\n')


if __name__ == '__main__':
    demo()
