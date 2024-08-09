import jax
import jax.numpy as jnp
import optax
from time import perf_counter as tpc
import os

def protes(f, d, n, m=None, k=100, k_top=10, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None, with_info_p=False,
           with_info_i_opt_list=False, with_info_full=False, sample_ext=None):
    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'k_top': k_top,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': []})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        '''key is a pseudo-random number generator'''
        rng, key = jax.random.split(rng)
        ''' d is dim of N1, r is rank key is for generating numbers from distribution....
        P is the initial TT-tensor with g1,g2:d-1,gd 3 components'''
        P = _generate_initial(d, n, r, key)
    elif len(P[1].shape) != 4:
        ''' Because we are generating all g2:d-1 together so we ar passing 4 things to it--- (d-2, r, n, r)'''
        raise ValueError('Initial P tensor should have special format')

    if with_info_p:
        info['P'] = P

    optim = optax.adam(lr)
    #creating object for optimizer adam
    state = optim.init(P)

    # changing function of type jit - just in time
    interface_matrices = jax.jit(_interface_matrices)
    sample = jax.jit(jax.vmap(_sample, (None, None, None, None, 0)))
    likelihood = jax.jit(jax.vmap(_likelihood, (None, None, None, None, 0)))

    @jax.jit
    def loss(P_cur, I_cur):
        Pl, Pm, Pr = P_cur
        Zm = interface_matrices(Pm, Pr)
        l = likelihood(Pl, Pm, Pr, Zm, I_cur)
        return jnp.mean(-l)

    loss_grad = jax.grad(loss)

    @jax.jit
    def optimize(state, P_cur, I_cur):
        grads = loss_grad(P_cur, I_cur)
        updates, state = optim.update(grads, state)
        P_cur = jax.tree_util.tree_map(lambda p, u: p + u, P_cur, updates)
        return state, P_cur

    is_new = None

    while True:
        if sample_ext:
            I = sample_ext(P, k, seed)
            seed += k
        else:
            # Pl = g1, Pm = g2:d-1, Pr = gd
            Pl, Pm, Pr = P
            # returning normalizing g2:d after some operation
            Zm = interface_matrices(Pm, Pr)
            # giving two new PRNG as default value is 2
            rng, key = jax.random.split(rng)
            # I is indices for which we will choose x ---- dim is (d,1)
            I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k))

        y = f(I)
        if y is None:
            break
        if len(y) == 0:
            continue

        y = jnp.array(y)
        # initially m is 0 we are counting number of iteration
        info['m'] += y.shape[0]

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)
        # not more than m iterations
        if info['m_max'] and info['m'] >= info['m_max']:
            break
        # sorting y for finding top k values
        ind = jnp.argsort(y, kind='stable')
        # selecting topk values
        ind = (ind[::-1] if is_max else ind)[:k_top]

        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])
        
        if with_info_p:
            info['P'] = P

        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    if is_new is not None:
        _log(info, log, is_new, is_end=True)

    return info['i_opt'], info['y_opt']


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)
    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]


def _interface_matrices(Ym, Yr):
    """Compute the "interface matrices" for the TT-tensor."""
    def body(Z, Y_cur):
        Z = jnp.sum(Y_cur, axis=1) @ Z
        Z /= jnp.linalg.norm(Z)
        return Z, Z
    # normalizing all the g2:d
    Z, Zr = body(jnp.ones(1), Yr)
    _, Zm = jax.lax.scan(body, Z, Ym, reverse=True)

    return jnp.vstack((Zm, Zr))


def _likelihood(Yl, Ym, Yr, Zm, i):
    """Compute the likelihood in a multi-index i for TT-tensor."""
    def body(Q, data):
        I_cur, Y_cur, Z_cur = data

        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, I_cur, :])
        Q /= jnp.linalg.norm(Q)

        return Q, G[I_cur]

    Q, yl = body(jnp.ones(1), (i[0], Yl, Zm[0]))
    Q, ym = jax.lax.scan(body, Q, (i[1:-1], Ym, Zm[1:]))
    Q, yr = body(Q, (i[-1], Yr, jnp.ones(1)))

    y = jnp.hstack((jnp.array(yl), ym, jnp.array(yr)))
    return jnp.sum(jnp.log(jnp.array(y)))


class Log:
    def __init__(self, fpath='../Results_new_fun/ss_protes.txt'):
        self.fpath = fpath
        self.is_new = True

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'a') as f:
            f.write(text + '\n')
        self.is_new = False

def _log(info, log=False, is_new=False, is_end=False):
    """Print current optimization result to output."""
    if not log or (not is_new and not is_end):
        return

    text = f'protes > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e}'

    if is_end:
        text += ' <<< DONE'

    print(text) if isinstance(log, bool) else log(text)
    t = f' m {info["m"]} | t {info["t"]} | y {info["y_opt"]} | x {info["i_opt"]}' 
    log = Log()
    log(t)


def _process(P, I, y, info, with_info_i_opt_list, with_info_full):
    """Check the current batch of function values and save the improvement."""
    ind_opt = jnp.argmax(y) if info['is_max'] else jnp.argmin(y)

    i_opt_curr = I[ind_opt, :]
    y_opt_curr = y[ind_opt]

    is_new = info['y_opt'] is None
    is_new = is_new or info['is_max'] and info['y_opt'] < y_opt_curr
    is_new = is_new or not info['is_max'] and info['y_opt'] > y_opt_curr

    if is_new:
        info['i_opt'] = i_opt_curr
        info['y_opt'] = y_opt_curr

    if is_new or with_info_full:
        info['m_opt_list'].append(info['m'])
        info['y_opt_list'].append(info['y_opt'])

        if with_info_i_opt_list or with_info_full:
            info['i_opt_list'].append(info['i_opt'].copy())

    if with_info_full:
        info['P_list'].append([G.copy() for G in P])
        info['I_list'].append(I.copy())
        info['y_list'].append(y.copy())

    return is_new


def _sample(Yl, Ym, Yr, Zm, key):
    """Generate sample according to given probability TT-tensor."""
    def body(Q, data):
        key_cur, Y_cur, Z_cur = data
        ''' The line G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur) uses Einstein summation notation in JAX to perform a contraction operation on the arrays Q, Y_cur, and Z_cur.

           Here's how it works:
           
           Q, Y_cur, and Z_cur are the input arrays.
           'r,riq,q->i' specifies the Einstein summation convention for the operation. Each character represents a dimension, and the arrow (->) indicates the output dimension.
           r corresponds to a dimension in Q.
           riq corresponds to dimensions in Y_cur.
           q corresponds to a dimension in Z_cur.
           i corresponds to the output dimension.
           The letters represent indices that are contracted (summed) over. The output array G will have the shape determined by the remaining non-contracted dimensions.
           In simpler terms, the operation can be described as follows:
           
           For each element of the output array G, an element-wise multiplication is performed between the corresponding elements of Q, Y_cur, and Z_cur.
           Then, the results are summed over the r and q dimensions and multiplied by the riq dimension, resulting in a single value for each element of G.
           The resulting array G will have a shape determined by the dimensions not specified in the output, which is just the i dimension in this case.'''
               
        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        i = jax.random.choice(key_cur, jnp.arange(Y_cur.shape[1]), p=G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, i, :])
        Q /= jnp.linalg.norm(Q)

        return Q, i

    keys = jax.random.split(key, len(Ym) + 2)

    Q, il = body(jnp.ones(1), (keys[0], Yl, Zm[0]))
    Q, im = jax.lax.scan(body, Q, (keys[1:-1], Ym, Zm[1:]))
    Q, ir = body(Q, (keys[-1], Yr, jnp.ones(1)))

    il = jnp.array(il, dtype=jnp.int32)
    ir = jnp.array(ir, dtype=jnp.int32)
    return jnp.hstack((il, im, ir))



# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
# import seaborn as sns


# sns.set_context('paper', font_scale=2.5)
# sns.set_style('white')
# sns.mpl.rcParams['legend.frameon'] = 'False'


# from jax.config import config
# config.update('jax_enable_x64', True)
# import os
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'


# import jax.numpy as jnp



# from teneva_bm import *

# bms = [
#     BmFuncChung(d = int(1.E+4) , n = 16, name ='F-01'),

#     BmFuncDixon(d = int(1.E+4) , n = 16, name ='F-02'), 

#     BmFuncPathological(d = int(1.E+4) , n = 16, name ='F-03'),
#     BmFuncPinter(d = int(1.E+4), n = 16, name ='F-04'), 
#     BmFuncPowell(d = int(1.E+4), n = 16, name ='F-05'), 

#     BmFuncQing(d = int(1.E+4), n = 16, name ='F-06'),
#     BmFuncRosenbrock(d = int(1.E+4), n = 16, name ='F-07'),

#     BmFuncSalomon(d = int(1.E+4), n = 16, name ='F-08'), 
#     BmFuncSphere(d = int(1.E+4), n = 16, name ='F-09'), 
#     BmFuncSquares(d = int(1.E+4), n = 16, name ='F-10'),
#     BmFuncTrid(d = int(1.E+4), n = 16, name ='F-11'), 
#     BmFuncTrigonometric(d = int(1.E+4), n = 16, name ='F-12'), 
#     BmFuncWavy(d = int(1.E+4), n = 16, name ='F-13'), 
#     BmFuncYang(d = int(1.E+4), n = 16, name ='F-14'),

# ]


# BM_FUNC = ['F-01', 'F-02', 'F-03', 'F-04', 'F-05', 'F-06', 'F-07', 'F-08', 'F-09', 'F-10', 'F-11', 'F-12', 'F-13', 'F-14']

# def prep_bm_func(bm):
#     shift = np.random.randn(bm.d) / 10
#     a_new = bm.a - (bm.b-bm.a) * shift
#     b_new = bm.b + (bm.b-bm.a) * shift
#     bm.set_grid(a_new, b_new)
#     bm.prep()
#     return bm


# import random
# import time
# import pandas as pd
# import numpy as np 

# def calc(m=int(1.E+4), seed=0):
#     log = Log()
#     i_opt = np.zeros(len(BM_FUNC))
#     y_opt = np.zeros(len(BM_FUNC))

#     d = int(1.E+4)            # Dimension
#     n = 11             # Mode size
#     seed = [random.randint(0, 100) for _ in range(len(BM_FUNC))]

#     for f in bms:
#         if f.name in BM_FUNC:
#             # We carry out a small random shift of the function's domain,
#             # so that the optimum does not fall into the middle of the domain:
#             f = prep_bm_func(f)
#         else:
#             f.prep()
#         t_start = time.time()
#         log(f"\n fun {f.name} | d {d} |\n")
#         i_opt, y_optk = protes(f, d, n, m, log=True, k = 100, k_top = 10)
#         time_taken = (time.time() - t_start)
#         print(f'\n {f.name} Function: {f} \n | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')

# # calc()



