from submodlib import LogDeterminantFunction, FacilityLocationFunction
from time import perf_counter as tpc
import jax
import jax.numpy as jnp
import optax
import numpy as np
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
jax.default_device(jax.devices('cpu')[0])
import warnings

warnings.filterwarnings('ignore')

def subset_submod_pts(f, d, n, m=None, k=100, k_gd=1, lr=5.E-2, r=5, seed=0,
           is_max=False, log=False, info={}, P=None, with_info_p=False,
           with_info_i_opt_list=False, with_info_full=False, sample_ext=None, subset_size = 10, sub_fun = 'FL'):
    time = tpc()
    info.update({'d': d, 'n': n, 'm_max': m, 'm': 0, 'k': k, 'subset_size': subset_size,
        'k_gd': k_gd, 'lr': lr, 'r': r, 'seed': seed, 'is_max': is_max,
        'is_rand': P is None, 't': 0, 'i_opt': None, 'y_opt': None,
        'm_opt_list': [], 'i_opt_list': [], 'y_opt_list': [], 'Submodular_function': sub_fun})
    if with_info_full:
        info.update({
            'P_list': [], 'I_list': [], 'y_list': []})

    rng = jax.random.PRNGKey(seed)

    if P is None:
        rng, key = jax.random.split(rng)
        P = _generate_initial(d, n, r, key)
    elif len(P[1].shape) != 4:
        raise ValueError('Initial P tensor should have special format')

    if with_info_p:
        info['P'] = P

    optim = optax.adam(lr)
    state = optim.init(P)

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
            Pl, Pm, Pr = P
            Zm = interface_matrices(Pm, Pr)
            rng, key = jax.random.split(rng)
            I = sample(Pl, Pm, Pr, Zm, jax.random.split(key, k))


        if sub_fun == 'FL':
            obj = FacilityLocationFunction(n=k, data=I, mode="dense", metric="euclidean")
        elif sub_fun == 'LD':
            obj = LogDeterminantFunction(n=k, data=I, mode="dense", metric="euclidean", lambdaVal=1)
        subset = obj.maximize(budget=subset_size, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        ind = [i[0] for i in subset]
        I = jnp.array([I[i] for i in ind])
        y = f(I)

        if y is None:
            break
        if len(y) == 0:
            continue

        y = jnp.array(y)
        info['m'] += k

        is_new = _process(P, I, y, info, with_info_i_opt_list, with_info_full)

        if info['m_max'] and info['m'] >= info['m_max']:
            break

        ind = jnp.argsort(y, kind='stable')

        for _ in range(k_gd):
            state, P = optimize(state, P, I[ind, :])

        if with_info_p:
            info['P'] = P

        info['t'] = tpc() - time
        _log(info, log, is_new)

    info['t'] = tpc() - time
    if is_new is not None:
        _log(info, log, is_new, is_end=True)

    return info['i_opt'], info['y_opt'], info['t'], info['m']


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
    def __init__(self, fpath='../Results/ss_protes_fl.txt'):
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

    text = f'SSPTS > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y_opt"]:-11.4e} |'
    if is_end:
        text += ' <<< DONE'

    print(text) if isinstance(log, bool) else log(text)
    if info["m"] < 10000:
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



import matplotlib as mpl
import numpy as np
import os
import pickle
import sys
from time import perf_counter as tpc

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': False,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


import jax.numpy as jnp



from teneva_bm import *

bms = [
    BmFuncChung(d = int(1.E+4) , n = 16, name ='F-01'),

    BmFuncDixon(d = int(1.E+4) , n = 16, name ='F-02'), 

    BmFuncPathological(d = int(1.E+4) , n = 16, name ='F-03'),
    BmFuncPinter(d = int(1.E+4), n = 16, name ='F-04'), 
    BmFuncPowell(d = int(1.E+4), n = 16, name ='F-05'), 

    BmFuncQing(d = int(1.E+4), n = 16, name ='F-06'),
    BmFuncRosenbrock(d = int(1.E+4), n = 16, name ='F-07'),

    BmFuncSalomon(d = int(1.E+4), n = 16, name ='F-08'), 
    BmFuncSphere(d = int(1.E+4), n = 16, name ='F-09'), 
    BmFuncSquares(d = int(1.E+4), n = 16, name ='F-10'),
    BmFuncTrid(d = int(1.E+4), n = 16, name ='F-11'), 
    BmFuncTrigonometric(d = int(1.E+4), n = 16, name ='F-12'), 
    BmFuncWavy(d = int(1.E+4), n = 16, name ='F-13'), 
    BmFuncYang(d = int(1.E+4), n = 16, name ='F-14'),

]


BM_FUNC = ['F-01', 'F-02', 'F-03', 'F-04', 'F-05', 'F-06', 'F-07', 'F-08', 'F-09', 'F-10', 'F-11', 'F-12', 'F-13', 'F-14']


def prep_bm_func(bm):
    shift = np.random.randn(bm.d) / 10
    a_new = bm.a - (bm.b-bm.a) * shift
    b_new = bm.b + (bm.b-bm.a) * shift
    bm.set_grid(a_new, b_new)
    bm.prep()
    return bm


import random
import time
import pandas as pd

def calc(m=int(1.E+4), seed=0):
    log = Log()
    i_opt = np.zeros(len(BM_FUNC))
    y_opt = np.zeros(len(BM_FUNC))

    d = int(1.E+4)             # Dimension
    n = 11             # Mode size
    seed = [random.randint(0, 100) for _ in range(len(BM_FUNC))]
    m_value = []
    y_value = []
    t_value = []
    for f in bms:
        if f.name in BM_FUNC:
            # We carry out a small random shift of the function's domain,
            # so that the optimum does not fall into the middle of the domain:
            f = prep_bm_func(f)
        else:
            f.prep()
        
        log(f"\n {f.name} | d {d} \n")
    
        i_opt, y_optk, t, m = subset_submod_pts(f, d, n, m, log=True, sub_fun = 'FL')
    #     log(f"\n {f.name} | m {m} | y {y_optk} | t {t} |  x {i_opt}\n")
    #     m_value.append(m)
    #     y_value.append(y_optk)
    #     t_value.append(t)
    #     # print(f'\n {f.name} Function: {f} \n | y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f}\n\n')
    #     # log(f'\nNumber of black boxes: {i} \n {f.name}:  y opt = {y_optk:-11.4e} | time = {time_taken:-10.4f} | x opt = {i_opt} \n ')
    # df = pd.DataFrame(columns = ['m','y','t'])
    # df['m'] = m_value
    # df['y'] = y_value
    # df['t'] = t_value
    # df.to_csv("../Results_new_fun/FL_ssprotes.csv")

if __name__ == '__main__':
    calc()
