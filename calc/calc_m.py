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


from constr import ind_tens_max_ones


from teneva_bm import *

bms = [
    # BmFuncAckley(d=7, n=16, name='P-01'),
    # BmFuncAlpine(d=7, n=16, name='P-02'),
    # BmFuncExp(d=7, n=16, name='P-03'),
    # BmFuncGriewank(d=7, n=16, name='P-04'),
    # BmFuncMichalewicz(d=7, n=16, name='P-05'),
    # BmFuncPiston(d=7, n=16, name='P-06'),
    # BmFuncQing(d=7, n=16, name='P-07'),
    # BmFuncRastrigin(d=7, n=16, name='P-08'),
    # BmFuncSchaffer(d=7, n=16, name='P-09'),
    # BmFuncSchwefel(d=7, n=16, name='P-10'), 

    ### new analytic functions
    #### need upgarded version

    # BmFuncChung(d = 7, n = 16, name ='P-21'),

    # BmFuncDixon(d = 7, n = 16, name ='P-22'), 

    # BmFuncPathological(d = 7, n = 16, name ='P-23'),
    # BmFuncPinter(d = 7, n = 16, name ='P-24'), 
    # BmFuncPowell(d = 7, n = 16, name ='P-25'), 

    # BmFuncQing(d = 7, n = 16, name ='P-26'),
    # BmFuncRosenbrock(d = 7, n = 16, name ='P-27'),

    # BmFuncSalomon(d = 7, n = 16, name ='P-28'), 
    # BmFuncSphere(d = 7, n = 16, name ='P-29'), 
    # BmFuncSquares(d = 7, n = 16, name ='P-30'),
    # BmFuncTrid(d = 7, n = 16, name ='P-31'), 
    # BmFuncTrigonometric(d = 7, n = 16, name ='P-32'), 
    # BmFuncWavy(d = 7, n = 16, name ='P-33'), 
    # BmFuncYang(d = 7, n = 16, name ='P-34'),

    
    
    # BmQuboMaxcut(d=50, name='P-11'),
    # BmQuboMvc(d=50, name='P-12'),
    # BmQuboKnapQuad(d=50, name='P-13'),
    # BmQuboKnapAmba(d=50, name='P-14'),

    BmOcSimple(d=25, name='P-15'),
    BmOcSimple(d=50, name='P-16'),
    BmOcSimple(d=100, name='P-17'),

    BmOcSimpleConstr(d=25, name='P-18'),
    BmOcSimpleConstr(d=50, name='P-19'),
    BmOcSimpleConstr(d=100, name='P-20'),
]


BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
                'P-08', 'P-09', 'P-10', 'P-21', 'P-22','P-23', 'P-24', 
                'P-25', 'P-26', 'P-27', 'P-28', 'P-29', 'P-30', 
                'P-31', 'P-32', 'P-33', 'P-34']
BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
BM_OC        = ['P-15', 'P-16', 'P-17']
BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']


from opti import *
Optis = {
    'FED PROTES': OptifedProtes,
    'Noisy PROTES': OptiProtesNoisy,
    # 'Noisy_protes': OptiProtesNoisyComp,
    'PROTES': OptiProtes,
    'BS-1': OptiTTOpt,
    'BS-2': OptiOptimatt,
    'BS-3': OptiOPO,
    'BS-4': OptiPSO,
    'BS-5': OptiNB,
    'BS-6': OptiSPSA,
    'BS-7': OptiPortfolio,
}


class Log:
    def __init__(self, fpath='../Results/log_protes_baselines.txt'):
        self.fpath = fpath
        self.is_new = True

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False


def calc(m=int(1.E+4), seed=0):
    log = Log()
    res = {}

    for bm in bms:
        # np.random.seed(seed)
        if bm.name in BM_FUNC:
            # We carry out a small random shift of the function's domain,
            # so that the optimum does not fall into the middle of the domain:
            bm = _prep_bm_func(bm)
        else:
            bm.prep()

        # log(bm.info())
        res[bm.name] = {}
        a = np.random.randint(0, 1000)
        for opti_name, Opti in Optis.items():
            # np.random.seed(seed)
            opti = Opti(name=opti_name)
            opti.prep(bm.get, bm.d, bm.n, m, is_f_batch=True)

            if bm.name in BM_OC_CONSTR and opti_name == 'Our' or opti_name == 'my':
                # Problem with constraint for PROTES (we use the initial
                # approximation of the special form in this case):
                P = ind_tens_max_ones(bm.d, 3, opti.opts_r)
                Pl = jnp.array(P[0], copy=True)
                Pm = jnp.array(P[1:-1], copy=True)
                Pr = jnp.array(P[-1], copy=True)
                P = [Pl, Pm, Pr]
                opti.opts(P=P)
            if opti_name == 'mw':
                def optimize_function():
                    opti.optimize()
                    return opti.y
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(optimize_function))
                y_values = zip(*results)
                opti.y = np.min(y_values)
                print("opti.y",opti.y)
            else:
                opti.optimize()
            log(f'{bm.name}:  {opti.info()}')
            # log(f"{bm.name} : opti.m_list : {opti.m_list} \n opti.y_list {opti.y_list}")
            res[bm.name][opti.name] = [opti.m_list, opti.y_list, opti.y]
            _save(res)

        log('\n\n')


def _load(fpath='res.pickle'):
    with open(fpath, 'rb') as f:
        res = pickle.load(f)
    return res


def _prep_bm_func(bm):
    shift = np.random.randn(bm.d) / 10
    a_new = bm.a - (bm.b-bm.a) * shift
    b_new = bm.b + (bm.b-bm.a) * shift
    bm.set_grid(a_new, b_new)
    bm.prep()
    return bm


def _save(res, fpath='res.pickle'):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # df = pd.DataFrame(columns = [""])
    calc()
    # for i in range(1000,10001,1000):
    #     calc(m = i)
