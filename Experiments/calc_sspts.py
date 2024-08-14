### remove temp folders:
### find . -mindepth 1 -maxdepth 1 -type d ! -name 'opti' ! -name 'figures' ! -name 'wfiles'  -exec rm -r {} +

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
    
    BmFuncChung(d = 10000, n = 16, name ='F-01'),

    # BmFuncDixon(d = 10000, n = 16, name ='F-02'), 

    # BmFuncPathological(d = 10000, n = 16, name ='F-03'),
    # BmFuncPinter(d = 10000, n = 16, name ='F-04'), 
    # BmFuncPowell(d = 10000, n = 16, name ='F-05'), 

    # BmFuncQing(d = 10000, n = 16, name ='F-06'),
    # BmFuncRosenbrock(d = 7, n = 16, name ='F-07'),

    # BmFuncSalomon(d = 10000, n = 16, name ='F-08'), 
    # BmFuncSphere(d = 10000, n = 16, name ='F-09'), 
    # BmFuncSquares(d = 10000, n = 16, name ='F-10'),
    # BmFuncTrid(d = 10000, n = 16, name ='F-11'), 
    # BmFuncTrigonometric(d = 10000, n = 16, name ='F-12'), 
    # BmFuncWavy(d = 10000, n = 16, name ='F-13'), 
    # BmFuncYang(d = 10000, n = 16, name ='F-14'),

    
    
    # BmQuboMaxcut(d=10000, name='P-11'),
    # BmQuboMvc(d=10000, name='P-12'),
    # BmQuboKnapQuad(d=10000, name='P-13'),
    # BmQuboKnapAmba(d=10000, name='P-14'),

    # BmOcSimple(d=10000, name='P-15'),
    # BmOcSimple(d=10000, name='P-16'),
    # BmOcSimple(d=10000, name='P-17'),

    # BmOcSimpleConstr(d=10000, name='P-18'),
    # BmOcSimpleConstr(d=10000, name='P-19'),
    # BmOcSimpleConstr(d=10000, name='P-20'),
]


### suplimentary metrial functions
# bms = [
#     BmFuncAckley(d=10000, n=16, name='P-01'),
#     BmFuncAlpine(d=10000, n=16, name='P-02'),
#     BmFuncExp(d=10000, n=16, name='P-03'),
#     BmFuncGriewank(d=10000, n=16, name='P-04'),
#     BmFuncMichalewicz(d=10000, n=16, name='P-05'),
#     BmFuncPiston(d=10000, n=16, name='P-06'),
#     BmFuncQing(d=10000, n=16, name='P-07'),
#     BmFuncRastrigin(d=10000, n=16, name='P-08'),
#     BmFuncSchaffer(d=10000, n=16, name='P-09'),
#     BmFuncSchwefel(d=10000, n=16, name='P-10'), 
# ]
# BM_FUNC      = ['P-01', 'P-02', 'P-03', 'P-04', 'P-05', 'P-06', 'P-07',
#                 'P-08', 'P-09', 'P-10']


BM_FUNC = ['F-01', 'F-02', 'F-03', 'F-04', 'F-05', 'F-06', 'F-07', 'F-08', 'F-09', 
           'F-10', 'F-11', 'F-12', 'F-13', 'F-14']
BM_QUBO      = ['P-11', 'P-12', 'P-13', 'P-14']
BM_OC        = ['P-15', 'P-16', 'P-17']
BM_OC_CONSTR = ['P-18', 'P-19', 'P-20']


from opti import *
Optis = {
    'SSPTSFL' : Optissptsfl,
    'SSPTSLD' : Optissptsld,
    'BS-0': OptiProtes
}


class Log:
    def __init__(self, fpath='../Results/sspts_results.txt'):
        self.fpath = fpath
        self.is_new = True

        if os.path.dirname(self.fpath):
            os.makedirs(os.path.dirname(self.fpath), exist_ok=True)

    def __call__(self, text):
        print(text)
        with open(self.fpath, 'w' if self.is_new else 'a') as f:
            f.write(text + '\n')
        self.is_new = False


def calc_sspts_fun(m=int(1.E+4), seed=0):
    log = Log()
    res = {}

    for bm in bms:
        np.random.seed(seed)
        if bm.name in BM_FUNC:
            # We carry out a small random shift of the function's domain,
            # so that the optimum does not fall into the middle of the domain:
            bm = _prep_bm_func(bm)
        else:
            bm.prep()

        log(bm.info())
        res[bm.name] = {}
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
           
            opti.optimize()
            log(opti.info())
            res[bm.name][opti.name] = [opti.m_list, opti.y_list, opti.y]
            _save(res)

        log('\n\n')


def text():
    res = _load()

    text =  '\n\n% ' + '='*50 + '\n' + '% [START] Auto generated data \n\n'

    for i, (bm, item) in enumerate(res.items(), 1):
        if i in [11, 15, 18]:
            text += '\n\\hline\n'
        if i == 1:
            text += '\\multirow{10}{*}{\\parbox{1.6cm}{Analytic Functions}}\n'
        if i == 11:
            text += '\\multirow{3}{*}{QUBO}\n'
        if i == 15:
            text += '\\multirow{3}{*}{Control}\n'
        if i == 18:
            pass
            # text += '\\multirow{3}{*}{\parbox{1.67cm}{Control +constr.}}\n'

        text += f'    & {bm}\n'
        vals = np.array([v[2] for v in item.values()])
        for v in vals:
            if v < 1.E+40:
                text += f'        & {v:-8.1e}\n'
            else:
                text += f'        & Fail\n'
        text += f'    \\\\ \n'
    text += '\n\n\\hline\n\n'
    text += '\n% [END] Auto generated data \n% ' + '='*50 + '\n\n'
    print(text)


def _load(fpath='../Results/sspts_res.pickle'):
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


def _save(res, fpath='../Results/res.pickle'):
    with open(fpath, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    calc_sspts_fun()