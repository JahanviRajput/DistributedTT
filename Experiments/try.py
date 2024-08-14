import matplotlib as mpl
import numpy as np
from time import perf_counter as tpc


mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
    'font.size': 12,
    'text.usetex': True,
})


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


import sys
sys.path.append("../")
from protes import protes_fed_learning
from protes import protes

def func_build(d=7, n=16):
    """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""

    a = -32.768         # Grid lower bound
    b = +32.768         # Grid upper bound

    par_a = 20.         # Standard parameter values for Ackley function
    par_b = 0.2
    par_c = 2.*np.pi

    def func(I):
        """Target function: y=f(I); [samples,d] -> [samples]."""
        X = I / (n - 1) * (b - a) + a

        y1 = np.sqrt(np.sum(X**2, axis=1) / d)
        y1 = - par_a * np.exp(-par_b * y1)

        y2 = np.sum(np.cos(par_c * X), axis=1)
        y2 = - np.exp(y2 / d)

        y3 = par_a + np.exp(1.)

        return y1 + y2 + y3

    return d, n, func

d, n, f = func_build() # Target function

# Number of requests to the objective function:
m = int(1.E+4)

# The batch size for optimization:
k_list = [50, 100, 150, 200, 250]

# Number of selected candidates for all batches:
k_top_list = [5, 10, 15, 20, 25]

# TT-rank of the probability TT-tensor:
r_list = [3, 5, 7]

# Number of gradient steps:
k_gd = 1

# Learning rate:
lr = 5.E-2


result = {}
for r in r_list:
    result[r] = {}
    for k in k_list:
        result[r][k] = {}
        for k_top in k_top_list:
            t = tpc()
            i_opt, y_opt = protes(f, d, n, m, k=k, k_top=k_top, k_gd=k_gd, lr=lr, r=r, seed=0)
            t = tpc() - t
            print(f'>>> r = {r:-5d} | k = {k:-5d} | k_top = {k_top:-5d} | y = {y_opt:-12.6e} | t = {t:-8.2f}')
            result[r][k][k_top] = {'y': y_opt, 't': t}


def prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()

    if leg:
        ax.legend(loc='lower left', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)

colors = ['#00B454', '#080c7c', '#FFB300', '#DDE83C', '#8d230d']
marker = ['s', '*', 'D', 'o', 'p']
    
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
plt.subplots_adjust(wspace=0.35, hspace=0.35)
axs = axs.flatten()

num = -1
for r in r_list:
    num += 1
    ax = axs[num]
    ax.set_xlabel('Top candidates (k)')
    ax.set_title(f'TT-rank (R) = {r}')    
    
    for i, k in enumerate(k_list):
        x = k_top_list
        y = [result[r][k][x_cur]['y'] for x_cur in x]
        ax.plot(x, y, label=f'Batch size (K) = {k:-3d}',
            linestyle='--',
            marker=marker[i], markersize=10, linewidth=2, color=colors[i])
    
    prep_ax(ax, xlog=False, ylog=False, leg=(num==0),
        xint=True, xticks=k_top_list)
    ax.set_ylim(8.30607, 8.306081)

plt.savefig('../Results_new_fun/Fig/check_ackley.png', bbox_inches='tight')