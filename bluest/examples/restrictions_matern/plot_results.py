from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
import subprocess
import os
import sys

################################################# PYPLOT SETUP ###############################################

# change the round factor if you like
r = 1

screens = [l.split()[-3:] for l in subprocess.check_output(
    ["xrandr"]).decode("utf-8").strip().splitlines() if " connected" in l]

sizes = []
for s in screens:
    w = float(s[0].replace("mm", "")); h = float(s[2].replace("mm", "")); d = ((w**2)+(h**2))**(0.5)
    sizes.append([2*round(n/25.4, r) for n in [w, h, d]])

gnfntsz = 35
fntsz = 28
ttlfntsz = 30

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{amsmath}\usepackage{bm}\usepackage{relsize}\DeclareMathOperator{\EPS}{\mathlarger{\mathlarger{\mathlarger{\varepsilon}}}}')
plt.rc('font', family='serif', size=gnfntsz)
plt.rc('xtick', labelsize=gnfntsz)     
plt.rc('ytick', labelsize=gnfntsz)

def newfig(fign,small=False):
    figsize=sizes[0][:-1]
    if small: figsize[0] /= 2
    fig = plt.figure(fign, figsize=figsize)
    fig.patch.set_facecolor("white")
    return fig

###################################### LOAD DATA FILES ########################################

save = False
efficiency = True

EPS = 0.0018621360085025829
#BUDGET = 417320/9 #46368.888888888888888888888888888888
#BUDGET = 24843.33333333333333333*Nrestr
BUDGET = 317994.6666666667

eps_list = []
bud_list = []
for Nrestr in [2,4,8,16]:
    data = np.load("./estimator_sample_data%d.npz" % Nrestr, allow_pickle=True)
    eps = data['eps'].flatten()[0]
    bud = data['budget'].flatten()[0]
    eps_list.append(eps)
    bud_list.append(bud)

eps = eps_list[0]
bud = bud_list[0]

fig = newfig(0)
c_list = eps['c_list']
v_list = eps['v_list']

costs = [np.median(item) for item in c_list]
v0 = np.median(v_list[0])

efficiency_list = []
for i in range(len(v_list)):
    efficiency_list.append(np.log10(costs[0]*EPS**2/(np.array(c_list[i])*np.array(v_list[i])**2)))
    v_list[i] = np.sort(v_list[i])/EPS

if efficiency:
    v1 = plt.violinplot(efficiency_list, positions=1*np.arange(1, len(v_list)+1))
    plt.ylabel(r"estimator efficiency")
    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
else:
    plt.axhline(y=1, color='gold', linestyle='-', linewidth=3)
    v1 = plt.violinplot(v_list, positions=1*np.arange(1, len(v_list)+1))
    ax = fig.gca()
    plt.yscale('log')
plt.title("Minimize cost")

labels = ["Ex", "Est"] + ["d = %d" % (i+1) for i in range(len(v_list)-2)]
ax.xaxis.set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(1*np.arange(1, len(labels) + 1), labels=labels)

for b in v1['bodies']:
    # get the center, then modify the paths to not go further left than the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    #b.set_color('r')

if save: fig.savefig("figure_restrictions_matern_eps.pdf", format='pdf', dpi=600, bbox_inches = "tight")

fig = newfig(1)
c_list = bud['c_list']
v_list = bud['v_list']

costs = [np.median(item) for item in c_list]
v0 = np.median(v_list[0])
EPS = np.median(np.sort(v_list[0]))

efficiency_list = []
for i in range(len(v_list)):
    efficiency_list.append(np.log10(BUDGET*v0**2/(np.array(c_list[i])*np.array(v_list[i])**2)))
    v_list[i] = np.sort(v_list[i])/EPS

if efficiency:
    v1 = plt.violinplot(efficiency_list, positions=1*np.arange(1, len(v_list)+1))
    plt.ylabel(r"estimator efficiency")
    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
else:
    plt.axhline(y=1, color='gold', linestyle='-', linewidth=3)
    v1 = plt.violinplot(v_list, positions=1*np.arange(1, len(v_list)+1))
    ax = fig.gca()
    plt.yscale('log')

plt.title("Minimize variance")

labels = ["Ex", "Est"] + ["d = %d" % (i+1) for i in range(len(v_list)-2)]
ax.xaxis.set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(1*np.arange(1, len(labels) + 1), labels=labels)

for b in v1['bodies']:
    # get the center, then modify the paths to not go further left than the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    #b.set_color('r')

if save: fig.savefig("figure_restrictions_matern_budget.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()
