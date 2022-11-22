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

gnfntsz = 25
fntsz = 25
ttlfntsz = 25

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

save = True
dmax = 3

EPS = 0.0018621360085025829
#BUDGET = 417320/9 #46368.888888888888888888888888888888
#BUDGET = 24843.33333333333333333*Nrestr
BUDGET = 317994.6666666667

Nrestr_list = [2,4,8,16]
eps_list = []
bud_list = []
for Nrestr in Nrestr_list:
    data = np.load("./estimator_sample_data%d.npz" % Nrestr, allow_pickle=True)
    eps = data['eps'].flatten()[0]
    bud = data['budget'].flatten()[0]
    eps_list.append(eps)
    bud_list.append(bud)

fig = newfig(0)
plt.subplot(2,2,1)
v0 = np.mean(np.array([item for bud in bud_list for item in bud['v_list'][0]]))

c_list = [np.array(bud['c_list'][1]) for bud in bud_list] #+ [np.array([item for bud in bud_list for item in bud['c_list'][0]])]
v_list = [np.array(bud['v_list'][1]) for bud in bud_list] #+ [np.array([item for bud in bud_list for item in bud['v_list'][0]])]

efficiency_list = []
for i in range(len(v_list)):
    efficiency_list.append(np.log10(BUDGET*v0**2/(np.array(c_list[i])*np.array(v_list[i])**2)))

v1 = plt.violinplot(efficiency_list, positions=1*np.arange(1, len(v_list)+1))
plt.ylabel(r"estimator efficiency")
plt.xlabel(r"$n_{\text{HF}}$", labelpad=-20)
ax = fig.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.title("Approach a)", fontsize=ttlfntsz)

labels = Nrestr_list
ax.xaxis.set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(1*np.arange(1, len(labels) + 1), labels=labels)

for b in v1['bodies']:
    # get the center, then modify the paths to not go further left than the center
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    #b.set_color('r')

#if save: fig.savefig("figure_restrictions_matern_budget_a.pdf", format='pdf', dpi=600, bbox_inches = "tight")

for d in range(dmax):
    #fig = newfig(1+d)
    plt.subplot(2,2,2+d)
    v0 = np.mean(np.array([item for bud in bud_list for item in bud['v_list'][0]]))

    c_list = [np.array(bud['c_list'][2+d]) for bud in bud_list]
    v_list = [np.array(bud['v_list'][2+d]) for bud in bud_list]

    efficiency_list = []
    for i in range(len(v_list)):
        efficiency_list.append(np.log10(BUDGET*v0**2/(np.array(c_list[i])*np.array(v_list[i])**2)))

    v1 = plt.violinplot(efficiency_list, positions=1*np.arange(1, len(v_list)+1))
    plt.ylabel(r"estimator efficiency")
    plt.xlabel(r"$n_{\text{HF}}$", labelpad=-20)
    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if d > 0: ax.set_yticks(np.array([-1,0]))

    plt.title("Approach b), $d=%d$" % (d+1), fontsize=ttlfntsz)

    labels = Nrestr_list
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(1*np.arange(1, len(labels) + 1), labels=labels)

    for b in v1['bodies']:
        # get the center, then modify the paths to not go further left than the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        #b.set_color('r')

    #if save: fig.savefig("figure_restrictions_matern_budget_b%d.pdf" % d, format='pdf', dpi=600, bbox_inches = "tight")

plt.tight_layout()
plt.subplots_adjust(left=None,
                    bottom=None,
                    right=None,
                    top=None,
                    wspace=None,
                    hspace=0.2)

if save: fig.savefig("figure_restrictions_matern_budget.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()

###########################################################################

#fig = newfig(0)
#c_list = eps['c_list']
#v_list = eps['v_list']
#
#costs = [np.median(item) for item in c_list]
#v0 = np.median(v_list[0])
#
#efficiency_list = []
#for i in range(len(v_list)):
#    efficiency_list.append(np.log10(costs[0]*EPS**2/(np.array(c_list[i])*np.array(v_list[i])**2)))
#    v_list[i] = np.sort(v_list[i])/EPS
#
#if efficiency:
#    v1 = plt.violinplot(efficiency_list, positions=1*np.arange(1, len(v_list)+1))
#    plt.ylabel(r"estimator efficiency")
#    ax = fig.gca()
#    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#else:
#    plt.axhline(y=1, color='gold', linestyle='-', linewidth=3)
#    v1 = plt.violinplot(v_list, positions=1*np.arange(1, len(v_list)+1))
#    ax = fig.gca()
#    plt.yscale('log')
#plt.title("Minimize cost")
#
#labels = ["Ex", "Est"] + ["d = %d" % (i+1) for i in range(len(v_list)-2)]
#ax.xaxis.set_tick_params(direction='out')
#ax.xaxis.set_ticks_position('bottom')
#ax.set_xticks(1*np.arange(1, len(labels) + 1), labels=labels)
#
#for b in v1['bodies']:
#    # get the center, then modify the paths to not go further left than the center
#    m = np.mean(b.get_paths()[0].vertices[:, 0])
#    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
#    #b.set_color('r')
#
#if save: fig.savefig("figure_restrictions_matern_eps.pdf", format='pdf', dpi=600, bbox_inches = "tight")

