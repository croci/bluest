from numpy import array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
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

gnfntsz = 40
fntsz = 35
ttlfntsz = 40

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

#costs = np.array([3.515200e+07, 2.299968e+06, 1.572160e+05, 4.394000e+06, 2.874960e+05, 1.965200e+04, 8.000000e+00, 4.000000e+00, 2.000000e+00, 4.000000e+00, 2.000000e+00, 1.000000e+00])
costs = np.array([6.04870956e+00, 1.53544166e+00, 3.95492548e-01, 3.02435478e+00, 7.67720829e-01, 1.97746274e-01, 1.16321338e-02, 5.81606688e-03, 2.90803344e-03, 5.81606688e-03, 2.90803344e-03, 1.45401672e-03]); costs = costs/min(costs)

BLUE_models,BLUE_data = ([[6], [7], [8], [9], [10], [11], [2, 9], [2, 10], [2, 11], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [7, 8], [7, 9], [7, 10], [7, 11], [8, 9], [8, 10], [8, 11], [9, 10], [9, 11], [10, 11], [2, 6, 9], [2, 6, 10], [2, 6, 11], [2, 7, 9], [2, 7, 10], [2, 7, 11], [2, 8, 9], [2, 8, 10], [2, 8, 11], [2, 9, 10], [2, 9, 11], [2, 10, 11], [6, 7, 8], [6, 7, 9], [6, 7, 10], [6, 7, 11], [6, 8, 9], [6, 8, 10], [6, 8, 11], [6, 9, 10], [6, 9, 11], [6, 10, 11], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 9, 10], [7, 9, 11], [7, 10, 11], [8, 9, 10], [8, 9, 11], [8, 10, 11], [9, 10, 11], [1, 2, 6, 10], [1, 2, 7, 9], [1, 2, 7, 10], [1, 2, 7, 11], [1, 2, 8, 9], [1, 2, 8, 10], [1, 2, 8, 11], [2, 6, 7, 9], [2, 6, 7, 10], [2, 6, 7, 11], [2, 6, 8, 9], [2, 6, 8, 10], [2, 6, 8, 11], [2, 6, 9, 10], [2, 6, 9, 11], [2, 6, 10, 11], [2, 7, 8, 9], [2, 7, 8, 10], [2, 7, 8, 11], [2, 7, 9, 10], [2, 7, 9, 11], [2, 7, 10, 11], [2, 8, 9, 10], [2, 8, 9, 11], [2, 8, 10, 11], [2, 9, 10, 11], [6, 7, 8, 9], [6, 7, 8, 10], [6, 7, 8, 11], [6, 7, 9, 10], [6, 7, 9, 11], [6, 7, 10, 11], [6, 8, 9, 10], [6, 8, 9, 11], [6, 8, 10, 11], [6, 9, 10, 11], [7, 8, 9, 10], [7, 8, 9, 11], [7, 8, 10, 11], [7, 9, 10, 11], [8, 9, 10, 11], [1, 2, 6, 7, 9], [1, 2, 6, 7, 10], [1, 2, 6, 7, 11], [1, 2, 6, 8, 9], [1, 2, 6, 8, 10], [1, 2, 6, 8, 11], [1, 2, 6, 10, 11], [1, 2, 7, 8, 9], [1, 2, 7, 8, 10], [1, 2, 7, 8, 11], [1, 2, 7, 9, 10], [1, 2, 7, 9, 11], [1, 2, 7, 10, 11], [1, 2, 8, 9, 10], [1, 2, 8, 9, 11], [1, 2, 8, 10, 11], [2, 5, 6, 7, 9], [2, 5, 6, 7, 10], [2, 5, 6, 7, 11], [2, 5, 6, 8, 9], [2, 5, 6, 8, 10], [2, 5, 6, 8, 11], [2, 5, 7, 8, 9], [2, 5, 7, 8, 10], [2, 5, 7, 8, 11], [2, 6, 7, 8, 9], [2, 6, 7, 8, 10], [2, 6, 7, 8, 11], [2, 6, 7, 9, 10], [2, 6, 7, 9, 11], [2, 6, 7, 10, 11], [2, 6, 8, 9, 10], [2, 6, 8, 9, 11], [2, 6, 8, 10, 11], [2, 6, 9, 10, 11], [2, 7, 8, 9, 10], [2, 7, 8, 9, 11], [2, 7, 8, 10, 11], [2, 7, 9, 10, 11], [2, 8, 9, 10, 11], [6, 7, 8, 9, 10], [6, 7, 8, 9, 11], [6, 7, 8, 10, 11], [6, 7, 9, 10, 11], [6, 8, 9, 10, 11], [7, 8, 9, 10, 11], [1, 2, 6, 7, 8, 9], [1, 2, 6, 7, 8, 10], [1, 2, 6, 7, 8, 11], [1, 2, 6, 7, 9, 10], [1, 2, 6, 7, 9, 11], [1, 2, 6, 7, 10, 11], [1, 2, 6, 8, 9, 10], [1, 2, 6, 8, 9, 11], [1, 2, 6, 8, 10, 11], [1, 2, 7, 8, 9, 10], [1, 2, 7, 8, 9, 11], [1, 2, 7, 8, 10, 11], [1, 2, 7, 9, 10, 11], [1, 2, 8, 9, 10, 11], [2, 5, 6, 7, 8, 9], [2, 5, 6, 7, 8, 10], [2, 5, 6, 7, 8, 11], [2, 5, 6, 7, 9, 10], [2, 5, 6, 7, 9, 11], [2, 5, 6, 7, 10, 11], [2, 5, 6, 8, 9, 10], [2, 5, 6, 8, 9, 11], [2, 5, 6, 8, 10, 11], [2, 5, 7, 8, 9, 10], [2, 5, 7, 8, 9, 11], [2, 5, 7, 8, 10, 11], [2, 6, 7, 8, 9, 10], [2, 6, 7, 8, 9, 11], [2, 6, 7, 8, 10, 11], [2, 6, 7, 9, 10, 11], [2, 6, 8, 9, 10, 11], [2, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [0, 1, 2, 4, 5, 6, 8], [0, 1, 2, 4, 5, 7, 8], [0, 1, 2, 5, 6, 7, 8], [0, 1, 2, 5, 6, 7, 11], [0, 1, 2, 5, 6, 8, 11], [0, 1, 2, 5, 7, 8, 10], [0, 1, 2, 5, 7, 8, 11], [0, 1, 2, 5, 8, 10, 11], [1, 2, 5, 6, 7, 8, 9], [1, 2, 5, 6, 7, 8, 10], [1, 2, 5, 6, 7, 8, 11], [1, 2, 6, 7, 8, 9, 10], [1, 2, 6, 7, 8, 9, 11], [1, 2, 6, 7, 8, 10, 11], [1, 2, 6, 7, 9, 10, 11], [1, 2, 6, 8, 9, 10, 11], [1, 2, 7, 8, 9, 10, 11], [2, 5, 6, 7, 8, 9, 10], [2, 5, 6, 7, 8, 9, 11], [2, 5, 6, 7, 8, 10, 11], [2, 5, 6, 7, 9, 10, 11], [2, 5, 6, 8, 9, 10, 11], [2, 5, 7, 8, 9, 10, 11], [2, 6, 7, 8, 9, 10, 11]], {'samples': array([0, 0, 0, ..., 0, 0, 0]), 'errors': array([0.00448546, 0.00834885, 0.01971731, 0.02251689, 0.00057583]), 'total_cost': 448096.4042166486})

BLUE_data["samples"] = np.load("./samples.npz")["samples"]

#([[8], [9], [10], [11], [7, 8], [7, 11], [8, 10], [8, 11], [9, 10], [9, 11], [10, 11], [6, 7, 8], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 9, 10], [7, 10, 11], [8, 9, 10], [8, 9, 11], [8, 10, 11], [9, 10, 11], [6, 7, 8, 9], [6, 7, 8, 10], [6, 7, 8, 11], [7, 8, 9, 10], [7, 8, 9, 11], [7, 8, 10, 11], [8, 9, 10, 11], [0, 1, 2, 5, 6], [0, 1, 2, 5, 11], [1, 2, 5, 8, 9], [2, 5, 7, 8, 11], [2, 6, 7, 8, 11], [5, 6, 7, 8, 11], [6, 7, 8, 9, 10], [6, 7, 8, 9, 11], [6, 7, 8, 10, 11], [7, 8, 9, 10, 11]], {'samples': array([0, 0, 0, ..., 0, 0, 5]), 'errors': array([0.03708262, 0.08300076, 0.19666174, 0.16321427, 0.00442241]), 'total_cost': 1720056232.0})

mlmc_models,mlmc_data = ([0, 1, 2], {'samples': array([  13554,   53973, 1617340]), 'errors': [0.004730865520657373, 0.008348642017634926, 0.016644892744555485, 0.032717515660550485, 0.0008363705168721898], 'total_cost': 804502.5198109777})

#([0, 1, 2], {'samples': array([   54,   424, 22701]), 'errors': [0.04011971078071171, 0.08347365348026371, 0.14777862818344045, 0.2864895813011106, 0.007241811996561267], 'total_cost': 6442354848.0})

mfmc_models, mfmc_data = ([0, 1, 2], {'samples': array([   5816,   33017, 1179987]), 'errors': [0.005452948466399274, 0.008348641643150844, 0.018768851644428867, 0.03590933647703095, 0.0009427933182597826], 'total_cost': 552551.0374409306, 'alphas': [array([0.99557222, 0.98641834]), array([0.9655126 , 0.90311733]), array([0.98729136, 0.96257713]), array([0.98575044, 0.95792625]), array([0.98972301, 0.9697663 ])]})

#([0, 1, 2, 11], {'samples': array([     19,     206,   13348, 2135801]), 'errors': [0.03658980150277759, 0.08323013534926027, 0.18475076828477818, 0.1566195813231975, 0.004200370009129618], 'total_cost': 3242336377.0, 'alphas': [array([0.99557222, 0.98641834, 5.55477848]), array([0.9655126 , 0.90311733, 1.01745367]), array([0.98729136, 0.96257713, 0.01684894]), array([0.98575044, 0.95792625, 0.36851226]), array([0.98972301, 0.9697663 , 0.48704786])]})

eps = np.array([0.00591726, 0.00834864, 0.01971628, 0.03892175, 0.00101537])

##############################################################################################

err = 0.2

def get_samples(method_models,sample_vals):
    out_samples = np.zeros(13)
    for i,item in enumerate(method_models):
        if isinstance(item, int): item = [item]
        for model in item:
            out_samples[model] += sample_vals[i]

    out_samples[1:] = costs*out_samples[:-1]
    out_samples[0] = (sum(costs*out_samples[1:]))*1e-6
    return out_samples

all_models = np.arange(-1,12); all_models[0] = -2

sample_vals = BLUE_data["samples"]; sample_vals = sample_vals[sample_vals > 0]
blue_samples = get_samples(BLUE_models, sample_vals)
mlmc_samples = get_samples(mlmc_models, mlmc_data["samples"])
mfmc_samples = get_samples(mfmc_models, mfmc_data["samples"])

################################### FIGURE 2 #################################

Np = 3
which_model = []
for l in range(12):
    problem_n = l%Np
    model_to_run = ['HH', 'FN'][(l//Np)%2]
    pde = bool((l//Np) < 2)
    if pde: which_model.append(problem_n)
    else:   which_model.append(3)
    print(l, ": ", model_to_run, " ", ["PDE", "ODE"][int(not pde)], ". mesh size: ", ["fine", "medium", "coarse"][problem_n], ".  which model: ", which_model[-1])

fig = newfig(0)
ax = fig.gca()
colors = ["tab:red", "skyblue", "limegreen", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
colors = ['#eff3ff','#bdd7e7','#6baed6','#2171b5'][::-1] #["#e6550d", "#3182bd", "#fdd0a2", "#deebf7"]
blues = ['steelblue', 'deepskyblue', 'skyblue']
reds = ['indianred', 'salmon', 'lightsalmon']
greens = ['forestgreen','limegreen','lightgreen']
hatches = ['xxx', "\\\\\\", "///", "", 'xxx', "\\\\\\", "///", "", 'xxx', "\\\\\\", "///", ""]
width = 0.8/1.2

which_blue_models = np.argwhere(blue_samples[1:]>0).flatten()
reduced_blue_samples = blue_samples[1:][which_blue_models]
aggregated_blue_samples = np.zeros((4,))
for i in range(len(reduced_blue_samples)):
    aggregated_blue_samples[which_model[which_blue_models[i]]] += reduced_blue_samples[i]

bottom = 0
for i in range(len(aggregated_blue_samples)):
    ax.barh('MLBLUE', aggregated_blue_samples[i], left=bottom, height=width, hatch="", log=False, color=colors[i], edgecolor='black')
    bottom += aggregated_blue_samples[i]

reduced_mlmc_samples = mlmc_samples[1:][mlmc_models]
bottom = 0
for i in range(len(reduced_mlmc_samples)):
    ax.barh('MLMC', reduced_mlmc_samples[i], left=bottom, height=width, hatch="", log=False, color=colors[which_model[mlmc_models[i]]], edgecolor='black')
    bottom += reduced_mlmc_samples[i]

reduced_mfmc_samples = mfmc_samples[1:][mfmc_models]
bottom = 0
for i in range(len(reduced_mfmc_samples)):
    ax.barh('MFMC', reduced_mfmc_samples[i], left=bottom, height=width, hatch="", log=False, color=colors[which_model[mfmc_models[i]]], edgecolor='black')
    bottom += reduced_mfmc_samples[i]

plt.xlabel("Sampling cost", labelpad=10)
#plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

patch_list = []
patch_list.append(Patch(edgecolor="black", facecolor=colors[0], hatch="", label="fine mesh"))
patch_list.append(Patch(edgecolor="black", facecolor=colors[1], hatch="", label="medium mesh"))
patch_list.append(Patch(edgecolor="black", facecolor=colors[2], hatch="", label="coarse mesh"))
patch_list.append(Patch(edgecolor="black", facecolor=colors[3], hatch="", label="ODE"))
leg = ax.legend(handles=patch_list, loc=9, bbox_to_anchor=(0.5,0), ncol=2,  title="Model type", fontsize=fntsz, title_fontsize=fntsz, handler_map = {list: HandlerTuple(None)})

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

fig.subplots_adjust(bottom=0.26)

fig.savefig("figure_HH_new.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()

import sys; sys.exit(0)

fig = newfig(0)
colors = ["steelblue", "tab:red", "tab:green", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
width = 0.8/3
plt.bar(all_models-width, mlmc_samples, width=width, label=r"MLMC",   log=True, color="tab:orange")
plt.bar(all_models,       blue_samples, width=width, label=r"MLBLUE", log=True, color="steelblue")
plt.bar(all_models+width, mfmc_samples, width=width, label=r"MFMC",   log=True, color="tab:green")
plt.xlabel("Model")
plt.ylabel("Log(sampling cost)")
plt.xticks(all_models, labels=[r"$C_{\text{tot}} \times 10^{-6}$", "fine\n0", "med\n1", "coarse\n2", "fine\n3", "med\n4", "coarse\n4", "fine\n6", "med\n7", "coarse\n8", "fine\n9", "med\n10", "coarse\n11"])
plt.legend(title=r"Tol = $0.2$",framealpha=1,loc=9)
plt.axvline(x=-1, linewidth=3, color="black")
plt.axvline(x=2.5, linewidth=3, color="black")
plt.axvline(x=5.5, linewidth=3, color="black")
plt.axvline(x=8.5, linewidth=3, color="black")
plt.text(1.,1.75e7, "H-H PDE", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))
plt.text(4.,1.75e7, "F-N PDE", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))
plt.text(7.,1.75e7, "H-H ODE", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))
plt.text(10.,1.75e7, "F-N ODE", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))

#plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()

fig.savefig("figure_HH.pdf", format='pdf', dpi=600, bbox_inches = "tight")
#plt.show()

print(blue_samples[0], mlmc_samples[0], mfmc_samples[0])
