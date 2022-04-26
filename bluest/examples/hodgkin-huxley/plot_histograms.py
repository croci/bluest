from numpy import array
import numpy as np
import matplotlib.pyplot as plt
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

costs = np.array([3.515200e+07, 2.299968e+06, 1.572160e+05, 4.394000e+06, 2.874960e+05, 1.965200e+04, 8.000000e+00, 4.000000e+00, 2.000000e+00, 4.000000e+00, 2.000000e+00, 1.000000e+00])

BLUE_models,BLUE_data = ([[8], [9], [10], [11], [7, 8], [7, 11], [8, 10], [8, 11], [9, 10], [9, 11], [10, 11], [6, 7, 8], [7, 8, 9], [7, 8, 10], [7, 8, 11], [7, 9, 10], [7, 10, 11], [8, 9, 10], [8, 9, 11], [8, 10, 11], [9, 10, 11], [6, 7, 8, 9], [6, 7, 8, 10], [6, 7, 8, 11], [7, 8, 9, 10], [7, 8, 9, 11], [7, 8, 10, 11], [8, 9, 10, 11], [0, 1, 2, 5, 6], [0, 1, 2, 5, 11], [1, 2, 5, 8, 9], [2, 5, 7, 8, 11], [2, 6, 7, 8, 11], [5, 6, 7, 8, 11], [6, 7, 8, 9, 10], [6, 7, 8, 9, 11], [6, 7, 8, 10, 11], [7, 8, 9, 10, 11]], {'samples': array([0, 0, 0, ..., 0, 0, 5]), 'errors': array([0.03708262, 0.08300076, 0.19666174, 0.16321427, 0.00442241]), 'total_cost': 1720056232.0})
BLUE_data["samples"] = np.load("./samples.npz")["samples"]

mlmc_models,mlmc_data = ([0, 1, 2], {'samples': array([   54,   424, 22701]), 'errors': [0.04011971078071171, 0.08347365348026371, 0.14777862818344045, 0.2864895813011106, 0.007241811996561267], 'total_cost': 6442354848.0})

mfmc_models, mfmc_data =  ([0, 1, 2, 11], {'samples': array([     19,     206,   13348, 2135801]), 'errors': [0.03658980150277759, 0.08323013534926027, 0.18475076828477818, 0.1566195813231975, 0.004200370009129618], 'total_cost': 3242336377.0, 'alphas': [array([0.99557222, 0.98641834, 5.55477848]), array([0.9655126 , 0.90311733, 1.01745367]), array([0.98729136, 0.96257713, 0.01684894]), array([0.98575044, 0.95792625, 0.36851226]), array([0.98972301, 0.9697663 , 0.48704786])]})

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
