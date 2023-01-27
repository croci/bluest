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

# 0:  (64, f, f). cost = 22.60032949
# 1:  (64, f, c). cost = 15.75815486
# 2:  (64, c, f). cost = 17.81515651
# 3:  (64, c, c). cost = 10.95222405
# 4:  (32, f, f). cost = 6.09950577
# 5:  (32, f, c). cost = 4.2777594
# 6:  (32, c, f). cost = 4.75914333
# 7:  (32, c, c). cost = 2.94629325
# 8:  (16, f, f). cost = 1.70378913
# 9:  (16, f, c). cost = 1.29555189
# 10: (16, c, f). cost = 1.40823723
# 11: (16, c, c). cost = 1


costs = np.array([514.2223183391003, 251.11670479549565, 316.3014055537184, 119.7527351628473, 37.42560553633218, 18.103253298467695, 22.869635334918478, 8.697236420727346, 2.9814995092105474, 1.6884537576736363, 1.9949462010013241, 1.0])

BLUE_data = {'models': [[9], [11], [8, 11], [9, 11], [10, 11], [8, 10, 11], [9, 10, 11], [7, 8, 9, 10, 11], [0, 7, 8, 9, 10, 11]], 'samples': array([      1, 1262042,   73877,  120362,   43849,    6748,   36070, 2677,     545]), 'errors': array([1.58928417e-03, 5.30358334e-05, 6.05170379e-05, 3.51303639e-03, 1.76118693e-04, 8.20450343e-06]), 'total_cost': 2553327.252334362}
BLUE_models = BLUE_data["models"]

mlmc_data = {'models': [0, 4, 8, 9], 'samples': array([   1140,   51211,  414394, 1680836]), 'errors': [0.0015341914973012135, 5.114254425422785e-05, 5.740520684626049e-05, 0.00306297188195172, 0.0001555048352823073, 7.362804126011787e-06], 'total_cost': 6576345.495880601}
mlmc_models = mlmc_data["models"]

#mfmc_models, mfmc_data = ([0, 4, 8, 9], {'samples': array([   131,   3775,  32284, 296653]), 'errors': [0.0036277556484884347, 0.0001095471583401799, 0.00013793971941145668, 0.00802001344683736, 0.00046222763424068884, 1.8142914261703772e-05], 'total_cost': 805784.3873325866, 'alphas': [array([1.0055886 , 1.02455459, 1.02216117]), array([1.00564508, 1.0905163 , 1.13111891]), array([0.99799941, 1.0355617 , 1.03525985]), array([1.00292252, 1.01379495, 1.15066571]), array([1.00451247, 1.04232768, 1.5492539 ]), array([1.0022737 , 0.9981801 , 1.04077654])]})
mfmc_data = {'models': [0, 4, 8, 11], 'samples': array([    884,   25994,  322647, 1954776]), 'errors': [0.0014139705335469274, 5.232821449565933e-05, 5.3853429585337654e-05, 0.00312461444574503, 0.00016029463332432244, 7.303857262220127e-06], 'total_cost': 4344161.591871439, 'alphas': [array([1.0055886 , 1.02455459, 1.30802092]), array([1.00564508, 1.13111891, 0.65796737]), array([0.99799941, 1.0355617 , 0.98074927]), array([1.00292252, 1.01379495, 1.15453911]), array([1.00451247, 1.04232768, 1.53998143]), array([1.0022737 , 0.9981801 , 0.99237681])]}
mfmc_models = mfmc_data["models"]

plot_samples = False
if plot_samples: costs = 1

def get_samples(method_models,sample_vals):
    out_samples = np.zeros(13)
    for i,item in enumerate(method_models):
        if isinstance(item, int): item = [item]
        for model in item:
            out_samples[model] += sample_vals[i]

    out_samples[1:] = costs*out_samples[:-1]
    out_samples[0] = (sum(out_samples[1:]))/5
    return out_samples

all_models = np.arange(-1,12); all_models[0] = -2

sample_vals = BLUE_data["samples"]; sample_vals = sample_vals[sample_vals > 0]
blue_samples = get_samples(BLUE_models, sample_vals)
mlmc_samples = get_samples(mlmc_models, mlmc_data["samples"])
mfmc_samples = get_samples(mfmc_models, mfmc_data["samples"])

################################### FIGURE 2 #################################

fig = newfig(2)
ax = fig.gca()
colors = ["tab:red", "skyblue", "limegreen", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
colors = ["#3182bd", "#9ecae1", "#deebf7"]
blues = ['steelblue', 'deepskyblue', 'skyblue']
reds = ['indianred', 'salmon', 'lightsalmon']
greens = ['forestgreen','limegreen','lightgreen']
hatches = ['xxx', "\\\\\\", "///", "", 'xxx', "\\\\\\", "///", "", 'xxx', "\\\\\\", "///", ""]
width = 0.8/1.2

which_blue_models = np.argwhere(blue_samples[1:]>0).flatten()
reduced_blue_samples = blue_samples[1:][which_blue_models]
bottom = 0
for i in range(len(reduced_blue_samples)):
    ax.barh('MLBLUE', reduced_blue_samples[i], left=bottom, height=width, hatch=hatches[which_blue_models[i]], log=False, color=colors[which_blue_models[i]//4], edgecolor='black')
    bottom += reduced_blue_samples[i]

reduced_mlmc_samples = mlmc_samples[1:][mlmc_models]
bottom = 0
for i in range(len(reduced_mlmc_samples)):
    ax.barh('MLMC', reduced_mlmc_samples[i], left=bottom, height=width, hatch=hatches[mlmc_models[i]], log=False, color=colors[mlmc_models[i]//4], edgecolor='black')
    bottom += reduced_mlmc_samples[i]

reduced_mfmc_samples = mfmc_samples[1:][mfmc_models]
bottom = 0
for i in range(len(reduced_mfmc_samples)):
    ax.barh('MFMC', reduced_mfmc_samples[i], left=bottom, height=width, hatch=hatches[mfmc_models[i]], log=False, color=colors[mfmc_models[i]//4], edgecolor='black')
    bottom += reduced_mfmc_samples[i]

plt.xlabel("Sampling cost", labelpad=3)
plt.ticklabel_format(axis='x', style='sci', scilimits=(int(np.log10(bottom)),int(np.log10(bottom))))

patch_list = []
patch_list.append(Patch(edgecolor="black", facecolor=colors[0], hatch="", label="Fine mesh"))
patch_list.append(Patch(edgecolor="black", facecolor=colors[1], hatch="", label="Medium mesh"))
patch_list.append(Patch(edgecolor="black", facecolor=colors[2], hatch="", label="Coarse mesh"))
patch_list.append(Patch(edgecolor="black", facecolor="white", hatch="xx", label="LR: Left \& Right"))
patch_list.append(Patch(edgecolor="black", facecolor="white", hatch="\\\\", label="LR: Left"))
patch_list.append(Patch(edgecolor="black", facecolor="white", hatch="//", label="LR: Right"))
patch_list.append(Patch(edgecolor="black", facecolor="white", hatch="", label="LR: None"))
leg = ax.legend(handles=patch_list, loc=9, bbox_to_anchor=(0.5,0), ncol=3,  title="Global mesh size and local refinement (LR) around cylinders", fontsize=fntsz, title_fontsize=fntsz, handler_map = {list: HandlerTuple(None)})

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

fig.subplots_adjust(bottom=0.26)

if plot_samples:
    fig.savefig("figure_NS_new_samples.pdf", format='pdf', dpi=600, bbox_inches = "tight")
else:
    fig.savefig("figure_NS_new.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()
