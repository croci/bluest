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

data = np.load("./samples.npz", allow_pickle=True)

costs = data['costs']

BLUE_models,BLUE_data = data["out_BLUE"]
mlmc_models,mlmc_data = data["out_MLMC"]
mfmc_models,mfmc_data = data["out_MFMC"]

def get_samples(method_models,sample_vals):
    out_samples = np.zeros(7+1)
    for i,item in enumerate(method_models):
        if isinstance(item, (int,np.int64)): item = [item]
        for model in item:
            out_samples[model] += sample_vals[i]

    out_samples[1:] = costs*out_samples[:-1]
    out_samples[0] = sum(out_samples[1:])
    return out_samples

all_models = np.arange(-1,7); all_models[0] = -2

sample_vals = BLUE_data["samples"]; sample_vals = sample_vals[sample_vals > 0]
blue_samples = get_samples(BLUE_models, sample_vals)
mlmc_samples = get_samples(mlmc_models, mlmc_data["samples"])
mfmc_samples = get_samples(mfmc_models, mfmc_data["samples"])

fig = newfig(0)
logscale = True
ax = fig.gca()
colors = ['#eff3ff','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594'][::-1]
width = 0.8/1.2

which_blue_models = np.argwhere(blue_samples[1:]>0).flatten()
reduced_blue_samples = blue_samples[1:][which_blue_models]

bottom = int(logscale)
for i in range(len(reduced_blue_samples)):
    ax.barh('MLBLUE', reduced_blue_samples[i], left=bottom, height=width, hatch="", log=logscale, color=colors[i], edgecolor='black')
    bottom += reduced_blue_samples[i]

reduced_mlmc_samples = mlmc_samples[1:][mlmc_models]
bottom = int(logscale)
for i in range(len(reduced_mlmc_samples)):
    ax.barh('MLMC', reduced_mlmc_samples[i], left=bottom, height=width, hatch="", log=logscale, color=colors[mlmc_models[i]], edgecolor='black')
    bottom += reduced_mlmc_samples[i]

reduced_mfmc_samples = mfmc_samples[1:][mfmc_models]
bottom = int(logscale)
for i in range(len(reduced_mfmc_samples)):
    ax.barh('MFMC', reduced_mfmc_samples[i], left=bottom, height=width, hatch="", log=logscale, color=colors[mfmc_models[i]], edgecolor='black')
    bottom += reduced_mfmc_samples[i]

plt.xlabel("Sampling cost", labelpad=10)
if not logscale:
    plt.ticklabel_format(axis='x', style='sci', scilimits=(int(np.log10(bottom)),int(np.log10(bottom))))
else:
    plt.xlim([10**3,10**(1+int(np.log10(mfmc_samples[0])))])

patch_list = []
for i in range(7):
    patch_list.append(Patch(edgecolor="black", facecolor=colors[i], hatch="", label=r"$\Delta x = 2^{-%d}$" % (7-i)))
leg = ax.legend(handles=patch_list, loc=9, bbox_to_anchor=(0.5,0), ncol=4,  title="Mesh size", fontsize=fntsz, title_fontsize=fntsz, handler_map = {list: HandlerTuple(None)})

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

fig.subplots_adjust(bottom=0.26)

fig.savefig("figure_restrictions_matern.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()
