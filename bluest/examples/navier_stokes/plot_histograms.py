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


costs = np.array([22.60032949, 15.75815486, 17.81515651 ,10.95222405 , 6.09950577 , 4.27775947 ,4.75914333 , 2.94629325 , 1.70378913 , 1.29555189,  1.40823723,1.])

BLUE_models,BLUE_data = ([[9], [11], [8, 11], [9, 10], [9, 11], [10, 11], [0, 5, 8], [0, 5, 10], [5, 8, 11], [5, 9, 10], [5, 9, 11]], {'samples': array([     0,      0,      0,      0,      0,      0,      0,      0,
            0, 214940,      0,  30402,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,   4004,      1,   2451,    275,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,    103,      0,    238,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,    238,   3862,   1069,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0]), 'errors': array([3.89161402e-03, 1.10001616e-04, 1.47843994e-04, 8.64736888e-03,
       5.95024297e-04, 1.98229767e-05]), 'total_cost': 371312.56079077424})

mlmc_models,mlmc_data = ([0, 4, 9], {'samples': array([   466,  28238, 248932]), 'errors': [0.003910636319553277, 0.00010334704011652857, 0.00014633779082842227, 0.008325307757169267, 0.0005230664985657133, 1.916529423849866e-05], 'total_cost': 505273.92158154864})

mfmc_models, mfmc_data = ([0, 4, 9], {'samples': array([   385,  19651, 260233]), 'errors': [0.0038718621429504638, 0.0001063939981311602, 0.00014711399823005036, 0.008565965195069327, 0.0005070270166749007, 1.934214475044101e-05], 'total_cost': 465707.8708401978, 'alphas': [array([1.0055886 , 1.02216117]), array([1.00564508, 1.0905163 ]), array([0.99799941, 1.03525985]), array([1.00292252, 1.15066571]), array([1.00451247, 1.5492539 ]), array([0.9981801 , 1.04077654])]})

err = 8.e-3

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

######################### FIRST PLOT ####################################
#
#fig = newfig(0)
#ax = fig.gca()
#colors = ["steelblue", "tab:red", "tab:green", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
#blues = ['steelblue', 'deepskyblue', 'skyblue']
#reds = ['indianred', 'salmon', 'lightsalmon']
#greens = ['forestgreen','limegreen','lightgreen']
#hatches = ['xxxx', "\\\\\\\\", "////", "", 'xxx', "\\\\\\", "///", "", 'xx', "\\\\", "//", ""]
#width = 0.8/1.2
#
#which_blue_models = np.argwhere(blue_samples[1:]>0).flatten()
#reduced_blue_samples = blue_samples[1:][which_blue_models]
#bottom = 0
#for i in range(len(reduced_blue_samples)):
#    ax.barh('MLBLUE', reduced_blue_samples[i], left=bottom, height=width, hatch=hatches[which_blue_models[i]], log=False, color=blues[which_blue_models[i]//4], edgecolor='black')
#    bottom += reduced_blue_samples[i]
#
#reduced_mlmc_samples = mlmc_samples[1:][mlmc_models]
#bottom = 0
#for i in range(len(reduced_mlmc_samples)):
#    ax.barh('MLMC', reduced_mlmc_samples[i], left=bottom, height=width, hatch=hatches[mlmc_models[i]], log=False, color=reds[mlmc_models[i]//4], edgecolor='black')
#    bottom += reduced_mlmc_samples[i]
#
#reduced_mfmc_samples = mfmc_samples[1:][mfmc_models]
#bottom = 0
#for i in range(len(reduced_mfmc_samples)):
#    ax.barh('MFMC', reduced_mfmc_samples[i], left=bottom, height=width, hatch=hatches[mfmc_models[i]], log=False, color=greens[mfmc_models[i]//4], edgecolor='black')
#    bottom += reduced_mfmc_samples[i]
#
#plt.xlabel("Sampling cost", labelpad=3)
#plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#
#patch_list = []
#patch_list.append([Patch(edgecolor="black", facecolor=cols[0], hatch="xxxx", label="Fine") for cols in [blues, reds, greens]])
#patch_list.append([Patch(edgecolor="black", facecolor=cols[1], hatch="xxx", label="Medium") for cols in [blues, reds, greens]])
#patch_list.append([Patch(edgecolor="black", facecolor=cols[2], hatch="xx", label="Coarse") for cols in [blues, reds, greens]])
##patch_list.append(Patch(edgecolor="black", facecolor="white", hatch="", label="Coarsest"))
#leg = ax.legend(handles=patch_list, labels=['Fine', 'Medium', 'Coarse'], loc=9, bbox_to_anchor=(0.25,0), title="Global refinement", ncol=2, fontsize=fntsz, title_fontsize=fntsz, handler_map = {list: HandlerTuple(None)}, handlelength=5)
#
#patch_list2 = []
#patch_list2.append(Patch(edgecolor="black", facecolor="white", hatch="xx", label="Left + Right"))
#patch_list2.append(Patch(edgecolor="black", facecolor="white", hatch="\\\\", label="Left"))
#patch_list2.append(Patch(edgecolor="black", facecolor="white", hatch="//", label="Right"))
#patch_list2.append(Patch(edgecolor="black", facecolor="white", hatch="", label="None"))
#ax.legend(handles=patch_list2, loc=9, bbox_to_anchor=(0.78,0), title="Cylinder refinement", ncol=2, fontsize=fntsz, title_fontsize=fntsz)
#ax.add_artist(leg)
#
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top')
#
#fig.subplots_adjust(bottom=0.22)
#
#fig.savefig("figure_NS_new1.pdf", format='pdf', dpi=600, bbox_inches = "tight")
#
######################### SECOND PLOT ####################################
#
#fig = newfig(1)
#
#ax = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
#
#colors = ["steelblue", "tab:red", "tab:green", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
#blues = ['steelblue', 'deepskyblue', 'skyblue']
#reds = ['indianred', 'salmon', 'lightsalmon']
#greens = ['forestgreen','limegreen','lightgreen']
#width = 0.8/1.2
#
#which_blue_models = np.argwhere(blue_samples[1:]>0).flatten()
#reduced_blue_samples = blue_samples[1:][which_blue_models]
#very_reduced_blue_samples = np.concatenate([reduced_blue_samples[:2], [sum(reduced_blue_samples[2:])]])
#bottom = 0
#for i in range(len(very_reduced_blue_samples)):
#    ax.barh('MLBLUE', very_reduced_blue_samples[i], left=bottom, height=width, log=False, color=blues[which_blue_models[i]//4], edgecolor='black')
#    bottom += very_reduced_blue_samples[i]
#
#reduced_mlmc_samples = mlmc_samples[1:][mlmc_models]
#bottom = 0
#for i in range(len(reduced_mlmc_samples)):
#    ax.barh('MLMC', reduced_mlmc_samples[i], left=bottom, height=width, log=False, color=reds[mlmc_models[i]//4], edgecolor='black')
#    bottom += reduced_mlmc_samples[i]
#
#reduced_mfmc_samples = mfmc_samples[1:][mfmc_models]
#bottom = 0
#for i in range(len(reduced_mfmc_samples)):
#    ax.barh('MFMC', reduced_mfmc_samples[i], left=bottom, height=width, log=False, color=greens[mfmc_models[i]//4], edgecolor='black')
#    bottom += reduced_mfmc_samples[i]
#
#plt.xlabel("Sampling cost", labelpad=3)
#plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#
#patch_list = []
#patch_list.append([Patch(edgecolor="black", facecolor=cols[0], label="Fine") for cols in [blues, reds, greens]])
#patch_list.append([Patch(edgecolor="black", facecolor=cols[1], label="Medium") for cols in [blues, reds, greens]])
#patch_list.append([Patch(edgecolor="black", facecolor=cols[2], label="Coarse") for cols in [blues, reds, greens]])
#leg = ax.legend(handles=patch_list, labels=['Fine', 'Medium', 'Coarse'], loc=9, bbox_to_anchor=(0.5,0), title="Global refinement", ncol=4, fontsize=fntsz, title_fontsize=fntsz, handler_map = {list: HandlerTuple(None)}, handlelength=4)
#
#ax.xaxis.tick_top()
#ax.xaxis.set_label_position('top')
#
#fig.subplots_adjust(bottom=0.22)
#
#colors = ['tab:orange', 'pink', 'gold', 'mediumaquamarine']
#labels = ['Right + Left', 'Left', 'Right', 'None']
#sizes_blue = reduced_blue_samples[2:]/sum(reduced_blue_samples[2:])*100
#
#for i in range(2):
#    ax = plt.subplot2grid((3, 4), (i, 3), colspan=1, rowspan=1)
#    ax.set_aspect('equal')
#    ax.pie([100], autopct='', startangle=90, textprops={'fontsize': fntsz}, colors=colors[1:])
#    if i == 0:
#        ax.set_title('Usage of local\nrefinement models', fontsize=ttlfntsz+2)
#
#ax = plt.subplot2grid((3, 4), (2, 3), colspan=1, rowspan=1)
#ax.set_aspect('equal')
#ax.pie(sizes_blue, autopct='', startangle=90, textprops={'fontsize': fntsz}, explode=(0,0.1,0,0), colors=colors)
#
#patch_list = []
#patch_list.append(Patch(edgecolor=None, facecolor=colors[0], hatch="", label="L + R"))
#patch_list.append(Patch(edgecolor=None, facecolor=colors[1], hatch="", label="Left"))
#patch_list.append(Patch(edgecolor=None, facecolor=colors[2], hatch="", label="Right"))
#patch_list.append(Patch(edgecolor=None, facecolor=colors[3], hatch="", label="None"))
#ax.legend(handles=patch_list, loc=9, bbox_to_anchor=(0.45,0), title="Cylinder refinement", ncol=2, fontsize=fntsz, title_fontsize=fntsz)
#
#plt.subplots_adjust(left=0.11, bottom=0.2, right=0.9, top=0.89, wspace=0.5, hspace=0.2)
#
#fig.savefig("figure_NS_new2.pdf", format='pdf', dpi=600, bbox_inches = "tight")
#
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
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

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

fig.savefig("figure_NS_new3.pdf", format='pdf', dpi=600, bbox_inches = "tight")

plt.show()

import sys; sys.exit(0)

fig = newfig(0)
colors = ["steelblue", "tab:red", "tab:green", "tab:orange", "tab:cyan", "tab:brown", "gold", "slategrey"]
width = 0.8/3
plt.bar(all_models-width, mlmc_samples, width=width, label=r"MLMC%, \hspace{25.5pt} $C_{\text{tot}}=3.5\times 10^7$", log=False, color="tab:orange")
plt.bar(all_models, blue_samples, width=width, label=r"MLBLUE%, $C_{\text{tot}}=3.7\times 10^6$", log=False, color="steelblue")
plt.bar(all_models+width, mfmc_samples, width=width, label=r"MFMC%, \hspace{25pt} $C_{\text{tot}}=1.5\times 10^7$", log=False, color="tab:green")
plt.xlabel("Model")
plt.ylabel("Sampling cost")
plt.xticks(all_models, labels=[r"$\frac{1}{5}C_{\text{tot}}$", "$(f,f)$\n0", "$(f,c)$\n1", "$(c,f)$\n2", "$(c,c)$\n3", "$(f,f)$\n4", "$(f,c)$\n5", "$(c,f)$\n6", "$(c,c)$\n7", "$(f,f)$\n8", "$(f,c)$\n9", "$(c,f)$\n10", "$(c,c)$\n11"])
plt.legend(title=r"Tol = $8\times 10^{-3}$",framealpha=1,loc=9)
plt.axvline(x=-1, linewidth=3, color="black")
plt.axvline(x=3.5, linewidth=3, color="black")
plt.axvline(x=7.5, linewidth=3, color="black")
plt.text(1.5,1.75e5, "finest grid", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))
plt.text(5.5,1.75e5, "medium grid", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))
plt.text(9.5,1.75e5, "coarse grid", horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=1, edgecolor='0.8'))

plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()

fig.savefig("figure_NS.pdf", format='pdf', dpi=600, bbox_inches = "tight")
#plt.show()

print(blue_samples[0], mlmc_samples[0], mfmc_samples[0])
