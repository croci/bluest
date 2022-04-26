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
    out_samples[0] = (sum(costs*out_samples[1:]))/5
    return out_samples

all_models = np.arange(-1,12); all_models[0] = -2

sample_vals = BLUE_data["samples"]; sample_vals = sample_vals[sample_vals > 0]
blue_samples = get_samples(BLUE_models, sample_vals)
mlmc_samples = get_samples(mlmc_models, mlmc_data["samples"])
mfmc_samples = get_samples(mfmc_models, mfmc_data["samples"])

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
