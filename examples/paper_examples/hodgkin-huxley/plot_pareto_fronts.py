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

data = dict(np.load("./pareto_front_results.npz"))

taus = data["taus"]
costs = data["costs"]
errors = data["errors"]

print(taus)
mask = np.argwhere(np.logical_and(taus>1.1e-5, taus < 1.1e6)).flatten()
print(taus[mask])

fig = newfig(0)

plt.loglog(costs[mask], errors[mask], 'o', linewidth=5, color='tab:green', markerfacecolor='tab:blue', markeredgecolor='tab:blue', markeredgewidth=5, markersize=20)

plt.xlim([min(costs[mask])/2.5, max(costs[mask])*2])
plt.ylim([min(errors[mask])/2, max(errors[mask])*2])

plt.xlabel('Total cost')
plt.ylabel('Normalized error')

plt.tight_layout()

fig.savefig("figure_pareto_front.pdf", format='pdf', dpi=600, bbox_inches = "tight")

#plt.show()
