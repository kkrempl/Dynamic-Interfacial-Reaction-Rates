#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 8

@author: Kevin Krempl
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress

from ixdat.techniques.deconvolution import Kernel, DecoMeasurement

data = DecoMeasurement.read("RawData/Measurement1.pkl", reader="EC_MS")

# Calibration of O2 (M32)
tspansO2 = [[8280, 8300], [8385, 8390], [8470, 8480]]
t_bg = [8200, 8215]
signal = []
current = []

for tspan in tspansO2:
    _, sig_values = data.grab_signal("M32", tspan=tspan, t_bg=t_bg)
    signal.append(np.mean(sig_values))
    _, curr_values = data.grab_current(tspan=tspan)
    current.append(np.mean(curr_values))

fit = linregress(signal, np.absolute(current))

calib_O2 = fit[0]  # Calibration constant in mA/A (partial current/signal current)

# Calibration of H2 (M2)
tspansH2 = [[21300, 21310], [21360, 21369], [21400, 21410]]
t_bg = [21500, 21510]
signal = []
current = []

for tspan in tspansH2:
    print(tspan)
    _, sig_values = data.grab_signal("M2", tspan=tspan, t_bg=t_bg)
    signal.append(np.mean(sig_values))
    _, curr_values = data.grab_current(tspan=tspan)
    current.append(np.mean(curr_values))

fit = linregress(signal, np.absolute(current))

calib_H2 = fit[0]

# Extract and plot O2 impulse responses from measurement
tspansO2impulses = [
    [18850, 18875],
    [18880, 18905],
    [18910, 18945],
    [18960, 18995],
    [19015, 19050],
    [19065, 19105],
    [19125, 19160],
]
t_bg = [19220, 19229]

fig1 = plt.figure(figsize=(8, 3.5))
axO2 = fig1.add_subplot(122)
cmap = matplotlib.cm.get_cmap("Greens")

for i in range(len(tspansO2impulses) - 1):
    kernel = data.extract_kernel(
        "M32", tspan=tspansO2impulses[i], cutoff_pot=1.4, t_bg=t_bg
    )

    charge = kernel.sig_area * calib_O2 / 0.196  # in mC/cm^2
    moles = np.round(charge / (4 * 96485) * 1e6, decimals=2)  # in nmol/cm^2
    kernel.plot(ax=axO2, label=r"${} $".format(moles), color=cmap((i + 3) / 9))

# Plot modeled impulse response
params = {
    "diff_const": 2.1e-9,
    "work_dist": 150e-6,
    "volflow_cap": 1.37e-10,
    "vol_gas": 1e-10,
    "henry_vola": 33,
}

model_kernel = Kernel(parameters=params)
model_kernel.plot(
    ax=axO2,
    norm=True,
    color="black",
    linestyle="dashed",
    label="modeled",
)

# Format figure
axO2.set_ylabel(r"$\mathsf{\frac{h(t)}{\int h(t)}}$ / [s$^{-1}$]")
axO2.set_title("norm. impulse response (O$_2$)")
axO2.set_xlabel("time / [s]")
axO2.set_xlim(0, 30)
axO2.set_ylim(-0.025, 0.15)
axO2.set_yticks([])
leg = axO2.legend(
    frameon=True,
    fancybox=False,
    title=r"$\mathsf{n_{O_2}}$ / [$\mathsf{\frac{nmol}{cm^2}}$]",
    edgecolor="black",
    shadow=False,
)

# axO2.text(
#     -0.1, 1.05, r"\textbf{b)}", 
#     transform=axO2.transAxes, size=10, fontweight="bold"
# )

# Extract and plot H2 impulses from measurement
tspansH2impulses = [
    [20170, 20188],
    [20200, 20225],
    [20230, 20250],
    [20255, 20275],
    [20280, 20305],
    [20310, 20335],
    [20350, 20380],
]
t_bg = [20100, 20120]

axH2 = fig1.add_subplot(121)
cmap = matplotlib.cm.get_cmap("Blues")

for i in range(len(tspansH2impulses) - 1):
    kernel = data.extract_kernel(
        "M2", tspan=tspansH2impulses[i], cutoff_pot=-0.03, t_bg=t_bg
    )

    charge = kernel.sig_area * calib_H2 / 0.196  # in mC/cm^2
    moles = np.round(charge / (2 * 96485) * 1e6, decimals=2)  # in nmol/cm^2
    kernel.plot(ax=axH2, label=r"${} $".format(moles), color=cmap((i + 3) / 9))

# Plot modeled impulse response
params = {
    "diff_const": 5.05e-9,
    "work_dist": 150e-6,
    "volflow_cap": 1.37e-10,
    "vol_gas": 1e-10,
    "henry_vola": 52,
}
model_kernel = Kernel(parameters=params)
model_kernel.plot(
    ax=axH2,
    norm=True,
    color="black",
    linestyle="dashed",
    label="modeled",
)

# Format figure
axH2.set_xlabel("time / [s]")
axH2.set_xlim(0, 16)
axH2.set_ylim(-0.05, 0.3)
# axH2.text(
#     -0.1, 1.05, r"\textbf{a)}", transform=axH2.transAxes, size=10, fontweight="bold"
# )
axH2.set_yticks([])
axH2.set_ylabel(
    r"$\mathsf{\frac{h(t)}{\int h(t)}}$ / [s$^{-1}$]"
)
leg = axH2.legend(
    frameon=True,
    fancybox=False,
    title=r"$\mathsf{n_{H_2}}$ / [$\mathsf{\frac{nmol}{cm^2}}$]",
    edgecolor="black",
    shadow=False,
)
axH2.set_title("norm. impulse response (H$_2$)")
plt.tight_layout()
fig1.savefig("Plots/comparisonH2BG_norm.png")
# fig1.savefig("Plots/comparisonH2BG_norm.eps", dpi=1000, format="eps")
