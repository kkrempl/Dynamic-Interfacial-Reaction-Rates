#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri June 12

@author: Kevin Krempl
"""
from pathlib import Path
from ixdat.techniques.deconvolution import Kernel
import numpy as np
import matplotlib.pyplot as plt
import pickle
import colorcet as cc

plt.close("all")

# Create figure
# params = {"text.latex.preamble": [r"\usepackage{siunitx}", "\sisetup{detect-all}"]}
# plt.rcParams.update(params)
fig = plt.figure(figsize=(7.5 / 2, 3.5))
ax = fig.add_subplot(111)

# Construct meshgrid for heatmap
x_sampling_freq = np.logspace(-1, 1, 100)  # Sampling frequency
y_working_dist = np.linspace(100, 200, 50)  # Working distance
X, Y = np.meshgrid(x_sampling_freq, y_working_dist)


# Function to calculate condition number for a given sampling frequency and working distance.
@np.vectorize
def get_cond_number(sampling_freq, working_dist):
    # Define mass transport parameters.
    params = {
        "diff_const": 5.05e-9,
        "work_dist": working_dist * 1e-6,
        "vol_gas": 1.37e-10,
        "volflow_cap": 1e-10,
        "henry_vola": 52,
    }

    model_kernel = Kernel(parameters=params)

    kernel_matrix = model_kernel.calculate_kernel(
        dt=1 / sampling_freq, duration=40, matrix=True
    )

    cond = np.linalg.cond(kernel_matrix)
    print(str(cond))
    return cond


# Calculate condition number for the heatmap or load precalculated from pickle.
with open(Path(__file__).parent / "RawData/condition_numbers.pkl", "rb") as f:
    Z = pickle.load(f)
# Z = get_cond_number(X, Y)
Z[Z > 100] = 100

# Create and format heatmap plot
Z[Z > 100] = 100  # can't get `vmax=100` colorbar to work right otherwise.
cp = plt.contourf(X, Y, Z, 100, cmap=cc.cm.rainbow)
# for c in cp.collections:
#     c.set_edgecolor("face")
cbar = plt.colorbar(cp, ticks=[1, 20, 40, 60, 80, 100])
cbar.ax.set_ylim([0, 100])
cbar.ax.set_yticklabels([r"$1$", r"$20$", r"$40$", r"$60$", r"$80$", r"$>100$"])
plt.xscale("log")
ax.set_title("Condition number")
ax.set_ylabel(r"Working distance $L$ / [$\mu$m]")
ax.set_xlabel(r"Sampling frequency $f$ / [$Hz$]")
plt.tight_layout()
plt.savefig("Plots/heatmap.png")
# plt.savefig("Plots/heatmap.png", dpi=1000, format="png")
