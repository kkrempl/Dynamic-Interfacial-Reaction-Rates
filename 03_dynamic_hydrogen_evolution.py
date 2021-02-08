#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 8

@author: Kevin Krempl
"""

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.stats import linregress

from ixdat.techniques.deconvolution import Kernel, DecoMeasurement

# Import measurement data
data1 = DecoMeasurement.read("RawData/Measurement1.pkl", reader="EC_MS")

data2 = DecoMeasurement.read("RawData/Measurement2.pkl", reader="EC_MS")

# Calibration of H2 (M2) of Measurement1
tspansH2 = [[21300, 21310], [21360, 21369], [21400, 21410]]
t_bg = [21500, 21510]
signal = []
current = []

for tspan in tspansH2:
    _, sig_values = data1.grab_signal("M2", tspan=tspan, t_bg=t_bg)
    signal.append(np.mean(sig_values))
    _, curr_values = data1.grab_current(tspan=tspan)
    current.append(np.mean(curr_values))

fit = linregress(signal, np.absolute(current))

calib1_H2 = fit[0]
data1.calibration = {"M2": calib1_H2}

# Calibration of H2 (M2) of Measurement2
tspansH2 = [[12340, 12345], [12300, 12305], [12250, 12255]]
t_bg = [12400, 12405]
signal = []
current = []

for tspan in tspansH2:
    _, sig_values = data2.grab_signal("M2", tspan=tspan, t_bg=t_bg)
    signal.append(np.mean(sig_values))
    _, curr_values = data2.grab_current(tspan=tspan)
    current.append(np.mean(curr_values))

fit = linregress(signal, np.absolute(current))

calib2_H2 = fit[0]
data2.calibration = {"M2": calib2_H2}

# RHE calibration of RE and surface area normalization of current
data1.calibrate(RE_vs_RHE=0.05, A_el=0.196)
data2.calibrate(RE_vs_RHE=0.05, A_el=0.196)

# Define model parameters and deconvolute signal of Measurement1
params = {
    "diff_const": 5.05e-9,
    "work_dist": 150e-6,
    "volflow_cap": 1.37e-10,
    "vol_gas": 1e-10,
    "henry_vola": 52,
}
model_kernel = Kernel(parameters=params)
tspan = [20815, 21000]
t_bg = [21500, 21510]

t_partcurr, v_partcurr = data1.grab_partial_current(
    "M2", model_kernel, tspan=tspan, t_bg=t_bg, snr=50
)
v_partcurr = -v_partcurr / 0.196  # Normalize to 0.196 electrode surface area
t_curr, v_curr = data1.grab_current(tspan=tspan)
t_curr = t_curr - t_partcurr[0]
t_pot, v_pot = data1.grab_potential(tspan=tspan)
t_pot = t_pot - t_partcurr[0]
t_sig, v_sig = data1.grab_cal_signal("M2", tspan=tspan, t_bg=t_bg)
t_sig = t_sig - t_partcurr[0]
v_sig = -v_sig / 0.196
t_partcurr = t_partcurr - t_partcurr[0]

# Plot and format
fig = plt.figure(figsize=(8, 3.5))

ax12 = fig.add_subplot(222)
ax12.set_xticklabels([])
ax12.set_ylabel(r"$J_{H_2}$ / [mA cm$^{-2}$]")
ax12.set_ylim(-0.8, 0.04)
ax12.set_xlim(0, 35)
ax12.plot(t_sig, v_sig, color=(0.5, 0.9, 1, 1), label="calibrated")
ax12.plot(
    t_partcurr,
    v_partcurr,
    color=(0.03137254901960784, 0.301914648212226, 0.588404459823145, 1.0),
    label="deconvoluted",
)
ax12.annotate(
    "deconvoluted",
    xytext=(20, -0.40),
    xy=(t_partcurr[100], v_partcurr[100]),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax12.annotate(
    "measured",
    xytext=(25, -0.25),
    xy=(t_sig[130], v_sig[130]),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
# ax12.annotate(
#     r"\textbf{b)}", xy=(0.505, 0.95), xytext=(0.505, 0.95), xycoords="figure fraction"
# )


ax32 = fig.add_subplot(224)
ax32.set_ylim(-0.8, 0.04)
ax32.set_ylabel(r"$J_{tot}$ / [$\mathsf{\frac{mA}{cm^2}}$]")
ax32.set_xlabel("time / [s]")
ax32.set_xlim(0, 35)
ax32.plot(t_curr, v_curr, color="black")
ax32.annotate(
    "", xy=(7.6, -0.1), xytext=(5.1, -0.1), arrowprops=dict(arrowstyle="<-", lw=0.5)
)

ax42 = ax32.twinx()
ax42.set_ylabel(r"$U_{RHE}$ / [$\mathsf{V}$]")
ax42.set_ylim(-0.05, 0.5)
ax42.plot(t_pot, v_pot, color="red")
ax42.annotate(
    "",
    xy=(7.6, 0.22),
    xytext=(10.2, 0.22),
    arrowprops=dict(arrowstyle="<-", color="red", lw=0.5),
)


# Define model parameters and deconvolute signal of Measurement2
params = {
    "diff_const": 5.05e-9,
    "work_dist": 140e-6,
    "volflow_cap": 1.52e-10,
    "vol_gas": 1e-10,
    "henry_vola": 52,
}
model_kernel = Kernel(parameters=params)
tspan = [12136, 12380]
t_bg = [11735, 11760]

t_partcurr, v_partcurr = data2.grab_partial_current(
    "M2", model_kernel, tspan=tspan, t_bg=t_bg, snr=3.3
)
v_partcurr = -v_partcurr / 0.196  # Normalize to 0.196 electrode surface area
t_curr, v_curr = data2.grab_current(tspan=tspan)
t_curr = t_curr - t_partcurr[0]
t_pot, v_pot = data2.grab_potential(tspan=tspan)
t_pot = t_pot - t_partcurr[0]
v_pot = v_pot * 1e3  # unit conversion to mV
t_sig, v_sig = data2.grab_cal_signal("M2", tspan=tspan, t_bg=t_bg)
t_sig = t_sig - t_partcurr[0]
v_sig = -v_sig / 0.196
t_partcurr = t_partcurr - t_partcurr[0]

# Plot and format
ax1 = fig.add_subplot(221)
ax1.set_xticklabels([])
ax1.set_ylabel(r"$J_{H_2}$ / [$\mathsf{\frac{mA}{cm^2}}$]")
ax1.set_ylim(-0.25, 0.01)
ax1.set_xlim(0, 230)
ax1.plot(t_sig, v_sig, color=(0.5, 0.9, 1, 1), label="calibrated")
ax1.plot(
    t_partcurr,
    v_partcurr,
    color=(0.03137254901960784, 0.301914648212226, 0.588404459823145, 1.0),
    label="deconvoluted",
)
ax1.annotate(
    "deconvoluted",
    xytext=(95, -0.06),
    xy=(t_partcurr[1000], v_partcurr[1000]),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax1.annotate(
    "measured",
    xytext=(160, -0.14),
    xy=(t_sig[895], v_sig[900]),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax1.annotate(
    r"\textbf{a)}", xy=(0.02, 0.95), xytext=(0.02, 0.95), xycoords="figure fraction"
)

ax3 = fig.add_subplot(223)
ax3.set_ylim(-0.25, 0.01)
ax3.set_xlim(0, 230)
ax3.set_ylabel(r"$J_{tot}$ / [$\mathsf{\frac{mA}{cm^2}}$]")
ax3.set_xlabel("time / [s]")
ax3.plot(t_curr, v_curr, color="black")
ax3.annotate(
    "", xy=(40, -0.08), xytext=(24, -0.08), arrowprops=dict(arrowstyle="<-", lw=0.5)
)

ax4 = ax3.twinx()
ax4.set_ylabel(r"$U_{RHE}$ / [$\mathsf{mV}$]")
ax4.set_ylim(0, 100)
ax4.plot(t_pot, v_pot, color="red")
ax4.annotate(
    "",
    xy=(6, 40),
    xytext=(22, 40),
    arrowprops=dict(arrowstyle="<-", color="red", lw=0.5),
)

axins = inset_axes(ax1, width="30%", height="48%", loc=3)
axins.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False)
axins.plot(
    t_partcurr[:100],
    v_partcurr[:100],
    color=(0.03137254901960784, 0.301914648212226, 0.588404459823145, 1.0),
)
axins.plot(
    t_partcurr[240:340] - t_partcurr[240],
    v_partcurr[240:340] - v_partcurr[240],
    color=(0.03137254901960784, 0.301914648212226, 0.588404459823145, 1.0),
    linestyle="dotted",
)
axins.set_xlim([-1, 14])
axins.annotate(
    r"1st",
    xytext=(8, -0.01),
    xy=(4, -0.01),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
axins.annotate(
    r"2nd",
    xytext=(9, -0.03),
    xy=(3.2, -0.04),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)

plt.tight_layout()
# fig.savefig("Plots/HER_transients_all.eps", dpi=1000, format="eps")
fig.savefig("Plots/HER_transients_all.png")
