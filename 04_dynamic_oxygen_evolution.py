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
from scipy.integrate import cumtrapz

from ixdat.techniques.deconvolution import Kernel, DecoMeasurement

# Import measurement data
data = DecoMeasurement.read("RawData/Measurement3.pkl", reader="EC_MS")
data.__class__ = DecoMeasurement

# Calibration of O2 (M32)
tspansO2 = [[8100, 8130], [8200, 8230], [8500, 8550]]
t_bg = [7260, 7280]
signal = []
current = []

for tspan in tspansO2:
    _, sig_values = data.get_signal("M32", tspan=tspan, t_bg=t_bg)
    signal.append(np.mean(sig_values))
    _, curr_values = data.get_current(tspan=tspan)
    current.append(np.mean(curr_values))

fit = linregress(signal, np.absolute(current))

calib_O2 = fit[0]  # Calibration constant in mA/A (partial current/signal current)
data.calibration = {"M32": calib_O2}

# Define model parameters and deconvolute the oxygen signal
params = {
    "diff_const": 2.1e-9,
    "work_dist": 160e-6,
    "volflow_cap": 1.9e-10,
    "vol_gas": 1e-10,
    "henry_vola": 33,
}
model_kernel = Kernel(parameters=params)
tspan = [10600, 10800]
t_bg = [10800, 10900]

t_partcurr, v_partcurr = data.get_partial_current(
    "M32", model_kernel, tspan=tspan, t_bg=t_bg, snr=10
)
v_partcurr = v_partcurr / 0.196  # Normalize to 0.196 electrode surface area
t_curr, v_curr = data.get_current(tspan=tspan)
t_curr = t_curr - t_partcurr[0]
v_curr = v_curr / 0.196
t_pot, v_pot = data.get_potential(tspan=tspan)
t_pot = t_pot - t_partcurr[0]
v_pot = v_pot + 0.05  # RHE calibration
t_sig, v_sig = data.get_calib_signal("M32", tspan=tspan, t_bg=t_bg)
t_sig = t_sig - t_partcurr[0]
v_sig = v_sig / 0.196
t_partcurr = t_partcurr - t_partcurr[0]

# Calculate oxide rate and thickness
v_curr_inter = np.interp(t_partcurr, t_curr, v_curr)
v_partcurr_oxide = v_curr_inter - v_partcurr
v_charge_oxide = cumtrapz(v_partcurr_oxide, t_partcurr, initial=0)
v_oxide_thickness = (
    v_charge_oxide / (4 * 96485) * 227 / 1000 / 11.25 * 1e8
)  # M_PtO2 = 227 g/mol, roh_PtO2 = 11.25 g/cm^3
v_oxide_thickness = v_oxide_thickness - v_oxide_thickness[0]


# Plot and format
fig = plt.figure(figsize=(7.5, 3.5))
gs = fig.add_gridspec(2, 8)

ax1 = fig.add_subplot(gs[0, :-4])
ax1.set_xticklabels([])
ax1.set_ylabel(r"$J_i$ / [$\mathsf{\frac{mA}{cm^2}}$]")
ax1.set_ylim(-0.01, 0.5)
ax1.plot(t_partcurr, v_partcurr_oxide, color="purple", label="$\mathsf{PtO_2}$")
ax1.plot(t_sig, v_sig, color="lightgreen")
ax1.plot(t_partcurr, v_partcurr, color="green", label=r"$\mathsf{O_2}$")
plt.tight_layout()
ax1.annotate(
    r"$\mathsf{O_2}$",
    xy=(25, 0.2),
    xytext=(40, 0.3),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax1.annotate(
    r"$\mathsf{O_2}$ measured",
    xy=(43, 0.1),
    xytext=(65, 0.2),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax1.annotate(
    r"$\mathsf{PtO_2}$",
    xy=(70, 0.01),
    xytext=(90, 0.1),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)
ax1.annotate(
    r"\textbf{a)}", xy=(0.02, 0.95), xytext=(0.02, 0.95), xycoords="figure fraction"
)

ax2 = ax1.twinx()
ax2.set_ylabel(r"$\mathsf{PtO_2}$ thickness / [$\mathsf{\r{A}}$]")
ax2.set_ylim(-0.2, 10)
ax2.plot(
    t_partcurr,
    v_oxide_thickness,
    color="purple",
    linestyle="dotted",
    label="$d_{PtO_2}$",
)
ax2.annotate(
    "",
    xy=(140, 8),
    xytext=(160, 8),
    arrowprops=dict(arrowstyle="<-", lw=0.5, color="purple"),
)

ax3 = fig.add_subplot(gs[1, :-4])

ax3.set_ylim(-0.01, 0.5)
ax3.set_ylabel(r"$J_{tot}$ / [$\mathsf{\frac{mA}{cm^2}}$]")
ax3.set_xlabel("time / [s]")
ax3.plot(t_curr, v_curr, color="black")
ax3.annotate(
    "", xy=(30, 0.2), xytext=(10, 0.2), arrowprops=dict(arrowstyle="<-", lw=0.5)
)

ax4 = ax3.twinx()
ax4.set_ylabel(r"$U_{RHE}$ / [$\mathsf{V}$]")
ax4.plot(t_pot, v_pot, color="red")
ax4.annotate(
    "",
    xy=(155, 1.55),
    xytext=(175, 1.55),
    arrowprops=dict(arrowstyle="<-", color="red", lw=0.5),
)


axOx = fig.add_subplot(gs[:, -3:])
axOx.set_ylim(0.01, 0.6)
axOx.set_xlim(0, 10)
axOx.set_xlabel(r"$\mathsf{PtO_2}$ thickness / [$\mathsf{\r{A}}$]")
axOx.set_ylabel(r"$J_{O_2}$ / [$\mathsf{\frac{mA}{cm^2}}$]")
axOx.semilogy(
    v_oxide_thickness,
    v_partcurr[-670:],
    linestyle="None",
    fillstyle="none",
    marker="s",
    color="red",
)

# Create linear fit to the semilog plot
coef = np.polyfit(v_oxide_thickness[-550:-150], np.log10(v_partcurr[-550:-150]), 1)
poly1d_fn = np.poly1d(coef)
axOx.plot(
    v_oxide_thickness[-570:-150], 10 ** (poly1d_fn(v_oxide_thickness[-570:-150])), "--k"
)


plt.tight_layout()
ax1.annotate(
    r"\textbf{b)}", xy=(0.62, 0.95), xytext=(0.62, 0.95), xycoords="figure fraction"
)
fig.savefig("Plots/OER_transients.eps", dpi=1000, format="eps")
