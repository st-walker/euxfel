# reference case of s2e for 250 pC and SASE1/SASE3 branch
# created by Igor Zagorodnov on 10 January 2020

# In order to use this script do following:
# 1) create directory data_dir+"/tws";
# 2) create directory data_dir+"/particles" and put there file "gun.npz";

import os
from copy import deepcopy
from math import pi
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ocelot.common.globals import m_e_GeV
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.io import load_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.gui.accelerator import plot_opt_func_reduced, show_e_beam
from ocelot.utils.section_track import SectionLattice

from euxfel.sections import A1, AH1, BC0, BC1, BC2, CL1, CL2, CL3, DL, L1, L2, L3, LH, SASE1, SASE2, STN10, T1, T3, T4

matplotlib.use('TkAgg')

print(os.getcwd())
data_dir = "./"

all_sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3, STN10, SASE1, T4]  # , SASE3, T4D]
sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3, STN10, SASE1, T4]
in_file = "gun.npz"

LoadRF = "RF_250_5_2M.txt"  # RF parameters
E40 = 14000  # final beam enenrgy
r1 = 4.1218  # deflecting radius in BC0
r2 = 8.3934  # deflecting radius in BC1
r3 = 14.4111  # deflecting radius in BC2
C10 = 3  # local compression in BC0
C20 = 7  # local compression in BC1
C30 = 400 / (C10 * C20)  # local compression in BC2
R2 = 0  # first derivative of the inverse compression function
R3 = 900  # second derivative of the inverse compression function

match_exec = True  # artificial matching
wake_exec = True  # wakes
SC_exec = True  # space charge
CSR_exec = True  # CSR
smooth_exec = False  # artificial smoothing

# design optics
tws0 = Twiss()
tws0.E = 0.005
tws0.beta_x = 0.286527307369
tws0.beta_y = 0.286527307369
tws0.alpha_x = -0.838833736086
tws0.alpha_y = -0.838833736086
section_lat = SectionLattice(sequence=all_sections, tws0=tws0, data_dir=data_dir)
lat = MagneticLattice(section_lat.elem_seq)
plot_opt_func_reduced(lat, section_lat.tws, top_plot=["Dx", "Dy"], legend=False)

# changing of r56 in the bunch compressors
config = {
    BC0: {"rho": r1},
    BC1: {"rho": r2},
    BC2: {"rho": r3},
}
section_lat.update_sections(sections, config=config)

# RF parameters
c = 299792458
grad = pi / 180
f = 1.3e9
k = 2 * pi * f / c
RFpars = np.loadtxt(LoadRF)
V11 = RFpars[0, 0]
fi11 = RFpars[0, 1] * grad
V13 = RFpars[1, 0]
fi13 = RFpars[1, 1] * grad
V21 = RFpars[2, 0]
fi21 = RFpars[2, 1] * grad
V31 = RFpars[3, 0]
fi31 = RFpars[3, 1] * grad
V41 = E40 - 2400
fi41 = 0

# configuration of physical processes defined in s2e_sections/sections.py
config = {
    A1: {"phi": fi11 / grad, "v": V11 / 8 * 1e-3, "SC": SC_exec, "smooth": True, "wake": wake_exec},
    AH1: {"phi": fi13 / grad, "v": V13 / 8 * 1e-3, "match": False, "SC": SC_exec, "wake": wake_exec},
    LH: {"match": True, "SC": SC_exec, "CSR": False, "wake": wake_exec},
    DL: {"match": match_exec, "SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    BC0: {"rho": r1, "match": match_exec, "SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    L1: {
        "match": match_exec,
        "phi": fi21 / grad,
        "v": V21 / 32 * 1e-3,
        "SC": SC_exec,
        "wake": wake_exec,
        "smooth": smooth_exec,
    },
    BC1: {"rho": r2, "match": match_exec, "SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    L2: {
        "match": match_exec,
        "phi": fi31 / grad,
        "v": V31 / 96 * 1e-3,
        "SC": SC_exec,
        "wake": wake_exec,
        "smooth": smooth_exec,
    },
    BC2: {"rho": r3, "match": match_exec, "SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    L3: {"phi": fi41 / grad, "v": V41 / 640 * 1e-3, "match": match_exec, "SC": SC_exec, "wake": wake_exec},
    CL1: {"match": match_exec, "SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    CL2: {"SC": SC_exec},
    CL3: {"SC": SC_exec, "CSR": CSR_exec, "wake": wake_exec},
    STN10: {"match": match_exec, "SC": False, "wake": wake_exec},
    SASE1: {"match": match_exec, "SC": False, "wake": wake_exec},
    T4: {"match": match_exec, "SC": False, "CSR": CSR_exec, "wake": wake_exec},
    T1: {"match": match_exec, "SC": False, "CSR": CSR_exec, "wake": wake_exec},
    SASE2: {"match": match_exec, "SC": False, "wake": wake_exec},
    T3: {"match": match_exec, "SC": False, "CSR": CSR_exec, "wake": wake_exec},
}
# tracking
p_array = load_particle_array(data_dir + "/particles/" + in_file)
s_start = deepcopy(p_array.s)
p_array = section_lat.track_sections(sections=sections, p_array=p_array, config=config, force_ext_p_array=True)

# collect tws for all sections
seq_global: List[Twiss] = []
tws_track_global: List[Twiss] = []
L = 0
for s in sections:
    sec = section_lat.dict_sections[s]
    seq_global.append(sec.lattice.sequence)
    for tws in sec.tws_track:
        tws.s += L
    tws_track_global = np.append(tws_track_global, sec.tws_track)
    L += sec.lattice.totalLen

# postprocessing
S = [tw.s + 3.2 for tw in section_lat.tws]
BetaX = [tw.beta_x for tw in section_lat.tws]
BetaY = [tw.beta_y for tw in section_lat.tws]
AlphaX = [tw.alpha_x for tw in section_lat.tws]
AlphaY = [tw.alpha_y for tw in section_lat.tws]
GammaX = [tw.gamma_x for tw in section_lat.tws]
GammaY = [tw.gamma_y for tw in section_lat.tws]
E = [tw.E for tw in section_lat.tws]


S_tr = np.array([tw.s for tw in tws_track_global])
S_tr = S_tr + s_start
BetaX_tr = [tw.beta_x for tw in tws_track_global]
BetaY_tr = [tw.beta_y for tw in tws_track_global]
AlphaX_tr = [tw.alpha_x for tw in tws_track_global]
AlphaY_tr = [tw.alpha_y for tw in tws_track_global]
EmitX_tr = [tw.emit_x for tw in tws_track_global]
EmitY_tr = [tw.emit_y for tw in tws_track_global]
E_tr = [tw.E for tw in tws_track_global]
sig_tau = np.sqrt([tw.tautau for tw in tws_track_global])
Q = np.sum(p_array.q_array)
current = c * Q / np.sqrt(2 * pi) / sig_tau
Sx = np.zeros(len(S_tr))
Sy = np.zeros(len(S_tr))
for i in range(len(S_tr)):
    gamma = E_tr[i - 1] / m_e_GeV
    EmitX_tr[i - 1] = EmitX_tr[i - 1] * gamma * 1e6
    EmitY_tr[i - 1] = EmitY_tr[i - 1] * gamma * 1e6
    Sx[i - 1] = current[i - 1] * BetaX_tr[i - 1] / (1.7045e04 * gamma * gamma * EmitX_tr[i - 1] * 1e-6)
    Sy[i - 1] = current[i - 1] * BetaY_tr[i - 1] / (1.7045e04 * gamma * gamma * EmitY_tr[i - 1] * 1e-6)

s0 = 0
s1 = 400
plt.figure()
plt.subplot(221)
plt.plot(S, BetaX, label='design')
plt.plot(S_tr, BetaX_tr, label='beam')
plt.legend()
plt.ylabel(r"$\beta_x$[m]")
plt.axis([s0, s1, -1, 100])
plt.subplot(222)
plt.plot(S, BetaY, S_tr, BetaY_tr)
plt.axis([s0, s1, -1, 100])
plt.subplot(223)
plt.plot(S, AlphaX, S_tr, AlphaX_tr)
plt.xlabel("s[m]")
plt.ylabel(r"$\alpha_x$")
plt.axis([s0, s1, -20, 20])
plt.subplot(224)
plt.plot(S, AlphaY, S_tr, AlphaY_tr)
plt.xlabel("s[m]")
plt.axis([s0, s1, -20, 20])

plt.figure()
plt.subplot(211)
plt.plot(S_tr, EmitX_tr, S_tr, EmitY_tr)
plt.grid(True)
plt.title("Emittances")
plt.xlabel("s[m]")
plt.ylabel(r"$\epsilon_x,\epsilon_y [\mu m]$")
plt.axis([s0, s1, 0, 1.7])
plt.subplot(212)
plt.plot(S, E, S_tr, E_tr)
plt.grid(True)
plt.title("Energy")
plt.xlabel("s[m]")
plt.ylabel(r"E [GeV]$")
# plt.axis([s0, s1, 0, 3])

plt.figure()
plt.subplot(211)
plt.plot(S_tr, sig_tau * 1e3)
plt.grid(False)
# plt.title("Sigma_tau");
plt.ylabel(r"$\sigma_z$ [mm]")
plt.axis([s0, s1, 0, 2.1])
plt.subplot(212)
plt.plot(S_tr, Sx, S_tr, Sy)
plt.plot(S_tr, Sx, "b", label=r"$k_x^{sc}$")
plt.plot(S_tr, Sy, "r", label=r"$k_y^{sc}$")
plt.legend()
plt.grid(False)
plt.xlabel("z[m]")
plt.axis([s0, s1, 0, 2])

show_e_beam(p_array, nparts_in_slice=5000, smooth_param=0.01, nfig=13, title="")
n_out = len(S_tr)
out = np.zeros((n_out, 8))
out[:, 0] = S_tr
out[:, 1] = AlphaX_tr
out[:, 2] = BetaX_tr
out[:, 3] = EmitX_tr
out[:, 4] = AlphaY_tr
out[:, 5] = BetaY_tr
out[:, 6] = EmitY_tr
out[:, 7] = E_tr
np.savetxt("Optics.txt", out)

plt.show()
