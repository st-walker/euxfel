# reference case of s2e for 250 pC and SASE1/SASE3 branch
# created by Igor Zagorodnov on 10 January 2020

# In order to use this script do following:
# 1) create directory data_dir+"/tws";
# 2) create directory data_dir+"/particles" and put there file "gun.npz";

import os
from copy import deepcopy
from math import pi
from typing import List

import numpy as np
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.io import load_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.gui.accelerator import plot_opt_func_reduced
from ocelot.utils.section_track import SectionLattice

from euxfel.plot import make_all_the_igor_plots
from euxfel.sections import (
    A1,
    AH1,
    BC0,
    BC1,
    BC2,
    CL1,
    CL2,
    CL3,
    CL3LUXE,
    DL,
    L1,
    L2,
    L3,
    LH,
    SASE1,
    SASE2,
    STN10,
    T1,
    T3,
    T4,
    T20LUXE,
)

IGOR_PLOTS = False

print(os.getcwd())
data_dir = "./"

all_sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3LUXE, T20LUXE]
# all_sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3, STN10, SASE1, T4, SASE3, T4D]
all_sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3LUXE, T20LUXE]
sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, L3, CL1, CL2, CL3LUXE, T20LUXE]
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

if IGOR_PLOTS:
    make_all_the_igor_plots(section_lat.tws, tws_track_global, p_array, s_start)

if T20LUXE in all_sections:
    t20 = section_lat.dict_sections[T20LUXE]
    parray_ip = t20.ip_dump.parray
