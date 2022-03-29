from math import pi

import matplotlib.pyplot as plt
import numpy as np
from ocelot.common.globals import m_e_GeV
from ocelot.gui.accelerator import show_e_beam


def make_all_the_igor_plots(tws, tws_track_global, p_array, s_start):

    c = 299792458
    # postprocessing
    S = [tw.s + 3.2 for tw in tws]
    BetaX = [tw.beta_x for tw in tws]
    BetaY = [tw.beta_y for tw in tws]
    AlphaX = [tw.alpha_x for tw in tws]
    AlphaY = [tw.alpha_y for tw in tws]
    # GammaX = [tw.gamma_x for tw in tws]
    # GammaY = [tw.gamma_y for tw in tws]
    E = [tw.E for tw in tws]

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

    # pass
