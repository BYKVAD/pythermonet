# from __future__ import annotations

# import numpy as np

# from ..physics.sources import ils, CSM
# from ..physics.thermal_resistances import Rp, RbMP, RbMPflc, Halley
# from ..system.gfunctions import compute_gfunction_pygfunction, compute_gfunction_infinite_medium
# from ..logging_config import get_logger

# logger = get_logger(__name__)

# SECONDS_IN_HOUR = 3600.0
# SECONDS_IN_MONTH = 24 * (365.25 / 12) * SECONDS_IN_HOUR
# SECONDS_IN_YEAR = 12 * SECONDS_IN_MONTH

# def _three_pulse_times(t_peak_h: float) -> np.ndarray:
#     # [30y+3m+peak, 3m+peak, peak]
#     return np.asarray([
#         30 * SECONDS_IN_YEAR + 3 * SECONDS_IN_MONTH + t_peak_h * SECONDS_IN_HOUR,
#         3 * SECONDS_IN_MONTH + t_peak_h * SECONDS_IN_HOUR,
#         t_peak_h * SECONDS_IN_HOUR
#     ], dtype=float)

# def _damped_amplitude_no_phase(A0: float, z_m: float, alpha_m2s: float) -> float:
#     """
#     Dæmpet amplitude ved dybde z, men uden faseforskydning (konservativ timing).
#     A(z) = A0 * exp(-z/delta), delta = sqrt(2*alpha/omega)
#     """
#     omega = 2 * np.pi / (365.25 * 24 * 3600)  # rad/s
#     delta = np.sqrt(2 * alpha_m2s / omega)    # m
#     return float(A0 * np.exp(-z_m / delta))

# def _dimension_grid_pipes(brine, net, aggLoad, t_H, t_C, Pr):
#     """
#     BEHOLDER oprindelig distributionsnet-model:
#     - g(segments) via CSM/ils + spejlled (Dirichlet ved overflade)
#     - rørmodstand via Rp
#     - beregner FPH/FPC, residual loads PHEH/PHEC, og net.T_dimv (vægtet)
#     """
#     doCooling = not np.isnan(aggLoad.P_s_C).any()
#     N_PG = len(net.L_segments)

#     # Soil diffusivity for shallow pipes
#     a_s = net.l_s_H / net.rhoc_s  # (m2/s) - bruges kun til grid pipe response

#     # Dæmpet amplitude ved rørdybde (uden fase)
#     TP = _damped_amplitude_no_phase(net.A, net.z_grid, a_s)

#     # Pipe thermal resistances
#     R_H = np.zeros(N_PG)
#     for i in range(N_PG):
#         R_H[i] = Rp(net.di_selected_H[i], net.d_selectedPipes_H[i], net.Re_selected_H[i], Pr, brine.l, net.l_p)

#     if doCooling:
#         R_C = np.zeros(N_PG)
#         for i in range(N_PG):
#             R_C[i] = Rp(net.di_selected_C[i], net.d_selectedPipes_C[i], net.Re_selected_C[i], Pr, brine.l, net.l_p)

#     # 3-pulse delta powers
#     dP_s_H = np.zeros(3)
#     dP_s_H[0] = aggLoad.P_s_H[0]
#     dP_s_H[1:] = np.diff(aggLoad.P_s_H)

#     if doCooling:
#         dP_s_C = np.zeros(3)
#         dP_s_C[0] = aggLoad.P_s_C[0]
#         dP_s_C[1:] = np.diff(aggLoad.P_s_C)

#     # Mirror-source kernel terms (vector length=3)
#     K1_H = ils(a_s, t_H, net.D_gridpipes) - ils(a_s, t_H, 2 * net.z_grid) - ils(a_s, t_H, np.sqrt(net.D_gridpipes**2 + 4 * net.z_grid**2))
#     if doCooling:
#         K1_C = ils(a_s, t_C, net.D_gridpipes) - ils(a_s, t_C, 2 * net.z_grid) - ils(a_s, t_C, np.sqrt(net.D_gridpipes**2 + 4 * net.z_grid**2))

#     # Segment-wise grid g and fractions
#     FPH_seg = np.zeros(N_PG)
#     T_tmp = np.zeros((N_PG, 3))

#     for i in range(N_PG):
#         G_grid_H = CSM(net.d_selectedPipes_H[i]/2, net.d_selectedPipes_H[i]/2, t_H, a_s) + K1_H
#         denom = np.dot(dP_s_H, (G_grid_H / net.l_s_H + R_H[i]))
#         FPH_seg[i] = (net.T0 - (aggLoad.Ti_H + aggLoad.To_H)/2 - TP) * net.L_segments[i] / denom

#         # three-pulse fluid temperatures in segment
#         Tseg = net.T0 - TP - FPH_seg[i] * np.cumsum(dP_s_H * (G_grid_H/net.l_s_H + R_H[i]) / net.L_segments[i])
#         # volume-weight segment contribution
#         T_tmp[i, :] = Tseg * (net.L_segments[i] * np.pi * net.di_selected_H[i]**2 / 4)

#     net.T_dimv = np.sum(T_tmp, axis=0) / net.V_brine

#     FPH = float(np.sum(FPH_seg))
#     PHEH = (1 - FPH) * dP_s_H

#     if doCooling:
#         FPC_seg = np.zeros(N_PG)
#         for i in range(N_PG):
#             G_grid_C = CSM(net.d_selectedPipes_C[i]/2, net.d_selectedPipes_C[i]/2, t_C, a_s) + K1_C
#             denom = np.dot(dP_s_C, (G_grid_C / net.l_s_C + R_C[i]))
#             FPC_seg[i] = ((aggLoad.Ti_C + aggLoad.To_C)/2 - net.T0 - TP) * net.L_segments[i] / denom

#         FPC = float(np.sum(FPC_seg))
#         PHEC = (1 - FPC) * dP_s_C
#     else:
#         FPC, PHEC = np.nan, None

#     logger.info("Grid thermal dimensioning complete. FPH=%.3f, doCooling=%s", FPH, doCooling)
#     logger.debug("net.T_dimv=%s, TP=%.3f", net.T_dimv, TP)

#     return net, TP, FPH, PHEH, FPC, PHEC, a_s


# def run_sourcedimensioning(brine, net, aggLoad, source_config):
#     """
#     Termisk dimensionering:
#     - Grid: analytisk (beholdt)
#     - BHE: pygfunction (koord-baseret)
#     - HHE: pygfunction i helrum-approx (D_large) + dæmpet amplitude uden fase
#     """
#     doCooling = not np.isnan(aggLoad.P_s_C).any()

#     t_H = _three_pulse_times(float(aggLoad.t_peak_H))
#     t_C = _three_pulse_times(float(aggLoad.t_peak_C))

#     # Fluid Pr
#     nu_f = brine.mu / brine.rho
#     a_f = brine.l / (brine.rho * brine.c)
#     Pr = nu_f / a_f

#     # 1) Grid (uændret model)
#     net, TP, FPH, PHEH, FPC, PHEC, a_s = _dimension_grid_pipes(brine, net, aggLoad, t_H, t_C, Pr)

#     # 2) Source sizing
#     if source_config.source == "BHE":
#         BHE = source_config

#         # --- hydraulics in BHE pipes (som før) ---
#         ri = BHE.r_p * (1 - 2 / BHE.SDR)
#         s_BHE = 2 * BHE.r_p + BHE.D_pipes

#         # Number of boreholes now comes from coordinates
#         coords = np.asarray(BHE.coordinates, dtype=float)
#         N_BHE = int(coords.shape[0])

#         Q_BHEmax_H = aggLoad.Qdim_H / N_BHE
#         v_BHEmax_H = Q_BHEmax_H / np.pi / ri**2
#         Re_BHEmax_H = brine.rho * v_BHEmax_H * (2 * ri) / brine.mu
#         BHE.Re_BHEmax_H = Re_BHEmax_H

#         if doCooling:
#             Q_BHEmax_C = aggLoad.Qdim_C / N_BHE
#             v_BHEmax_C = Q_BHEmax_C / np.pi / ri**2
#             Re_BHEmax_C = brine.rho * v_BHEmax_C * (2 * ri) / brine.mu
#             BHE.Re_BHEmax_C = Re_BHEmax_C

#         # Soil diffusivity around boreholes
#         alpha_ss = BHE.l_ss / BHE.rhoc_ss

#         # g-functions ONLY with pygfunction (no ICS)
#         # H_m is per-borehole length you iterate on (start guess etc.)
#         # If you already store a design length, use that. Otherwise set an initial guess.
#         # Here we assume BHE.H_m exists as "current guess" or nominal.
#         H_guess = float(getattr(BHE, "H_m", 100.0))
#         D_top = float(getattr(BHE, "D_m", 1.5))
#         r_b = float(BHE.r_b)

#         g_BHE_H = compute_gfunction_pygfunction(
#             coords_xy_m=coords,
#             time_s=t_H,
#             alpha_m2s=alpha_ss,
#             H_m=H_guess,
#             D_m=D_top,
#             r_b_m=r_b,
#             method=getattr(BHE, "pyg_method", "equivalent"),
#             options=getattr(BHE, "pyg_options", {"nSegments": 8}),
#         ).g

#         if doCooling:
#             g_BHE_C = compute_gfunction_pygfunction(
#                 coords_xy_m=coords,
#                 time_s=t_C,
#                 alpha_m2s=alpha_ss,
#                 H_m=H_guess,
#                 D_m=D_top,
#                 r_b_m=r_b,
#                 method=getattr(BHE, "pyg_method", "equivalent"),
#                 options=getattr(BHE, "pyg_options", {"nSegments": 8}),
#             ).g

#         # Borehole resistance (multipole) - beholdt (kan senere flyttes til pygfunction pipe model hvis ønsket)
#         Rb_H = RbMP(brine.l, net.l_p, BHE.l_g, BHE.l_ss, BHE.r_b, BHE.r_p, ri, s_BHE, Re_BHEmax_H, Pr)

#         # Initial length estimate (samme struktur som tidligere, men g fra pygfunction)
#         T0_BHE = net.T0
#         dTdz = BHE.q_geo / BHE.l_ss
#         a = dTdz / (2 * N_BHE)
#         b = T0_BHE - (aggLoad.Ti_H + aggLoad.To_H) / 2
#         c = -np.dot(PHEH, g_BHE_H / (2 * np.pi * BHE.l_ss) + Rb_H)
#         L_BHE_H_total = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

#         # Length iteration with length-dependent Rb + updated pygfunction g (per-borehole length)
#         eps = np.finfo(np.float64).eps
#         tol = 1e-4
#         iter_max = 50

#         dL = (L_BHE_H_total / N_BHE) * np.sqrt(eps)
#         Lcand = (L_BHE_H_total / N_BHE) + np.array([-dL, 0, dL])
#         Rb_cand = np.zeros(3)
#         err = np.ones(3)

#         Tbound_H = (aggLoad.Ti_H + aggLoad.To_H) / 2
#         N_iter = 0

#         while abs(err[1]) > tol and N_iter < iter_max + 1:
#             for i in range(3):
#                 # length-corrected Rb
#                 Rb_cand[i] = RbMPflc(
#                     brine.l, net.l_p, BHE.l_g, BHE.l_ss,
#                     brine.rho, brine.c,
#                     BHE.r_b, BHE.r_p, ri,
#                     Lcand[i], s_BHE,
#                     Q_BHEmax_H, Re_BHEmax_H, Pr
#                 )

#                 # update g with pygfunction for this candidate length
#                 g_i = compute_gfunction_pygfunction(
#                     coords_xy_m=coords,
#                     time_s=t_H,
#                     alpha_m2s=alpha_ss,
#                     H_m=float(Lcand[i]),
#                     D_m=D_top,
#                     r_b_m=r_b,
#                     method=getattr(BHE, "pyg_method", "equivalent"),
#                     options=getattr(BHE, "pyg_options", {"nSegments": 8}),
#                 ).g

#                 err[i] = (
#                     T0_BHE + dTdz * Lcand[i] / 2
#                     - (np.dot(PHEH, g_i / (2*np.pi*BHE.l_ss) + Rb_cand[i])) / (Lcand[i] * N_BHE)
#                     - Tbound_H
#                 )

#             L_new = Halley(Lcand[1], dL, err[0], err[1], err[2])
#             Lcand = L_new + np.array([-dL, 0, dL])
#             N_iter += 1

#         if N_iter > iter_max - 1:
#             logger.warning("BHE heating: convergence failed. Falling back to initial estimate.")
#             # fallback: use initial estimate without length-corrected iteration
#             L_final_per = float(L_BHE_H_total / N_BHE)
#             Rb_final = float(Rb_H)
#         else:
#             L_final_per = float(Lcand[1])
#             Rb_final = float(Rb_cand[1])

#         L_BHE_H_total = L_final_per * N_BHE
#         BHE.Rb_H = Rb_final
#         BHE.L_BHE_H = float(L_BHE_H_total)
#         BHE.FPH = float(FPH)
#         BHE.V_brine = float(2 * L_BHE_H_total * np.pi * ri**2)

#         # final g for reporting and T_dimv
#         g_final = compute_gfunction_pygfunction(
#             coords_xy_m=coords,
#             time_s=t_H,
#             alpha_m2s=alpha_ss,
#             H_m=L_final_per,
#             D_m=D_top,
#             r_b_m=r_b,
#             method=getattr(BHE, "pyg_method", "equivalent"),
#             options=getattr(BHE, "pyg_options", {"nSegments": 8}),
#         ).g

#         BHE.T_dimv = T0_BHE + dTdz * L_final_per / 2 - np.cumsum((PHEH * (g_final/(2*np.pi*BHE.l_ss) + Rb_final)) / L_BHE_H_total)

#         if doCooling:
#             # Cooling branch mirrors heating (same pattern)
#             Rb_C = RbMP(brine.l, net.l_p, BHE.l_g, BHE.l_ss, BHE.r_b, BHE.r_p, ri, s_BHE, BHE.Re_BHEmax_C, Pr)

#             b = -T0_BHE + (aggLoad.Ti_C + aggLoad.To_C) / 2
#             c = -np.dot(PHEC, g_BHE_C / (2 * np.pi * BHE.l_ss) + Rb_C)
#             L_BHE_C_total = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

#             dL = (L_BHE_C_total / N_BHE) * np.sqrt(eps)
#             Lcand = (L_BHE_C_total / N_BHE) + np.array([-dL, 0, dL])
#             Rb_cand = np.zeros(3)
#             err = np.ones(3)

#             Tbound_C = (aggLoad.Ti_C + aggLoad.To_C) / 2
#             N_iter = 0

#             while abs(err[1]) > tol and N_iter < iter_max + 1:
#                 for i in range(3):
#                     Rb_cand[i] = RbMPflc(
#                         brine.l, net.l_p, BHE.l_g, BHE.l_ss,
#                         brine.rho, brine.c,
#                         BHE.r_b, BHE.r_p, ri,
#                         Lcand[i], s_BHE,
#                         Q_BHEmax_C, BHE.Re_BHEmax_C, Pr
#                     )

#                     g_i = compute_gfunction_pygfunction(
#                         coords_xy_m=coords,
#                         time_s=t_C,
#                         alpha_m2s=alpha_ss,
#                         H_m=float(Lcand[i]),
#                         D_m=D_top,
#                         r_b_m=r_b,
#                         method=getattr(BHE, "pyg_method", "equivalent"),
#                         options=getattr(BHE, "pyg_options", {"nSegments": 8}),
#                     ).g

#                     err[i] = (
#                         T0_BHE + dTdz * Lcand[i] / 2
#                         + (np.dot(PHEC, g_i / (2*np.pi*BHE.l_ss) + Rb_cand[i])) / (Lcand[i] * N_BHE)
#                         - Tbound_C
#                     )

#                 L_new = Halley(Lcand[1], dL, err[0], err[1], err[2])
#                 Lcand = L_new + np.array([-dL, 0, dL])
#                 N_iter += 1

#             if N_iter > iter_max - 1:
#                 logger.warning("BHE cooling: convergence failed. Falling back to initial estimate.")
#                 L_final_per = float(L_BHE_C_total / N_BHE)
#                 Rb_final = float(Rb_C)
#             else:
#                 L_final_per = float(Lcand[1])
#                 Rb_final = float(Rb_cand[1])

#             L_BHE_C_total = L_final_per * N_BHE
#             BHE.Rb_C = Rb_final
#             BHE.L_BHE_C = float(L_BHE_C_total)
#             BHE.FPC = float(FPC)

#         logger.info("BHE sizing complete. N=%s, L_BHE_H_total=%.1f m", N_BHE, BHE.L_BHE_H)
#         return BHE


#     if source_config.source == "HHE":
#         HHE = source_config

#         # HHE model via pygfunction in "infinite medium" approximation:
#         # - Use coords of "equivalent elements" (loops or representatives)
#         # - Use D_large to enforce full-space kernel (no half-space BC)
#         coords = np.asarray(HHE.coordinates, dtype=float)
#         N = int(coords.shape[0])

#         alpha_ss = HHE.l_s / HHE.rhoc_s  # soil diffusivity used for seasonal wave and kernel
#         r_b = float(getattr(HHE, "r_b", 0.05))  # “equivalent radius” (engineering parameter)
#         D_large = float(getattr(HHE, "D_large_m", 100.0))

#         # Dæmpet amplitude ved faktisk burial depth (uden fase)
#         z_bury = float(getattr(HHE, "z_bury_m", net.z_grid))
#         TP_hhe = _damped_amplitude_no_phase(net.A, z_bury, alpha_ss)

#         # Effective length parameter for pygfunction kernel
#         H_guess = float(getattr(HHE, "H_m", 200.0))

#         g_H = compute_gfunction_infinite_medium(
#             coords_xy_m=coords,
#             time_s=t_H,
#             alpha_m2s=alpha_ss,
#             H_m=H_guess,
#             r_b_m=r_b,
#             D_large_m=D_large,
#             method=getattr(HHE, "pyg_method", "equivalent"),
#             options=getattr(HHE, "pyg_options", {"nSegments": 8}),
#         ).g

#         # Konservativ temperaturgrænse med TP_hhe (uden fase)
#         Tbound_H = (aggLoad.Ti_H + aggLoad.To_H) / 2

#         # Første længde-estimat (samme form som BHE, men uden geo-gradient)
#         # Her antager vi ingen geo-gradient og ingen borehole resistance; du kan tilføje Rp/Rb analogt hvis du ønsker
#         # L_total = dot(PHEH, G) / (T0 - Tbound - TP)
#         denom = (net.T0 - Tbound_H - TP_hhe)
#         if denom <= 0:
#             raise ValueError("HHE heating: Non-positive temperature driving potential (check T0/Tbound/TP).")

#         L_total_H = float(np.dot(PHEH, g_H / (2*np.pi*HHE.l_s))) / denom  # note: scaling is engineering choice
#         HHE.L_HHE_H = L_total_H
#         HHE.FPH = float(FPH)

#         if doCooling:
#             g_C = compute_gfunction_infinite_medium(
#                 coords_xy_m=coords,
#                 time_s=t_C,
#                 alpha_m2s=alpha_ss,
#                 H_m=H_guess,
#                 r_b_m=r_b,
#                 D_large_m=D_large,
#                 method=getattr(HHE, "pyg_method", "equivalent"),
#                 options=getattr(HHE, "pyg_options", {"nSegments": 8}),
#             ).g

#             Tbound_C = (aggLoad.Ti_C + aggLoad.To_C) / 2
#             denomC = (Tbound_C - net.T0 - TP_hhe)
#             if denomC <= 0:
#                 raise ValueError("HHE cooling: Non-positive temperature driving potential (check T0/Tbound/TP).")

#             L_total_C = float(np.dot(PHEC, g_C / (2*np.pi*HHE.l_s))) / denomC
#             HHE.L_HHE_C = L_total_C
#             HHE.FPC = float(FPC)

#         logger.info("HHE sizing complete (pygfunction full-space approx). N=%s, D_large=%.1f m", N, D_large)
#         return HHE

#     raise ValueError(f"Unknown source type: {source_config.source}")
