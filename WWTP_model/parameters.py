# Define the lower and upper bounds for the variables
LB = [530, 1, 265.0, 265.0, 1, 100, 10, 0.2, 0.2, 0.5, 10, 10, 0.01, 10, 10, 0.001, 0, 0, 0, 1, 1, 0.01, 1, 1, 1, 0.01,
      0, 0, 0.01, 1e-06, 0.1, 0.001, 5, 5, 5, 0.001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 100, 6, 6, 10, 0.1, 10, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
      0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 2]

UB = [2650, 530, 2650, 1060, 108, 10000, 1000, 5, 5, 2, 10000, 10000, 100, 1000, 100, 10, 100, 100, 10000, 10000, 10000,
      1000, 10000, 10000, 10000, 500, 500, 500, 10000, 1000, 100, 100, 10000, 10000, 10000, 100, 10, 50, 1000, 1000,
      1000, 50, 200000, 8, 8, 10000, 50, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,
      10000, 10000, 100000, 100000, 100000, 1000000, 2, 300, 2, 1000, 1000, 10000, 10000, 10000, 1000, 10000, 10000,
      10000, 125, 10000, 10000, 10000, 100, 10000, 10000, 10000, 35, 10000, 10000, 10000, 100, 1000, 1000, 1000, 100,
      1000, 1000, 1000, 15, 5, 10]

# Parameters
parameters = {
    'u_H': 6,
    'Y_H': 0.666,
    'K_S': 20,
    'K_OH': 0.2,
    'eta_g': 0.8,
    'K_NO': 0.5,
    'k_h': 3,
    'K_X': 0.03,
    'eta_h': 0.4,
    'f_p': 0.08,
    'b_H': 0.62,
    'b_A': 0.04,
    'u_A': 0.8,
    'K_NH': 1,
    'K_OA': 0.4,
    'Y_A': 0.24,
    'i_XB': 0.086,
    'k_a': 0.08,
    'i_XP': 0.06,
    'T': 20,
    'P_O2': 0.21,
    'dens': 999.96 * (2.29e-2 * 20) - (5.44e-3 * 20**2),
    'beta': 0.95,
    'Henry': (708 * 20) + 25700,
    'SOST': 1777.8 * 0.95 * 999.96 * 0.21 / ((708 * 20) + 25700),
    'fracO2': 0.21,
    'alfa': 0.8,
    'eta': 0.07,
    'icv': 1.48,
    'f_BOD': 0.66,
    'IVLD': 150,
    'TSSr_max': 1200 * 1000 / 150,
    'TSSr_max_p': (1200 / 150 + 2) * 1000,
    'SRT': 20,
    'S_O': 2,
    'beta_TSS': 2,
    'beta_COD': 1,
    'beta_BOD': 2,
    'beta_TKN': 20,
    'beta_NO': 20,
    'Qinf': 530,
    'Q_P': 54,
    'S_alkinf': 7,
    'X_Iinf': 90,
    'X_IIinf': 18.3,
    'X_Sinf': 168.75,
    'S_Sinf': 112.5,
    'S_I': 12.5,
    'S_Oinf': 0,
    'S_NOinf': 0,
    'S_NHinf': 11.7,
    'S_NDinf': 0.63,
    'X_NDinf': 1.251,
    'X_BHinf': 0,
    'X_BAinf': 0,
    'X_Pinf': 0,
    'TKNinf': 11.7 + 0.63 + 1.251 + 0.086 * (0 + 0) + 0.06 * (0 + 90),
    'Ninf': 11.7 + 0.63 + 1.251 + 0.086 * (0 + 0) + 0.06 * (0 + 90) + 0,
    'Xinf': 90 + 168.75 + 0 + 0 + 0,
    'Sinf': 12.5 + 112.5,
    'CODinf': 90 + 168.75 + 0 + 0 + 0 + 12.5 + 112.5,
    'VSSinf': (90 + 168.75 + 0 + 0 + 0) / 1.48,
    'TSSinf': ((90 + 168.75 + 0 + 0 + 0) / 1.48) + 18.3,
    'BODinf': 0.66 * (112.5 + 168.75 + 0 + 0),
    'v1_0': 274,
    'v_0': 410,
    'r_h': 0.0004,
    'f_ns': 0.001,
    'r_P': 0.0025,
    'ST_t': 2000,
    'COD_law': 125,
    'TSS_law': 35,
    'N_law': 15,
}
