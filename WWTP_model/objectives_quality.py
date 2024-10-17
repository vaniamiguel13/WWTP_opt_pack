from WWTP_model.parameters import parameters
import numpy as np

def quality_function(x):
    # Extract variables from x
    Q, Qw, Qr, Qef, Qr_p, V_a, A_s, h3, h4, r_p, X_I, X_Ir, X_Ief, S_Sent, S_S, S_Oent, S_NOent, S_NO, X_BHent, X_BH, X_BHr, X_BHef, \
    X_Sent, X_S, X_Sr, X_Sef, X_BAent, X_BA, X_BAr, X_BAef, S_NHent, S_NH, X_Pent, X_P, X_Pr, X_Pef, S_NDent, S_ND, X_NDent, \
    X_ND, X_NDr, X_NDef, G_s, S_alkent, S_alk, SSI, SSIef, SSIr, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9, ST10, v_dn, v_up, \
    v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10, J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, HRT, KLa, r, Sent, S, \
    Xent, X, Xr, Xef, CODent, COD, CODr, CODef, VSSent, VSS, VSSr, VSSef, TSSent, TSS, TSSr, TSSef, BODent, BOD, BODr, BODef, \
    TKNent, TKN, TKNr, TKNef, Nent, N, Nr, Nef, h, S_O = x

    # Constants
    beta_TSS = parameters['beta_TSS']
    beta_COD = parameters['beta_COD']
    beta_BOD = parameters['beta_BOD']
    beta_TKN = parameters['beta_TKN']
    beta_NO = parameters['beta_NO']

    # Objective functions
    f2 = 1 / 1000 * (beta_TSS * TSSef + beta_COD * CODef + beta_BOD * BODef + beta_TKN * TKNef + beta_NO * S_NO) * Qef

    return np.array([f2])
