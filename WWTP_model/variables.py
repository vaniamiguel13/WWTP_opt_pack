import numpy as np

# Initial values for the variables
initial_values = np.array(
    [4000, 100, 2000, 1900, 100, 1000, 1000, 2, 0.5, 0.5, 727.3, 950.571, 1e-05, 50, 10, 1, 1e-06,
     1e-06, 0.2, 350, 711.2, 1e-05, 3000, 350, 806.714, 1e-05, 1e-05, 1e-06, 1.9e-6, 1e-05, 10,
     7.5, 1500, 90, 174.52, 1e-05, 0.5, 0.5, 200, 20, 20, 0.5, 10000, 7, 7, 1500, 0.1, 3500, 10,
     20, 40, 70, 200, 300, 350, 350, 2000, 4000, 21, 19, 7.941, 11.77, 15.92, 22.09, 34.25, 67.15,
     179.1, 179.1, 179.1, 116.6, 0.08737, 0.1836, 0.3291, 0.6257, 1.505, 6.057, 59.05, 59.05,
     59.06, 0, 3.5, 100, 1, 80, 50, 5000, 1000, 4440, 1e-04, 5500, 1600, 5500, 80, 3500, 1050,
     3500, 1e-06, 5000, 1800, 5000, 10, 2000, 500, 2000, 30, 350, 106, 350, 10, 350, 106, 350,
     15, 3.5, 2])

# Define variables as a dictionary
variables = {
    'Q': initial_values[0],
    'Qw': initial_values[1],
    'Qr': initial_values[2],
    'Qef': initial_values[3],
    'Qr_p': initial_values[4],
    'V_a': initial_values[5],
    'A_s': initial_values[6],
    'h3': initial_values[7],
    'h4': initial_values[8],
    'r_p': initial_values[9],
    'X_I': initial_values[10],
    'X_Ir': initial_values[11],
    'X_Ief': initial_values[12],
    'S_Sent': initial_values[13],
    'S_S': initial_values[14],
    'S_Oent': initial_values[15],
    'S_NOent': initial_values[16],
    'S_NO': initial_values[17],
    'X_BHent': initial_values[18],
    'X_BH': initial_values[19],
    'X_BHr': initial_values[20],
    'X_BHef': initial_values[21],
    'X_Sent': initial_values[22],
    'X_S': initial_values[23],
    'X_Sr': initial_values[24],
    'X_Sef': initial_values[25],
    'X_BAent': initial_values[26],
    'X_BA': initial_values[27],
    'X_BAr': initial_values[28],
    'X_BAef': initial_values[29],
    'S_NHent': initial_values[30],
    'S_NH': initial_values[31],
    'X_Pent': initial_values[32],
    'X_P': initial_values[33],
    'X_Pr': initial_values[34],
    'X_Pef': initial_values[35],
    'S_NDent': initial_values[36],
    'S_ND': initial_values[37],
    'X_NDent': initial_values[38],
    'X_ND': initial_values[39],
    'X_NDr': initial_values[40],
    'X_NDef': initial_values[41],
    'G_s': initial_values[42],
    'SSI': initial_values[43],
    'SSIef': initial_values[44],
    'SSIr': initial_values[45],
    'HRT': initial_values[46],
    'r': initial_values[47],
    'ST1': initial_values[48],
    'ST2': initial_values[49],
    'ST3': initial_values[50],
    'ST4': initial_values[51],
    'ST5': initial_values[52],
    'ST6': initial_values[53],
    'ST7': initial_values[54],
    'ST8': initial_values[55],
    'ST9': initial_values[56],
    'ST10': initial_values[57],
    'v_dn': initial_values[58],
    'v_up': initial_values[59],
    'v_s1': initial_values[60],
    'v_s2': initial_values[61],
    'v_s3': initial_values[62],
    'v_s4': initial_values[63],
    'v_s5': initial_values[64],
    'v_s6': initial_values[65],
    'v_s7': initial_values[66],
    'v_s8': initial_values[67],
    'v_s9': initial_values[68],
    'v_s10': initial_values[69],
    'J1': initial_values[70],
    'J2': initial_values[71],
    'J3': initial_values[72],
    'J4': initial_values[73],
    'J5': initial_values[74],
    'J6': initial_values[75],
    'J7': initial_values[76],
    'J8': initial_values[77],
    'J9': initial_values[78],
    'J10': initial_values[79],
    'S_alkent': initial_values[80],
    'S_alk': initial_values[81],
    'KLa': initial_values[82],
    'Sent': initial_values[83],
    'S': initial_values[84],
    'Xent': initial_values[85],
    'X': initial_values[86],
    'Xr': initial_values[87],
    'Xef': initial_values[88],
    'CODent': initial_values[89],
    'COD': initial_values[90],
    'CODr': initial_values[91],
    'CODef': initial_values[92],
    'VSSent': initial_values[93],
    'VSS': initial_values[94],
    'VSSr': initial_values[95],
    'VSSef': initial_values[96],
    'TSSent': initial_values[97],
    'TSS': initial_values[98],
    'TSSr': initial_values[99],
    'TSSef': initial_values[100],
    'BODent': initial_values[101],
    'BOD': initial_values[102],
    'BODr': initial_values[103],
    'BODef': initial_values[104],
    'TKNent': initial_values[105],
    'TKN': initial_values[106],
    'TKNr': initial_values[107],
    'TKNef': initial_values[108],
    'Nent': initial_values[109],
    'N': initial_values[110],
    'Nr': initial_values[111],
    'Nef': initial_values[112],
    'h': initial_values[113],
    'S_O': initial_values[114]
}

