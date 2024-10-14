import pandas as pd
# Load the CSV file
file_path = 'x_output.csv'
data = pd.read_csv(file_path)

# Define the new headers
new_headers = [
    "Q", "Qw", "Qr", "Qef", "Qr_p", "V_a", "A_s", "h3", "h4", "r_p", "X_I", "X_Ir", "X_Ief", "S_Sent", "S_S", "S_Oent",
    "S_NOent", "S_NO", "X_BHent", "X_BH", "X_BHr", "X_BHef", "X_Sent", "X_S", "X_Sr", "X_Sef", "X_BAent", "X_BA",
    "X_BAr", "X_BAef", "S_NHent", "S_NH", "X_Pent", "X_P", "X_Pr", "X_Pef", "S_NDent", "S_ND", "X_NDent", "X_ND",
    "X_NDr", "X_NDef", "G_s", "S_alkent", "S_alk", "SSI", "SSIef", "SSIr", "ST1", "ST2", "ST3", "ST4", "ST5", "ST6",
    "ST7", "ST8", "ST9", "ST10", "v_dn", "v_up", "v_s1", "v_s2", "v_s3", "v_s4", "v_s5", "v_s6", "v_s7", "v_s8",
    "v_s9", "v_s10", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "HRT", "KLa", "r", "Sent", "S",
    "Xent", "X", "Xr", "Xef", "CODent", "COD", "CODr", "CODef", "VSSent", "VSS", "VSSr", "VSSef", "TSSent", "TSS",
    "TSSr", "TSSef", "BODent", "BOD", "BODr", "BODef", "TKNent", "TKN", "TKNr", "TKNef", "Nent", "N", "Nr", "Nef",
    "h", "S_O"
]

# Apply the new headers to the dataframe
data.columns = new_headers

# Save the updated dataframe to a new CSV
output_file_path = '../results/x_output_updated_500000.csv'
data.to_csv(output_file_path, index=False)