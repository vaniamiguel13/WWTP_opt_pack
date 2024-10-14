import pandas as pd

# Load the CSV files
ceq_data = pd.read_csv('ceq_output.csv')
c_data = pd.read_csv('c_output.csv')

# List of indexes to delete
indexes_to_delete = [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]

# Delete rows with the remembered indexes
cleaned_ceq_data = ceq_data.drop(index=indexes_to_delete, errors='ignore')
cleaned_c_data = c_data.drop(index=indexes_to_delete, errors='ignore')

# Save the cleaned data back to CSV files
cleaned_ceq_data.to_csv('cleaned_ceq_output.csv', index=False)
cleaned_c_data.to_csv('cleaned_c_output.csv', index=False)

print("Rows deleted and cleaned files saved successfully.")
