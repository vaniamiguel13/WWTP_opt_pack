import pandas as pd

# Load the CSV file
file_path = 'ceq_output.csv'
data = pd.read_csv(file_path)


# Function to find the top N maximum absolute values and their columns, excluding duplicates from the same column
def find_top_n_max_abs_values(df, n):
    result = []
    temp_df = df.copy()

    for _ in range(n):
        max_val = temp_df.abs().max().max()
        max_col = temp_df.abs().max().idxmax()
        result.append((max_val, max_col))

        # Drop the column with the current max absolute value to exclude it from the next iterations
        temp_df = temp_df.drop(columns=[max_col])

    return result


# Get the top 3 maximum absolute values and their columns
top_3_values = find_top_n_max_abs_values(data, 30)

# Print the results
for i, (value, column) in enumerate(top_3_values, start=1):
    print(f"{i}th largest absolute value: {value} in column: {column}")
