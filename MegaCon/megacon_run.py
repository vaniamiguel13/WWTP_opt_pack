import numpy as np
import matplotlib.pyplot as plt
from megacon_b import MEGAcon  # Adjust the import according to your file structure
from objectives import cost_function
from constraints2 import constraints
from parameters import LB, UB
from variables import initial_values
import pandas as pd

# Problem definition
Problem = {
    'ObjFunction': cost_function,
    'Variables': len(LB),
    'Objectives': 2,
    'LB': LB,
    'UB': UB,
    'Constraints': constraints
}

# Options definition
Options = {
    'PopSize': 150,
    'Verbosity': 2,
    'CTol': 1e-4,
    'CeqTol': 1e2,
    'MaxObj': 500000,
    "MaxGen": 500000
}

# Initial guess to include in the initial population
# InitPop = [{'x': np.array([2, 2])}]
initial_population = [{'x': np.array(initial_values)}]

# Running MEGAcon
x, fx, S = MEGAcon(Problem, initial_population, Options)

# Convert the array to a DataFrame
df_f = pd.DataFrame(fx["f"], columns=['f1(x)', 'f2(x)'])

# Save the DataFrame to a CSV file without the index
df_f.to_csv('f_output_500000.csv', index=False)
df_c = pd.DataFrame(fx["c"], columns=["c"])
df_c.to_csv("c_output_500000.csv", index=False)

# Determine column names, e.g., Column1, Column2, ..., Column99
column_names = [f"Column{i+1}" for i in range(fx["ceq"].shape[1])]

# Convert the array to a DataFrame with these column names
df_ceq = pd.DataFrame(fx["ceq"], columns=column_names)

# Save the DataFrame to a CSV file without the index
df_ceq.to_csv('ceq_output_500000.csv', index=False)

# Determine column names, e.g., Column1, Column2, ..., Column99
column_names_x = [f"Column{i+1}" for i in range(x.shape[1])]

# Convert the array to a DataFrame with these column names
df_x = pd.DataFrame(x, columns=column_names_x)

# Save the DataFrame to a CSV file without the index
df_x.to_csv('x_output_500000.csv', index=False)

print("value of x\n", x)
print("value of f\n", fx["f"])
print("value of c\n", fx["c"])
print("value of ceq\n", fx["ceq"])
print("value of S\n", S)

# Plotting the results
plt.scatter(fx['f'][:, 0], fx['f'][:, 1], marker='x')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
# plt.title('Pareto Front')
plt.show()
