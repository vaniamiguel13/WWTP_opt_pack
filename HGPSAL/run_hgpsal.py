import numpy as np
from hgpsal import HGPSAL
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from MegaCon.objectives_cost import cost_function
from MegaCon.objectives_quality import quality_function
from MegaCon.constraints2 import constraints
from MegaCon.parameters import LB, UB
from MegaCon.variables import initial_values
import pandas as pd

# # Problem definition
# Problem = {
#     'ObjFunction': cost_function,
#     'Variables': len(LB),
#     'Objectives': 2,
#     'LB': LB,
#     'UB': UB,
#     'Constraints': constraints
# }

# # Options definition
# Options = {
#     'PopSize': 150,
#     'Verbosity': 2,
#     'CTol': 1e-4,
#     'CeqTol': 1e-2,
#     'MaxObj': 1000000,
#     "MaxGen": 1000000
# }


# Clear all equivalent in Python
Problem = {}
Options = {}
LB = np.array(LB)
UB = np.array(UB)

# Example with constraints
Problem['nvar'] = 115
Problem['Variables'] = len(LB)
Problem['ObjFunction'] = quality_function
Problem['LB'] = LB
Problem['UB'] = UB
Problem['Constraints'] = constraints
Problem['x0'] = np.array(initial_values)
Options['verbose'] = 1
Options["Verbosity"] = 1
Options['max_objfun'] = 5000
Options["MaxGen"] = 5000
Options["pop_size"] = 150
Problem["x"] = np.array(initial_values)
Options["maxit"] = 50000
Options["maxet"] = 50000
# Assuming HGPSAL_old is defined elsewhere, the equivalent Python call would be:
x, fx, c, ceq, la, stats = HGPSAL(Problem, Options)


# Convert the array to a DataFrame

# Save the DataFrame to a CSV file without the index
df_c = pd.DataFrame(c,columns=["c"])
df_c.to_csv("c_output_qi3.csv",index= False)

# Determine column names, e.g., Column1, Column2, ..., Column99
try:
    column_names = [f"Column{i+1}" for i in range(ceq.shape[1])]
except:
    column_names = ["ceq"]

# Convert the array to a DataFrame with these column names
df_ceq = pd.DataFrame(ceq, columns=column_names)

# Save the DataFrame to a CSV file without the index
df_ceq.to_csv('ceq_output_qi3.csv', index=False)
# Determine column names, e.g., Column1, Column2, ..., Column99
try:
    column_names = [f"Column{i+1}" for i in range(x.shape[1])]
except:
    column_names = ["x"]

# Convert the array to a DataFrame with these column names
df_x = pd.DataFrame(x, columns=column_names)
df_x.to_csv("x_qi3.csv",index= False)

df_fx = pd.DataFrame(fx, columns=column_names)
df_fx.to_csv("fx_qi3.csv",index= False)

print("value of x\n",x)
print("value of fx\n",fx)
print("value of c\n",c)
print("value of la\n",la)
print("value of ceq\n",ceq)
print("shape of f\n",fx.shape)
print("shape of c\n",c.shape)
print("shape of ceq\n",ceq.shape)
print("stats\n",stats)

