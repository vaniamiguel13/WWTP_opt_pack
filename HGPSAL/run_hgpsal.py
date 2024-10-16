import numpy as np
from hgpsal import HGPSAL
# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2
# Define the constraints
def constraints(x):
    c = [x[0]**2 + x[1]**2 - 1] # Inequality: x0^2 + x1^2 <= 1
    ceq = [x[0] + x[1] - 1] # Equality: x0 + x1 = 1
    return np.array(c), np.array(ceq)

# Problem definition
Problem = {
'ObjFunction': objective_function,
'Variables': 2,
'LB': [-2, -2],
'UB': [2, 2],
'Constraints': constraints,
'x0': [2, 1]
}
# Options definition
Options = {
'pop_size': 50,
'max_objfun': 10000,
'verbose': 0
}
# Running HGPSAL
x, fx, c, ceq, la, stats = HGPSAL(Problem, Options)
print("Solution:", x)
print("Objective Value:", fx)
print("Inequality Constraints:", c)
print("Equality Constraints:", ceq)
print("Lagrangian Value:", la)