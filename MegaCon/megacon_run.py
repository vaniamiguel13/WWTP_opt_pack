import numpy as np
from megacon import MEGAcon

# Define the objective functions
def multi_objective(x):
    f1 = x[0]**2 + x[1]**2
    f2 = (x[0] - 1)**2 + x[1]**2
    return np.array([f1, f2])
# Define the constraints
def constraints(x):
    c = [x[0]**2 + x[1]**2 - 1] # Inequality: x0^2 + x1^2 <= 1
    ceq = [x[0] + x[1] - 1] # Equality: x0 + x1 = 1
    return np.array(c), np.array(ceq)
# Problem definition
Problem = {
'ObjFunction': multi_objective,
'Variables': 2,
'Objectives': 2,
'LB': [-2, -2],
'UB': [5, 5],
'Constraints': constraints
}

# Options definition
Options = {
'PopSize': 50,
'Verbosity': 1,
'CTol': 1e-4,
'CeqTol': 1e-2,
'MaxObj': 1000,
'MaxGen': 5000
}
# Initial guess to include in the initial population
initial_population = [{'x': np.array([2, 3])}]

# Running MEGAcon
x, fx, S = MEGAcon(Problem, initial_population, Options)
print("Solutions:", x)
print("Objective Values:", fx['f'])
print("Inequality Constraints:", fx['c'])
print("Equality Constraints:", fx['ceq'])