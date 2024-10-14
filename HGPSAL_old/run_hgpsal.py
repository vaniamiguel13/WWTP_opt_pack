import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective_function(x: np.ndarray) -> float:
    x = np.asarray(x)  # Convert input to a NumPy array if it's not already
    return float(np.sum(x ** 2))

# Define the constraints function
def constraints(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)  # Convert input to a NumPy array if it's not already
    return np.array([np.sum(x ** 2) - 4]), np.array([np.sum(x) - 1])

# Define the bounds for each variable
bounds = [(-5, 5), (-5, 5)]  # Bounds for x[0] and x[1]

# Initial guess
x0 = [0.8, 0.2]  # Starting point

# Define the constraints in the format required by scipy
constr = [
    {'type': 'ineq', 'fun': lambda x: 4 - np.sum(np.asarray(x) ** 2)},  # Inequality constraint
    {'type': 'eq', 'fun': lambda x: np.sum(np.asarray(x)) - 1}         # Equality constraint
]

# Call the minimize function
result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constr)

# Print the result
if result.success:
    print("\nConstrained test:")
    print(f"Solution: {result.x}")
    print(f"Objective value: {result.fun}")
    print(f"Inequality constraints: {4 - np.sum(result.x ** 2)}")  # Should be <= 0
    print(f"Equality constraints: {np.sum(result.x) - 1}")  # Should be = 0
    print(f"Number of function evaluations: {result.nfev}")  # Number of function evaluations
else:
    print("Optimization failed:", result.message)

# Assertions for testing purposes
c, ceq = constraints(result.x)
assert c[0] <= 0.0, f"Inequality constraint violation: {c[0]}"
assert np.isclose(ceq[0], 0.0, atol=1e-5), f"Equality constraint violation: {ceq[0]}"

# Expected objective value calculation
expected_fx = objective_function([0.8, 0.2])
assert np.isclose(result.fun, expected_fx, atol=1e-5)
