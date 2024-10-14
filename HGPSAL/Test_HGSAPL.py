import unittest
import numpy as np
from HGPSAL.hgpsal import HGPSAL
from typing import List, Tuple, Any

class TestHGPSAL(unittest.TestCase):

    def test_unconstrained(self):
        def objective_function(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        def constraints(x: np.ndarray) -> Tuple[List[float], List[float]]:
            return [], []

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': np.array([-5, -5]),
            'UB': np.array([5, 5]),
            'Constraints': constraints,
            'x0': np.array([0.5, 0.5])
        }

        options = {
            'pop_size': 200,
            'max_objfun': 150000,
            'verbose': 0,
            'pmut': 0.1,
            'epsilon1': 1e-9,
            'epsilon2': 1e-9,
            'maxit': 1500,
            'maxet': 1500
        }

        num_starts = 7
        best_x = None
        best_fx = float('inf')

        for _ in range(num_starts):
            x, fx, c, ceq, la, stats = HGPSAL(problem, options)

            if fx < best_fx:
                best_x = x
                best_fx = fx

        print(f"Unconstrained test results:")
        print(f"Best HGPSAL solution: {best_x}, fx: {best_fx}")
        print(f"Number of function evaluations: {stats['objfun']}")

        self.assertLess(best_fx, 1e-6, f"Objective value {best_fx} not less than 1e-6")
        np.testing.assert_array_almost_equal(best_x, np.array([0, 0]), decimal=5)

    def test_bounded(self):
        def objective_function(x: np.ndarray) -> float:
            return float(np.sum((x - np.array([2, 1])) ** 2))

        def constraints(x: np.ndarray) -> Tuple[List[float], List[float]]:
            return [], []

        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([5.0, 5.0])

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': lower_bounds,
            'UB': upper_bounds,
            'Constraints': constraints,
            'x0': np.array([0.5, 0.5])
        }

        options = {
            'pop_size': 150,
            'max_objfun': 75000,
            'verbose': 1,
            'pmutation': 0.1,
            'epsilon1': 1e-9,
            'epsilon2': 1e-9,
            'maxit': 1000,  # Added
            'maxet': 1000   # Added
        }

        x, fx, c, ceq, la, stats = HGPSAL(problem, options)

        print(f"\nBounded test results:")
        print(f"Solution: {x}")
        print(f"Objective value: {fx}")

        self.assertLess(fx, 1e-3, f"Objective value {fx} not less than 1e-3")
        np.testing.assert_array_almost_equal(x, np.array([2, 1]), decimal=2,
                                             err_msg=f"Solution {x} not close to [2, 1] with decimal=2")

        self.assertTrue(np.all(x >= problem['LB']) and np.all(x <= problem['UB']),
                        f"Solution {x} is outside the bounds [{problem['LB']}, {problem['UB']}]")

        print(f"Number of function evaluations: {stats['objfun']}")
        print(f"Number of iterations: {stats['extit']}")

    def test_constrained(self):
        def objective_function(x: np.ndarray) -> float:
            return float(np.sum(x ** 2))

        def constraints(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.array([np.sum(x ** 2) - 4]), np.array([np.sum(x) - 1])

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': [-5, -5],
            'UB': [5, 5],
            'Constraints': constraints,
            'x0': [0.8, 0.2]
        }

        options = {
            'pop_size': 150,
            'max_objfun': 750000,
            'verbose': 0,
            'pmut': 0.1,
            'epsilon1': 1e-9,
            'epsilon2': 1e-9,
            'maxit': 750,
            'maxet': 750
        }

        x, fx, c, ceq, la, stats = HGPSAL(problem, options)

        print(f"Solution: {x}")
        print(f"Objective value: {fx}")
        print(f"Inequality constraints: {c}")
        print(f"Equality constraints: {ceq}")
        print(f"Number of function evaluations: {stats['objfun']}")

        self.assertLessEqual(c[0], 1e-6, f"Inequality constraint violation: {c[0]}")
        self.assertAlmostEqual(ceq[0], 0.0, delta=1e-6, msg=f"Equality constraint violation: {ceq[0]}")

        expected_solution = np.array([0.5, 0.5])
        expected_fx = np.sum(expected_solution ** 2)
        self.assertAlmostEqual(fx, expected_fx, delta=1e-6,
                               msg=f"Objective value {fx} not close to expected {expected_fx}")

        expected_c = np.sum(expected_solution ** 2) - 4
        self.assertAlmostEqual(c[0], expected_c, delta=1e-6,
                               msg=f"Inequality constraint value {c[0]} not close to expected {expected_c}")

    def test_Rast(self):
        def objective_function(x: np.ndarray) -> float:
            return float(20 + np.sum(x ** 2) - 10 * np.sum(np.cos(2 * np.pi * x)))

        def constraints(x: np.ndarray) -> Tuple[List[float], List[float]]:
            return [-x[0] - 5, x[0] - 5, -x[1] - 5, x[1] - 5], []

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': np.array([-5.12, -5.12]),
            'UB': np.array([5.12, 5.12]),
            'Constraints': constraints,
            'x0': np.array([0, 0])
        }

        options = {
            'pop_size': 150,
            'max_objfun': 75000,
            'verbose': 0,
            'pmut': 0.2,
            'maxit': 1000,  # Added
            'maxet': 1000   # Added
        }

        num_starts = 7
        best_x = None
        best_fx = float('inf')

        for i in range(num_starts):
            x, fx, c, ceq, la, stats = HGPSAL(problem, options)

            if fx < best_fx:
                best_x = x
                best_fx = fx
                best_stats = stats

            print(f"Run {i + 1}: x = {x}, fx = {fx}")

        print(f"\nRastrigin test results:")
        print(f"Best solution: {best_x}")
        print(f"Best objective value: {best_fx}")

        x_known = np.array([0., 0.])
        fx_known = 0.0

        self.assertTrue(np.allclose(best_x, x_known, atol=5e-2),
                        f"Solution {best_x} is not close to the known global minimum {x_known}")

        self.assertLess(best_fx, 0.5,
                        f"Objective value {best_fx} is not close enough to the known minimum {fx_known}")

        self.assertTrue(np.all(best_x >= problem['LB']) and np.all(best_x <= problem['UB']),
                        "Solution is outside the bounds")

        print(f"Number of function evaluations: {best_stats['objfun']}")
        print(f"Number of iterations: {best_stats['extit']}")

        c, ceq = problem['Constraints'](best_x)
        self.assertTrue(all(ci <= 1e-6 for ci in c), "Inequality constraints are not satisfied")
        self.assertTrue(all(abs(cei) <= 1e-6 for cei in ceq), "Equality constraints are not satisfied")

    def test_Rosenbrock(self):
        def objective_function(x: np.ndarray) -> float:
            return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

        def constraints(x: np.ndarray) -> Tuple[List[float], List[float]]:
            return [], []

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': np.array([-2, -2]),
            'UB': np.array([2, 2]),
            'Constraints': constraints,
            'x0': np.array([0, 0])
        }

        options = {
            'pop_size': 250,
            'max_objfun': 1500000,
            'verbose': 0,
            'pmut': 0.2,
            'maxit': 750,
            'maxet': 750
        }

        num_starts = 7
        best_x = None
        best_fx = float('inf')

        for _ in range(num_starts):
            x, fx, c, ceq, la, stats = HGPSAL(problem, options)

            if fx < best_fx:
                best_x = x
                best_fx = fx

        print(f"Original Rosenbrock test results:")
        print(f"Best solution: {best_x}")
        print(f"Best objective value: {best_fx}")

        x_known = np.array([1.0, 1.0])
        fx_known = 0.0

        self.assertTrue(np.allclose(best_x, x_known, atol=1e-2),
                        f"Solution {best_x} is not close to the known global minimum {x_known}")

        self.assertLess(best_fx, 1e-3,
                        f"Objective value {best_fx} is not close enough to the known minimum {fx_known}")

        self.assertTrue(np.all(best_x >= problem['LB']) and np.all(best_x <= problem['UB']),
                        "Solution is outside the bounds")

        print(f"Number of function evaluations: {stats['objfun']}")
        print(f"Number of iterations: {stats['extit']}")

    def test_himmelblau(self):
        def objective_function(x: np.ndarray) -> float:
            return float((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)

        def constraints(x: np.ndarray) -> Tuple[List[float], List[float]]:
            return [], []

        problem = {
            'ObjFunction': objective_function,
            'Variables': 2,
            'LB': np.array([-5, -5]),
            'UB': np.array([5, 5]),
            'Constraints': constraints,
            'x0': np.array([0, 0])
        }

        options = {
            'pop_size': 150,
            'max_objfun': 750000,
            'verbose': 0,
            'pmut': 0.1,
            'epsilon1': 1e-9,
            'epsilon2': 1e-9,
            'maxit': 1000,  # Added
            'maxet': 1000   # Added
        }

        x, fx, c, ceq, la, stats = HGPSAL(problem, options)

        self.assertLess(fx, 1e-2)  # More stringent condition

        minima = [
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126])
        ]

        self.assertTrue(any(np.allclose(x, m, atol=5e-2) for m in minima),
                        f"Solution {x} is not close to any known minimum")

if __name__ == '__main__':
    unittest.main()