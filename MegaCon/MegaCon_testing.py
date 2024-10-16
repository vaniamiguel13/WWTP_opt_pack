import unittest
import numpy as np
from MegaCon.megacon import MEGAcon

class TestMega(unittest.TestCase):

    def test_zdt1(self):
        def zdt1_objective(x):
            f1 = x[0]
            g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
            h = 1 - np.sqrt(f1 / g)
            f2 = g * h
            return np.array([f1, f2])
        
        def zdt1_constraints(x):
            return [], []

        problem = {
            'ObjFunction': zdt1_objective,
            'Variables': 30,
            'Objectives': 2,
            'LB': [0] * 30,
            'UB': [1] * 30,
            'Constraints': zdt1_constraints,
            'x0': np.random.rand(30)
        }

        options = {
            'PopSize': 100,
            'MaxGen': 250,
            'CTol': 1e-4,
            'CeqTol': 1e-2,
            'MaxObj': 10000
        }

        initial_population = [{'x': np.array(problem['x0'])}]

        x, fx, S = MEGAcon(problem, initial_population, options)

        self.assertTrue(len(fx['f']) > 0)
        self.assertTrue(np.all(fx['f'][:, 0] >= 0))
        self.assertTrue(np.all(fx['f'][:, 0] <= 1))
        self.assertTrue(np.all(fx['f'][:, 1] >= 0))
        self.assertTrue(np.all(fx['f'][:, 1] <= 10))

    def test_dtlz1(self):
        def dtlz1_objective(x):
            k = len(x) - 5
            g = 100 * (k + sum((x[5:] - 0.5)**2 - np.cos(20 * np.pi * (x[5:] - 0.5))))
            f1 = 0.5 * x[0] * x[1] * (1 + g)
            f2 = 0.5 * x[0] * (1 - x[1]) * (1 + g)
            f3 = 0.5 * (1 - x[0]) * (1 + g)
            return np.array([f1, f2, f3])

        def dtlz1_constraints(x):
            return [], []

        problem = {
            'ObjFunction': dtlz1_objective,
            'Variables': 7,
            'Objectives': 3,
            'LB': [0] * 7,
            'UB': [1] * 7,
            'Constraints': dtlz1_constraints,
            'x0': np.random.rand(7)
        }

        options = {
            'PopSize': 100,
            'MaxGen': 250,
            'CTol': 1e-4,
            'CeqTol': 1e-2,
            'MaxObj': 100000
        }

        initial_population = [{'x': np.array(problem['x0'])}]

        x, fx, S = MEGAcon(problem, initial_population, options)



        self.assertTrue(len(fx['f']) > 0)
        self.assertTrue(np.all(fx['f'][:, 0] >= 0))
        self.assertTrue(np.all(fx['f'][:, 1] >= 0))
        self.assertTrue(np.all(fx['f'][:, 2] >= 0))

    def test_unconstrained(self):
        # Define objective 
        def func(x):
            f1 = x[0]**2 + x[1]**2
            f2 = (x[0] - 1)**2 + x[1]**2
            return np.array([f1, f2])

        # No constraints
        def constraints(x):
            return [], []

        # Create problem
        problem = {
            'ObjFunction': func,
            'Variables': 2,
            'Objectives': 2,
            'LB': [-5, -5],
            'UB': [5, 5],
            'Constraints': constraints,
            'x0': [0, 0]
        }

        options = {
            'PopSize': 100,
            'MaxGen': 100,
            'CTol': 1e-4,
            'CeqTol': 1e-2,
            'MaxObj': 1000
        }

        initial_population = [{'x': np.array(problem['x0'])}]

        x, fx, S = MEGAcon(problem, initial_population, options)

        # Check solution
        self.assertTrue(len(fx['f']) > 0)
        self.assertTrue(np.all(fx['f'][:, 0] >= 0))
        self.assertTrue(np.all(fx['f'][:, 1] >= 0))

    def test_constrained(self):
        # Define objective 
        def func(x):
            f1 = x[0]**2 + x[1]**2
            f2 = (x[0] - 1)**2 + x[1]**2
            return np.array([f1, f2])

        # Define constraints
        def constraints(x):
            return [x[0]**2 + x[1]**2 - 1], [x[0] + x[1] - 1]  # Inequality: x0^2 + x1^2 <= 1, Equality: x0 + x1 = 11

        # Create problem
        problem = {
            'ObjFunction': func,
            'Variables': 2,
            'Objectives': 2,
            'LB': [-5, -5],
            'UB': [5, 5],
            'Constraints': constraints,
            'x0': [1, 0.5]
        }

        options = {
            'PopSize': 100,
            'MaxGen': 1000,
            'CTol': 1e-4,
            'CeqTol': 1e-2,
            'MaxObj': 1000,
        }

        initial_population = [{'x': np.array(problem['x0'])}]

        x, fx, S = MEGAcon(problem, initial_population, options)

        # Check solution
        self.assertTrue(len(fx['f']) > 0)
        self.assertTrue(np.all(fx['f'][:, 0] >= 0))
        self.assertTrue(np.all(fx['f'][:, 1] >= 0))
        self.assertTrue(np.all([c <= 0 for c in fx['c']]))  # Inequality constraints satisfied
        self.assertTrue(np.all([abs(ceq) <= 1e-1 for ceq in fx['ceq']]))  # Equality constraints satisfied

if __name__ == '__main__':
    unittest.main()