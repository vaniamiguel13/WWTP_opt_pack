import numpy as np
import time
from penalty2 import penalty2
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D


def Projection(Problem: Dict[str, Any], x: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Project a solution vector or population onto the feasible region defined by the problem bounds.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including bounds and variables.
    - x (Union[List[float], np.ndarray]): Solution vector or population to be projected.

    Returns:
    - np.ndarray: Projected solution vector or population.
    """
    x = np.array(x)  # Ensure x is a numpy array
    if x.ndim == 1:
        # If x is a 1D array (single solution)
        for i in range(Problem['Variables']):
            if x[i] < Problem['LB'][i]:
                x[i] = Problem['LB'][i]
            if x[i] > Problem['UB'][i]:
                x[i] = Problem['UB'][i]
    elif x.ndim == 2:
        # If x is a 2D array (population of solutions)
        for i in range(x.shape[0]):
            for j in range(Problem['Variables']):
                if x[i, j] < Problem['LB'][j]:
                    x[i, j] = Problem['LB'][j]
                if x[i, j] > Problem['UB'][j]:
                    x[i, j] = Problem['UB'][j]
    else:
        raise ValueError("Input x should be either 1D or 2D array")
    return x


def plot_population(Problem: Dict[str, Any], Population: Dict[str, np.ndarray], *args: Any) -> None:
    """
    Plot the current population of solutions for visualization purposes.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including bounds, objective function, and verbosity.
    - Population (Dict[str, np.ndarray]): Dictionary containing the current population of solutions and their objective function values.
    - *args (Any): Additional arguments passed to the objective function.
    """
    if Problem['Verbose']:
        print('rGA is alive...')
        if Problem['Verbose'] == 2:
            fig = plt.figure(figsize=(12, 6))

            # Objective function surface plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Objective function')
            x_range = np.arange(Problem['LB'][0], Problem['UB'][0], (Problem['UB'][0] - Problem['LB'][0]) / 80)
            y_range = np.arange(Problem['LB'][1], Problem['UB'][1], (Problem['UB'][1] - Problem['LB'][1]) / 80)
            xx, yy = np.meshgrid(x_range, y_range)

            zz = np.zeros(xx.shape)
            for i in range(xx.shape[0]):
                for j in range(yy.shape[1]):
                    zz[i, j] = Problem['ObjFunction']([xx[i, j], yy[i, j]], *args)
            ax1.plot_surface(xx, yy, zz, cmap='viridis')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('f(x)')

            # Population contour plot
            ax2 = fig.add_subplot(122)
            ax2.set_title(f'Population at generation: {Problem["Stats"]["GenCounter"]}')
            contour = ax2.contour(xx, yy, zz)
            plt.colorbar(contour)
            ax2.grid(True)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.plot(Population['x'][:, 0], Population['x'][:, 1], '.', markersize=10)

            plt.draw()
            plt.pause(1)
            plt.close(fig)


def rGA(Problem: Dict[str, Any], InitialPopulation: Union[None, List[Dict[str, Any]]] = None,
        Options: Union[None, Dict[str, Any]] = None, *args: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Main function for the robust Genetic Algorithm (rGA).

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including objective function, bounds, and other parameters.
    - InitialPopulation (Union[None, List[Dict[str, Any]]], optional): Initial population of solutions. Defaults to None.
    - Options (Union[None, Dict[str, Any]], optional): Options for the algorithm. Defaults to None.
    - *args (Any): Additional arguments passed to the objective function.

    Returns:
    - Tuple[np.ndarray, float, Dict[str, Any]]: Best chromosome, its objective value, and run data (statistics).
    """

    rGAVersion = '1.1b'

    DefaultOpt = {
        'MaxObj': 2000, 'MaxGen': 2000, 'PopSize': 40, 'EliteProp': 0.1, 'TourSize': 2,
        'Pcross': 0.9, 'Icross': 20, 'Pmut': 0.1, 'Imut': 20, 'CPTolerance': 1.0e-6,
        'CPGenTest': 0.01, 'Verbosity': 0
    }

    if not Problem:
        raise ValueError('rGA:Arguments', 'Invalid number of arguments. Type "help rGA" to obtain help.')

    if Problem == 'defaults':
        return DefaultOpt

    if not isinstance(Problem, dict):
        raise ValueError('rGA:StructProblem', 'First parameter must be a struct.')

    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('rGA:ObjMissing', 'Objective function name is missing.')

    if not isinstance(Problem['LB'], (np.ndarray, list)) or not isinstance(Problem['UB'], (np.ndarray, list)):
        raise ValueError('rGA:Bounds', 'LB and UB must be either numpy arrays or lists.')

    if len(Problem['LB']) != len(Problem['UB']):
        raise ValueError('rGA:BoundsSize', 'Lower bound and upper bound arrays length mismatch.')

    if 'Variables' not in Problem or not Problem['Variables']:
        Problem['Variables'] = len(Problem['LB'])

    if Problem['Variables'] < 0 or Problem['Variables'] > np.size(Problem['LB']):
        raise ValueError('rGA:VariablesNumber', 'Number of variables do not agree with bound constraints.')

    start_time = time.time()

    MaxGenerations = GetOption('MaxGen', Options, DefaultOpt)
    MaxEvals = GetOption('MaxObj', Options, DefaultOpt)
    Pop = GetOption('PopSize', Options, DefaultOpt)
    Elite = GetOption('EliteProp', Options, DefaultOpt)
    Tour = GetOption('TourSize', Options, DefaultOpt)
    Pc = GetOption('Pcross', Options, DefaultOpt)
    Ic = GetOption('Icross', Options, DefaultOpt)
    Pm = GetOption('Pmut', Options, DefaultOpt)
    Im = GetOption('Imut', Options, DefaultOpt)

    Problem['Verbose'] = GetOption('Verbosity', Options, DefaultOpt)
    Problem['Tolerance'] = GetOption('CPTolerance', Options, DefaultOpt)
    Problem['GenTest'] = GetOption('CPGenTest', Options, DefaultOpt)

    Problem['Stats'] = {'ObjFunCounter': 0}

    Problem, Population = InitPopulation(Problem, InitialPopulation, Pop, *args)

    try:
        temp = np.hstack((Population['x'], Population['f'][:, None]))
    except ValueError as e:
        temp = np.hstack((Population['x'], Population['f'].reshape(-1, 1)))
    temp = temp[temp[:, -1].argsort()]
    Population['x'] = temp[:, :-1]
    Population['f'] = temp[:, -1]
    Problem['Stats']['GenCounter'] = 0

    plot_population(Problem, Population, *args)

    Problem['Stats']['Best'] = [Population['f'][0]]
    Problem['Stats']['Worst'] = [Population['f'][Pop - 1]]
    Problem['Stats']['Mean'] = [np.mean(Population['f'])]
    Problem['Stats']['Std'] = [np.std(Population['f'])]

    while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
        if Problem['Stats']['GenCounter'] > 0 and Problem['Stats']['GenCounter'] % int(
                Problem['GenTest'] * MaxGenerations) == 0 and \
                abs(Problem['Stats']['Best'][Problem['Stats']['GenCounter']] -
                    Problem['Stats']['Best'][
                        Problem['Stats']['GenCounter'] - int(Problem['GenTest'] * MaxGenerations)]) < Problem[
            'Tolerance']:
            print(
                'Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest generations')
            break

        Problem['Stats']['GenCounter'] += 1

        elitesize = int(Pop * Elite)
        pool = Pop - elitesize
        parent_chromosome = tournament_selection(Population, pool, Tour)

        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)

        Population['x'][elitesize:] = offspring_chromosome['x'][:pool]
        Population['x'] = Projection(Problem, Population['x'])

        for i in range(elitesize, Pop):
            Problem, Population['f'][i] = ObjEval(Problem, Population['x'][i], *args)

        temp = np.hstack((Population['x'], Population['f'][:, None]))
        temp = temp[temp[:, -1].argsort()]
        Population['x'] = temp[:, :-1]
        Population['f'] = temp[:, -1]

        Problem['Stats']['Best'].append(Population['f'][0])
        Problem['Stats']['Worst'].append(Population['f'][Pop - 1])
        Problem['Stats']['Mean'].append(np.mean(Population['f']))
        Problem['Stats']['Std'].append(np.std(Population['f']))

        fig, ax = plt.subplots()
        p, = ax.plot(Population["x"][:, 0], Population["x"][:, 1], 'b.')  # Initial plot

        if Problem["Verbose"]:
            # Search illustration: plots the population if the number of variables is 2
            if Problem["Verbose"] == 2:
                time.sleep(0.2)
                ax.set_title(f'Population at generation: {Problem["Stats"]["GenCounter"]}')
                ax.set_xlabel('XData')
                ax.set_ylabel('YData')
                p.set_xdata(Population["x"][:, 0])
                p.set_ydata(Population["x"][:, 1])
                plt.draw()
                plt.show()
                plt.pause(0.1)
                plt.close()
        plt.close(fig)

    elapsed_time = time.time() - start_time

    if Problem['Stats']['GenCounter'] >= MaxGenerations or Problem['Stats']['ObjFunCounter'] >= MaxEvals:
        print('Maximum number of iterations or objective function evaluations reached')

    BestChrom = Population['x'][0]
    BestChromObj = Population['f'][0]
    RunData = Problem['Stats']

    if Problem["Verbose"]:
        # Search illustration: plots the population if the number of variables is 2
        if Problem["Verbose"] == 2:
            time.sleep(0.2)
            ax.set_title(f'Population at generation: {Problem["Stats"]["GenCounter"]}')
            ax.set_xlabel('XData')
            ax.set_ylabel('YData')
            p.set_xdata(Population["x"][:, 0])
            p.set_ydata(Population["x"][:, 1])
            plt.draw()
            plt.pause(0.1)
            plt.close()


    return BestChrom, BestChromObj, RunData


def InitPopulation(Problem: dict, initial_population: Union[List[Dict[str, np.ndarray]], np.ndarray], Size: int,
                   *args) -> Tuple[dict, dict]:
    """
    Initializes the population.

    Args:
        Problem (dict): A dictionary containing problem-specific parameters and functions.
        initial_population (Union[List[Dict[str, np.ndarray]], np.ndarray]): Initial population or starting point.
        Size (int): Population size.
        *args: Additional arguments passed to the objective function.

    Returns:
        tuple:
            - dict: Updated problem dictionary with statistics.
            - dict: Initialized population.
    """
    # Initialize the population
    Population = {
        'x': [],
        'f': []
    }

    if initial_population is not None:
        if isinstance(initial_population, np.ndarray):
            # Convert single NumPy array to list of dict
            initial_population = [{'x': initial_population}]
        elif not isinstance(initial_population, list):
            raise ValueError('Initial population must be defined as a list of dictionaries or a NumPy array.')

        # Check for size
        if len(initial_population) > Size:
            # User provided an initial population greater than the parent population size
            raise ValueError('Initial population size must be less than or equal to PopSize.')

        for individual in initial_population:
            x = Bounds(individual['x'], Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            Problem, f = ObjEval(Problem, x, *args)
            Population['x'].append(x)
            Population['f'].append(f)

    # Randomly generate the remaining population
    for i in range(len(Population['x']), Size):
        x = np.array([np.random.uniform(low, high) for low, high in
                      zip(Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])])
        Problem, f = ObjEval(Problem, x, *args)
        Population['f'].append(f)
        Population['x'].append(x)

    Population['f'] = np.array(Population['f'])
    Population['x'] = np.array(Population['x'])

    return Problem, Population

def tournament_selection(chromosomes: Dict[str, np.ndarray], pool_size: int, tour_size: int) -> Dict[str, np.ndarray]:
    """
    Perform tournament selection to choose parent chromosomes.

    Parameters:
    - chromosomes (Dict[str, np.ndarray]): Dictionary containing the population of solutions and their objective values.
    - pool_size (int): Number of parents to select.
    - tour_size (int): Size of each tournament.

    Returns:
    - Dict[str, np.ndarray]: Selected parent chromosomes.
    """

    pop = chromosomes['x'].shape[0]
    P = {'x': [], 'f': []}
    for i in range(pool_size):
        candidates = np.random.choice(pop, tour_size, replace=False)
        fitness = chromosomes['f'][candidates]
        min_candidate = candidates[np.argmin(fitness)]
        P['x'].append(chromosomes['x'][min_candidate])
        P['f'].append(chromosomes['f'][min_candidate])
    P['x'] = np.array(P['x'])
    P['f'] = np.array(P['f'])
    return P


def genetic_operator(Problem: dict, parent_chromosome: dict, pc: float, pm: float, mu: float, mum: float) -> dict:
    """
    Applies genetic operators (crossover and mutation) to generate offspring.

    Args:
        Problem (dict): A dictionary containing problem-specific parameters and functions.
        parent_chromosome (dict): The parent chromosomes.
        pc (float): Crossover probability.
        pm (float): Mutation probability.
        mu (float): Distribution index for crossover.
        mum (float): Distribution index for mutation.

    Returns:
        dict: The offspring chromosomes.
    """
    N, V = parent_chromosome['x'].shape

    child = np.zeros((N, V))
    p = 0

    while p < N:
        parent_1_idx = np.random.randint(0, N)
        parent_2_idx = np.random.randint(0, N)
        while parent_1_idx == parent_2_idx:
            parent_2_idx = np.random.randint(0, N)

        parent_1 = parent_chromosome["x"][parent_1_idx, :]
        parent_2 = parent_chromosome["x"][parent_2_idx, :]

        if np.random.rand() < pc:
            child_1 = np.zeros(V)
            child_2 = np.zeros(V)
            for j in range(V):
                u = np.random.rand()
                if u <= 0.5:
                    bq = (2 * u) ** (1 / (mu + 1))
                else:
                    bq = (1 / (2 * (1 - u))) ** (1 / (mu + 1))
                child_1[j] = 0.5 * ((1 + bq) * parent_1[j] + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * ((1 - bq) * parent_1[j] + (1 + bq) * parent_2[j])
            child_1 = Bounds(child_1, Problem['LB'], Problem['UB'])
            child_2 = Bounds(child_2, Problem['LB'], Problem['UB'])
        else:
            child_1 = parent_1
            child_2 = parent_2

        if np.random.rand() < pm:
            for j in range(V):
                if np.random.rand() < pm:
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_1[j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_1 = Bounds(child_1, Problem['LB'], Problem['UB'])

        if np.random.rand() < pm:
            for j in range(V):
                if np.random.rand() < pm:
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2 * r) ** (1 / (mum + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                    child_2[j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_2 = Bounds(child_2, Problem['LB'], Problem['UB'])

        child[p, :] = Projection(Problem, child_1[:V])
        if p + 1 < N:
            child[p + 1, :] = child_2[:V]
        p += 2

    P = {'x': child}
    return P


def Bounds(X: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Enforce boundary constraints on a solution vector.

    Parameters:
    - X (np.ndarray): Solution vector.
    - L (np.ndarray): Lower bounds.
    - U (np.ndarray): Upper bounds.

    Returns:
    - np.ndarray: Solution vector with boundary constraints applied.
    """

    X = np.clip(X, L, U)
    return X


def ObjEval(Problem: Dict[str, Any], x: np.ndarray, *args: Any) -> Tuple[Dict[str, Any], float]:
    """
    Evaluate the objective function for a given solution.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including objective function and statistics.
    - x (np.ndarray): Solution vector to evaluate.
    - *args (Any): Additional arguments passed to the objective function.

    Returns:
    - Tuple[Dict[str, Any], float]: Updated problem dictionary with statistics and the objective value of the solution.
    """
    try:
        ObjValue = Problem['ObjFunction'](x, *args)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise ValueError('rGA:ObjectiveError',
                         f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return Problem, ObjValue


def lag(x: np.ndarray, Problem: Dict[str, Any], alg: Any) -> Any:
    """
    Compute the Lagrangian penalty for a solution.

    Parameters:
    - x (np.ndarray): Solution vector.
    - Problem (Dict[str, Any]): Dictionary containing problem definition.
    - alg (Any): Algorithm-specific parameters.

    Returns:
    - Any: Lagrangian penalty value.
    """
    Value = penalty2(Problem, x, alg)
    return Value['la']


def GetOption(Option: str, Options: Union[None, Dict[str, Any]], DefaultOpt: Dict[str, Any]) -> Any:
    """
    Retrieve an option value from the provided options dictionary, or use the default value if not specified.

    Parameters:
    - Option (str): Option name.
    - Options (Union[None, Dict[str, Any]]): Dictionary of provided options. Defaults to None.
    - DefaultOpt (Dict[str, Any]): Dictionary of default options.

    Returns:
    - Any: Retrieved option value.
    """
    if Options is None or not isinstance(Options, dict):
        return DefaultOpt[Option]
    return Options.get(Option, DefaultOpt[Option])

