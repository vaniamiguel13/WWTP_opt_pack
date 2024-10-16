from typing import Any, Dict, List, Tuple, Union
import numpy as np
import time


def HJ(Problem: Dict[str, Any], x0: Union[List[float], np.ndarray], delta: Union[List[float], np.ndarray] = None,
       Options: Dict[str, Any] = None, *args: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
        Implements the Hooke and Jeeves optimization algorithm.

        Parameters:
        - Problem (Dict[str, Any]): Dictionary containing problem definition including the objective function and bounds.
        - x0 (Union[List[float], np.ndarray]): Initial approximation of the solution.
        - delta (Union[List[float], np.ndarray], optional): Initial step size for the exploratory moves. Defaults to an array of ones with the same shape as `x0`.
        - Options (Dict[str, Any], optional): Options for the algorithm, including maximum number of evaluations, iterations, tolerance, and step size reduction factor. Defaults to `None`.
        - *args (Any): Additional arguments to be passed to the objective function.

        Returns:
        - Tuple[np.ndarray, float, Dict[str, Any]]: Final solution vector, objective function value at the solution, and run data.
    """

    DefaultOpt = {'MaxObj': 2000, 'MaxIter': 200, 'DeltaTol': 1e-6, 'Theta': 0.5}
    HJVersion = '0.1'

    if Problem is None or x0 is None:
        raise ValueError('HJ:AtLeastOneInput',
                         'HJ requests at least two inputs (Problem definition and initial approximation).')

    x0 = np.array(x0).flatten()
    if delta is None:
        delta = np.ones_like(x0)
    else:
        delta = np.array(delta).flatten()

    if Options is None:
        Options = {}

    MaxEval = GetOption('MaxObj', Options, DefaultOpt)
    MaxIt = GetOption('MaxIter', Options, DefaultOpt)
    DelTol = GetOption('DeltaTol', Options, DefaultOpt)
    theta = GetOption('Theta', Options, DefaultOpt)

    start_time = time.time()

    Problem['Stats'] = {'Algorithm': 'Hooke and Jeeves', 'Iterations': 0, 'ObjFunCounter': 0, 'BestObj': float('inf')}

    x = Projection(Problem, x0)
    fx, Problem = ObjEval(Problem, x, *args)

    e = np.eye(len(x0))
    s = np.zeros_like(x0)
    rho = 0

    while np.linalg.norm(delta) > DelTol and Problem['Stats']['ObjFunCounter'] < MaxEval and Problem['Stats'][
        'Iterations'] < MaxIt:
        s, Problem = Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *args)
        x_trial = Projection(Problem, x + s)
        fx1, Problem = ObjEval(Problem, x_trial, *args)

        # Update best objective found
        if fx1 < Problem['Stats']['BestObj']:
            Problem['Stats']['BestObj'] = fx1

        rho = fx - fx1
        if rho > 0:
            x = x_trial
            fx = fx1
            #print(f"Iteration {Problem['Stats']['Iterations']}: Improved to {fx} at {x}")
        else:
            delta = delta * theta

        Problem['Stats']['Iterations'] += 1

    # Final termination messages
    if Problem['Stats']['Iterations'] >= MaxIt:
        Problem['Stats']['Message'] = 'HJ: Maximum number of iterations reached'
    if Problem['Stats']['ObjFunCounter'] >= MaxEval:
        Problem['Stats']['Message'] = 'HJ: Maximum number of objective function evaluations reached'
    if np.linalg.norm(delta) <= DelTol:
        Problem['Stats']['Message'] = 'HJ: Stopping due to step size norm inferior to tolerance'

    print(Problem['Stats']['Message'])
    # print(f"Best Objective Value Found: {Problem['Stats']['BestObj']} at {x}")
    Problem['Stats']['Time'] = time.time() - start_time
    RunData = Problem['Stats']

    return x, fx, RunData

def Exploratory_Moves(Problem: Dict[str, Any], s: np.ndarray, delta: np.ndarray, e: np.ndarray, x: np.ndarray, fx: float, rho: float, *args: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform exploratory moves in the Hooke and Jeeves algorithm.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition.
    - s (np.ndarray): Current search direction.
    - delta (np.ndarray): Step size for exploratory moves.
    - e (np.ndarray): Identity matrix for coordinate direction.
    - x (np.ndarray): Current solution vector.
    - fx (float): Objective function value at the current solution.
    - rho (float): Improvement in objective function value.
    - *args (Any): Additional arguments to be passed to the objective function.

    Returns:
    - Tuple[np.ndarray, Dict[str, Any]]: Updated search direction and problem dictionary.
    """
    if rho > 0:
        x_new = Projection(Problem, x + s)
        min_val, Problem = ObjEval(Problem, x_new, *args)
        rho = fx - min_val
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_val, rho, *args)
    if rho <= 0:
        s = np.zeros_like(s)
        rho = 0
        min_val = fx
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_val, rho, *args)
    return s, Problem


def Coordinate_Search(Problem: Dict[str, Any], s: np.ndarray, delta: np.ndarray, e: np.ndarray, x: np.ndarray, min_val: float, rho: float, *args: Any) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Perform coordinate search in the Hooke and Jeeves algorithm.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition.
    - s (np.ndarray): Current search direction.
    - delta (np.ndarray): Step size for coordinate search.
    - e (np.ndarray): Identity matrix for coordinate direction.
    - x (np.ndarray): Current solution vector.
    - min_val (float): Minimum value of the objective function found.
    - rho (float): Improvement in objective function value.
    - *args (Any): Additional arguments to be passed to the objective function.

    Returns:
    - Tuple[np.ndarray, float, Dict[str, Any]]: Updated search direction, improvement in objective function value, and problem dictionary.
    """
    for i in range(len(x)):
        s1 = s + delta * e[:, i]
        x1 = Projection(Problem, x + s1)
        fx1, Problem = ObjEval(Problem, x1, *args)

        if fx1 < min_val:
            rho = min_val - fx1
            min_val = fx1
            s = s1
        else:
            s1 = s - delta * e[:, i]
            x1 = Projection(Problem, x + s1)
            fx1, Problem = ObjEval(Problem, x1, *args)

            if fx1 < min_val:
                rho = min_val - fx1
                min_val = fx1
                s = s1
    return s, rho, Problem


def GetOption(Option: str, Options: Dict[str, Any], DefaultOpt: Dict[str, Any]) -> Any:
    """
    Retrieve the value of an option from the options dictionary, or use a default value if the option is not set.

    Parameters:
    - Option (str): The name of the option to retrieve.
    - Options (Dict[str, Any]): Dictionary of user-specified options.
    - DefaultOpt (Dict[str, Any]): Dictionary of default options.

    Returns:
    - Any: The value of the option.
    """
    return Options.get(Option, DefaultOpt[Option])


def Projection(Problem: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """
    Project a solution vector onto the feasible region defined by the problem bounds.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition, including variable bounds.
    - x (np.ndarray): Solution vector to be projected.

    Returns:
    - np.ndarray: Projected solution vector.
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


def ObjEval(Problem: Dict[str, Any], x: np.ndarray, *args: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate the objective function at a given solution vector.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition, including the objective function.
    - x (np.ndarray): Solution vector at which to evaluate the objective function.
    - *args (Any): Additional arguments to be passed to the objective function.

    Returns:
    - Tuple[float, Dict[str, Any]]: Objective function value and updated problem dictionary.
    """
    try:
        ObjValue = Problem['ObjFunction'](x, *args)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise ValueError('rGA:ObjectiveError',
                         f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return ObjValue, Problem
