import numpy as np
from typing import Any, Dict, List, Tuple, Union


def penalty2(Problem: Dict[str, Any], x: Union[List[float], np.ndarray], alg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the augmented Lagrangian penalty for a solution.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition, including objective and constraint functions.
    - x (Union[List[float], np.ndarray]): Solution vector.
    - alg (Dict[str, Any]): Dictionary of algorithm-specific parameters, including Lagrange multipliers and penalty terms.

    Returns:
    - Dict[str, Any]: Dictionary containing the objective function value, constraint values, and augmented Lagrangian penalty.
    """

    # Value = {}
    #     # x = np.array(x)
    #     # Value['fx'] = ObjEval(Problem, x)
    #     # Value['c'], Value['ceq'] = ConsEval(Problem, x)
    #     #
    #     # term1 = np.sum(alg['lambda'] * Value['ceq'])
    #     # term2 = np.sum(Value['ceq'] ** 2)  # Remove [0]
    #     # term3 = np.sum(np.maximum(0, alg['ldelta'] + Value['c'] / alg['miu']) ** 2 - alg['ldelta'] ** 2)  # Remove [0]
    #     # Value['la'] = Value['fx'] + term1 + term2 / (2 * alg['miu']) + alg['miu'] * term3 / 2
    #     #
    #     # return Value~
    Value = {}
    x = np.array(x)
    Value['fx'] = ObjEval(Problem, x)
    Value['c'], Value['ceq'] = ConsEval(Problem, x)

    # Ensure all values are numpy arrays
    Value['c'] = np.array(Value['c'])
    Value['ceq'] = np.array(Value['ceq'])
    alg['lambda'] = np.array(alg['lambda'])
    alg['ldelta'] = np.array(alg['ldelta'])

    term1 = np.sum(alg['lambda'] * Value['ceq'])
    term2 = np.sum(Value['ceq'] ** 2)
    term3 = np.sum(np.maximum(0, alg['ldelta'] + Value['c'] / alg['miu']) ** 2 - alg['ldelta'] ** 2)
    Value['la'] = Value['fx'] + term1 + term2 / (2 * alg['miu']) + alg['miu'] * term3 / 2

    return Value


def ObjEval(Problem: Dict[str, Any], x: np.ndarray, *varargin: Any) -> float:
    """
    Evaluate the objective function for a given solution.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including the objective function.
    - x (np.ndarray): Solution vector to evaluate.
    - *varargin (Any): Additional arguments passed to the objective function.

    Returns:
    - float: The objective value of the solution.
    """
    try:
        ObjValue = Problem['ObjFunction'](x, *varargin)
    except Exception as e:
        raise ValueError(f'augLagr:ObjectiveError',
                         f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return ObjValue


def ConsEval(Problem: Dict[str, Any], x: np.ndarray, *varargin: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the constraints for a given solution.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition including the constraints function.
    - x (np.ndarray): Solution vector to evaluate.
    - *varargin (Any): Additional arguments passed to the constraints function.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the inequality constraints and equality constraints.
    """

    try:
        c, ceq = Problem['Constraints'](x, *varargin)
    except Exception as e:
        raise ValueError(f'augLagr:ConstraintsError',
                         f'Cannot continue because user supplied function constraints failed with the following error:\n{e}')
    return c, ceq
