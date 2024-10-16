import numpy as np
import time
from penalty2 import penalty2
from HJ import HJ
from rGA import rGA
from typing import Any, Dict, List, Tuple, Union


def GetOption(Option: str, Options: Dict[str, Any], DefaultOpt: Dict[str, Any]) -> Any:
    """
    Retrieve an option value from a dictionary of options, with a fallback to a default value.

    Parameters:
    - Option (str): The name of the option to retrieve.
    - Options (Dict[str, Any]): Dictionary of user-provided options.
    - DefaultOpt (Dict[str, Any]): Dictionary of default options.

    Returns:
    - Any: The value of the specified option.
    """

    try:
        Value = Options.get(Option, DefaultOpt[Option])
    except KeyError:
        Value = DefaultOpt[Option]
    return Value


def Projection(Problem: Dict[str, Any], x: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Project a solution onto the feasible region defined by lower and upper bounds.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition, including variable bounds.
    - x (Union[List[float], np.ndarray]): Solution vector or population of solutions.

    Returns:
    - np.ndarray: The projected solution(s).
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


def mega_(lambda_: np.ndarray, ldelta: np.ndarray, miu: float, teta_tol: float) -> float:
    """
    Compute a scaling factor for the augmented Lagrangian algorithm.

    Parameters:
    - lambda_ (np.ndarray): Lagrange multipliers for equality constraints.
    - ldelta (np.ndarray): Lagrange multipliers for inequality constraints.
    - miu (float): Penalty parameter.
    - teta_tol (float): Tolerance parameter.

    Returns:
    - float: The computed scaling factor.
    """

    return 1 / max(1, (1 + np.linalg.norm(lambda_) + np.linalg.norm(ldelta) + (1 / miu)) / teta_tol)


def lag(x: Union[List[float], np.ndarray], Problem: Dict[str, Any], alg: Dict[str, Any]) -> float:
    """
    Evaluate the augmented Lagrangian for a given solution.

    Parameters:
    - x (Union[List[float], np.ndarray]): Solution vector.
    - Problem (Dict[str, Any]): Dictionary containing problem definition.
    - alg (Dict[str, Any]): Dictionary of algorithm-specific parameters.

    Returns:
    - float: The value of the augmented Lagrangian.
    """

    Value = penalty2(Problem, x, alg)
    return Value['la']


def HGPSAL(Problem: Dict[str, Any], options: Dict[str, Any], *varargin: Any) -> Tuple[
    np.ndarray, float, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Hybrid Genetic and Pattern Search Augmented Lagrangian algorithm.

    Parameters:
    - Problem (Dict[str, Any]): Dictionary containing problem definition.
    - options (Dict[str, Any]): Dictionary of options for the algorithm.
    - *varargin (Any): Additional arguments passed to the objective and constraints functions.

    Returns: - Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, Dict[str, Any]]: Solution vector,
    objective function value, inequality constraints, equality constraints, augmented Lagrangian value, and statistics.
    """

    HGPSALversion = 0.3

    DefaultOpt = {
        'lambda_min': -1e12, 'lambda_max': 1e12, 'teta_tol': 1e12, 'miu_min': 1e-12, 'miu0': 1, 'csi': 0.5, 'eta0': 1,
        'ffeas': 1, 'gps': 1, 'niu': 1.0, 'zeta': 0.001, 'epsilon1': 1e-4, 'epsilon2': 1e-8, 'suficient': 1e-4,
        'method': 0,
        'omega0': 1, 'alfaw': 0.9, 'alfa_eta': 0.9 * 0.9, 'betaw': 0.9, 'beta_eta': 0.5 * 0.9, 'gama1': 0.5,
        'teta_miu': 0.5,
        'pop_size': 40, 'elite_prop': 0.1, 'tour_size': 2, 'pcross': 0.9, 'icross': 20, 'pmut': 0.1, 'imut': 20,
        'gama': 1, 'delta': 1, 'teta': 0.5, 'eta_asterisco': 1.0e-2, 'epsilon_asterisco': 1.0e-6,
        'cp_ga_test': 0.1, 'cp_ga_tol': 1.0e-6, 'delta_tol': 1e-6, 'maxit': 100, 'maxet': 200, 'max_objfun': 20000,
        'verbose': 0
    }

    if not Problem.get('Variables'):
        raise ValueError('HGPSAL:nMissing', 'Problem dimension is missing.')

    if Problem.get('x0') is None:
        Problem['x0'] = []

    x0 = np.array(Problem['x0'])

    if Problem.get('LB') is None:
        raise ValueError('HGPSAL:lbMissing', 'Problem lower bounds are missing.')
    lb = np.array(Problem['LB'])

    if Problem.get('UB') is None:
        raise ValueError('HGPSAL:ubMissing', 'Problem upper bounds are missing.')
    ub = np.array(Problem['UB'])

    if not Problem.get('ObjFunction'):
        raise ValueError('HGPSAL:ObjMissing', 'Objective function name is missing.')

    if not Problem.get('Constraints'):
        raise ValueError('HGPSAL:ConstraintsMissing', 'Function constraints are missing.')

    if not options:
        opt = DefaultOpt
        opt['pop_size'] = min(20 * Problem['Variables'], 200)
        print('HGPSAL: rGA population size set to %d\n' % opt["pop_size"])
        opt["pmut"] = 1 / Problem['Variables']
        print('HGPSAL: rGA mutation probability set to %f\n' % opt["pmut"])
    else:
        opt = {}
        if 'pop_size' not in options:
            options['pop_size'] = min(20 * Problem['Variables'], 200)
        print('HGPSAL: rGA population size set to %d\n' % options["pop_size"])

        if 'pmut' not in options:
            options["pmut"] = 1 / Problem['Variables']
        print('HGPSAL: rGA mutation probability set to %f\n' % options["pmut"])

        for key in DefaultOpt:
            opt[key] = GetOption(key, options, DefaultOpt)

    start_time = time.time()
    if len(x0) == 0:
        x = []
        for i in range(Problem['Variables']):
            if lb[i] > -np.inf and ub[i] < np.inf:
                x.append(np.random.rand() * (ub[i] - lb[i]) + lb[i])
            else:
                if lb[i] <= -np.inf and ub[i] >= np.inf:
                    x.append(20 * (np.random.rand() - 0.5))
                else:
                    if lb[i] <= -np.inf:
                        x.append(ub[i] - abs(2 * np.random.rand() * ub[i]))
                    else:
                        x.append(lb[i] + abs(2 * np.random.rand() * lb[i]))
    else:
        x = x0

    x = Projection(Problem, x)

    try:
        fx = Problem['ObjFunction'](x, *varargin)
    except Exception as e:
        raise ValueError('augLagr:ConstraintsError',
                         f'Cannot continue because user supplied objective function failed with the following error:\n{e}')

    try:
        c, ceq = Problem['Constraints'](x, *varargin)
    except Exception as e:
        raise ValueError('augLagr:ConstraintsError',
                         f'Cannot continue because user supplied function constraints failed with the following error:\n{e}')

    Problem['m'] = len(ceq)
    Problem['p'] = len(c)

    if opt['verbose']:
        print('Initial guess:')
        print(x)

    alg = {}
    alg['lambda'] = np.ones(Problem['m']) if Problem['m'] else np.array([])
    alg['ldelta'] = np.ones(Problem['p']) if Problem['p'] else np.array([])

    alg['miu'] = opt['miu0']
    alg['alfa'] = min(alg['miu'], opt['gama1'])
    alg['omega'] = opt['omega0'] * pow(alg['alfa'], opt['alfaw'])
    alg['epsilon'] = alg['omega'] * mega_(alg['lambda'], alg['ldelta'], alg['miu'], opt['teta_tol'])
    alg['eta'] = opt['eta0'] * pow(alg['alfa'], opt['alfa_eta'])

    if opt['delta'] == 0:
        alg['delta'] = []
        for j in range(Problem['Variables']):
            if len(x0) == 0 or x0[j] == 0:
                alg['delta'].append(opt['gama'])
            else:
                alg['delta'].append(x0[j] * opt['gama'])
    elif opt['delta'] == 1:
        alg['delta'] = [1] * Problem['Variables']
    else:
        raise ValueError('Invalid option for delta, input a valid option (0 or 1)')

    c = np.array(c)
    ceq = np.array(ceq)
    stats = {'extit': 0, 'objfun': 0}
    stats['x'] = x
    stats['fx'] = fx
    if len(c) > 0:
        stats['c'] = c
    if len(ceq) > 0:
        stats['ceq'] = ceq

    stats['history'] = [('Iter', 'fx rGA', 'nf rGA', 'fx HJ', 'nf HJ')]
    global_search = 1

    prev_fx = np.inf
    prev_x = np.full_like(x, np.inf)

    while stats['extit'] <= opt['maxet'] and stats['objfun'] <= opt['max_objfun']:
        stats['extit'] += 1
        stats['history'].append((stats['extit'],))

        Probl = Problem.copy()
        Probl['ObjFunction'] = lag

        if global_search:
            InitialPopulation = [{'x': x}]
            Options = {
                'PopSize': opt['pop_size'], 'EliteProp': opt['elite_prop'], 'TourSize': opt['tour_size'],
                'Pcross': opt['pcross'], 'Icross': opt['icross'], 'Pmut': opt['pmut'], 'Imut': opt['imut'],
                'CPTolerance': alg['epsilon'], 'CPGenTest': opt['cp_ga_test'], 'MaxGen': opt['maxit'],
                'MaxObj': opt['max_objfun'], 'Verbosity': opt['verbose']
            }



            x, fval, RunData = rGA(Probl, InitialPopulation, Options, Problem, alg)
            x = Projection(Problem, x)
            stats['objfun'] += RunData['ObjFunCounter']
            stats['history'][-1] += (fval, RunData['ObjFunCounter'])

            if opt['verbose']:
                print(f'GA external it: {stats["extit"]}')
                print(x)
                print(fval)
                print(RunData)

        Options = {'MaxIter': opt['maxit'], 'MaxObj': opt['max_objfun'], 'DeltaTol': alg['epsilon'],
                   'Theta': opt['teta']}
        x, fval, Rundata = HJ(Probl, x, alg['delta'], Options, Problem, alg)
        stats['objfun'] += Rundata['ObjFunCounter']
        stats['history'][-1] += (fval, Rundata['ObjFunCounter'])

        if opt['verbose']:
            print(f'HJ external it: {stats["extit"]}')
            print(x)
            print(fval)
            print(Rundata)

        Value = penalty2(Problem, x, alg)
        c = np.array(Value['c'])
        ceq = np.array(Value['ceq'])
        fx = Value['fx']
        la = Value['la']

        stats['x'] = x
        stats['fx'] = fx
        if len(c) > 0:
            stats['c'] = c
        if len(ceq) > 0:
            stats['ceq'] = ceq

        if opt['verbose']:
            print(x)
            print(fx)
            print(c)
            print(ceq)
            print(la)

        if len(alg['lambda']) == 0 and len(alg['ldelta']) == 0:
            break

        max_i = np.max(np.abs(ceq)) if len(ceq) > 0 else 0
        v = np.max(np.maximum(c, alg['ldelta'] * np.abs(c))) if len(c) > 0 else 0

        norma_lambda = np.linalg.norm(alg['lambda'])
        norma_x = np.linalg.norm(x)

        if len(ceq) > 0:
            alg['lambda'] = np.clip(alg['lambda'] + ceq / alg['miu'], opt['lambda_min'], opt['lambda_max'])
        if len(c) > 0:
            alg['ldelta'] = np.clip(np.maximum(0, alg['ldelta'] + c / alg['miu']), opt['lambda_min'], opt['lambda_max'])

        if max_i <= alg['eta'] * (1 + norma_x) and v <= alg['eta'] * (1 + norma_lambda):
            if (alg['epsilon'] < opt['epsilon_asterisco'] and max_i <= opt['eta_asterisco'] * (1 + norma_x) and
                    v <= opt['eta_asterisco'] * (1 + norma_lambda) and global_search == 0):
                stats['message'] = 'HGPSAL: Tolerance of constraints violations satisfied.'
                print(stats['message'])
                break
            else:
                alg['alfa'] = min(alg['miu'], opt['gama1'])
                alg['omega'] = alg['omega'] * pow(alg['alfa'], opt['betaw'])
                alg['epsilon'] = alg['omega'] * mega_(alg['lambda'], alg['ldelta'], alg['miu'], opt['teta_tol'])
                alg['eta'] = alg['eta'] * pow(alg['alfa'], opt['beta_eta'])
        else:
            alg['miu'] = max(min(alg['miu'] * opt['csi'], pow(alg['miu'], opt['teta_miu'])), opt['miu_min'])
            alg['alfa'] = min(alg['miu'], opt['gama1'])
            alg['omega'] = opt['omega0'] * pow(alg['alfa'], opt['alfaw'])
            alg['epsilon'] = alg['omega'] * mega_(alg['lambda'], alg['ldelta'], alg['miu'], opt['teta_tol'])
            alg['eta'] = opt['eta0'] * pow(alg['alfa'], opt['alfa_eta'])
            global_search = 1

        # Check for convergence
        if stats['extit'] > 1:
            if np.abs(fx - prev_fx) < opt['epsilon1'] and np.linalg.norm(x - prev_x) < opt['epsilon2']:
                stats['message'] = 'HGPSAL: Convergence achieved.'
                print(stats['message'])
                break

        prev_fx = fx
        prev_x = x.copy()

    if stats['extit'] > opt['maxet']:
        stats['message'] = 'HGPSAL: Maximum number of external iterations reached.'
        print(stats['message'])
    if stats['objfun'] > opt['max_objfun']:
        stats['message'] = 'HGPSAL: Maximum number objective function evaluations reached.'
        print(stats['message'])

    elapsed_time = time.time() - start_time
    print("Total Time Taken ", elapsed_time)

    return x, fx, c, ceq, la, stats
