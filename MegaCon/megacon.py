import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Any, Dict, List, Tuple, Union


def MEGAcon(Problem: Dict[str, Any], InitialPopulation: List[Dict[str, Any]], Options: Dict[str, Any], *args: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Executes the MEGA optimization algorithm.

    Args:
        Problem (dict): A dictionary containing problem-specific parameters and functions.
        InitialPopulation (list): List of dictionaries representing the initial population.
        Options (dict): A dictionary containing options and parameters for the algorithm.
        *args: Additional arguments passed to the objective and constraint functions.

    Returns:
        tuple:
            - np.ndarray: The non-dominated points.
            - dict: A dictionary containing the front points.
            - dict: Run data statistics.
    """
    MEGAVersion = 'MEGA v1.0'
    print(MEGAVersion)

    DefaultOpt = {
        'MaxObj': 2000, 'MaxGen': 1000, 'PopSize': 40, 'Elite': 0.1, 
        'TourSize': 2, 'Pcross': 0.9, 'Icross': 20, 'Pmut': 0.1, 'Imut': 20, 'Sigma': 0.1,
        'CPTolerance': 1.0e-6, 'CPGenTest': 0.01, 'CTol': 1e-2, 'CeqTol': 1e-2, 'NormType': np.inf, 'NormCons': 1, 'Verbosity': 0
    }

    if len(args) == 0 and Options is None:
        raise ValueError('Invalid number of arguments. Type "help MEGA" to obtain help.')

    if len(args) == 1 and Options is None and Problem == 'defaults':
        return DefaultOpt

    if not isinstance(Problem, dict):
        raise ValueError('First parameter must be a struct (dictionary in Python).')

    if Options is None:
        Options = {}
    if InitialPopulation is None:
        InitialPopulation = []

    if 'ObjFunction' not in Problem or not Problem['ObjFunction']:
        raise ValueError('Objective function name is missing.')

    if not isinstance(Problem['LB'], (list, np.ndarray)) or not isinstance(Problem['UB'], (list, np.ndarray)) or len(Problem['LB']) == 0 or len(Problem['UB']) == 0:
        raise ValueError('Population relies on finite bounds on all variables.')

    if len(Problem['LB']) != len(Problem['UB']):
        raise ValueError('Lower bound and upper bound arrays length mismatch.')

    if 'Variables' not in Problem or not Problem['Variables']:
        Problem['Variables'] = len(Problem['LB'])

    if Problem['Variables'] < 0 or Problem['Variables'] > len(Problem['LB']):
        raise ValueError('Number of variables do not agree with bound constraints')

    Conflag = 1 if 'Constraints' in Problem and Problem['Constraints'] else 0

    if 'Objectives' not in Problem or not Problem['Objectives']:
        Problem['Objectives'] = 2

    MaxGenerations = GetOption('MaxGen', Options, DefaultOpt)
    MaxEvals = GetOption('MaxObj', Options, DefaultOpt)
    Pop = GetOption('PopSize', Options, DefaultOpt)
    Elite = GetOption('Elite', Options, DefaultOpt)
    if Elite:
        eliteinf = max(2, int(np.ceil(Elite / 2 * Pop)))
        elitesup = min(Pop - 2, int(np.floor((1 - Elite / 2) * Pop)))
        print(f'MEGA: MEGA elite size set to the interval {eliteinf} and {elitesup}')
    Tour = GetOption('TourSize', Options, DefaultOpt)
    Pc = GetOption('Pcross', Options, DefaultOpt)
    Ic = GetOption('Icross', Options, DefaultOpt)
    if 'Pmut' not in Options:
        Pm = 1 / Problem['Variables']
        print(f'MEGA: MEGA mutation probability set to {Pm}')
    else:
        Pm = GetOption('Pmut', Options, DefaultOpt)
    Im = GetOption('Imut', Options, DefaultOpt)
    if 'Sigma' not in Options:
        print(f'MEGA: MEGA niching radius will be adapted during the search {Pm}')
        Sigma = 0
    else:
        Sigma = GetOption('Sigma', Options, DefaultOpt)
    CTol = GetOption('CTol', Options, DefaultOpt)
    CeqTol = GetOption('CeqTol', Options, DefaultOpt)
    NormType = GetOption('NormType', Options, DefaultOpt)
    NormCons = GetOption('NormCons', Options, DefaultOpt)

    Problem['Verbose'] = GetOption('Verbosity', Options, DefaultOpt)
    Problem['Tolerance'] = GetOption('CPTolerance', Options, DefaultOpt)
    Problem['GenTest'] = GetOption('CPGenTest', Options, DefaultOpt)

    start_time = time.time()

    Problem['Stats'] = {
        'ObjFunCounter': 0,
        'ConCounter': 0,
        'GenCounter': 0,
        'N1Front': [0],
        'NFronts': [0]
    }

    Problem, Population = InitPopulation(Problem, InitialPopulation, Pop, Conflag, CTol, CeqTol, NormType, NormCons, *args)
    Population = RankPopulation(Population, Elite, Sigma, NormType)

    temp = np.hstack((Population['x'], Population['f'], Population['c'], Population['ceq'], Population['Feasible'].reshape(-1, 1), Population['Rank'].reshape(-1, 1), Population['Fitness'].reshape(-1, 1)))
    temp = temp[temp[:, -1].argsort()]
    Population['x'] = temp[:, :Population['x'].shape[1]]
    Population['f'] = temp[:, Population['x'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1]]
    Population['c'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]]
    Population['ceq'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1]]
    Population['Feasible'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1]]
    Population['Rank'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1] + 1]
    Population['Fitness'] = temp[:, -1]

    Problem['Stats']['N1Front'][0] = len(np.where(Population['Rank'] == 1)[0])
    Problem['Stats']['NFronts'][0] = np.max(Population['Rank'])

    if Problem['Verbose']:
        print('MEGA is running...')
        if np.sum(Population['Feasible']) == 0:
            print(f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front= {Problem["Stats"]["N1Front"][0]}  Number of fronts= {Problem["Stats"]["NFronts"][0]}  All points are unfeasible. Best point: {np.linalg.norm(np.maximum(0, Population["c"][0, :]), NormType) + np.linalg.norm(np.abs(Population["ceq"][0, :]), NormType)}')
        else:
            print(f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front= {Problem["Stats"]["N1Front"][0]}  Number of fronts= {Problem["Stats"]["NFronts"][0]}')
        if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
            p, pp = draw_illustration(Problem, Population['x'][:, 0], Population['x'][:, 1], Population['f'][:, 0], Population['f'][:, 1], *args)

    while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
        Problem['Stats']['GenCounter'] += 1

        if Elite:
            pool = int(np.floor(max(eliteinf, min(elitesup, Pop - len(np.where(Population['Rank'][Population['Feasible'] == 1] == 1)[0])))))
        else:
            pool = Pop

        parent_chromosome = tournament_selection(Population, pool, Tour)
        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)
        Population['x'][-pool:] = offspring_chromosome['x'][:pool]

        for i in range(Pop - pool, Pop):
            Problem, Population['f'][i, :] = ObjEval(Problem, Population['x'][i, :], *args)
            if Conflag:
                Problem, Population['c'][i, :], Population['ceq'][i, :] = ConEval(Problem, Population['x'][i, :], *args)
            else:
                Population['c'][i, :] = 0
                Population['ceq'][i, :] = 0
                Population['Feasible'][i] = 1

        if Conflag:
            for i in range(Pop - pool, Pop):
                if NormCons:
                    maxc = np.maximum(0, np.min(Population['c'], axis=0))
                    maxc[maxc == 0] = 1
                    maxceq = np.abs(np.min(Population['ceq'], axis=0))
                    maxceq[maxceq == 0] = 1
                    Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i, :]) / maxc, NormType) <= CTol and np.linalg.norm(np.abs(Population['ceq'][i, :]) / maxceq, NormType) <= CeqTol)
                else:
                    Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i, :]), NormType) <= CTol and np.linalg.norm(np.abs(Population['ceq'][i, :]), NormType) <= CeqTol)

        Population = RankPopulation(Population, Elite, Sigma, NormType)

        temp = np.hstack((Population['x'], Population['f'], Population['c'], Population['ceq'], Population['Feasible'].reshape(-1, 1), Population['Rank'].reshape(-1, 1), Population['Fitness'].reshape(-1, 1)))
        temp = temp[temp[:, -1].argsort()]
        Population['x'] = temp[:, :Population['x'].shape[1]]
        Population['f'] = temp[:, Population['x'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1]]
        Population['c'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]]
        Population['ceq'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1]:Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1]]
        Population['Feasible'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1]]
        Population['Rank'] = temp[:, Population['x'].shape[1] + Population['f'].shape[1] + Population['c'].shape[1] + Population['ceq'].shape[1] + 1]
        Population['Fitness'] = temp[:, -1]

        Problem['Stats']['N1Front'].append(len(np.where(Population['Rank'] == 1)[0]))
        Problem['Stats']['NFronts'].append(np.max(Population['Rank']))

        if Problem['Verbose']:
            if np.sum(Population['Feasible']) == 0:
                print(f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front= {Problem["Stats"]["N1Front"][-1]}  Number of fronts= {Problem["Stats"]["NFronts"][-1]}  All points are unfeasible. Best point: {np.linalg.norm(np.maximum(0, Population["c"][0, :]), NormType) + np.linalg.norm(np.abs(Population["ceq"][0, :]), NormType)}')
            else:
                print(f'Gen: {Problem["Stats"]["GenCounter"] + 1}  No. points in 1st front= {Problem["Stats"]["N1Front"][-1]}  Number of fronts= {Problem["Stats"]["NFronts"][-1]}')
            if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
                update_illustration(p, pp, Problem['Stats']['GenCounter'], Population['x'][:, 0], Population['x'][:, 1], Population['f'][:, 0], Population['f'][:, 1])

    elapsed_time = time.time() - start_time
    print(f'Elapsed time: {elapsed_time:.2f} seconds')

    if Problem['Stats']['GenCounter'] >= MaxGenerations or Problem['Stats']['ObjFunCounter'] >= MaxEvals:
        print('Maximum number of iterations or objective function evaluations reached')

    RunData = Problem['Stats']

    NonDomPoint = []
    FrontPoint = {'f': [], 'c': [], 'ceq': []}
    for i in range(Pop):
        if Population['Rank'][i] == 1:
            NonDomPoint.append(Population['x'][i, :])
            FrontPoint['f'].append(Population['f'][i, :])
            FrontPoint['c'].append(Population['c'][i, :])
            FrontPoint['ceq'].append(Population['ceq'][i, :])

    NonDomPoint = np.array(NonDomPoint)
    FrontPoint['f'] = np.array(FrontPoint['f'])
    FrontPoint['c'] = np.array(FrontPoint['c'])
    FrontPoint['ceq'] = np.array(FrontPoint['ceq'])

    if Problem['Verbose']:
        if Problem['Verbose'] == 2 and Problem['Variables'] == 2 and Problem['Objectives'] == 2:
            terminate_illustration(p, pp, NonDomPoint[:, 0], NonDomPoint[:, 1], FrontPoint['f'][:, 0], FrontPoint['f'][:, 1], len(NonDomPoint))
        print(Problem['Stats'])

    return NonDomPoint, FrontPoint, RunData


def ObjEval(problem: Dict[str, Any], x: np.ndarray, *args: Any) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates the objective function.

    Args:
        problem (dict): A dictionary containing problem-specific parameters and functions.
        x (np.ndarray): The decision variables.
        *args: Additional arguments passed to the objective function.

    Returns:
        tuple:
            - dict: Updated problem dictionary with statistics.
            - np.ndarray: Objective function value.
    """
    try:
        obj_value = problem['ObjFunction'](x, *args)
        # update counter
        problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise RuntimeError(
            f"Cannot continue because the user-supplied objective function failed with the following error:\n{e}"
        )
    return problem, obj_value
def Bounds(X: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Applies bounds to the decision variables.

    Args:
        X (np.ndarray): The decision variables.
        L (np.ndarray): The lower bounds.
        U (np.ndarray): The upper bounds.

    Returns:
        np.ndarray: The bounded decision variables.
    """
    for i in range(len(X)):
        if X[i] < L[i]:
            X[i] = L[i]
        if X[i] > U[i]:
            X[i] = U[i]
    return X

def ConEval(problem: Dict[str, Any], x: np.ndarray, *args: Any) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates the constraints.

    Args:
        problem (dict): A dictionary containing problem-specific parameters and functions.
        x (np.ndarray): The decision variables.
        *args: Additional arguments passed to the constraint function.

    Returns:
        tuple:
            - dict: Updated problem dictionary with statistics.
            - np.ndarray: Inequality constraints.
            - np.ndarray: Equality constraints.
    """
    problem['Stats']['ConCounter'] += 1
    
    try:
        c, ceq = problem['Constraints'](x, *args)
        
        # Ensure c is a numpy array
        if not isinstance(c, np.ndarray):
            c = np.array(c)
        
        # Ensure ceq is a numpy array
        if not isinstance(ceq, np.ndarray):
            ceq = np.array(ceq)
        
        # If c is an empty array, set it to an array with a single zero element
        if c.size == 0:
            c = np.array([0])
        
        # If ceq is an empty array, set it to an array with a single zero element
        if ceq.size == 0:
            ceq = np.array([0])
    
    except Exception as e:
        raise RuntimeError(
            f"Cannot continue because the user-supplied function constraints failed with the following error:\n{e}"
        )
    
    return problem, c, ceq


def tournament_selection(chromosomes: Dict[str, np.ndarray], pool_size: int, tour_size: int) -> Dict[str, np.ndarray]:
    """
    Performs tournament selection.

    Args:
        chromosomes (dict): The current population of chromosomes.
        pool_size (int): The size of the selection pool.
        tour_size (int): The size of the tournament.

    Returns:
        dict: Selected parent chromosomes.
    """
    pop = chromosomes['x'].shape[0]
    P = {'x': np.zeros((pool_size, chromosomes['x'].shape[1]))}
    candidate = np.zeros(tour_size, dtype=int)
    fitness = np.zeros(tour_size)
    
    for i in range(pool_size):
        for j in range(tour_size):
            candidate[j] = np.random.randint(1, pop+1)
            if j > 1:
                while np.any(candidate[:j] == candidate[j]):
                    candidate[j] = np.random.randint(1, pop+1)
            fitness[j] = chromosomes['Fitness'][candidate[j]-1]
        min_candidate = np.where(fitness == np.min(fitness))[0]
        P['x'][i, :] = chromosomes['x'][candidate[min_candidate[0]]-1, :]
    
    return P

def genetic_operator(Problem: Dict[str, Any], parent_chromosome: Dict[str, np.ndarray], pc: float, pm: float, mu: float, mum: float) -> Dict[str, np.ndarray]:
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
        parent_1 = np.random.randint(1, N+1)
        parent_2 = np.random.randint(1, N+1)
        while parent_1 == parent_2:
            parent_2 = np.random.randint(1, N+1)
        parent_1 = parent_chromosome['x'][parent_1-1, :]
        parent_2 = parent_chromosome['x'][parent_2-1, :]

        if np.random.rand() < pc:
            child_1 = np.zeros(V)
            child_2 = np.zeros(V)
            for j in range(V):
                u = np.random.rand()
                if u <= 0.5:
                    bq = (2*u)**(1/(mu+1))
                else:
                    bq = (1/(2*(1 - u)))**(1/(mu+1))
                child_1[j] = 0.5 * ((1 + bq) * parent_1[j] + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * ((1 - bq) * parent_1[j] + (1 + bq) * parent_2[j])
            child_1 = np.clip(child_1, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            child_2 = np.clip(child_2, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
        else:
            child_1 = parent_1
            child_2 = parent_2

        if np.random.rand() < np.sqrt(pm):
            for j in range(V):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2*r)**(1/(mum+1)) - 1
                    else:
                        delta = 1 - (2*(1 - r))**(1/(mum+1))
                    child_1[j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_1 = np.clip(child_1, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])

        if np.random.rand() < np.sqrt(pm):
            for j in range(V):
                if np.random.rand() < np.sqrt(pm):
                    r = np.random.rand()
                    if r < 0.5:
                        delta = (2*r)**(1/(mum+1)) - 1
                    else:
                        delta = 1 - (2*(1 - r))**(1/(mum+1))
                    child_2[j] += (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_2 = np.clip(child_2, Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])

        child[p, :] = child_1[:V]
        if p + 1 < N:
            child[p + 1, :] = child_2[:V]
        p += 2

    P = {'x': child}
    return P


def InitPopulation(Problem: Dict[str, Any], InitialPopulation: List[Dict[str, Any]], Size: int, Conflag: int, CTol: float, CeqTol: float, NormType: Union[int, float], NormCons: int, *args: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Initializes the population.

    Args:
        Problem (dict): A dictionary containing problem-specific parameters and functions.
        InitialPopulation (list): List of dictionaries representing the initial population.
        Size (int): Population size.
        Conflag (int): Flag indicating if constraints are present.
        CTol (float): Constraint tolerance.
        CeqTol (float): Equality constraint tolerance.
        NormType (int or float): Norm type for constraint evaluation.
        NormCons (int): Flag indicating if normalization is applied.
        *args: Additional arguments passed to the objective and constraint functions.

    Returns:
        tuple:
            - dict: Updated problem dictionary with statistics.
            - dict: Initialized population.
    """
    if InitialPopulation and not isinstance(InitialPopulation, list):
        raise ValueError('Initial population must be defined in a structure (list of dictionaries).')
    else:
        Population = {
            'x': np.zeros((Size, Problem['Variables'])),
            'f': np.zeros((Size, Problem['Objectives'])),
            #'f': np.zeros((Size, Problem['Variables'])),

            'c': np.zeros((Size, 1)),  # Adjust the shape based on constraints
            'ceq': np.zeros((Size, 99)),  # Adjust the shape based on constraints
            'Feasible': np.zeros(Size, dtype=bool),
            'Rank': np.zeros(Size, dtype=int),
            'Fitness': np.zeros(Size)
        }

        if len(InitialPopulation) > Size:
            raise ValueError('Initial population size must be inferior to PopSize.')

        # Copy the initial population and initialize them
        for i, ind in enumerate(InitialPopulation):
            #Bounds(InitialPopulation(i).x,Problem.LB(1:Problem.Variables),Problem.UB(1:Problem.Variables))
            Population['x'][i, :] = Bounds(ind['x'], Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            Problem, Population['f'][i, :] = ObjEval(Problem, Population['x'][i, :], *args)
            if Conflag:
                Problem, Population['c'][i, :], Population['ceq'][i, :] = ConEval(Problem, Population['x'][i, :], *args)
            else:
                Population['c'][i, :] = 0
                Population['ceq'][i, :] = 0
                Population['Feasible'][i] = True
            Population['Rank'][i] = 0

        # Randomly generate the remaining population
        for i in range(len(InitialPopulation), Size):
            Population['x'][i, :] = np.array(Problem['LB'][:Problem['Variables']]) + np.array((Problem['UB'][:Problem['Variables']]) - np.array(Problem['LB'][:Problem['Variables']])) * np.random.rand(Problem['Variables'])
            Problem, Population['f'][i, :] = ObjEval(Problem, Population['x'][i, :], *args)
            if Conflag:
                Problem, Population['c'][i, :], Population['ceq'][i, :] = ConEval(Problem, Population['x'][i, :], *args)
            else:
                Population['c'][i, :] = 0
                Population['ceq'][i, :] = 0
                Population['Feasible'][i] = True
            Population['Rank'][i] = 0

        if Conflag:
            for i in range(Size):
                if NormCons:
                    maxc = np.minimum(1, np.max(Population['c'], axis=0))
                    maxc[maxc == 0] = 1
                    maxceq = np.abs(np.min(Population['ceq'], axis=0))
                    maxceq[maxceq == 0] = 1
                    Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i, :]) / maxc, ord=NormType) <= CTol and
                                                 np.linalg.norm(np.abs(Population['ceq'][i, :]) / maxceq, ord=NormType) <= CeqTol)
                else:
                    Population['Feasible'][i] = (np.linalg.norm(np.maximum(0, Population['c'][i, :]), ord=NormType) <= CTol and
                                                 np.linalg.norm(np.abs(Population['ceq'][i, :]), ord=NormType) <= CeqTol)

        Population['Feasible'] = Population['Feasible'].reshape(-1, 1)
        Population['Rank'] = Population['Rank'].reshape(-1, 1)
        Population['Fitness'] = Population['Rank']

    return Problem, Population

def RankPopulation(Population: Dict[str, np.ndarray], elite: float, sigma: float, NormType: Union[int, float]) -> Dict[str, np.ndarray]:
    """
    Ranks the population based on non-domination and sharing.

    Args:
        Population (dict): The population to be ranked.
        elite (float): Elite fraction.
        sigma (float): Sharing parameter.
        NormType (int or float): Norm type for constraint evaluation.

    Returns:
        dict: Ranked population.
    """
    pop = Population['f'].shape[0]
    obj = Population['f'].shape[1]

    # Compute rank
    IP = np.where(Population['Feasible'] == 1)[0]
    rank = 1
    fk = 1

    if IP.size > 0:
        P = np.hstack((Population['f'][IP, :], IP.reshape(-1, 1)))
        while P.shape[0] > 0:
            ND = nondom(P, 1)  # non-dominated and indices
            P = np.array([x for x in P if not any((x == ND).all(axis=1))])
            for i in range(ND.shape[0]):
                Population['Rank'][int(ND[i, obj])] = rank
            rank += 1

        I = np.where(Population['Rank'] == 1)[0]
        ideal = np.min(Population['f'][I, :], axis=0)
        J = np.argmin(Population['f'][I, :], axis=0)
        if sigma == 0:
            nadir = np.max(Population['f'][I, :], axis=0)
            dnorm = np.linalg.norm(nadir - ideal)
            if dnorm == 0:
                dnorm = np.linalg.norm(np.max(Population['f'], axis=0) - np.min(Population['f'], axis=0))
            sigma = 2 * dnorm * (pop - np.floor(elite * pop / 2) - 1) ** (-1 / (obj - 1))

        # Compute sharing values
        if sigma != 0:
            frente = 1
            while frente < rank:
                I = np.where(Population['Rank'] == frente)[0]
                LI = len(I)
                for i in range(LI):
                    if I[i] not in J:
                        nc = 0
                        for j in range(LI):
                            nc += share(np.linalg.norm(Population['f'][I[i], :] - Population['f'][I[j], :]), sigma)
                        Population['Fitness'][I[i]] = fk * nc
                    else:
                        Population['Fitness'][I[i]] = fk
                fk = np.floor(np.max(Population['Fitness'][I]) + 1)
                frente += 1
        else:
            Population['Fitness'] = Population['Rank']
    else:
        Population['Rank'][:] = 1
        Population['Fitness'][:] = 1

    # Handle unfeasible solutions
    IP = np.where(Population['Feasible'] == 0)[0]
    for i in range(len(IP)):
        Population['Rank'][IP[i]] = rank
        Population['Fitness'][IP[i]] = fk + np.linalg.norm(np.maximum(0, Population['c'][IP[i], :]), ord=NormType) + np.linalg.norm(np.abs(Population['ceq'][IP[i], :]), ord=NormType)

    return Population

def share(dist: float, sigma: float) -> float:
    """
    Calculates the sharing value based on distance and sigma.

    Args:
        dist (float): The distance.
        sigma (float): The sharing parameter.

    Returns:
        float: Sharing value.
    """
    if dist <= sigma:
        sh = 1 - (dist / sigma) ** 2
    else:
        sh = 0
    return sh

def dom(x: np.ndarray, y: np.ndarray) -> int:
    """
    Determines if one solution dominates another.

    Args:
        x (np.ndarray): Solution 1.
        y (np.ndarray): Solution 2.

    Returns:
        int: 1 if x dominates y, 0 otherwise.
    """
    m = len(x)  # numero de objectivos

    for i in range(m):
        if x[i] > y[i]:
            return 0  # x nao domina y

    for i in range(m):
        if x[i] < y[i]:
            return 1  # x domina y

    return 0  # x nao domina y

def nondom(P: np.ndarray, nv: int) -> np.ndarray:
    """
    Finds non-dominated solutions.

    Args:
        P (np.ndarray): Population matrix.
        nv (int): Number of variables.

    Returns:
        np.ndarray: Non-dominated solutions.
    """
    n = P.shape[0]  # numero de pontos
    m = P.shape[1] - nv  # numero de objectivos

    PL = []
    i = 0  # para percorrer pontos

    while i < n:
        cand = 1  # solucao candidata a nao dominada
        j = 0
        while j < n and cand == 1:
            if j != i and dom(P[j, :m], P[i, :m]) == 1:  # se solucao j domina solucao i
                cand = 0  # solucao i deixa de ser candidata a nao dominada
            j += 1
        if cand == 1:  # se a solucao i e nao dominada
            PL.append(P[i, :])  # acrescentar a PL
        i += 1

    return np.array(PL)

def cnondom(P: np.ndarray, nv: int, F: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Finds constrained non-dominated solutions.

    Args:
        P (np.ndarray): Population matrix.
        nv (int): Number of variables.
        F (np.ndarray): Feasibility array.
        C (np.ndarray): Constraint violation array.

    Returns:
        np.ndarray: Constrained non-dominated solutions.
    """
    n = P.shape[0]  # numero de pontos
    m = P.shape[1] - nv  # numero de objectivos

    PL = []
    i = 0  # para percorrer pontos

    while i < n:
        cand = 1  # solucao candidata a nao dominada
        j = 0
        while j < n and cand == 1:
            if j != i:
                if F[j] == 1 and F[i] == 0:  # j é admissivel e i não, então j domina i
                    cand = 0
                else:
                    if F[j] == 1 and F[i] == 1 and C[j] < C[i]:  # ambas não admissíveis e j viola menos do que i, então j domina i
                        cand = 0
                    else:
                        if dom(P[j, :m], P[i, :m]) == 1:  # se solucao j domina solucao i
                            cand = 0  # solucao i deixa de ser candidata a nao dominada
            j += 1
        if cand == 1:  # se a solucao i e nao dominada
            PL.append(P[i, :])  # acrescentar a PL
        i += 1

    return np.array(PL)

def GetOption(Option: str, Options: Dict[str, Any], DefaultOpt: Dict[str, Any]) -> Any:
    """
    Retrieves an option value from user-provided options or defaults.

    Args:
        Option (str): The option key.
        Options (dict): User-provided options.
        DefaultOpt (dict): Default options.

    Returns:
        Any: The option value.
    """
    # Check for user provided options
    if Options is None or not isinstance(Options, dict):
        # User does not provide Options
        return DefaultOpt[Option]
    
    # Try the option provided by user
    try:
        Value = Options[Option]
    except KeyError:
        Value = None

    # Option not provided by user
    if Value is None:
        Value = DefaultOpt[Option]

    return Value

def draw_illustration(Problem: Dict[str, Any], X: np.ndarray, Y: np.ndarray, F1: np.ndarray, F2: np.ndarray, *args: Any) -> Tuple[Any, Any]:
    """
    Draws initial illustrations for the optimization process.

    Args:
        Problem (dict): A dictionary containing problem-specific parameters and functions.
        X (np.ndarray): Decision variables X.
        Y (np.ndarray): Decision variables Y.
        F1 (np.ndarray): Objective function 1 values.
        F2 (np.ndarray): Objective function 2 values.
        *args: Additional arguments.

    Returns:
        tuple:
            - Any: Plot handle for decision variables.
            - Any: Plot handle for objective functions.
    """
    plt.close('all')
    fig = plt.figure(1, figsize=(12, 10))

    xx, yy = np.meshgrid(
        np.linspace(Problem['LB'][0], Problem['UB'][0], 80),
        np.linspace(Problem['LB'][1], Problem['UB'][1], 80)
    )
    z1 = np.zeros_like(xx)
    z2 = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            f = eval(Problem['ObjFunction'])([xx[i, j], yy[i, j]], *args)
            z1[i, j] = f[0]
            z2[i, j] = f[1]

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_surface(xx, yy, z1, alpha=0.7)
    ax1.plot_surface(xx, yy, z2, alpha=0.7)
    ax1.set_title('Objective function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f1(x) and f2(x)')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.contour(xx, yy, z1, colors='b')
    ax2.contour(xx, yy, z2, colors='r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    p, = ax2.plot(X, Y, 'bo')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(z1, z2, 'b.')
    ax3.set_xlabel('f1(x)')
    ax3.set_ylabel('f2(x)')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Population at generation: 0')
    ax4.set_xlabel('f1(x)')
    ax4.set_ylabel('f2(x)')
    ax4.grid(True)
    pp, = ax4.plot(F1, F2, 'bo')

    plt.draw()
    plt.pause(0.1)

    return p, pp

def update_illustration(p: Any, pp: Any, gen: int, X: np.ndarray, Y: np.ndarray, F1: np.ndarray, F2: np.ndarray) -> None:
    """
    Updates the illustration with new generation data.

    Args:
        p (Any): Plot handle for decision variables.
        pp (Any): Plot handle for objective functions.
        gen (int): Current generation number.
        X (np.ndarray): Decision variables X.
        Y (np.ndarray): Decision variables Y.
        F1 (np.ndarray): Objective function 1 values.
        F2 (np.ndarray): Objective function 2 values.
    """
    plt.suptitle(f'Population at generation: {gen}')
    p.set_xdata(X)
    p.set_ydata(Y)
    p.set_color('b')
    pp.set_xdata(F1)
    pp.set_ydata(F2)
    pp.set_color('b')
    plt.draw()
    plt.pause(0.1)

def terminate_illustration(p: Any, pp: Any, X: np.ndarray, Y: np.ndarray, F1: np.ndarray, F2: np.ndarray, ND: int) -> None:
    """
    Terminates the illustration with final results.

    Args:
        p (Any): Plot handle for decision variables.
        pp (Any): Plot handle for objective functions.
        X (np.ndarray): Decision variables X.
        Y (np.ndarray): Decision variables Y.
        F1 (np.ndarray): Objective function 1 values.
        F2 (np.ndarray): Objective function 2 values.
        ND (int): Number of non-dominated solutions.
    """
    plt.suptitle(f'Search terminated. Number of nondominated solutions: {ND}')
    p.set_xdata(X)
    p.set_ydata(Y)
    p.set_color('k')
    pp.set_xdata(F1)
    pp.set_ydata(F2)
    pp.set_color('k')
    plt.draw()
    plt.pause(0.1)
