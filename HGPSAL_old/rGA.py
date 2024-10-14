import numpy as np
import time
from penalty2 import penalty2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Define the main function rGA
def plot_population(Problem, Population, *args):
    if Problem['Verbose']:
        print('rGA is alive...')
        if Problem['Verbose'] == 2:
            fig = plt.figure(figsize=(12, 6))
            
            # Objective function surface plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Objective function')
            xx, yy = np.meshgrid(
                np.linspace(Problem['LB'][0], Problem['UB'][0], 81),
                np.linspace(Problem['LB'][1], Problem['UB'][1], 81)
            )
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
            ax2.set_title(f'Population at generation:')
            contour = ax2.contour(xx, yy, zz)
            plt.colorbar(contour)
            ax2.grid(True)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.plot(Population['x'][:, 0], Population['x'][:, 1], '.', markersize=10)
            
            plt.draw()
            plt.show()
def rGA(Problem, InitialPopulation=None, Options=None, *varargin):
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

    if not isinstance(Problem['LB'], np.ndarray) or not isinstance(Problem['UB'], np.ndarray) or \
        not Problem['LB'].any() or not Problem['UB'].any():
        raise ValueError('rGA:Bounds', 'Population relies on finite bounds on all variables.')

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

    Problem, Population = InitPopulation(Problem, InitialPopulation, Pop, *varargin)

    temp = np.hstack((Population['x'], Population['f'].reshape(-1, 1)))
    temp = temp[temp[:, -1].argsort()]
    Population['x'] = temp[:, :-1]
    Population['f'] = temp[:, -1]
    
    plot_population(Problem, Population, *varargin)


    Problem['Stats']['GenCounter'] = 0
    Problem['Stats']['Best'] = [Population['f'][0]]
    Problem['Stats']['Worst'] = [Population['f'][Pop - 1]]
    Problem['Stats']['Mean'] = [np.mean(Population['f'])]
    Problem['Stats']['Std'] = [np.std(Population['f'])]

    while Problem['Stats']['GenCounter'] < MaxGenerations and Problem['Stats']['ObjFunCounter'] < MaxEvals:
        if Problem['Stats']['GenCounter'] > 0 and Problem['Stats']['GenCounter'] % int(Problem['GenTest'] * MaxGenerations) == 0 and \
                abs(Problem['Stats']['Best'][Problem['Stats']['GenCounter']] - 
                    Problem['Stats']['Best'][Problem['Stats']['GenCounter'] - int(Problem['GenTest'] * MaxGenerations)]) < Problem['Tolerance']:
            print('Stopping due to objective function improvement inferior to CPTolerance in the last CPGenTest generations')
            break

        Problem['Stats']['GenCounter'] += 1

        elitesize = int(Pop * Elite)
        pool = Pop - elitesize
        parent_chromosome = tournament_selection(Population, pool, Tour)

        offspring_chromosome = genetic_operator(Problem, parent_chromosome, Pc, Pm, Ic, Im)

        Population['x'][elitesize:] = offspring_chromosome['x'][:pool]

        for i in range(elitesize, Pop):
            Problem, Population['f'][i] = ObjEval(Problem, Population['x'][i], *varargin)

        temp = np.hstack((Population['x'], Population['f'][:, None]))
        temp = temp[temp[:, -1].argsort()]
        Population['x'] = temp[:, :-1]
        Population['f'] = temp[:, -1]

        Problem['Stats']['Best'].append(Population['f'][0])
        Problem['Stats']['Worst'].append(Population['f'][Pop - 1])
        Problem['Stats']['Mean'].append(np.mean(Population['f']))
        Problem['Stats']['Std'].append(np.std(Population['f']))

        plot_population(Problem, Population, *varargin)
   

    elapsed_time = time.time() - start_time

    if Problem['Stats']['GenCounter'] >= MaxGenerations or Problem['Stats']['ObjFunCounter'] >= MaxEvals:
        print('Maximum number of iterations or objective function evaluations reached')

    BestChrom = Population['x'][0]
    BestChromObj = Population['f'][0]
    RunData = Problem['Stats']

    if Problem['Verbose'] and Problem['Verbose'] == 2:
        # Visualization of search process
        pass

    return BestChrom, BestChromObj, RunData

def InitPopulation(Problem,InitPopulation,Size,*args):
    # Initialize the population
    Population = {
        'x': [],
        'f': []
    }
    if InitPopulation and not isinstance(InitPopulation, list):
        raise ValueError('Initial population must be defined in a dictionary.')
    else:
        # Check for size
        if len(InitPopulation) > Size:
            # User provided an initial population greater than the parent population size
            raise ValueError('Initial population size must be less than or equal to PopSize.')
        
        
        
       
        for i in range(len(InitPopulation)):
            x = Bounds(InitPopulation[i]['x'], Problem['LB'][:Problem['Variables']], Problem['UB'][:Problem['Variables']])
            Problem,f = ObjEval(Problem, x, *args)
            Population['x'].append(x)
            Population['f'].append(f)
        # Randomly generate the remaining population
    for i in range(len(InitPopulation), Size):
        x = Problem['LB'][:Problem['Variables']] + \
                           (Problem['UB'][:Problem['Variables']] - Problem['LB'][:Problem['Variables']]) * np.random.rand(1, Problem['Variables'])
        #x = Problem['LB'][:Problem['Variables']] + (Problem['UB'][:Problem['Variables']] - Problem['LB'][:Problem['Variables']]) * np.random.rand(Problem['Variables'])
        x= x.flatten()
        Problem,f = ObjEval(Problem, x, *args)
        Population['f'].append(f)
        Population['x'].append(x)

    Population['f'] = np.array(Population['f'])
    Population['x'] = np.array(Population['x'])

    return Problem, Population
    
def tournament_selection(chromosomes, pool_size, tour_size):
    pop = chromosomes['x'].shape[0]
    P = {'x': [], 'f': []}
    for i in range(pool_size):
        candidates = np.random.choice(pop, tour_size, replace=False)
        fitness = chromosomes['f'][candidates]
        min_candidate = candidates[np.argmin(fitness)]
        P['x'].append(chromosomes['x'][min_candidate])
        P['f'].append(chromosomes['f'][min_candidate])
    P['x'] = np.array(P['x'])
    P['f'] = np.array(P['f']).reshape
    return P

def genetic_operator(Problem, parent_chromosome, pc, pm, mu, mum):
    N, V = parent_chromosome['x'].shape

    p = 0
    child = np.zeros((N, V))
    while p < N:
        # SBX (Simulated Binary Crossover) applied with probability pc
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
                child_1[j] = 0.5 * (((1 + bq) * parent_1[j]) + (1 - bq) * parent_2[j])
                child_2[j] = 0.5 * (((1 - bq) * parent_1[j]) + (1 + bq) * parent_2[j])
            
            child_1 = Bounds(child_1, Problem['LB'], Problem['UB'])
            child_2 = Bounds(child_2, Problem['LB'], Problem['UB'])
        else:
            child_1 = parent_1
            child_2 = parent_2

        # Polynomial mutation applied with probability pm
        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                child_1[j] = child_1[j] + (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_1 = Bounds(child_1, Problem['LB'], Problem['UB'])

        for j in range(V):
            if np.random.rand() < pm:
                r = np.random.rand()
                if r < 0.5:
                    delta = (2 * r) ** (1 / (mum + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - r)) ** (1 / (mum + 1))
                child_2[j] = child_2[j] + (Problem['UB'][j] - Problem['LB'][j]) * delta
            child_2 = Bounds(child_2, Problem['LB'], Problem['UB'])

        child[p, :] = child_1[:V]
        child[p + 1, :] = child_2[:V]
        p += 2

    P = {'x': child}
    return P
def Bounds(X, L, U):
    X = np.clip(X, L, U)
    return X

def ObjEval(Problem, x, *varargin):
    try:
        ObjValue = Problem['ObjFunction'](x, *varargin)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise ValueError('rGA:ObjectiveError', f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return Problem, ObjValue

def lag(x, Problem, alg):
    Value = penalty2(Problem, x, alg)
    return Value['la']
def GetOption(Option, Options, DefaultOpt):
    if Options is None or not isinstance(Options, dict):
        return DefaultOpt[Option]
    return Options.get(Option, DefaultOpt[Option])
