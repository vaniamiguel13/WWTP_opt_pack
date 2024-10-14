import numpy as np
import time

def HJ(Problem, x0, delta=None, Options=None, *varargin):
    DefaultOpt = {'MaxObj': 2000, 'MaxIter': 200, 'DeltaTol': 1e-6, 'Theta': 0.5}
    HJVersion = '0.1'

    if Problem is None or x0 is None:
        raise ValueError('HJ:AtLeastOneInput', 'HJ requests at least two inputs (Problem definition and initial approximation).')

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

    Problem['Stats'] = {'Algorithm': 'Hooke and Jeeves', 'Iterations': 0, 'ObjFunCounter': 0}

    x = Projection(Problem, x0)
    fx, Problem = ObjEval(Problem, x, *varargin)

    e = np.eye(len(x0))

    s = np.zeros_like(x0)
    rho = 0

    while np.linalg.norm(delta) > DelTol and Problem['Stats']['ObjFunCounter'] < MaxEval and Problem['Stats']['Iterations'] < MaxIt:
        s, Problem = Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *varargin)
        x_trial = x + s
        x_trial = Projection(Problem, x_trial)
        fx1, Problem = ObjEval(Problem, x_trial, *varargin)
        rho = fx - fx1
        if rho > 0:
            x = x_trial
            fx = fx1
        else:
            delta = delta * theta
        Problem['Stats']['Iterations'] += 1

    if Problem['Stats']['Iterations'] >= MaxIt:
        Problem['Stats']['Message'] = 'HJ: Maximum number of iterations reached'
    if Problem['Stats']['ObjFunCounter'] >= MaxEval:
        Problem['Stats']['Message'] = 'HJ: Maximum number of objective function evaluations reached'
    if np.linalg.norm(delta) <= DelTol:
        Problem['Stats']['Message'] = 'HJ: Stopping due to step size norm inferior to tolerance'
    
    print(Problem['Stats']['Message'])
    Problem['Stats']['Time'] = time.time() - start_time
    RunData = Problem['Stats']

    return x, fx, RunData

def Exploratory_Moves(Problem, s, delta, e, x, fx, rho, *varargin):
    if rho > 0:
        min_val, Problem = ObjEval(Problem, x + s, *varargin)
        rho = fx - min_val
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_val, rho, *varargin)
    if rho <= 0:
        s = np.zeros_like(s)
        rho = 0
        min_val = fx
        s, rho, Problem = Coordinate_Search(Problem, s, delta, e, x, min_val, rho, *varargin)
    return s, Problem

def Coordinate_Search(Problem, s, delta, e, x, min_val, rho, *varargin):
    for i in range(len(x)):
        s1 = s + delta * e[:, i]
        x1 = x + s1
        x1 = Projection(Problem, x1)
        fx1, Problem = ObjEval(Problem, x1, *varargin)

        if fx1 < min_val:
            rho = min_val - fx1
            min_val = fx1
            s = s1
        else:
            s1 = s - delta * e[:, i]
            x1 = x + s1
            x1 = Projection(Problem, x1)
            fx1, Problem = ObjEval(Problem, x1, *varargin)

            if fx1 < min_val:
                rho = min_val - fx1
                min_val = fx1
                s = s1
    return s, rho, Problem

def GetOption(Option, Options, DefaultOpt):
    return Options.get(Option, DefaultOpt[Option])

def Projection(Problem, x):
    for i in range(len(x)):
        if x[i] < Problem['LB'][i]:
            x[i] = Problem['LB'][i]
        if x[i] > Problem['UB'][i]:
            x[i] = Problem['UB'][i]
    return x

def ObjEval(Problem, x, *varargin):
    try:
        ObjValue = Problem['ObjFunction'](x, *varargin)
        Problem['Stats']['ObjFunCounter'] += 1
    except Exception as e:
        raise ValueError('rGA:ObjectiveError', f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return ObjValue, Problem
