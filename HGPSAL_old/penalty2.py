import numpy as np

def penalty2(Problem, x, alg):
    Value = {}
    Value['fx'] = ObjEval(Problem, x)
    Value['c'], Value['ceq'] = ConsEval(Problem, x)

    term1 = np.sum(alg['lambda'] * Value['ceq'])
    term2 = np.sum(Value['ceq'][0] ** 2)
    term3 = np.sum(np.maximum(0, alg['ldelta'] + Value['c'][0] / alg['miu']) ** 2 - alg['ldelta'] ** 2)
    Value['la'] = Value['fx'] + term1 + term2 / (2 * alg['miu']) + alg['miu'] * term3 / 2

    return Value

def ObjEval(Problem, x, *varargin):
    try:
        ObjValue = Problem['ObjFunction'](x, *varargin)
    except Exception as e:
        raise ValueError(f'augLagr:ObjectiveError', f'Cannot continue because user supplied objective function failed with the following error:\n{e}')
    return ObjValue

def ConsEval(Problem, x, *varargin):
    try:
        c, ceq = Problem['Constraints'](x, *varargin)
    except Exception as e:
        raise ValueError(f'augLagr:ConstraintsError', f'Cannot continue because user supplied function constraints failed with the following error:\n{e}')
    return c, ceq
