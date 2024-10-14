import numpy as np

def ZDT1(x):
    f1 = x[0]
    
    g = 1 + 9 * (np.sum(x) - x[0]) / (len(x) - 1)
    
    f2 = g * (1 - np.sqrt(x[0] / g))
    
    return np.array([f1, f2])
