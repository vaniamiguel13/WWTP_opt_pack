import numpy as np

def zdt1_con(x):
    c = [(x[0] ** 2 - 0.5)]
    ceq = []
    # ceq = [x[0] + x[1] - 0.5]  # Uncomment if needed
    return np.array(c), np.array(ceq)
