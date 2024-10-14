def Rast_constr(x):
    c = [(x[0] - 2.5) ** 2 + (x[1] - 2.5) ** 2 - 4]
    ceq = [x[0] + x[1] - 7]
    return c, ceq