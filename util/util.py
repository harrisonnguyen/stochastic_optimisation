import numpy as np
from scipy.optimize import minimize

f1 = lambda x,y: 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y
df1_dy = lambda x,y: 12*x*y - 2*x**2 - 7*x

f2 = lambda x,y: 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y -5*x

# the cost function we would like to minimise/maximise
def evaluate(x,y, func, weights=None,find_mean=True):
    function = func(x,y)
    if find_mean:
        if weights is not None:
            return np.average(function,axis=0,weights=weights)
        else:
            return np.mean(function,axis=0)
    else:
        return function
        
def df_dy(x,y):
    return df1_dy(x,y)
    
def SAA_minimise(samples, func, p0=0.0):
    res = minimize(evaluate,p0,args=(samples,func))
    return res.x
