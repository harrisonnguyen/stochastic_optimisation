import sys
sys.path.insert(0, '../')
import numpy as np
import scipy.stats as st
from util import optimiser

import pandas as pd

def f_mean(*args,**kwargs):
    return np.mean(f(*args,**kwargs),axis=0)

def f(x,y,alpha,beta,gamma):
    return x**2+alpha*x**2*y+beta*x*y**2+gamma*x*y

def main():
    cost_coefs = pd.read_csv("cost_function_coeffs.csv")
    chosen_functions = [0,1,2,4,5,7,9,15,16,18]
    # Define the true distribution
    beta = st.beta(a = 2,b=2,loc=-1,scale=2)

    N = [10,25,50]
    file_name = "beta_experiments_final2.csv"
    try:
        df = pd.read_csv(file_name,index_col=0)
    except:
        df = pd.DataFrame(columns=['w_star','method','function','N'])
        df.to_csv(file_name)


    n_iter = 200
    parameter_bounds =((1,None),(1,None),(-1,-1),(2,2))
    for z in range(len(chosen_functions)):
        coefs = cost_coefs.iloc[chosen_functions[z]].values
        print(coefs)
        new_f = lambda x,y: f_mean(x,y,coefs[0],coefs[1],coefs[2])
        saa = optimiser.SAA(new_f)
        bagging = optimiser.BaggingSolver(400,
                                          objective_function=new_f)
        mle = optimiser.MLESolver(st.beta,
                                  parameter_bounds=parameter_bounds,
                                  objective_function=new_f,
                                 n_to_sample=10000)
        bayes = optimiser.BetaBayesianSolver(objective_function=new_f,
                                            n_to_sample=10000)
        kde = optimiser.KDESolver(10000,
                                  objective_function=new_f)
        methods = [bagging,mle,saa,kde,bayes]
        for k in range(n_iter):
            for j in range(len(N)):
                samples = beta.rvs(N[j])
                for ele in methods:
                    w_star = ele.solve(samples,initial_conditions=0.0)[0]
                    results = {'w_star': w_star,'method':ele.__str__(),
                           'function':chosen_functions[z]+1,'N':N[j]}
                    df = df.append(results,ignore_index=True)
            # save the results so far
            df.to_csv(file_name)
if __name__ == '__main__':
    main()
