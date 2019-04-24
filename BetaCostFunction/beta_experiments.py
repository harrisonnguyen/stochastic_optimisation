import sys
sys.path.insert(0, '../')
import numpy as np
import scipy.stats as st
from stochastic_optimisation.util import optimiser
import argparse
import os
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture

import pandas as pd

def f_mean(*args,**kwargs):
    return np.mean(f(*args,**kwargs),axis=0)

def f(x,y,alpha,beta,gamma):
    return x**2+alpha*x**2*y+beta*x*y**2+gamma*x*y

def main(args):
    file_name = os.path.join("BetaCostFunction",args.dist + "_" + args.file)
    n_iter = args.n_iter
    n_samples = args.samples


    if args.dist == "beta1":
        # Define the true distribution
        dist = st.beta(a = 2,b=2,loc=-1,scale=2)
    elif args.dist == "beta2":
        # Define the true distribution
        dist = st.beta(a = 5,b=5,loc=-1,scale=2)
    elif args.dist == "beta3":
        # Define the true distribution
        dist = st.beta(a = 2,b=5,loc=-1,scale=2)
    elif "gmm" in args.dist:
        if args.dist=="gmm1":
            mu1=-0.5
            sigma1=0.15
            mu2=0.4
            sigma2=0.3
            p=0.6
        elif args.dist=="gmm2":
            mu1=-0.1
            sigma1=0.3
            mu2=0.4
            sigma2=0.1
            p=0.7
        dist = GaussianMixture(n_components=2,covariance_type="spherical")
        dist.weights_=np.array([p,1-p])
        dist.means_=np.array([[mu1],[mu2]])
        dist.covariances_=np.array([sigma1**2,sigma2**2])
        dist.precisions_cholesky_ =  np.linalg.cholesky(np.linalg.inv([[sigma1**2,0],[0,sigma2**2]]))


    chosen_functions = [0,1,2,4,5,7,9,15,16,18]
    parameter_bounds =None
    #parameter_bounds =((1,None),(1,None),(None,None),(None,None))
    cost_coefs = pd.read_csv(args.coef_file)

    N = [10,25,50]
    try:
        df = pd.read_csv(file_name,index_col=0)
    except:
        df = pd.DataFrame(columns=['w_star','method','function',
                        'N',"distribution"])
        df.to_csv(file_name)

    for z in range(len(chosen_functions)):
        coefs = cost_coefs.iloc[chosen_functions[z]].values
        new_f = lambda x,y: f_mean(x,y,coefs[0],coefs[1],coefs[2])
        saa = optimiser.SAA(new_f)
        bagging = optimiser.BaggingSolver(400,
                                          objective_function=new_f)
        if "gmm" in args.dist:
            mle = optimiser.GMMSolver(False,2,
                              objective_function=new_f,
                             n_to_sample=n_samples)
            bayes = optimiser.GMMSolver(True,2,
                              objective_function=new_f,
                             n_to_sample=n_samples)
        else:
            mle = optimiser.MLESolver(st.beta,
                                      parameter_bounds=parameter_bounds,
                                      objective_function=new_f,
                                     n_to_sample=n_samples,
                                     floc=-1,
                                     fscale=2)
            bayes = optimiser.BetaBayesianSolver(objective_function=new_f,
                                                n_to_sample=n_samples)
        kde = optimiser.KDESolver(n_samples,
                                  objective_function=new_f)
        methods = [bagging,mle,saa,kde,bayes]
        for k in range(n_iter):
            for j in range(len(N)):
                if "gmm" in args.dist:
                    samples = dist.sample(N[j])[0].reshape(-1)
                else:
                    samples = dist.rvs(N[j])
                for ele in methods:
                    w_star = ele.solve(samples,initial_conditions=0.0)[0]
                    results = {'w_star': w_star,'method':ele.__str__(),
                           'function':chosen_functions[z]+1,'N':N[j],
                           "distribution":args.dist}
                    df = df.append(results,ignore_index=True)
            # save the results so far
            df.to_csv(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform portfolio experiments')
    parser.add_argument('--samples', dest='samples',default=10000, type=int, help='number of samples for the computation of the expectation')
    parser.add_argument('n_iter', type=int, help='number of iteration to perform experiments')
    parser.add_argument('--file', dest='file', default='experiment_results.csv', help='name of the file to write/load')
    parser.add_argument('--dist', dest='dist', help='underlying true distribution')
    parser.add_argument('--coef_file', dest='coef_file', default='BetaCostFunction/cost_function_coeffs.csv', help='file where coefficients are located')
    args = parser.parse_args()
    main(args)
