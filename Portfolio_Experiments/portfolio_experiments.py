import sys
sys.path.insert(0, '../')
import numpy as np
import scipy.stats as st
from stochastic_optimisation.util import optimiser
from stochastic_optimisation.util import distribution
import argparse
import os
import pandas as pd

def mean_variance(weights,mean,cov,gamma,average=False):
    var = np.dot(np.dot(weights,cov),weights)
    mean = 1.0/gamma*np.dot(mean,weights)
    if average:
        return np.mean(var-mean,axis=0)
    else:
        return var-mean

def compute_mean_variance(samples,values=()):
    if not values:
        mean = np.mean(samples,axis=0)
        cov = np.cov(samples.T)
        return (mean,cov)
    else:
        return values

def sum_weights(weights):
    return np.sum(weights)-1

def fix_string(x):
    return np.array([float(z) for z in x[1:-1].split(' ') if z is not ""])

def main(args):
    file_name = os.path.join("Portfolio_Experiments","covar"+str(args.covariance) +"_" + args.file)
    n_iter = args.n_iter
    n_samples = args.samples

    mu = np.array([0.0,0.0,0.0,0.0,0.0])

    # Define the true distribution
    covar = np.loadtxt(os.path.join("Portfolio_Experiments",
                                    "covar"+str(args.covariance)+".csv"),
                            delimiter=",")
    covar = np.triu(covar.T,1) + covar

    dist = distribution.Multivariate_T(mu,covar,3.5)

    N = [50,100,200]
    try:

        df = pd.read_csv(file_name,index_col=0)
        df['w_star'] = df['w_star'].apply(lambda x:fix_string(x))

    except:
        df = pd.DataFrame(columns=['w_star','method',
                        'N',"distribution"])
        df.to_csv(file_name)
    constraints = [{'type':'eq','fun':sum_weights}]
    gamma = 1
    init_weights = np.ones((5))/5
    bounds = [(0,None)]*5

    saa = optimiser.SAA(objective_function=mean_variance,
                    constraints=constraints,
                    bounds=bounds,
                   helper_func=compute_mean_variance)
    bagger = optimiser.BaggingSolver(iterations=200,
                                 objective_function=mean_variance,
                                constraints=constraints,
                                 bounds=bounds,
                                helper_func=compute_mean_variance)
    mle = optimiser.MLESolver(distribution.Multivariate_T,
                         objective_function=mean_variance,
                                constraints=constraints,
                                 bounds=bounds,
                                helper_func=compute_mean_variance,
                                n_to_sample=n_samples)
    kde = optimiser.KDESolver(n_to_sample=n_samples,
                              objective_function=mean_variance,
                              constraints=constraints,
                              bounds=bounds,
                              helper_func=compute_mean_variance,)
    #bayes = optimiser.MvGaussianSolver(n_to_sample=int(n_samples/10),
    #                          objective_function=mean_variance,
    #                        constraints=constraints,
    #                          bounds=bounds,)
    bayes = optimiser.MvStudentTBayesianSolver(n_to_sample=int(n_samples/10),
                              objective_function=mean_variance,
                            constraints=constraints,
                              bounds=bounds,)
    methods = [bagger,mle,saa,kde,bayes]
    for k in range(n_iter):
        for j in range(len(N)):
            samples = dist.rvs(N[j])
            for ele in methods:

                if ele.__str__() == "Bayesian":
                    w_star = ele.solve(
                                samples,
                                initial_conditions=init_weights,
                                additional_args=(gamma,True))
                else:

                    w_star = ele.solve(
                            samples,
                            initial_conditions=init_weights,
                            additional_args=(gamma,False))
                results = {'w_star': w_star,
                            'method':ele.__str__(),
                            'N':N[j],
                            "distribution":args.covariance}
                df = df.append(results,ignore_index=True)
        # save the results so far
        df.to_csv(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform portfolio experiments')
    parser.add_argument('--samples', dest='samples',default=10000, type=int, help='number of samples for the computation of the expectation')
    parser.add_argument('n_iter', type=int, help='number of iteration to perform experiments')
    parser.add_argument('--file', dest='file', default='portfolio_results.csv', help='name of the file to write/load')
    parser.add_argument('--covariance', dest='covariance', help='underlying true distribution covariance choice')
    args = parser.parse_args()
    main(args)
