import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import pandas as pd
import argparse

def mean_variance(weights,mean,cov,gamma,average=False):
    var = np.dot(np.dot(weights,cov),weights)
    mean = 1.0/gamma*np.dot(mean,weights)
    if average:
        return np.mean(var-mean,axis=0)
    else:
        return var-mean

def sum_weights(weights):
    return np.sum(weights)-1

def mle_mv_gaussian(mean_est,cov_est,gamma,average=False):
    init_weights = np.ones((5))/5
    constraints = [{'type':'eq','fun':sum_weights}]
    res = minimize(mean_variance,init_weights,args=(mean_est,cov_est,gamma,
                                                    average),
                  constraints=constraints,
                  bounds=[(0,None)]*5)
    return res.x

def compute_posterior(n,sample_mean,sample_cov,n_samples):
    # the priors
    mu0 = np.array([27]*5)
    m = 1
    Psi = 5*np.eye(5)
    nu0 = 5

    posterior_sigma_scale = Psi+n*sample_cov+m*n/(n+m)*np.dot(sample_mean-mu0,
                                                          (sample_mean-mu0).T)
    posterior_sigma_nu = n+nu0
    post_invwish = st.invwishart(scale=posterior_sigma_scale,
                                df=posterior_sigma_nu)
    cov_array = []
    mean_array = []
    for k in range(n_samples):

        cov_array.append(post_invwish.rvs())
        posterior_mean_mu = (n*sample_mean+m*mu0)/(m+n)
        #mean_wishart = posterior_sigma_scale/(posterior_sigma_nu-sample_mean.shape[0]-1)
        posterior_mean_sigma = 1/(m+n)*cov_array[k]
        mean_array.append(st.multivariate_normal.rvs(posterior_mean_mu,
                                                    posterior_mean_sigma))
    return mean_array,cov_array

def main(args):
    N_TRIALS = args.n_iter
    N_SAMPLES = args.samples
    file_name = args.file

    gamma = [0.1,1.0,5.0]
    N = [50,100,200]

    mu = np.array([26.11 , 25.21 , 28.90 , 28.68 , 24.18])
    covar = np.array([[3.715 , 3.730 , 4.420 , 3.606 , 3.673],
                     [3.730 , 3.908 , 4.943 , 3.732 , 3.916],
                    [4.420 , 4.943 , 8.885 , 4.378 , 5.010],
                      [3.606 , 3.732 , 4.378 , 3.930 , 3.789],
                      [3.673 , 3.916 , 5.010 , 3.789 , 4.027]])
    model = st.multivariate_normal(mean=mu,cov=covar)

    df = pd.DataFrame(columns=['w_star','method','gamma','N'])

    for l in range(N_TRIALS):
        for j in range(len(gamma)):
            for i in range(len(N)):
                n = N[i]
                samples = model.rvs(n)
                sample_mean = np.mean(samples,axis=0)
                sample_cov = np.cov(samples.T)

                # perform mle
                weights = mle_mv_gaussian(sample_mean,
                                        sample_cov,
                                        gamma[j])
                results = {'w_star': weights,'method':'SAA','gamma':gamma[j],'N':N[i]}

                mean_array,cov_array = compute_posterior(
                                        n,
                                        sample_mean,
                                        sample_cov,
                                        N_SAMPLES)
                weights = mle_mv_gaussian(
                        mean_array,
                        sample_cov,
                        gamma[j],
                        True)

                results = {'w_star': weights,'method':'Bayesian','gamma':gamma[j],'N':N[i]}
                df = df.append(results,ignore_index=True)
        df.to_csv(file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform portfolio experiments')
    parser.add_argument('--samples', dest='samples',default=2000, type=int, help='number of samples for the computation of the expectation')
    parser.add_argument('n_iter', type=int, help='number of iteration to perform experiments')
    parser.add_argument('--file', dest='file', default='portfolio_results.csv', help='name of the file to write/load')
    args = parser.parse_args()
    main(args)
