import numpy as np
import scipy.stats as st
import pymc3 as pm
import pandas as pd
import pymc3.sampling as sampling
import argparse
from scipy.optimize import minimize
def bayesian_beta(samples,loc=-1,scale=2,n_samples=1000):
    # known scaling
    scaled_samples = (samples - loc)/(scale)
    with pm.Model() as beta_model:
        alpha_dist = pm.Uniform('alpha',lower=1.0, upper=7.0)
        beta_dist = pm.Uniform('beta',lower=1.0, upper=7.0)
        observed = pm.Beta('obs',alpha=alpha_dist,beta=beta_dist,
                             observed=scaled_samples)
        #map_estimate = pm.find_MAP(model=beta_model)

        # Using Metropolis Hastings Sampling
        step = pm.Metropolis()

        # Sample from the posterior using the sampling method
        beta_trace = pm.sample(n_samples, step=step, njobs=4,progressbar=False,cores=4)
    return beta_model,beta_trace

def main(args):
    N_TRIALS = args.n_iter
    N_MCMC_SAMPLES = args.mcmc_samples
    file_name = args.file


    try:
        df = pd.read_csv(file_name,index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['method','N','x_star'])

    Y1 = st.beta(a = 2,b=2,loc=-1,scale=2)
    f1 = lambda x,y: -x*y + 5*x**2 + 2*x
    # concave function in x
    f1_mean = lambda x,y: np.mean(f1(x,y),axis=0)

    N = [10,20,50]
    for i in range(N_TRIALS):
        for j in range(len(N)):
            samples = Y1.rvs(N[j])
            # compute saa

            res = minimize(f1_mean,0.0,args=(samples))
            results ={'method':'SAA','N':N[j],'x_star':res.x[0]}
            df = df.append(results,ignore_index=True)

            # compute bayesuian
            model,trace = bayesian_beta(samples,n_samples=N_MCMC_SAMPLES)
            #new_samples = np.array(run_ppc(trace, samples=200, model=model)['obs'])
            obs = sampling.sample_ppc(trace,samples=50,model=model,size=500,progressbar=False)
            new_samples = np.reshape(obs['obs'],-1,1)
            res = minimize(f1_mean,res.x,args=(new_samples*2-1))

            results ={'method':'Bayes','N':N[j],'x_star':res.x[0]}
            df = df.append(results,ignore_index=True)
        df.to_csv(file_name)

    samples = Y1.rvs(7000)
    # compute saa
    res = minimize(f1_mean,0.0,args=(samples))
    true = res.x[0]
    df = df.append(results,ignore_index=True)

    samples = Y1.rvs(10000)
    func = lambda x: np.mean(f1(x,samples))
    df.loc[:,'expected_cost'] = df.loc[:,'x_star'].apply(func)

    df.to_csv(file_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform wasserstein experiments')
    parser.add_argument('--mcmc_samples', dest='mcmc_samples',default=1000, type=int, help='number of samples for the mcmc trace')
    parser.add_argument('n_iter', type=int, help='number of iteration to perform experiments')
    parser.add_argument('--file', dest='file', default='bayes_results.csv', help='name of the file to write/load')
    args = parser.parse_args()
    main(args)
