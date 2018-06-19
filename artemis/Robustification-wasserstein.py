

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.stats import wasserstein_distance
import csv
import argparse


def main(args):

    # Define the true distribution
    Y = beta(a = 3,b=3,loc=-1,scale=2)

    # the cost function we would like to minimise/maximise
    def f(x,y, weights=None,find_mean=True):
        function = 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y
        if find_mean:
            if weights is not None:
                return np.average(function,axis=0,weights=weights)
            else:
                return np.mean(function,axis=0)
        else:
            return function



    # the cost function we would like to minimise/maximise
    def f_convex(x,y, weights=None,find_mean=True):
        function = 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y -5*x
        if find_mean:
            if weights is not None:
                return np.average(function,axis=0,weights=weights)
            else:
                return np.mean(function,axis=0)
        else:
            return function


    def df_dy(x,y):
        return 12*x*y - 2*x**2 - 7*x


    def SAA_minimise(samples, func, p0=0.0):
        res = minimize(func,p0,args=(samples))
        return res.x

    def f_maximise(y,x):
        function = 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y
        return -np.mean(function,axis=0)


    def f_convex_maximise(y,x):
        function = 3*x**2 + 6*x*y**2 - 2*x**2*y - 7*x*y - 5*x
        return -np.mean(function,axis=0)


    def wasserstein_constraint(Sx,Sy,delta):
        distance = wasserstein_distance(Sx,Sy)
        return  delta-distance


    def wasserstein_robustification(y,x_star,func,constraint,bound):
        res = minimize(func,y,args=(x_star),constraints=constraint,bounds=bound)
        return res.x


    # In[99]:


    # Some experimental hyper parameters

    # number of iterations to perform each experiment
    n_iter = args.n_iter

    # number of samples for each experiment
    N = args.N

    delta = [0.01,0.015,0.02]


    # In[114]:


    #%%time
    # SAA_results = np.zeros((n_iter,len(N)))
    # wasserstein_results = np.zeros((n_iter,len(N),len(delta)))

    #start_time = timeit.default_timer()
    
    file_name = 'wasser_N'+str(N)+'.csv'
    append_style = args.style

    with open(file_name, append_style) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['SAA'] + delta)
        # define the bounds
        bound = [(-1,1)]*N
        for k in range(n_iter):
            # obtain the samples
            S = Y.rvs(N)
            S = np.sort(S)
            # obtain result for SAA
            x_star = SAA_minimise(S,f)
            # perform robustification for different delta
            wasser_results = []
            for i in range(len(delta)):
                constraint = {'type':'ineq','fun':wasserstein_constraint,'args':[S,delta[i]]}
                new_S = wasserstein_robustification(S,x_star,f_maximise,constraint,bound)
                temp = SAA_minimise(new_S,f)
                wasser_results += [float(temp)]
            #result_list = [float(x_star)] + 
            writer.writerow([float(x_star)] + wasser_results)
            if k%100 == 0:
                print "At iteration %d" %k


    # In[115]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform wasserstein experiments')
    parser.add_argument('--N', dest='N', type=int, help='number of samples taken from beta distribution')
    parser.add_argument('--n_iter', dest='n_iter', type=int, default=100, help='number of iteration to perform experiments')
    parser.add_argument('--style', dest='style', default='w', help='value or either write (w) or append (w)')
    args = parser.parse_args()
    main(args)




