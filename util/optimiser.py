import numpy as np
from scipy.optimize import minimize
import pymc3 as pm
import pymc3.sampling as sampling
import theano
from sklearn.utils import resample
from scipy.misc import logsumexp

from scipy.stats import gaussian_kde
class SAA(object):
    def __init__(self,
                 objective_function,
                 constraints=(),
                 bounds=None,
                 helper_func=None):
        self.objective_function = objective_function
        self.constraints = constraints
        self.bounds = bounds
        self.helper_func = helper_func

    def solve(self,samples,initial_conditions,additional_args=(),*args,**kwargs):
        # do any preprocessing on the samples
        if self.helper_func:
            new_args = self.helper_func(samples,*args,**kwargs)
        else:
            new_args = (samples,)
        additional_args = new_args + additional_args
        res = minimize(self.objective_function,
                       initial_conditions,
                       args=additional_args,
                      constraints=self.constraints,
                      bounds=self.bounds)
        return res.x

    def __str__(self):
        return 'SAA'

class BaggingSolver(SAA):
    def __init__(self,iterations=30,*args,**kwargs):
        super(BaggingSolver,self).__init__(*args,**kwargs)
        self.iterations=iterations

    def solve(self,samples,*args,**kwargs):
        result_array = []
        sample_size = samples.shape[0]
        for i in range(self.iterations):
            #indexs = np.random.choice(range(sample_size),
            #                          size=sample_size,
            #                          replace=True)
            #bootstrapped_samples = samples[indexs]
            bootstrapped_samples = resample(samples,n_samples=sample_size)
            result_array.append(super(BaggingSolver,self).solve(bootstrapped_samples,*args,**kwargs))
        return np.mean(result_array,axis=0)

    def __str__(self):
        return 'Bagging'

class MLESolver(SAA):
    def __init__(self,distribution,parameter_bounds=None,
                n_to_sample=5000,*args,**kwargs):
        self.floc = kwargs.pop("floc",None)
        self.fscale = kwargs.pop("fscale",None)
        super(MLESolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.distribution = distribution
        self.parameter_bounds = parameter_bounds


    def solve(self,samples,*args,**kwargs):

        parameters = self.distribution.fit(samples,floc=self.floc,fscale=self.fscale)

        if self.parameter_bounds is not None:
            def beta_loglikelihood(params,x):
                log_pdf = self.distribution.logpdf(x, params[0], params[1], loc=params[2], scale=params[3])
                return -logsumexp(log_pdf,axis=0)
            parameters = minimize(beta_loglikelihood,parameters,args=(samples),
                                bounds=self.parameter_bounds,options={'maxiter':1000}).x


        new_samples = self.distribution(*parameters).rvs(self.n_to_sample)

        return super(MLESolver,self).solve(new_samples,*args,**kwargs)

    def __str__(self):
        return 'MLE'

class BetaBayesianSolver(SAA):
    def __init__(self,n_to_sample=1000,*args,**kwargs):
        super(BetaBayesianSolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.model = pm.Model()
        self.shared_data = theano.shared(np.ones(1)*0.5, borrow=True)
        with self.model:
            self.alpha_dist = pm.Uniform('alpha',lower=1.0, upper=7.0)
            self.beta_dist = pm.Uniform('beta',lower=1.0, upper=7.0)
            observed = pm.Beta('obs',alpha=self.alpha_dist,beta=self.beta_dist,
                                     observed=self.shared_data)
            self.step = pm.Metropolis()

    def solve(self,samples,loc=-1,scale=2,*args,**kwargs):
        scaled_samples = (samples - loc)/(scale)
        with self.model:

            self.shared_data.set_value(scaled_samples)
            # Sample from the posterior using the sampling method
            trace = pm.sample(self.n_to_sample, step=self.step, njobs=4,progressbar=False,cores=4,verbose=-1)
        obs = sampling.sample_ppc(trace,samples=50,model=self.model,size=500,progressbar=False)
        new_samples = np.reshape(obs['obs'],-1,1)*scale+loc
        return super(BetaBayesianSolver,self).solve(new_samples,*args,**kwargs)

    def __str__(self):
        return 'Bayesian'

class KDESolver(SAA):
    def __init__(self,n_to_sample=1000,*args,**kwargs):
        super(KDESolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample

    def solve(self,samples,*args,**kwargs):
        kde = gaussian_kde(samples)
        new_samples = kde.resample(self.n_to_sample)
        return super(KDESolver,self).solve(np.squeeze(new_samples),*args,**kwargs)

    def __str__(self):
        return 'KDE'
