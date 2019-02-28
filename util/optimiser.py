import numpy as np
from scipy.optimize import minimize

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
        """
        if 'cov_est' not in kwargs:
            cov_est = np.cov(samples.T)
        else:
            cov_est = kwargs['cov_est']
        if 'mean_est' not in kwargs:
            mean_est = np.mean(samples,axis=0)
        else:
            mean_est = kwargs['mean_est']
            (mean_est,
                 cov_est,
                 kwargs['gamma'],
                 use_average)
        """
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
    def __init__(self,iterations=500,*args,**kwargs):
        super(BaggingSolver,self).__init__(*args,**kwargs)
        self.iterations=iterations

    def solve(self,samples,*args,**kwargs):
        result_array = []
        sample_size = samples.shape[0]
        for i in range(self.iterations):
            indexs = np.random.choice(range(sample_size),
                                      size=sample_size,
                                      replace=True)
            bootstrapped_samples = samples[indexs]
            result_array.append(super(BaggingSolver,self).solve(bootstrapped_samples,*args,**kwargs))
        return np.mean(result_array,axis=0)

    def __str__(self):
        return 'Bagging'

class MLESolver(SAA):
    def __init__(self,distribution,n_to_sample=5000,*args,**kwargs):
        super(MLESolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.distribution = distribution

    def solve(self,samples,*args,**kwargs):
        parameters = self.distribution.fit(samples)
        new_samples = self.distribution(*parameters).rvs(self.n_to_sample)
        return super(MLESolver,self).solve(new_samples,*args,**kwargs)

    def __str__(self):
        return 'MLE'
