import numpy as np
from scipy.optimize import minimize

class SAA(object):
    def __init__(self,
                 objective_function,
                 initial_conditions,
                 constraints=(),
                 bounds=None,
                 helper_func=None):
        self.objective_function = objective_function
        self.initial_conditions = initial_conditions
        self.constraints = constraints
        self.bounds = bounds
        self.helper_func = helper_func

    def solve(self,samples,additional_args=(),*args,**kwargs):
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
            new_args = self.helper_func(samples)
        else:
            new_args = (samples,)
        additional_args = new_args + additional_args
        res = minimize(self.objective_function,
                       self.initial_conditions,
                       args=additional_args,
                      constraints=self.constraints,
                      bounds=self.bounds)
        return res.x

class BaggingSolver(SAA):
    def __init__(self,iterations=500,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.iterations=iterations

    def solve(self,samples,*args,**kwargs):
        result_array = []
        sample_size = samples.shape[0]
        for i in range(self.iterations):
            indexs = np.random.choice(range(sample_size),
                                      size=sample_size,
                                      replace=True)
            bootstrapped_samples = samples[indexs]
            result_array.append(super().solve(bootstrapped_samples,*args,**kwargs))
        return np.mean(result_array,axis=0)

class MLESolver(SAA):
    def __init__(self,distribution,n_to_sample=5000,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.distribution = distribution

    def solve(self,samples,*args,**kwargs):
        parameters = self.distribution.fit(samples)
        print(parameters)
        new_samples = self.distribution(*parameters).rvs(self.n_to_sample)
        return super().solve(new_samples,*args,**kwargs)
