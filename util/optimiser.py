import numpy as np
from scipy.optimize import minimize
import pymc3 as pm
import pymc3.sampling as sampling
import theano
from sklearn.utils import resample
from scipy.misc import logsumexp
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import scipy.stats as st
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

class GMMSolver(SAA):
    def __init__(self,is_bayesian,n_components,covariance_type="spherical",
                n_to_sample=5000,*args,**kwargs):
        super(GMMSolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.is_bayesian=is_bayesian
        if is_bayesian:
            self.distribution = BayesianGaussianMixture(n_components=n_components,
                                                       covariance_type= covariance_type)
        else:
            self.distribution = GaussianMixture(n_components=n_components,
                                                       covariance_type= covariance_type)


    def solve(self,samples,*args,**kwargs):

        self.distribution.fit(samples.reshape(-1, 1))

        new_samples = self.distribution.sample(self.n_to_sample)[0].reshape(-1)

        return super(GMMSolver,self).solve(new_samples,*args,**kwargs)

    def __str__(self):
        if self.is_bayesian:
            return "Bayesian"
        else:
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

class MvGaussianSolver(SAA):
    def __init__(self,n_to_sample=1000,*args,**kwargs):
        super(MvGaussianSolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        # the priors
        self.mu0 = np.array([0]*5)
        self.m = 1
        self.Psi = 5*np.eye(5)
        self.nu0 = 5


    def solve(self,samples,initial_conditions,
                additional_args=None,*args,**kwargs):
        n = samples.shape[0]
        sample_mean = np.mean(samples,axis=0)
        sample_cov = np.cov(samples.T)

        posterior_sigma_scale = self.Psi+n*sample_cov+self.m*n/(n+self.m)*np.dot(sample_mean-self.mu0,(sample_mean-self.mu0).T)
        posterior_sigma_nu = n+self.nu0
        post_invwish = st.invwishart(scale=posterior_sigma_scale,
                                    df=posterior_sigma_nu)
        cov_array = []
        mean_array = []
        for k in range(self.n_to_sample):

            cov_array.append(post_invwish.rvs())
            posterior_mean_mu = (n*sample_mean+self.m*self.mu0)/(self.m+n)
            #mean_wishart = posterior_sigma_scale/(posterior_sigma_nu-sample_mean.shape[0]-1)
            posterior_mean_sigma = 1/(self.m+n)*cov_array[k]
            mean_array.append(st.multivariate_normal.rvs(posterior_mean_mu,
                                                        posterior_mean_sigma))
        args = (mean_array,cov_array)
        if additional_args:
            args = args + additional_args
        res = minimize(self.objective_function,
                        initial_conditions,
                        constraints=self.constraints,
                        bounds = self.bounds,
                        args=args)

        return res.x

    def __str__(self):
        return 'Bayesian'

class MvStudentTBayesianSolver(SAA):
    def __init__(self,n_to_sample=2000,*args,**kwargs):
        super(MvStudentTBayesianSolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample
        self.model = pm.Model()
        self.shared_data = theano.shared(np.zeros((5,5))*0.5, borrow=True)
        with self.model:
            sd_dist = pm.HalfCauchy.dist(beta=2.5)
            packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=5, sd_dist=sd_dist)
            chol = pm.expand_packed_triangular(5, packed_chol, lower=True)
            cov = pm.Deterministic('cov', theano.dot(chol,chol.T))
            self.mu_dist = pm.MvNormal("mu",mu=np.zeros(5),
                            chol=chol, shape=5)
            observed = pm.MvStudentT('obs',nu=7.0,
                                    mu=self.mu_dist,
                                    chol=chol,
                                     observed=self.shared_data)
            self.step = pm.Metropolis()

    def solve(self,samples,initial_conditions,additional_args,*args,**kwargs):
        with self.model:

            self.shared_data.set_value(samples)
            # Sample from the posterior using the sampling method
            trace = pm.sample(self.n_to_sample, step=self.step, njobs=4,progressbar=False,cores=4,verbose=-1)
        args = (trace["mu"][1000:],trace["cov"][1000:])
        if additional_args:
            args = args + additional_args
        res = minimize(self.objective_function,
                        initial_conditions,
                        constraints=self.constraints,
                        bounds = self.bounds,
                        args=args)

        return res.x

    def __str__(self):
        return 'Bayesian'

class KDESolver(SAA):
    def __init__(self,n_to_sample=1000,*args,**kwargs):
        super(KDESolver,self).__init__(*args,**kwargs)
        self.n_to_sample = n_to_sample

    def solve(self,samples,*args,**kwargs):
        if len(samples.shape)==2:
            samples = samples.T
        self.kde = gaussian_kde(samples)
        new_samples = self.kde.resample(self.n_to_sample)
        return super(KDESolver,self).solve(np.squeeze(new_samples.T),*args,**kwargs)

    def __str__(self):
        return 'KDE'
