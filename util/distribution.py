import numpy as np
from scipy.optimize import minimize
from scipy import special
class Multivariate_T(object):
    def __init__(self,mu,sigma,nu):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        nu = degrees of freedom
        '''
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    def rvs(self,M):
        '''
           M = # of samples to produce
        '''
        d = len(self.sigma)
        g = np.tile(np.random.gamma(self.nu/2.,2./self.nu,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),self.sigma,M)
        return self.mu + Z/np.sqrt(g)

    def logpdf(self,samples):
        n_dim = self.sigma.shape[0]
        try:
            cv_chol = linalg.cholesky(self.sigma, lower=True)
        except linalg.LinAlgError:
            raise ValueError("'covars' must be symmetric, "
                         "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (samples - self.mu).T, lower=True).T

        norm = (sp_gammaln((self.nu + n_dim) / 2.) - sp_gammaln(self.nu / 2.)
                - 0.5 * n_dim * np.log(self.nu * np.pi))
        inner = - (self.nu + n_dim) / 2. * np.log1p(np.sum(cv_sol ** 2, axis=1) / nu)
        log_prob = norm + inner - cv_log_det

        return log_prob

    def fit(samples,dof=7.0,iter=500,eps=1e-6,*args,**kwargs):
        D = samples.shape[1]
        N = samples.shape[0]
        cov = np.cov(samples,rowvar=False)
        mean = samples.mean(axis=0)
        mu = samples - mean[None,:]
        delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
        z = (dof + D) / (dof + delta)
        obj = [
            -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
            -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
        ]

        # iterate
        for i in range(iter):
            # M step
            mean = (samples * z[:,None]).sum(axis=0).reshape(-1,1) / z.sum()
            mu = samples - mean.squeeze()[None,:]
            cov = np.einsum('ij,ik->jk', mu, mu * z[:,None])/N

            # E step
            #delta = (mu * np.linalg.solve(cov,mu.T).T).sum(axis=1)
            delta = np.einsum('ij,ij->i', mu, np.linalg.solve(cov,mu.T).T)
            z = (dof + D) / (dof + delta)

            # store objective
            obj.append(
                -N*np.linalg.slogdet(cov)[1]/2 - (z*delta).sum()/2 \
                -N*special.gammaln(dof/2) + N*dof*np.log(dof/2)/2 + dof*(np.log(z)-z).sum()/2
            )

            if np.abs(obj[-1] - obj[-2]) < eps:
                break
        return mean.squeeze(),cov,dof
