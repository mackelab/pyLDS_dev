from __future__ import division

import numpy as np

from pybasicbayes.distributions import Regression
from autoregressive.distributions import AutoRegression
from pybasicbayes.util.general import any_none

# TODO: fix the resample() functionality to draw A, Sigma from priors
#       also for diagonal sigma (currently would sample full matrices
#       even if flag diag_sigma = True).

class Regression_diag(Regression):
    def __init__(
            self, nu_0=None,S_0=None,M_0=None,K_0=None,
            affine=False,
            A=None,sigma=None):

        self.affine = affine

        self.diag_sigma = self._check_shapes(A, sigma, nu_0, S_0, M_0, K_0)

        self.A = A
        
        if self.diag_sigma:
            self.sigma  = np.diag(sigma)
            self.dsigma = sigma
        else: 
            self.sigma  = sigma
            self.dsigma = None

        have_hypers = not any_none(nu_0,S_0,M_0,K_0)

        if have_hypers:
            self.natural_hypparam = self.mf_natural_hypparam = \
                self._standard_to_natural(nu_0,S_0,M_0,K_0)

        if A is sigma is None and have_hypers:
            self.resample()  # initialize from prior

    @staticmethod
    def _check_shapes(A, sigma, nu, S, M, K):
        is_2d = lambda x: isinstance(x, np.ndarray) and x.ndim == 2
        is_1d = lambda x: isinstance(x, np.ndarray) and x.ndim == 1
        not_none = lambda x: x is not None
        assert all(map(is_2d, filter(not_none, [A, S, M, K]))), \
            'Matrices must be 2D'
        assert sigma is None or is_2d(sigma) or is_1d(sigma), \
            'sigma must give full 2D matrix or its diagonal (1D)'

        get_dim = lambda x, i: x.shape[i] if x is not None else None
        get_dim_list = lambda pairs: filter(not_none, map(get_dim, *zip(*pairs)))
        is_consistent = lambda dimlist: len(set(dimlist)) == 1
        dims_agree = lambda pairs: is_consistent(get_dim_list(pairs))
        assert dims_agree([(A, 1), (M, 1), (K, 0), (K, 1)]), \
            'Input dimensions not consistent'
        assert dims_agree([(A, 0), (sigma, 0), (S, 0), (S, 1), (M, 0)]), \
            'Output dimensions not consistent'

        # set the diag_sigmaonal flag only if provided sigma is explicitly 1D

        if is_2d(sigma): 
            assert sigma.shape[0] == sigma.shape[1], \
                'sigma is not a square matrix'
            return False
        elif sigma is None:
            return False
        else:
            return True

    def max_likelihood(self,data,weights=None,stats=None, idx_grp=None):
        if stats is None:
            stats = self._get_statistics(data) if weights is None \
                else self._get_weighted_statistics(data,weights)
        try:

            n = stats[3]
            if isinstance(n, np.ndarray):

                self._ml_stitch(stats, idx_grp)

            elif not n > 0:
                self.broken = True
                return self

            else:
                self._ml_affine(stats) if self.affine else self._ml_linear(stats)

        except np.linalg.LinAlgError:            
            self.broken = True


        #assert np.allclose(self.sigma,self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        self._initialize_mean_field()

        return self

    def _ml_linear(self, stats):

        yyT, yxT, xxT, n = stats

        def symmetrize(A):
            return (A + A.T)/2.

        self.A = np.linalg.solve(xxT, yxT.T).T

        if self.diag_sigma:

            self.dsigma = (yyT - np.sum(self.A * yxT,1))/n

            self.sigma = np.diag(self.dsigma)

        else:

            self.sigma = (yyT - self.A.dot(yxT.T))/n
            self.sigma = 1e-10*np.eye(self.D_out) \
                + symmetrize(self.sigma)  # numerical

    def _ml_affine(self, stats):

        yyT, yxT, xxT, n = stats

        def symmetrize(A):
            return (A + A.T)/2.

        yxT, y = yxT[:,:-1], yxT[:,-1]
        xxT, x = xxT[:-1, :-1], xxT[:-1, -1]

        A = np.linalg.solve(xxT - np.outer(x,x)/n, (yxT - np.outer(y,x)/n).T).T
        b = (y - A.dot(x))/n
        self.A = np.hstack([A,np.atleast_2d(b).T])

        if self.diag_sigma:

            self.dsigma = (yyT \
                        - 2 * y * b \
                        + 2 * np.sum(A * (np.outer(b,x) - yxT),1)
                        + np.einsum('ij,ik,jk->i', A, A,xxT) )/n \
                        + b * b

            self.sigma = np.diag(self.dsigma)

        else:

            self.sigma = (yyT \
                        - 2 * symmetrize(np.outer(y, b)) \
                        - 2 * symmetrize( (yxT - np.outer(b,x)).dot(A.T) ) \
                        + A.dot(xxT.dot(A.T)) )/n \
                        + np.outer(b,b)


    def _ml_stitch(self, stats, idx_grp):

        yyT, yxT, xxT, n = stats

        yxT, y = yxT[:,:-1], yxT[:,-1]
        xxT, x = xxT[:-1, :-1, :], xxT[:-1, -1, :]

        A = np.empty((self.D_out,self.D_in-1))    
        b = np.empty(self.D_out)

        AxxTAT = np.zeros(self.D_out)
        bxT = np.zeros((self.D_out,self.D_in-1))

        for j in range(len(idx_grp)):
            ixg  = idx_grp[j]

            tmp = np.linalg.solve(xxT[:,:,j] - np.outer(x[:,j], x[:,j]) / n[j],
                (yxT[ixg,:] - np.outer(y[ixg], x[:,j]) / n[j]).T).T

            A[ixg,:] = np.linalg.solve(xxT[:,:,j] - np.outer(x[:,j], x[:,j]) / n[j],
                (yxT[ixg,:] - np.outer(y[ixg], x[:,j]) / n[j]).T).T
            b[ixg] = (y[ixg] - A[ixg,:].dot(x[:,j])) / n[j]


            bxT[ixg,:] += np.outer(b[ixg],x[:,j])
            AxxTAT[ixg] += np.einsum('ij,ik,jk->i', A[ixg,:], A[ixg,:], xxT[:,:,j])

        self.A = np.hstack([A,np.atleast_2d(b).T])

        self.dsigma = yyT - 2 * y * b + AxxTAT + 2 * np.sum(A * (bxT-yxT),1)
        for i in range(len(idx_grp)):
            self.dsigma[idx_grp[i]] /= n[i]
        self.dsigma += b*b
        
        self.dsigma += 1e-10*np.ones(self.D_out) # numerical
        self.sigma = np.diag(self.dsigma)                        


class AutoRegression_input(AutoRegression):

    def __init__(
            self, nu_0=None,S_0=None,M_0=None,K_0=None,
            affine=False,
            A=None,sigma=None):
        self.affine = affine

        self._check_shapes(A, sigma, nu_0, S_0, M_0, K_0)
        self.input = False

        self.A = A
        self.sigma = sigma

        have_hypers = not any_none(nu_0,S_0,M_0,K_0)

        if have_hypers:
            self.natural_hypparam = self.mf_natural_hypparam = \
                self._standard_to_natural(nu_0,S_0,M_0,K_0)

        if A is sigma is None and have_hypers:
            self.resample()  # initialize from prior

    def max_likelihood(self,data,weights=None,stats=None):
        if stats is None:
            stats = self._get_statistics(data) if weights is None \
                else self._get_weighted_statistics(data,weights)

        yyT, yxT, xxT, n = stats 

        if n > 0:
            try:
                def symmetrize(A):
                    return (A + A.T)/2.

                if self.input:
                    raise Exception('not yet implemented')
                else:
                    self.A = np.linalg.solve(xxT, yxT.T).T

                    self.sigma = (yyT \
                                - 2 * symmetrize(yxT.dot(self.A.T)) \
                                + self.A.dot(xxT.dot(self.A.T)) )/n 

                    self.sigma = 1e-10*np.eye(self.D_out) \
                        + symmetrize(self.sigma)  # numerical

            except np.linalg.LinAlgError:
                self.broken = True
        else:
            self.broken = True

        #assert np.allclose(self.sigma,self.sigma.T)
        #assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        self._initialize_mean_field()

        return self