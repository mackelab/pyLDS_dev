from __future__ import division

import numpy as np

from pybasicbayes.distributions import Regression

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

    def max_likelihood(self,data,weights=None,stats=None):
        if stats is None:
            stats = self._get_statistics(data) if weights is None \
                else self._get_weighted_statistics(data,weights)

        yyT, yxT, xxT, n = stats

        if n > 0:
            try:
                def symmetrize(A):
                    return (A + A.T)/2.

                if self.affine:

                    yxT, y = yxT[:,:-1], yxT[:,-1]
                    xxT, x = xxT[:-1, :-1], xxT[:-1, -1]

                    A = np.linalg.solve(xxT - np.outer(x,x)/n, (yxT - np.outer(y,x)/n).T).T
                    b = np.atleast_2d(y - A.dot(x)).T/n
                    self.A = np.hstack([A,b])

                    self.sigma = (yyT \
                                - 2 * symmetrize(np.outer(y, b)) \
                                - 2 * symmetrize(A.dot(yxT.T - np.outer(x,b))) \
                                + A.dot(xxT.dot(A.T)) )/n \
                                + np.outer(b,b)

                else:                    

                    self.A = np.linalg.solve(xxT, yxT.T).T

                    self.sigma = (yyT - self.A.dot(yxT.T))/n
                    self.sigma = 1e-10*np.eye(self.D_out) \
                        + symmetrize(self.sigma)  # numerical

            except np.linalg.LinAlgError:
                self.broken = True
        else:
            self.broken = True

        assert np.allclose(self.sigma,self.sigma.T)
        assert np.all(np.linalg.eigvalsh(self.sigma) > 0.)

        self._initialize_mean_field()

        return self