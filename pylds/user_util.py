import numpy as np
import scipy as sp

from models import LDS, DefaultLDS
from distributions import Regression_diag, AutoRegression_input
from obs_scheme import ObservationScheme
from scipy.linalg import solve_discrete_lyapunov as dlyap

def gen_pars(n, p, u_dim=0, 
             pars_in=None, 
             obs_scheme=None,
             gen_A='diagonal', lts=None,
             gen_B='random', 
             gen_Q='identity', 
             gen_mu0='random', 
             gen_V0='identity', 
             gen_C='random', 
             gen_d='scaled', 
             gen_R='fraction',
             diag_R_flag=True,
             x=None, y=None, u=None): 
    """ INPUT:
        n : dimensionality of latent states x
        p : dimensionality of observed states y
        u_dim : dimensionality of input states u
        pars_in:    None, or list/np.ndarray/dict containing no, some or all
                   of the desired parameters. This function will identify which
                   parameters were not handed over and will fill in the rest
                   according to selected paramters below.
        obs_scheme: observation scheme for given data, stored in dictionary
                   with keys 'sub_pops', 'obs_time', 'obs_pops'
        gen_A   : string specifying methods of parameter generation
        lts    : ndarray with one entry per latent time scale (i.e. n many)
        gen_B   :  ""
        gen_Q   :  ""
        gen_mu0 :  "" 
        gen_V0  :  ""
        gen_C   :  "" 
        gen_d   :  ""
        gen_R   : (see below for details)
        x: data array of latent variables
        y: data array of observed variables
        u: data array of input variables
        Generates parameters of an LDS, potentially by looking at given data.
        Can be used for for generating ground-truth parameters for generating
        data from an artificial experiment using the LDS, or for finding 
        parameter initialisations for fitting an LDS to data. Usage is slightly
        different in the two cases (see below). By nature of the wide range of
        applicability of the LDS model, this function contains many options
        (implemented as strings differing different cases, and arrays giving
         user-specified values such as timescale ranges), and is to be extended
        even further in the future.

    """
    totNumParams = 8 # an input-LDS model has 8 parameters: A,B,Q,mu0,V0,C,d,R

    """ pars optional inputs to this function """
    if (not lts is None and
        not (isinstance(lts, np.ndarray) and 
            (np.all(lts.shape==(n,)) or np.all(lts.shape==(n,1)))
            ) ):
        print('lts (latent time scales)')
        raise Exception(('variable lts has to be an ndarray of shape (n,)'
                         ' However, it is '), lts)


    if y is None:
        if not x is None:
            raise Exception(('provided latent state sequence x but not '
                             'observed data y.')) 
        if not u is None:
            raise Exception(('provided input sequence u but not '
                             'observed data y.')) 

    else: # i.e. if y is provided:
        if not (isinstance(y,np.ndarray) 
                and len(y.shape)==2 and y.shape[0]==p):
            raise Exception(('When providing optional input y, it has to be '
                             'an np.ndarray of dimensions (p,t_tot). '
                             'It is not.'))
        else:
            t_tot = y.shape[1] # take these values from y and compare with 

    if not (x is None or (isinstance(x,np.ndarray) 
                          and len(x.shape)==2 
                          and x.shape[0]==n
                          and x.shape[1]==t_tot) ):
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (n,t_tot). '
                         'It is not'))

    if not (u is None or (isinstance(u,np.ndarray) 
                          and len(u.shape)==2 
                          and u.shape[0]==u_dim
                          and u.shape[1]==t_tot) ):
        raise Exception(('When providing optional input x, it has to be an '
                         'np.ndarray of dimensions (n,t_tot). '
                         'It is not'))


    if (gen_C in ('PCA', 'flat_scaled')) or (gen_R == 'fractionObserved'):  
        cov_y = np.cov(y[:,:]-np.mean(y, 1).reshape(p,1)) 
        # Depending on the observation scheme, not all entries of the data
        # covariance are also interpretable, and the entries of cov_y for 
        # variable pairs (y_i,y_j) that were not observed together may indeed
        # contain NaN's of Inf's depending on the choice of representation of
        # missing data entries. Keep this in mind when selecting parameter 
        # initialisation methods such as gen_C=='PCA', which will work with 
        # the full matrix cov_y.
        # Note that the diagonal of cov_y should also be safe to use. 

    gen_pars_flags = np.ones(totNumParams, dtype=bool) # which pars to generate

    pars_out = {}

    """ parse (optional) user-provided true model parameters: """
    # allow comfortable use of dictionaries:
    if isinstance(pars_in, dict):

        if 'A' in pars_in:
            if np.all(pars_in['A'].shape==(n,n)): 
                pars_out['A']   = pars_in['A'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter A.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of A is '), pars_in['A'].shape)
            gen_pars_flags[0] = False
        if 'B' in pars_in:
            if u_dim > 0:        
                if np.all(pars_in['B'].shape==(n,u_dim)): 
                    pars_out['B']  = pars_in['B'].copy()
                else:
                    raise Exception(('Bad initialization for LDS parameter B.'
                                     'Shape not matching dimensionality of x,u'
                                     '. Given shape of B is '),pars_in[1].shape)
            else: # if we're not going to use B anyway, ...
                pars_out['B'] = np.array([0]) 
            gen_pars_flags[1] = False
        if 'Q' in pars_in:
            pars_out['Q'] = pars_in['Q']
            if np.all(pars_in['Q'].shape==(n,n)): 
                pars_out['Q']   = pars_in['Q'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter Q.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of Q is '), pars_in[2].shape)
            gen_pars_flags[2] = False
        if 'mu0' in pars_in:
            if pars_in['mu0'].size==n: 
                pars_out['mu0'] = pars_in['mu0'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter mu0.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of mu0 is '), pars_in['mu0'].shape)
            gen_pars_flags[3] = False
        if 'V0' in pars_in:
            if np.all(pars_in['V0'].shape==(n,n)): 
                pars_out['V0'] = pars_in['V0'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter V0.'
                                 'Shape not matching dimensionality of x. '
                                 'Given shape of V0 is '), pars_in['V0'].shape)
            gen_pars_flags[4] = False
        if 'C' in pars_in:
            if np.all(pars_in['C'].shape==(p,n)):
                pars_out['C'] = pars_in['C'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter C.'
                                 'Shape not matching dimensionality of y, x. '
                                 'Given shape of C is '), pars_in['C'].shape)
            gen_pars_flags[5] = False
        if 'd' in pars_in:
            if np.all(pars_in['d'].shape==(p,)):
                pars_out['d'] = pars_in['d'].copy()
            else:
                raise Exception(('Bad initialization for LDS parameter d.'
                                 'Shape not matching dimensionality of y, x. '
                                 'Given shape of d is '), pars_in['d'].shape)              
            gen_pars_flags[6] = False
        if 'R' in pars_in:
            if diag_R_flag:
                if np.all(pars_in['R'].shape==(p,)):
                    pars_out['R'] = pars_in['R'].copy()
                elif np.all(pars_in['R'].shape==(p,p)):
                    pars_out['R'] = pars_in['R'].copy().diagonal()
                else:
                    raise Exception(('Bad initialization for LDS '
                                     'parameter R. Shape not matching '
                                     'dimensionality of y. '
                                     'Given shape of R is '),pars_in['R'].shape)                  
            else:
                if np.all(pars_in['R'].shape==(p,p)):
                    pars_out['R'] = pars_in['R'].diagonal().copy()
                else:
                    raise Exception(('Bad initialization for LDS '
                                     'parameter R. Shape not matching '
                                     'dimensionality of y. '
                                     'Given shape of R is '),pars_in['R'].shape)      
            gen_pars_flags[7] = False

    elif not pars_in is None:
        raise Exception('provided input parameter variable pars_in has to be '
                        'a dictionary with (optional) key-value pairs for '
                        'desired parameter initialisations. For no specified, '
                        'initialisations, use {} or None. However, pars_in is',
                         pars_in)

    """ fill in missing parameters (could be none, some, or all) """
    # generate latent state tranition matrix A
    if gen_pars_flags[0]:
        if lts is None:
            lts = np.random.uniform(size=[n])
        if gen_A == 'diagonal':
            pars_out['A'] = np.diag(lts) # lts = latent time scales
        elif gen_A == 'full':
            #pars_out['A'] = np.diag(lts) # lts = latent time scales
            #if n == 3:
            #    W    = rand_rotation_matrix()
            #    pars_out['A'] = np.dot(np.dot(W, pars_out['A']), np.linalg.inv(W))
            #else:
            #    raise Exception('random rotation matrices for n != 3 not implemented')
            while True:
                pars_out['A'] = np.random.normal(size=(n,n))
                D, V = np.linalg.eig(pars_out['A'])
                D = (lts/np.abs(D)) * D
                pars_out['A'] = np.real(V.dot(np.diag(D).dot(np.linalg.inv(V))))
                if np.mean( (np.abs(np.linalg.eigvals(pars_out['A']) ) - lts)**2 ) < 0.01**2:
                    break

        elif gen_A == 'random':
            pars_out['A'] = np.random.normal(size=[n,n])            
        elif gen_A == 'zero':  # e.g. when fitting without dynamics
            pars_out['A'] = np.zeros((n,n))            
        else:
            raise Exception(('selected type for generating A not supported. '
                             'Selected type is '), gen_A)
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. Any rotation of A can be corrected for by rightmultiplying C with
    # the inverse rotation matrix. We do not wish to limit A to any certain
    # basis in latent space, but in a first approach may still initialise A as
    # diagonal matrix .     

    # generate latent state input matrix B
    if gen_pars_flags[1]:
        if gen_B == 'random':
            pars_out['B'] = np.random.normal(size=[n,u_dim])            
        elif gen_B == 'zero': # make sure is default if use_B_flag=False
            pars_out['B'] = np.zeros((n,u_dim))            
        else:

            raise Exception(('selected type for generating B not supported. '
                             'Selected type is '), gen_B)
    # Parameter B is never touched within the code unless use_B_flag == True,
    # hence we don't need to ensure its correct dimensionality if use_B_flag==False

    # generate latent state innovation noise matrix Q
    if gen_pars_flags[2]:                             # only one implemented standard 
        if gen_Q == 'identity':     # case: we can *always* rotate x
            pars_out['Q']    = np.identity(n)  # so that Q is the identity 
        else:

            raise Exception(('selected type for generating Q not supported. '
                             'Selected type is '), gen_Q)
    # There is inherent degeneracy in any LDS regarding the basis in the latent
    # space. One way to counter this is to set the latent covariance to unity.
    # We don't hard-fixate this, as it prevents careful study of when stitching
    # can really work. Nevertheless, we can still initialise parameters Q as 
    # unity matrices without commiting to any assumed structure in the  final
    # innovation noise estimate. 
    # Note that the initialisation choice for Q should be in agreement with the
    # initialisation of C! For instance when setting Q to the identity and 
    # when getting C from PCA, one should also normalise the rows of C with
    # the sqrt of the variances of y_i, i.e. really whiten the assumed 
    # latent covariances instead of only diagonalising them.            

    # generate initial latent state mean mu0
    if gen_pars_flags[3]:
        if gen_mu0 == 'random':
            pars_out['mu0']  = np.random.normal(size=[n])
        elif gen_mu0 == 'zero': 
            pars_out['mu0']  = np.zeros(n)
        else:
            raise Exception(('selected type for generating mu0 not supported. '
                             'Selected type is '), gen_mu0)
    # generate initial latent state covariance matrix V0
    if gen_pars_flags[4]:
        if gen_V0 == 'identity': 
            pars_out['V0'] = np.identity(n)  
        elif gen_V0 == 'stable':
            pars_out['V0'] = dlyap(pars_out['A'], pars_out['Q'])
        else:
            raise Exception(('selected type for generating V0 not supported. '
                             'Selected type is '), gen_V0)
    # Assuming long time series lengths, parameters for the very first time
    # step are usually of minor importance for the overall fitting result
    # unless they are overly restrictive. We by default initialise V0 
    # non-commitingly to the identity matrix (same as Q) and mu0 either
    # to all zero or with a slight random perturbation on that.   

    # generate emission matrix C
    if gen_pars_flags[5]:
        if gen_C == 'random': 
            pars_out['C'] = np.random.normal(size=[p, n])
        elif gen_C == 'flat':
            pars_out['C'] = np.ones((p,n))
        elif gen_C == 'flat_scaled':
            pars_out['C'] = np.sqrt(np.atleast_2d(np.diag(cov_y)).T) * \
                            np.ones((p,n))/n
        elif gen_C == 'PCA':
            if y is None:
                raise Exception(('tried to set emission matrix C from results '
                                 'of a PCA on the observed data without '
                                 'providing any data y'))            
            w, v = np.linalg.eig(cov_y)                           
            w = np.sort(w)[::-1]                 
            # note that we also enforce equal variance for each latent dim. :
            pars_out['C'] = np.dot(v[:, range(n)], 
                                np.diag(np.sqrt(w[range(n)])))  
        elif gen_C == 'PCA_subpop':
            if y is None:
                raise Exception(('tried to set emission matrix C from results '
                                 'of a PCA on the observed data without '
                                 'providing any data y'))            
            if obs_scheme is None:
                raise Exception('tried to set emission matrix C in chunks for '
                                'subpopuplation, but did not give observation '
                                'scheme')
            pars_out['C'] = np.zeros((p,n))
            for i in range(len(obs_scheme.sub_pops)):
                idx = obs_scheme.sub_pops[i]
                w, v = np.linalg.eig(cov_y[np.ix_(idx,idx)])                           
                w = np.sort(w)[::-1]
                # note that we do not rotate the latent spaces into a single
                # coordinate system:
                pars_out['C'][idx,:] = np.dot(v[:, range(n)], 
                                       np.diag(np.sqrt(w[range(n)])))  


        else:
            raise Exception(('selected type for generating C not supported. '
                             'Selected type is '), gen_C)
    # C in many cases is the single-most important parameter to properly 
    # initialise. If the data is fully observed, a basic and powerful solution
    # is to use PCA on the full data covariance (after attributing a certain 
    # fraction of variance to R). In stitching contexts, this however is not
    # possible. Finding a good initialisation in the context of incomplete data
    # observation is not trivial. 

    # check for resulting stationary covariance of latent states x
    Pi    = sp.linalg.solve_discrete_lyapunov(pars_out['A'], 
                                              pars_out['Q'])
    Pi_t  = np.dot(pars_out['A'].transpose(),Pi)  # time-lagged cov(y_t,y_{t-1})
    CPiC = np.dot(pars_out['C'], np.dot(Pi, pars_out['C'].transpose())) 

    # generate emission noise covariance matrix R
    if gen_pars_flags[7]:
        if gen_R == 'fraction':
            # set R_ii as 25% to 125% of total variance of y_i
            pars_out['R'] =(0.25+np.random.uniform(size=[p]))*CPiC.diagonal()
        elif gen_R == 'fractionObserved':
            if y is None:
                raise Exception(('tried to set emission noise covariance R as '
                                 'a fraction of data variance without '
                                 'providing any data y'))
            if p>1:
                pars_out['R']   = 0.1 * cov_y.diagonal()  
            else:
                pars_out['R']   = 0.1 * np.array(cov_y).reshape(1,)

        elif gen_R == 'identity':
            gen_R = np.ones(p)
        elif gen_R == 'zero':                        # very extreme case!
            gen_R = np.zeros(p)
        else:
            raise Exception(('selected type for generating R not supported. '
                             'Selected type is '), gen_R)
    # C and R should not be initialised independently! Following on the idea
    # of (diagonal) R being additive private noise for the individual variables
    # y_i, we can initialise R as being a certain fraction of the observed 
    # noise. When initialising R from data, we have to be carefull not to
    # attribute too much noise to R, as otherwise the remaining covariance 
    # matrix cov(y)-np.diag(R) might no longer be positive definite!

    # generate emission noise covariance matrix d
    if gen_pars_flags[6]:
        if gen_d == 'scaled':
            pars_out['d'] = (np.sqrt(
                                    np.mean(
                                            np.diag( CPiC
                                                   + np.diag(pars_out['R']
                                                   )
                                            )
                                    )
                            )
                            * np.random.normal(size=p))
        elif gen_d == 'random':
            pars_out['d'] = np.random.normal(size=p)
        elif gen_d == 'zero':
            pars_out['d'] = np.zeros(p)
        elif gen_d == 'mean':
            if y is None:
                raise Exception(('tried to set observation offset d as the '
                                 'data mean without providing any data y'))
            pars_out['d'] = np.mean(y,1) 
        else:
            raise Exception(('selected type for generating d not supported. '
                              'Selected type is '), gen_d)
    # A bad initialisation for d can spell doom for the entire EM algorithm,
    # as this may offset the estimates of E[x_t] far away from zero mean in
    # the first E-step, so as to capture the true offset present in data y. 
    # This in turn ruins estimates of the linear dynamics: all the eigenvalues
    # of A suddenly have to be close to 1 to explain the constant non-decaying
    # offset of the estimates E[x_t]. Hence the ensuing M-step will generate
    # a parameter solution that is immensely far away from optimal parameters,
    # and the algorithm most likely gets stuck in a local optimum long before
    # it found its way to any useful parameter settings (in fact, C and d of
    # the first M-step will adjust to the offset in the latent states and 
    # hence contribute to the EM algorithm sticking to latent offset and bad A)

    # collect options for debugging and experiment-tracking purposes
    options_init = {
                 'gen_A'   : gen_A,
                 'lts'    : lts,
                 'gen_B'   : gen_B,
                 'gen_Q'   : gen_Q,
                 'gen_mu0' : gen_mu0,
                 'gen_V0'  : gen_V0,
                 'gen_C'   : gen_C,
                 'gen_d'   : gen_d,
                 'gen_R'   : gen_R
                    }

    """ check validity (esp. of user-provided parameters), return results """

    return pars_out, options_init


def init_LDS_model(pars,data, obs_scheme):

    p = pars['C'].shape[0]
    n = pars['C'].shape[1]

    model = LDS(dynamics_distn=AutoRegression_input(
                  nu_0=n+1, S_0=n*np.eye(n), M_0=np.zeros((n, n)), K_0=n*np.eye(n),
                  A=pars['A'].copy(), sigma=pars['Q'].copy()),                
                emission_distn=Regression_diag(
                  nu_0=p+1, S_0=p*np.eye(p), M_0=np.zeros((p, n+1)), K_0=p*np.eye(n+1),
                  A=np.hstack((pars['C'].copy(), pars['d'].copy().reshape(p,1))), 
                  sigma=pars['R'].copy(),
                  affine=True))
    
    model.mu_init = pars['mu0'].copy()
    model.sigma_init = pars['V0'].copy()
    
    model.add_data(data)
    
    model.states_list[0].obs_scheme = obs_scheme
    
    return model


def collect_LDS_stats(model):
    stats =  {'mu_h' : model.states_list[0].smoothed_mus.copy(),
              'V_h'  : model.states_list[0].smoothed_sigmas.copy(),
              'yy': model.states_list[0].E_emission_stats[0],
              'yx': model.states_list[0].E_emission_stats[1],
              'xx': model.states_list[0].E_emission_stats[2],
              'extxtm1': model.states_list[0].E_addition_stats.copy()}
    pars = {'A' : model.A.copy(),
            'B' : 0,
            'Q' : model.sigma_states.copy(),
            'mu0': model.mu_init.copy(),
            'V0': model.sigma_init.copy(),
            'C' : model.C.copy(),
            'd' : model.d.copy(),
            'R' : model.sigma_obs.copy()}    
    return stats, pars

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
