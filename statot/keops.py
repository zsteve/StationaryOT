import ot
import numpy as np
import pykeops
from pykeops.numpy import LazyTensor, Vi, Vj
import scipy
from scipy.sparse.linalg.interface import IdentityOperator
from scipy.sparse.linalg import aslinearoperator, LinearOperator, gmres

dtype = "float32"

def set_dtype(d):
    """Set dtype to use in statot.keops
    
    """
    dtype = d

def form_cost(mu_spt, nu_spt, norm_factor = None, keops = True):
    """Form cost matrix (matrix of squared Euclidean distances)
    
    :param mu_spt: support of source measure
    :param nu_spt: support of target measure
    :param norm_factor: normalisation factor as a `float`, `None` or "mean"
    :param keops: whether to return a `LazyTensor` or `np.array`
    """
    norm = None
    if keops:
        C = ((Vi(mu_spt) - Vj(nu_spt))**2).sum(2)
        if norm_factor == "mean":
            norm = float(np.sum(C @ np.ones(C.shape[1], dtype = dtype))/(C.shape[0] * C.shape[1]))
    else:
        C = ot.utils.dist(mu_spt, nu_spt)
        if norm_factor == "mean":
            norm = C.mean()
    if norm_factor is None:
        norm = 1
    elif norm is None:
        norm = norm_factor
    return C/norm, norm

def sinkhorn(mu, nu, K, max_iter = 5000, err_check = 10, tol = 1e-9, verbose = False):
    """Sinkhorn algorithm compatible with KeOps LazyTensor
    
    :param mu: source distribution
    :param nu: target distribution
    :param K: Gibbs kernel as `LazyTensor`
    :param max_iter: maximum number of iterations
    :param err_check: interval for checking marginal error
    :param tol: tolerance (inf-norm on marginals)
    :param verbose: flag for verbose output
    """
    u = np.ones(mu.shape[0], dtype = dtype)
    v = np.ones(nu.shape[0], dtype = dtype)
    converged = False
    for i in range(max_iter):
        u = mu/(K @ v)
        v = nu/(K.T @ u)
        if i % err_check == 0:
            mu_tmp = u * (K @ v)
            nu_tmp = v * (K.T @ u)
            err = max(np.linalg.norm(mu - mu_tmp, ord = float('inf')), 
                        np.linalg.norm(nu - nu_tmp, ord = float('inf')))
            if verbose:
                print("Iteration %d, Error = %f" % (i, err))
            if err < tol:
                converged = True
                break
    if converged == False:
        print("Warning: sinkhorn failed to converge")
    return u, v

def get_QR_submat(u, K, v, X, sink_idx, eps, cost_norm_factor):
    """Compute Q (as LazyTensor) and R (as np.ndarray) matrices 
    
    :param u: dual potential for source distribution
    :param K: Gibbs kernel as `LazyTensor`
    :param v: dual potential for target distribution
    :param X: coordinates as np.ndarraya
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise. 
    :param eps: value of `eps` used for solving with `sinkhorn`
    :param cost_norm_factor: normalisation factor used in `form_cost`
    """
    gamma = (Vi(u.reshape(-1, 1)) * K * Vj(v.reshape(-1, 1))) 
    z = gamma @ np.ones(gamma.shape[1], dtype = dtype)  # row norm factor for full tr matrix
    # KeOps doesn't allow slicing of LazyTensors, so need to manually construct Q as submatrix of P
    K_q = (-form_cost(X[~sink_idx, :], X[~sink_idx, :], norm_factor = cost_norm_factor)[0]/eps).exp() 
    K_r = np.exp(-form_cost(X[~sink_idx, :], X[sink_idx, :], norm_factor = cost_norm_factor, keops = False)[0]/eps)
    Q = Vi((u/z)[~sink_idx].reshape(-1, 1)) * K_q * Vj(v[~sink_idx].reshape(-1, 1)) 
    R = (u/z)[~sink_idx].reshape(-1, 1) * K_r * v[sink_idx].reshape(1, -1)
    return Q, R

def compute_fate_probs(Q, R):
    """Compute fate probabilities from Q (LazyTensor) and R (np.ndarray)
    
    :param Q: transient part of transition matrix from `get_QR_submat`, as `LazyTensor`
    :param R: absorbing part of transition matrix. Should aggregate the columns across fates, 
                since the solver cannot solve multiple RHS at once. 
    """
    A = IdentityOperator(Q.shape) - aslinearoperator(Q)
    A.dtype = np.dtype(dtype)
    B = np.zeros(R.shape, dtype = dtype)
    for i in range(R.shape[1]):
        print("Solving fate probabilities for lineage %d" % i)
        out = gmres(A, R[:, i])
        if out[1] != 0:
            print("Warning: gmres returned error code %d" % out[1])
        B[:, i] = out[0]
    return B