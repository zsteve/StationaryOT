
import pykeops
from pykeops.numpy import LazyTensor, Vi, Vj

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

def sinkhorn_keops(mu, nu, K, max_iter = 5000, err_check = 10, tol = 1e-9, verbose = False):
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
