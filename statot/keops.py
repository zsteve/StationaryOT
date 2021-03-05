import ot
import numpy as np
import pykeops
from pykeops.numpy import LazyTensor, Vi, Vj
import scipy
from scipy.sparse.linalg.interface import IdentityOperator
from scipy.sparse.linalg import aslinearoperator, LinearOperator, gmres, cg

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
    """Sinkhorn algorithm for solving entropy-regularised optimal transport
        compatible with KeOps LazyTensor framework. 
    
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


def quad_ot_semismooth_newton(mu, nu, C, eps, max_iter = 50, theta = 0.1, kappa = 0.5, tol = 1e-3, eta = 1e-5, cg_max_iter = 500, verbose = False):
    """Semismooth Newton algorithm for solving quadratically regularised optimal transport
        compatible with KeOps LazyTensor framework. Uses the method from Algorithm 2 of 
        `Lorenz, D.A., Manns, P. and Meyer, C., 2019. 
        Quadratically regularized optimal transport. Applied Mathematics & Optimization, pp.1-31.`

    :param mu: source distribution
    :param nu: target distribution
    :param C: cost matrix as `LazyTensor`
    :param max_iter: maximum number of Newton steps
    :param theta: Armijo control parameter (choose in :math:`(0, 1)`)
    :param kappa: Armijo step scaling parameter (choose in :math:`(0, 1)`)
    :param tol: tolerance (inf-norm on marginals)
    :param eta: conjugate gradient regularisation parameter 
    :param cg_max_iter: maximum number of conjugate gradient iterations
    :param verbose: flag for verbose output

    """
    class NewtonMatrix:
        def __init__(self, sigma, eta):
            self.sigma = sigma
            self.eta = eta
            self.diag = np.concatenate([sigma @ np.ones(sigma.shape[1], dtype = dtype), 
                sigma.T @ np.ones(sigma.shape[0], dtype = dtype)]) + eta
            self.offdiag = [sigma, sigma.T]
            self.shape = (sum(self.sigma.shape), sum(self.sigma.shape))
            self.dtype = self.sigma.dtype
        def matvec(self, v):
            v1 = v[0:self.sigma.shape[0]]
            v2 = v[self.sigma.shape[0]:]
            return self.diag * v + np.concatenate([self.sigma @ v2 , self.sigma.T @ v1])
    def Phi(u, v, mu, nu, C, eps):
        X = ((Vi(u.reshape(-1, 1)) + Vj(v.reshape(-1, 1)) - C).relu())**2
        return (X @ np.ones(X.shape[1], dtype = dtype)).sum()/2 - eps*(np.dot(u, nu) + np.dot(v, mu))
    # initialise dual variables
    u = np.ones(mu.shape[0], dtype = dtype)
    v = np.ones(nu.shape[0], dtype = dtype)
    for i in range(max_iter):
        P = Vi(u.reshape(-1, 1)) + Vj(v.reshape(-1, 1)) - C
        sigma = P.relu().sign()
        gamma = P.relu()/eps
        # compute Newton matrix
        G = NewtonMatrix(sigma, eta = eta)
        G_op = aslinearoperator(G)
        x = np.concatenate([gamma @ np.ones(gamma.shape[1], dtype = dtype) - nu,
            gamma.T @ np.ones(gamma.shape[0], dtype = dtype) - mu])
        duv, cg_err = cg(G_op, -eps*x, maxiter = cg_max_iter)
        if (cg_err != 0 ) and verbose:
            print("Warning: cg failed to converge")
        du = duv[0:u.shape[0]]
        dv = duv[u.shape[0]:]
        d = eps*((gamma.T @ du).sum() + (gamma @ dv).sum() - (np.dot(du, nu) + np.dot(dv, mu)))
        # Armijo backtracking linesearch 
        t = 1
        Phi0 = Phi(u, v, mu, nu, C, eps)
        while Phi(u + t*du, v + t*dv, mu, nu, C, eps) >= Phi0 + t*theta*d:
            t *= kappa
            if (t == 0) and verbose: 
                print("Warning: underflow in step size t")
                break
        # make Newton step 
        u = u + t*du 
        v = v + t*dv
        err1 = np.linalg.norm(gamma @ np.ones(gamma.shape[1], dtype = dtype) - nu, ord = float("inf"))
        err2 = np.linalg.norm(gamma.T @ np.ones(gamma.shape[0], dtype = dtype) - mu, ord = float("inf"))
        if verbose:
            print("Iteration ", i, "Error = ", max(err1, err2))
        if max(err1, err2) < tol:
            break
    return u, v

def get_QR_submat_ent(u, K, v, X, sink_idx, eps, cost_norm_factor):
    """Compute Q (as LazyTensor) and R (as np.ndarray) matrices for 
        entropy-regularised OT dual potentials (u, v)
    
    :param u: dual potential for source distribution
    :param K: Gibbs kernel as `LazyTensor`
    :param v: dual potential for target distribution
    :param X: coordinates as np.ndarray
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

def get_QR_submat_quad(u, C, v, X, sink_idx, eps, cost_norm_factor):
    """Compute Q (as LazyTensor) and R (as np.ndarray) matrices for
        quadratically regularised OT dual potentials (u, v)
    
    :param u: dual potential for source distribution
    :param C: cost matrix as `LazyTensor`
    :param v: dual potential for target distribution
    :param X: coordinates as np.ndarray
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise. 
    :param eps: value of `eps` used for solving with `quad_ot_semismooth_newton`
    :param cost_norm_factor: normalisation factor used in `form_cost`
    """
    gamma = (Vj(u.reshape(-1, 1)) + Vi(v.reshape(-1, 1)) - C.T).relu()/eps
    z = gamma @ np.ones(gamma.shape[1], dtype = dtype)  # row norm factor for full tr matrix
    # KeOps doesn't allow slicing of LazyTensors, so need to manually construct Q as submatrix of P
    C_q = form_cost(X[~sink_idx, :], X[~sink_idx, :], norm_factor = cost_norm_factor)[0] 
    C_r = form_cost(X[sink_idx, :], X[~sink_idx, :], norm_factor = cost_norm_factor, keops = False)[0]
    Q = (Vj(u[~sink_idx].reshape(-1, 1)) + Vi(v[~sink_idx].reshape(-1, 1)) - C_q.T).relu()/eps
    R = np.maximum(u[sink_idx].reshape(1, -1) + v[~sink_idx].reshape(-1, 1) - C_r.T, 0)/eps
    return Q / Vi(z[~sink_idx].reshape(-1, 1)), R / z[~sink_idx].reshape(-1, 1)

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
