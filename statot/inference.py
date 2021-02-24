# statOT core 
# Author: Stephen Zhang (syz@math.ubc.ca)

import numpy as np
import pandas as pd
from numpy import linalg 
import ot
import copy
import pykeops
from pykeops.numpy import LazyTensor, Vi, Vj

dtype = "float32"

def form_cost(mu_spt, nu_spt, norm_factor = None, keops = True):
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

def sinkhorn_keops(mu, nu, C, eps, max_iter = 5000, err_check = 10, tol = 1e-9, verbose = False):
    K = (-C/eps).exp()
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
    return u, K, v

def statot(x, C = None, eps = None, method = "ent", g = None,
           dt = None, 
           maxiter = 5000, tol = 1e-9,
           verbose = False):
    """Fit statOT model
    
    :param x: input data -- `N` points of `M` dimensions in the form of a matrix with dimensions `(N, M)`
    :param C: cost matrix for optimal transport problem
    :param eps: regularisation parameter 
    :param method: choice of regularisation -- either "ent" (entropy) or "quad" (L2). "unbal" for unbalanced transport is not yet implemented. 
    :param g: numeric array of length `N`, containing the relative growth rates for cells.
    :param flow_rate: used only in the growth-free case (flow only)
    :param dt: choice of the time step over which to fit the model
    :param maxiter: max number of iterations for OT solver
    :param tol: relative tolerance for OT solver convergence
    :param verbose: detailed output on convergence of OT solver. 
    :return: gamma (optimal transport coupling), mu (source measure), nu (target measure)
    """
    mu_spt = x
    nu_spt = x
    mu = g**dt
    nu = mu.sum()/x.shape[0]*np.ones(x.shape[0])
    if method == "quad":
        gamma = ot.smooth.smooth_ot_dual(mu, nu, C, eps, numItermax = maxiter, stopThr = tol*mu.sum(), verbose = verbose)
    elif method == "ent":
        gamma = ot.sinkhorn(mu, nu, C, eps, numItermax = maxiter, stopThr = tol*mu.sum(), verbose = verbose)
    elif method == "ent_keops":
        u, K, v = sinkhorn_keops(np.array(mu, dtype = dtype), np.array(nu, dtype = dtype), C, eps, max_iter = maxiter, tol = tol*mu.sum(), verbose = verbose)
        gamma = (Vi(u.reshape(-1, 1)) * K * Vj(v.reshape(-1, 1))) 
    elif method == "unbal":
        print("method = 'unbal' not implemented yet")
    # manually check for convergence since POT is a bit iffy
    if verbose and method != "ent_keops":
        print("max inf-norm error for marginal agreement: ",
              max(np.linalg.norm(gamma.sum(1) - mu, ord = np.inf), np.linalg.norm(gamma.sum(0) - nu, ord = np.inf)))
    return gamma, mu, nu

def row_normalise(gamma, sink_idx = None):
    """Enforce sink condition and row normalise coupling to produce transition matrix

    :param gamma: coupling produced by `statot()`;
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise. 
        If provided, sets the transition distributions for all sinks to be the identity. 
    :return: transition matrix obtained by row-normalising the input `gamma`.
    """
    gamma_ = gamma.copy()
    if sink_idx is not None:
        gamma_[sink_idx, :] = 0
        gamma_[np.ix_(sink_idx, sink_idx)] = np.eye(sink_idx.sum())
    return (gamma_.T/gamma_.sum(1)).T

def gaussian_tr(C, h):
    """Form Gaussian (discrete heat flow) transition matrix of bandwidth h
    
    :param C: pairwise square distances
    :param h: bandwidth
    """
    return row_normalise(np.exp(-C/h))

# transliterated from Julia code
def _compute_NS(P, sink_idx):
    Q = P[np.ix_(~sink_idx, ~sink_idx)]
    S = P[np.ix_(~sink_idx, sink_idx)]
    # N = np.linalg.inv(np.eye(Q.shape[0]) - Q) BAD!!!!
    N = np.eye(Q.shape[0]) - Q
    return N, S

def compute_fate_probs(P, sink_idx):
    """Compute fate probabilities by individual sink cell

    :param P: transition matrix
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :return: matrix with dimensions `(N, S)` where `S` is the number of sink cells present.
    """
    N, S = _compute_NS(P, sink_idx)
    B = np.zeros((P.shape[0], sink_idx.sum()))
    B[~sink_idx, :] = np.linalg.solve(N, S)
    B[sink_idx, :] = np.eye(sink_idx.sum())
    return B

def compute_fate_probs_lineages(P, sink_idx, labels):
    """Compute fate probabilities by lineage

    :param P: transition matrix
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :param labels: string array of length `N` containing lineage names. Only those entries corresponding to sinks will be used.
    :return: matrix with dimensions `(N, L)` where `L` is the number of lineages with sinks.
    """
    B = compute_fate_probs(P, sink_idx)
    sink_labels = np.unique(labels[sink_idx])
    B_lineages = np.array([B[:, labels[sink_idx] == i].sum(1) for i in sink_labels]).T
    return B_lineages, sink_labels


def compute_conditional_mfpt(P, j, sink_idx):
    """Compute conditional mean first passage time MFPT(x_i -> x_j). Based on implementation in PBA.
    
    :param P: transition matrix
    :param j: index j of cell x_j
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :return: vector t_i containing MFPT(x_i -> x_)
    """
    new_idx = list(range(P.shape[0]))
    new_idx.remove(j)
    new_idx.append(j)
    new_P = P.copy()[new_idx, :][:, new_idx]
    new_sinks = sink_idx.copy()[new_idx]
    new_sinks[-1] = True
    new_P[-1, :] = 0
    new_P[-1, -1] = 1
    new_P = row_normalise(new_P)
    Q = new_P[np.ix_(~new_sinks, ~new_sinks)]
    S = new_P[np.ix_(~new_sinks, new_sinks)]
    N = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    B = np.dot(N, S)
    d = B[:, -1]
    mask = sink_idx.copy()
    mask[j] = True
    t = np.zeros(P.shape[0])
    t[~mask] = (1/d)*(N @ d)
    t[mask] = float('Inf')
    t[j] = 0
    return t

def velocity_from_transition_matrix(P, x, deltat):
    """Estimate velocity field from transition matrix (i.e. compute expected displacements)
    
    :param P: transition matrix
    :param x: input data -- `N` points of `M` dimensions in the form of a matrix with dimensions `(N, M)`
    :param deltat: timestep for which `P` was calculated.
    """
    return (P @ x - x)/deltat