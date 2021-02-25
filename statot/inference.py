# statOT core 
# Author: Stephen Zhang (syz@math.ubc.ca)

import numpy as np
import pandas as pd
from numpy import linalg 
import ot
import copy

def statot(x, C = None, eps = None, method = "ent", g = None,
           dt = None, 
           maxiter = 5000, tol = 1e-9,
           verbose = False):
    """Fit statOT model
    
    :param x: input data -- `N` points of `M` dimensions in the form of a matrix with dimensions `(N, M)`
    :param C: cost matrix for optimal transport problem
    :param eps: regularisation parameter 
    :param method: choice of regularisation -- either "ent" (entropy) or "quad" (L2). "unbal" for unbalanced transport is not yet implemented. 
                if "marginals", return just `mu` and `nu`.
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
    if method == "marginals":
        return mu, nu
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