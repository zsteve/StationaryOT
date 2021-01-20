# statOT core 
# Author: Stephen Zhang (syz@math.ubc.ca)

import numpy as np
import pandas as pd
from numpy import linalg 
import ot
import copy

def statot(x, source_idx, sink_idx, sink_weights, C = None, eps = None, method = "ent", g = None,
           flow_rate = None,
           dt = None, 
           maxiter = 5000, tol = 1e-9,
           verbose = False):
    """Fit statOT model
    
    :param x: input data -- `N` points of `M` dimensions in the form of a matrix with dimensions `(N, M)`
    :param source_idx: boolean array of length `N`, set to `True` for sources and `False` otherwise.
    :param sink_idx: boolean array of length `N`, set to `True` for sinks and `False` otherwise.
    :param sink_weights: numeric array of length `N`. Only the entries corresponding to sinks will be used.
    :param C: cost matrix for optimal transport problem
    :param eps: regularisation parameter 
    :param method: choice of regularisation -- either "ent" (entropy) or "quad" (L2). "unbal" for unbalanced transport is not yet implemented. 
    :param g: numeric array of length `N`, containing the relative growth rates for cells.
    :param flow_rate: used only in the growth-free case (flow only)
    :param dt: choice of the time step over which to fit the model
    :param maxiter: max number of iterations for OT solver
    :param tol: relative tolerance for OT solver convergence
    :param verbose: detailed output on convergence of OT solver. 
    :return: gamma, mu, nu
    """
    mu_spt = x
    nu_spt = x
    if g is None:
        mu = np.ones(mu_spt.shape[0])
        mu[sink_idx] = 0
        mu[source_idx] += (flow_rate*dt)/source_idx.sum()
        nu = np.ones(nu_spt.shape[0])
        nu[sink_idx] = (flow_rate*dt)*sink_weights[sink_idx]/sink_weights[sink_idx].sum()
    else:
        mu = g**dt
        mu[sink_idx] = 0
        nu = mu.sum()/x.shape[0]*np.ones(x.shape[0])
    if method == "quad":
        gamma = ot.smooth.smooth_ot_dual(mu, nu, C, eps, numItermax = maxiter, stopThr = tol*mu.sum(), verbose = verbose)
    elif method == "ent":
        gamma = ot.sinkhorn(mu, nu, C, eps, numItermax = maxiter, stopThr = tol*mu.sum(), verbose = verbose)
    elif method == "unbal":
        print("method = 'unbal' not implemented yet")
    # manually check for convergence since POT is a bit iffy
    if verbose:
        print("max inf-norm error for marginal agreement: ",
              max(np.linalg.norm(gamma.sum(1) - mu, ord = np.inf), np.linalg.norm(gamma.sum(0) - nu, ord = np.inf)))
    gamma[np.ix_(sink_idx, sink_idx)] = np.eye(sink_idx.sum())
    return gamma, mu, nu

def row_normalise(gamma):
    """Row normalise coupling to produce transition matrix

    :param gamma: coupling produced by `statot()`
    :return: transition matrix obtained by row-normalising the input `gamma`.
    """
    return (gamma.T/gamma.sum(1)).T

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
