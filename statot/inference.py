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
    gamma[np.ix_(sink_idx, sink_idx)] = np.eye(sink_idx.sum())
    return gamma, mu, nu

# transliterated from Julia code
def _compute_NS(P, sink_idx):
    Q = P[np.ix_(~sink_idx, ~sink_idx)]
    S = P[np.ix_(~sink_idx, sink_idx)]
    # N = np.linalg.inv(np.eye(Q.shape[0]) - Q) BAD!!!!
    N = np.eye(Q.shape[0]) - Q
    return N, S

def compute_fate_probs(P, sink_idx):
    N, S = _compute_NS(P, sink_idx)
    B = np.zeros((P.shape[0], sink_idx.sum()))
    B[~sink_idx, :] = np.linalg.solve(N, S)
    B[sink_idx, :] = np.eye(sink_idx.sum())
    return B

def compute_fate_probs_lineages(P, sink_idx, labels):
    B = compute_fate_probs(P, sink_idx)
    sink_labels = np.unique(labels[sink_idx])
    B_lineages = np.array([B[:, labels[sink_idx] == i].sum(1) for i in sink_labels]).T
    return B_lineages, sink_labels