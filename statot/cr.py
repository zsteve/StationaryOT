# statOT cellrank wrapper
# Author: Stephen Zhang (syz@math.ubc.ca)

import numpy as np
import pandas as pd
from numpy import linalg 
import ot
import copy
from scipy.sparse import spmatrix, csr_matrix
from statot.inference import statot, row_normalise

import cellrank
from cellrank import logging as logg
from cellrank.tl.kernels import Kernel, PrecomputedKernel
from cellrank.tl.kernels._base_kernel import (
    _LOG_USING_CACHE
)

def set_terminal_states(adata, sink_idx, labels, terminal_colors):
    """Set terminal states for CellRank API
    
    :param adata: `AnnData` object containing `N` cells. 
    :param sink_idx: string specifying the key in `adata.uns` to a boolean array of length `N`, set to `True` for sinks and `False` otherwise, or the array itself.
    :param labels: string array of length `N` containing lineage names. Only those entries corresponding to sinks will be used.
    :param terminal_colors: colors corresponding to terminal state labels.
    """
    adata.obs["terminal_states"] = None
    adata.obs.loc[sink_idx, "terminal_states"] = labels[sink_idx]
    adata.obs["terminal_states"] = adata.obs["terminal_states"].astype("category")
    adata.uns['terminal_states_colors'] = terminal_colors
    adata.uns['terminal_states_names'] = np.array(adata.obs.terminal_states.cat.categories)
    return 

class OTKernel(Kernel):
    """Kernel class allowing statOT method to be used from CellRank
    
    :param adata: `AnnData` object containing `N` cells. We can use any embedding for statOT, selected when calling `OTKernel.compute_transition_matrix()`. 
    :param source_idx: string specifying the key in `adata.uns` to a boolean array of length `N`, set to `True` for sources and `False` otherwise, or the array itself.
    :param sink_idx: string specifying the key in `adata.uns` to a boolean array of length `N`, set to `True` for sinks and `False` otherwise, or the array itself.
    :param g: string specifying the key in `adata.obs` to a numeric array of length `N`, containing the relative growth rates for cells, or the array itself.
    :param compute_cond_num: set to `True` to compute the condition number of the transition matrix. 
    """
    def __init__(self, adata, source_idx, sink_idx, g, compute_cond_num = False):
        super().__init__(adata, backward = False, compute_cond_num = compute_cond_num, check_connectivity = False)
        if isinstance(source_idx, str):
            self.source_idx = adata.uns[source_idx]
        else:
            assert adata.shape[0] == source_idx.shape[0], "Size of source_idx doesn't match adata!"
            self.source_idx = source_idx
        if isinstance(sink_idx, str):
            self.sink_idx = adata.uns[sink_idx]
        else:
            assert adata.shape[0] == sink_idx.shape[0], "Size of sink_idx doesn't match adata!"
            self.sink_idx = sink_idx
        if isinstance(g, str):
            self.g = adata.obs[g]
        elif g is not None:
            assert adata.shape[0] == g.shape[0], "Size of g doesn't match adata!"
            self.g = g
        else:
            self.g = g
    
    def compute_transition_matrix(self, eps, dt, expr_key = "X_pca", cost_norm_method = None, method = "ent", tol = 0, thresh = 0, maxiter = 5000, C = None, verbose = False):
        """Compute transition matrix using statOT 

        :param eps: regularisation parameter 
        :param dt: choice of the time step over which to fit the model
        :param expr_key: key to embedding to use in `adata.obsm`.
        :param cost_norm_method: cost normalisation method to use. use "mean" to ensure `mean(C) = 1`, or refer to `ot.utils.cost_normalization` in Python OT.
        :param thresh: threshold for output transition probabilities (no thresholding by default)
        :param maxiter: max number of iterations for OT solver
        :param C: cost matrix for optimal transport problem
        :param verbose: detailed output on convergence of OT solver. 
        """
        start = logg.info("Computing transition matrix using statOT")
        params = {"eps" : eps, "cost_norm_method" : cost_norm_method, "expr_key" : expr_key, "dt" : dt, "method" : method, "thresh" : thresh}
        if params == self.params:
            assert self.transition_matrix is not None, _ERROR_EMPTY_CACHE_MSG
            logg.debug(_LOG_USING_CACHE)
            logg.info("    Finish", time=start)
            return self
        self._params = params
        if C is None:
            C = ot.utils.dist(self.adata.obsm[expr_key])
            if cost_norm_method == "mean":
                C = C/C.mean()
            elif cost_norm_method is not None:
                C = ot.utils.cost_normalization(C, norm = cost_norm_method)
        gamma, mu, nu = statot(self.adata.obsm[expr_key], 
                                      C = C, 
                                      eps = eps, 
                                      method = method, 
                                      g = self.g, 
                                      dt = dt,
                                      maxiter = maxiter, 
                                      tol = tol, 
                                      verbose = verbose)
        transition_matrix = row_normalise(gamma, sink_idx = self.sink_idx) 
        if thresh is not None:
            transition_matrix[transition_matrix < thresh] = 0
            transition_matrix = (transition_matrix.T/transition_matrix.sum(1)).T
            
        self._transition_matrix = csr_matrix(transition_matrix)
        self._maybe_compute_cond_num()
        logg.info("    Finish", time=start)
        
    def copy(self) -> "OTKernel":
        k = OTKernel(
            copy(self.transition_matrix),
            adata=self.adata,
            source_idx = self.source_idx,
            sink_idx = self.sink_idx
        )
        k._params = self._params.copy()
        return k
