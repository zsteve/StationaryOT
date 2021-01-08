import numpy as np
import pandas as pd
from numpy import linalg 
import ot
import copy
from scipy.sparse import spmatrix, csr_matrix
from statot.inference import statot

import cellrank
from cellrank import logging as logg
from cellrank.tl.kernels import Kernel, PrecomputedKernel
from cellrank.tl.kernels._base_kernel import (
    _LOG_USING_CACHE
)

class OTKernel(Kernel):
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
        else:
            assert adata.shape[0] == g.shape[0], "Size of g doesn't match adata!"
            self.g = g
    
    def compute_transition_matrix(self, eps, dt, expr_key = "X_pca", cost_norm_method = None, sink_weights = None, method = "ent", thresh = 1e-9, C = None):
        start = logg.info("Computing transition matrix using statOT")
        params = {"eps" : eps, "cost_norm_method" : cost_norm_method, "expr_key" : expr_key, "dt" : dt, "sink_weights" : sink_weights, "method" : method, "thresh" : thresh}
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
        if sink_weights is None:
            sink_weights = np.ones(self.adata.shape[0])
        gamma, mu, nu = statot(self.adata.obsm[expr_key], self.source_idx, self.sink_idx,
                                      sink_weights, 
                                      C = C, 
                                      eps = eps, 
                                      method = method, 
                                      g = self.g, 
                                      dt = dt)
        # logg.error("statot() failed to converge, perhaps eps is too small?")

        transition_matrix = (gamma.T/gamma.sum(1)).T
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
