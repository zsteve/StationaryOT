# Stationary OT

Entropy-regularized optimal transport (OT) has been applied to infer cellular trajectories in time-course data[1]. Stationary OT extends this approach to snapshots of biological systems in equilibrium. A system is in equilibrium if you would expect the same proportions of populations in each snapshot, though individual cells progress along trajectories. 

We model biological processes as a diffusion-drift process subject to branching due to cell birth and death. To maintain equlibrium we define source and sink regions, where cells are created in the source regions and are absorbed in the sinks subject to birth and death rates. At a population level, the effects of growth, entry, and exit can be captured by a spatially dependent flux, R(x), and population dynamics can be described by a population balance partial differential equation:
![Population Balance PDE](https://github.com/zsteve/statOT/blob/main/aux_files/pop_balance_PDE.png)

where ![equilibrium equation](https://github.com/zsteve/statOT/blob/main/aux_files/equilibrium-eqn.png) by our equilibirum assumption.

This problem has previously been explored by Weinreb et. al who approach the problem by solving a system of linear equations to recover the potential. In contrast, stationary OT solves a convex optimization problem for the transition probabilities. This approach has the flexibility to allow additional information, such as RNA velocity, which may allow recovery of non-conservative dynamics such as oscillations. Combined with earlier OT approaches, stationary OT provides a framework for approaching both time-series and snapshot data.

This package provides the ability to run stationary OT on single-cell expression data to recover transition and fate probabilities. This package also provides the interface to use stationary OT with CellRank.

## Installation

For now, clone this repository and run `pip install .` in the top level directory. 

## Usage

### Inputs
1. **Projected Expression Matrix**: An n x m matrix of cells by expression data. We recommend projecting the data into X-Y dimensions using PCA.
2. **Sinks**: An n x 1 boolean vector indicating cells that leave the system.
3. **Sources**: An n x 1 boolean vector indicating cells that are where mass is entering the system. Typically, this will be the complement of the sink vector.
4. **Sink Weights**: An n x 1 vector indicating the weight of each sink, where sinks with higher weight will absorb more mass.
5. **Cost Matrix**: An n x n matrix specifiying the cost for each cell transporting to each other cell in the coupling, such as the Euclidean distance between cells in PCA coordinates.
6. **Growth Rates**: An n x 1 vector specifying the expected number of decendents of each cell in $dt = 1$.
7. **dt**: The timestep between snapshots.

### Quick Start

First use the `statot.statot` function to calculate transition probabilities for cells in a single timestep:

```python
statot(x, source_idx, sink_idx, sink_weights, C = None, eps = None, method = "ent", g = None,
           flow_rate = None,
           dt = None, 
           maxiter = 5000, tol = 1e-9)
```

To compute fate probabilities by sink, use `compute_fate_probs`:
```python
compute_fate_probs(P, sink_idx)
```

Finally, to compute the fate probabilities by lineage, use 
`compute_fate_probs_lineages`:
```python
compute_fate_probs_lineages(P, sink_idx, labels)
```
where `labels` should be a `np.array` of strings corresponding to the lineage annotation for each cell.
