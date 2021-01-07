# statOT
Dynamic inference from single-cell snapshots by optimal transport.

Placeholder README. Will be replaced soon.

## Installation

For now, clone this repository and run `pip install .` in the top level directory. 

## Usage

For now, the main function is the base `statot.statot()` function:

```python
statot(x, source_idx, sink_idx, sink_weights, C = None, eps = None, method = "ent", g = None,
           flow_rate = None,
           dt = None, 
           maxiter = 5000, tol = 1e-9)
```

To compute fate probabilities to each individual sink, use `compute_fate_probs`:
```python
compute_fate_probs(P, sink_idx)
```

Finally, to compute the fate probabilities by lineage, use 
`compute_fate_probs_lineages`:
```python
compute_fate_probs_lineages(P, sink_idx, labels)
```
where `labels` should be a `np.array` of strings corresponding to the lineage annotation for each cell.