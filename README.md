# Oscillatory gene-regulatory loop simulation dataset

This repository contains the simulation and inference code, archived input
matrices, ground-truth networks, and evaluation results for the 1,000 trials
reported in the associated manuscript.

## Simulation data and results

- `data/snapshots/trial_0001.csv` through `trial_1000.csv` are the unordered,
  observation-noise-perturbed expression matrices supplied to the inference
  pipeline. Rows represent simulated single-cell snapshots and columns
  represent genes.
- `data/ground_truth/network_edges.csv` records the source, target, and
  activation or repression type of every true regulatory edge. `trial_id`
  links each edge to its snapshot matrix and result records.
- `results/simulation_results.csv` contains one record per true loop and the outcomes of the
  co-oscillation, loop-gene selection, and BEP reconstruction stages.
  Records sharing a `trial_id` belong to the same simulated trial. A trial is
  end-to-end correct when every loop bearing that `trial_id` has
  `loop_correct_and_bep_correct=True`.
- `results/table_i_counts.csv` contains the numerators, denominators, and
  percentages used to produce Table I.


## Code

- `code/simulation.py` generates synthetic gene-regulatory networks, simulates
  their expression dynamics, and samples unordered noisy snapshots.
- `code/co_oscillating_genes.py` identifies groups of co-oscillating genes
  using the extremum-spread criterion.
- `code/loop_genes.py` selects the feedback-loop genes from each
  co-oscillating group using pairwise dispersion and graph-based analysis.
- `code/bep_algorithm.py` applies Bilateral Extremum Placement (BEP) and
  reconstructs the signed cyclic regulatory network.

The files are organized as importable Python modules corresponding to the
main stages described in the manuscript. They require Python 3.10 or later
and the packages listed in `requirements.txt`.

Install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```
