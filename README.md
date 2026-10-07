# ASPIRE

ASPIRE is a Julia package for preparing matrix product states (MPS) on a quantum computer with few
two-qubit gates. It implements ASPIRE, a gate-placement protocol that scores every admissible qubit
pair by its removable quantum mutual information, the exact decrease of the total correlation that
one two-qubit unitary on the pair can achieve, selects a set of disjoint pairs by maximum-weight
matching, applies the optimal (eigenbasis) gate to a working copy of the target and repeats. The
inverses of the applied gates form the preparation circuit, which can be refined by Procrustes
sweeps. Costs are counted in native two-qubit gates, including the SWAPs needed on a device with
limited connectivity. The baseline is the matrix product disentangler (MPD) of Ran, a
nearest-neighbour staircase of analytically computed gates, which the package wraps for
like-for-like comparison. The package also contains the certificate `Lambda` that bounds the
infidelity from the remaining correlation, the global depolarising noise model used for the
heavy-hex comparison, and the target states of the paper (lattice ground states, kicked-Ising
states, quantics function states, toy states).

## Installation

Julia 1.12 is required. From the root of a checkout of this repository:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

ASPIRE builds on MPSCircuits.jl (https://zenodo.org/records/23074431), which provides the circuit data
structures, the analytic MPD layers and the Procrustes gate updates. The version used for the results
is pinned in `Project.toml` and installed by the command above.

## Quick start

```julia
using ASPIRE, ITensorMPS, Random

set_max_bond_dim!(256)                                   # single truncation used throughout
psi = tfim_snake_state(4, 3; h=2.0, nsweeps=8, maxdim=[10, 20, 40], rng=MersenneTwister(1))

mpd = ran_iter_oall_curve(psi; Lmax=3, sweeps=0)         # MPD, 3 layers, no optimisation
cfg = ASPIREConfig(threshold=0.0, relative_threshold=0.0, max_gates=mpd.gates[end])
asp = aspire(psi; config=cfg)                           # ASPIRE with the same gate count

println("gates:  MPD ", mpd.gates[end], "   ASPIRE ", asp.gates[end])
println("eps:    MPD ", 1 - sqrt(mpd.fid[end]), "   ASPIRE ", asp.product_frame_amplitude_infidelity[end])
```

`asp.gate_log` holds the applied gates; `prep_circuit_from_gate_log` turns them into a preparation
circuit and `optimize_to_floor` refits it by Procrustes sweeps.

## Repository layout

| Path | Content |
|---|---|
| `src/ASPIRE.jl` | module that includes and re-exports the submodules below |
| `src/SimConfig.jl` | global truncation parameters (maximum bond dimension, SVD cutoff) |
| `src/FidelityMetrics.jl` | squared fidelity, amplitude fidelity and infidelity conventions |
| `src/Matching.jl` | maximum-weight matching (Edmonds' blossom algorithm) |
| `src/EntanglementBasics.jl`, `src/RDM.jl` | bond entropies, two-site reduced density matrices, mutual information |
| `src/HardwareCost.jl` | routing on a coupling graph, SWAP counting, running-layout cost, site orderings |
| `src/Correlation.jl` | total correlation, removable correlation, the certificate `Lambda` and its infidelity bounds |
| `src/Disentangling.jl` | two-qubit gates, eigenbasis disentanglers, greedy min-cut batching |
| `src/Selection.jl` | the ASPIRE selector (`ASPIREConfig`, `aspire_step`, `aspire`) |
| `src/ModelStates.jl` | TFIM and Heisenberg snake states, heavy-hex patches, kicked-Ising states |
| `src/ToyStates.jl`, `src/QuanticsStates.jl` | GHZ, Bell-pair and cluster states; quantics function states |
| `src/Saturation.jl` | saturation stopping rule for gate budgets |
| `src/TCount.jl` | T-gate count of a circuit under rotation synthesis models |
| `src/OptimizerCost.jl` | classical cost of a Procrustes sweep |
| `src/CircuitOptimization.jl` | preparation circuits, product-frame infidelity, Procrustes sweeps, convergence rule |
| `src/WarmStartSelection.jl` | gate selection by the Procrustes score and greedy selection with span penalty |
| `src/EnergyProcrustes.jl` | energy refinement of a circuit by coordinate updates |
| `src/Baselines.jl` | the MPD circuit and curve drivers, random Gaussian MPS |
| `src/Noise.jl` | global depolarising noise and the MPD routing charge |
| `notebooks/` | four Jupyter notebooks, see below |
| `experiments/` | experiment scripts and their settings, see `experiments/README.md` |
| `test/` | unit tests |

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Notebooks

- `01_aspire_vs_mpd`: ASPIRE against the MPD on lattice ground states, with the certified bound
- `02_gate_selection`: removable correlation of a pair, matching rules, thresholds
- `03_optimization`: Procrustes refitting, warm-start selection, energy refinement
- `04_hardware_and_toy_models`: connectivity, routing cost, heavy-hex noise, GHZ and quantics states

They run in Jupyter through IJulia (`julia --project=notebooks -e 'using IJulia; notebook()'`). Each
notebook activates its own environment, `notebooks/Project.toml`, in its first cell.

## Experiments

`experiments/` holds `settings.jl`, the parameters of each experiment, and one script per experiment
that writes tables and plots to `experiments/out/`. See `experiments/README.md`.

```
julia --project=experiments experiments/connectivity_radius.jl
```

## Conventions

- Entropies, mutual informations and the total correlation are in bits (`log2`).
- The infidelity of a preparation circuit `C` is `eps = 1 - |<psi|C|0>|`, the amplitude infidelity.
  The squared fidelity is `|<psi|C|0>|^2`, so `eps = 1 - sqrt(F)`.
- The native two-qubit gate is the CNOT. A generic two-qubit unitary costs 3 CNOTs and a SWAP costs 3
  CNOTs.
- A pair `(i, j)` at graph distance `s_ij + 1` costs `c_ij = 3 (1 + s_ij)` CNOTs. SWAPs are charged on
  the running layout: the qubit permutation left by one gate is kept for the next.
- Product-frame convention: a layer of single-qubit gates in front of the circuit is free. The
  `product_frame_*` quantities use the bond-dimension-one truncation of the residual state as the
  frame, which is an upper bound on the infidelity with the optimal product frame. The `zero_product_*`
  quantities use the fixed state `|0...0>`.
- Matching-based selection (`batch_mode=:matching`) is exact over the `shortlist` highest-weight
  admitted pairs; `batch_mode=:blossom` is exact over all admitted pairs.

## Reproducibility

The stored notebook outputs were produced with the default BLAS thread settings. Gate scores,
matchings, certificates and cost models reproduce to rounding error under any thread count. Circuits
refit with a few Procrustes sweeps depend on the order of floating-point summation, so their
infidelities can change with the BLAS thread count and with the library versions in `Manifest.toml`.

## License

GNU Affero General Public License v3.0 (AGPL-3.0), see `LICENSE`.

## Citation

If you use this code, please cite "Breaking the chain: geometry-native state preparation with ASPIRE" ([arXiv:2610.03528](https://arxiv.org/abs/2610.03528)).
