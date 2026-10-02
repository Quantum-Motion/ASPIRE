# Experiments

Scripts that run the ASPIRE pipeline and the matrix product disentangler (MPD) baseline with the
package API, and `settings.jl`, which holds the parameters of every script. See the accompanying
paper for the motivation and the definitions.

## Running

```
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments experiments/<script>.jl
```

Output goes to `experiments/out/<script>/` (ignored by git): CSV tables, `settings.json` with the
parameters of the run, and the plot as PNG and PDF. The first run includes package compilation.

## Scripts

| Script | What it computes | Runtime |
|---|---|---|
| `lattice_aspire_vs_mpd.jl` | TFIM ground states of 3x3 and 4x4 lattices: infidelity against gate count for both arms, T count, infidelity ratio at equal gates and gate saving at equal infidelity, and both arms refit by Procrustes sweeps at equal classical cost | 2 min |
| `converged_floors.jl` | Infidelity floors of the MPD and ASPIRE circuits of equal gate count after Procrustes sweeps run to convergence (Heisenberg 4x3, TFIM 4x3, random MPS) | 5 min |
| `connectivity_radius.jl` | Infidelity of ASPIRE at the MPD gate count against the connectivity radius, and the first radius at which it beats the MPD | 2 min |
| `heavy_hex_noise.jl` | Kicked Ising state on a heavy-hex hexagon under global depolarising noise: noisy infidelity against native gates, and best infidelity against the error rate | 1 min |
| `quantics_matched_budget.jl` | Correlated Gaussian quantics states with d variables of n bits: best infidelity of each arm within the MPD gate budget, and the ratio against d | 1 min |
| `advantage_decomposition.jl` | Infidelity against native CNOTs on a linear chain for the MPD, the MPD pattern with ASPIRE gates, nearest-neighbour ASPIRE, and routed, free and cost-aware ASPIRE on all pairs | 2 min |

Runtimes are wall times on a laptop with default thread settings.

## Cost model of each script

| Script | Gate counting |
|---|---|
| `lattice_aspire_vs_mpd.jl` | All-to-all. One gate per two-qubit unitary; the T count of the framed circuit of both arms is recorded. Procrustes cost is counted in bond-resolved chi^3 operations. |
| `converged_floors.jl` | All-to-all. Gate count only. |
| `connectivity_radius.jl` | All-to-all within the radius. Gate count only. |
| `heavy_hex_noise.jl` | One native gate per emitted unitary on the couplers. The MPD is charged `MPD_SWAPS_PER_GATE` SWAPs per gate, each SWAP one native gate. The threshold tau is applied to R/c with c = 3. |
| `quantics_matched_budget.jl` | All-to-all. One gate per two-qubit unitary. |
| `advantage_decomposition.jl` | Linear chain, running layout, c_ij = 3(1 + s_ij) CNOTs (3 per generic gate, 3 per SWAP). |

Infidelity is measured in the product frame (the bond-dimension-one truncation frame) in
`lattice_aspire_vs_mpd.jl`, `heavy_hex_noise.jl` and `quantics_matched_budget.jl`. The other scripts
score ASPIRE against the fixed `|0...0>`, while the MPD circuit carries its analytic frame.
