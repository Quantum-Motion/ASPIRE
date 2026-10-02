# Settings of the numerical experiments, and the helpers the scripts share.
#
# Each script has one named tuple of parameters (LATTICE, FLOORS, CONNECTIVITY, HEAVY_HEX,
# QUANTICS, DECOMPOSITION). The sizes are chosen so that every script finishes in a few minutes on
# a laptop.
#
# ASPIRE is the gate-placement protocol (removable quantum mutual information), MPD the matrix
# product disentangler. Infidelity is eps = 1 - |<psi|C|0>|. Entropies and correlations are in bits.
#
# Usage from a script:  include("settings.jl"); S = LATTICE

using Random, Printf, Statistics, LinearAlgebra
using ITensors, ITensorMPS
using ASPIRE
import JSON3

# ---------------------------------------------------------------------------------------------
# Selector configurations
# ---------------------------------------------------------------------------------------------

# Exact matching by branch and bound over the `shortlist` highest-weight admitted pairs, score R_ij,
# no admission thresholds, no cap on the layer width (it is bounded by the matching alone).
const SELECTOR_MATCHING = (batch_mode=:matching, shortlist=32, matching_max_edges=48,
    tau=0.0, rho=0.0, max_batch_size=nothing, cost_normalised=true, pair_gate_cost=3,
    accept_only=false, min_objective_gain=-1e6)

# Edmonds' blossom algorithm over all admitted pairs.
const SELECTOR_BLOSSOM = (batch_mode=:blossom, shortlist=32, matching_max_edges=48,
    tau=0.0, rho=0.0, max_batch_size=nothing, cost_normalised=false, pair_gate_cost=3,
    accept_only=false, min_objective_gain=-1e6)

# Greedy disjoint packing of pairs ordered by R_ij / (2|i-j|+1)^gamma, admitted if R_ij > 1e-12.
const SELECTOR_GREEDY = (select=:qmi, gate=:mi, min_gain=1e-12)

# Greedy min-cut batching on the mutual information I(i:j) with cross-pair bound kappa = 0.1 bits.
# The admission floors are lifted to zero if a layer would be empty.
const SELECTOR_MINCUT = (score=:mutual_information, tau=1e-4, rho=0.75, kappa=0.1,
    lift_empty_layer=true)

# ---------------------------------------------------------------------------------------------
# lattice_aspire_vs_mpd.jl: TFIM on square lattices, ASPIRE against MPD at equal gate count
# ---------------------------------------------------------------------------------------------

const LATTICE = (
    target=(kind=:tfim, J=1.0, h=2.0, dmrg_cutoff=1e-12),
    # One row per L x L lattice. `layers` is the number of MPD layers, `dmrg_maxdim` the bond
    # dimension ramp of the DMRG target.
    lattices=[
        (L=3, seeds=[1, 2], layers=4, dmrg_sweeps=8, dmrg_maxdim=[10, 20, 40]),
        (L=4, seeds=[1, 2], layers=4, dmrg_sweeps=8, dmrg_maxdim=[10, 20, 40]),
    ],
    # Single truncation used by the selector, the sweeps and the measurement.
    simulation=(max_bond_dim=128, cutoff=1e-10),
    mpd=(tolerance=1e-8, sweeps=0),
    # ASPIRE runs to the MPD gate count; candidate pairs are within distance 2L - 1 on the chain.
    aspire=merge(SELECTOR_MATCHING, (max_depth=400,)),
    # Procrustes sweeps with the rolling environment up to a fixed classical cost, counted in
    # bond-resolved chi^3 operations.
    optimizer=(environment=:rolling, chi3_budget=2e8, bond_resolved=true, max_sweeps=200),
    tcount=(total_error=1e-3,),      # synthesis error budget of the T count
)

# ---------------------------------------------------------------------------------------------
# converged_floors.jl: converged Procrustes floors at fixed layer count
# ---------------------------------------------------------------------------------------------

const FLOORS = (
    N=12,
    targets=[
        (name="heisenberg_4x3", kind=:heisenberg, Lx=4, Ly=3, J=1.0, Delta=1.0, seeds=[1]),
        (name="tfim_4x3",       kind=:tfim, Lx=4, Ly=3, J=1.0, h=2.0, seeds=[1]),
        (name="random_mps",     kind=:random_gaussian, N=12, seeds=[1, 2]),
    ],
    target_chi=16,
    dmrg_sweeps=10, dmrg_maxdim=[8, 16], dmrg_cutoff=1e-12,
    layers=[2, 3],                   # gates = layers x (N - 1)
    simulation=(max_bond_dim=64, cutoff=1e-12),
    mpd=(tolerance=1e-8,),
    # ASPIRE settings tried at each layer count: span penalty gamma with layer width `max_batch`
    # (4), plus width 1 for the gammas in `gamma_max_batch1`. The best floor is reported.
    aspire=merge(SELECTOR_GREEDY, (gamma_max_batch4=[0.0, 0.25], gamma_max_batch1=Float64[],
        max_batch=(4,))),
    # Convergence rule: an interval of the measurement schedule that lowers eps by less than
    # rtol x eps per sweep counts as flat; `patience` consecutive flat intervals end the run.
    # `cap` bounds the number of sweeps.
    optimizer=(environment=:telescoping, rtol=1e-4, patience=3, cap=120, trace_growth=1.15,
        bond_resolved=false),
)

# ---------------------------------------------------------------------------------------------
# connectivity_radius.jl: infidelity of ASPIRE against the connectivity radius
# ---------------------------------------------------------------------------------------------

const CONNECTIVITY = (
    N=12,
    targets=[
        (name="heisenberg_4x3", kind=:heisenberg, Lx=4, Ly=3, J=1.0, Delta=1.0,
            dmrg_maxdim=[4, 8, 16, 32]),
        (name="tfim_3x4", kind=:tfim, Lx=3, Ly=4, J=1.0, h=2.0, dmrg_maxdim=[4, 8, 16, 32, 64]),
    ],
    seeds=[17, 18, 19],
    dmrg_sweeps=16, dmrg_cutoff=1e-12,
    radii=[1, 2, 3, 4, 5, 6, 8, 11],
    mpd=(layers=6, tolerance=1e-8, sweeps=0),      # 6 layers x 11 gates = 66 gates
    aspire=SELECTOR_MINCUT,
    simulation=(max_bond_dim=512, cutoff=1e-10),
)

# ---------------------------------------------------------------------------------------------
# heavy_hex_noise.jl: kicked Ising on a heavy-hex fragment with global depolarising noise
# ---------------------------------------------------------------------------------------------

const HEAVY_HEX = (
    cells=1,                          # one decorated hexagon, N = 12 qubits
    ordering=:routing,
    theta_h=0.4, steps=2,
    simulation=(max_bond_dim=64, cutoff=1e-12),
    mpd=(layers=6, tolerance=1e-8, sweeps=2),
    # ASPIRE restricted to the couplers; every emitted pair gate is one native gate.
    aspire=merge(SELECTOR_MATCHING, (entanglement_weight=0.25, max_depth=60, sweeps=2,
        checkpoint_interval=2, gate_budget=66)),
    # Admission arms: no thresholds ("fill"), rho = 0.9, tau = 1e-3 (bits per 3-CNOT block).
    aspire_arms=[(name="fill", tau=0.0, rho=0.0), (name="rho0.9", tau=0.0, rho=0.9),
        (name="tau1e-3", tau=1e-3, rho=0.0)],
    # Global depolarising model F = q F0 + (1 - q) 2^-N with q = (1 - 16p/15)^M for M native gates.
    # The MPD is charged MPD_SWAPS_PER_GATE SWAPs per emitted gate for routing, each SWAP
    # counting `swap_gate_cost` native gates. `p` are the noise levels of the curves, `p_scan`
    # those of the best-infidelity panel, `x_max` the horizontal range of the curves.
    noise=(p=[3e-4, 1e-3, 3e-3], p_scan=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
        mpd_swaps_per_gate=MPD_SWAPS_PER_GATE, swap_gate_cost=1, x_max=95),
)

# ---------------------------------------------------------------------------------------------
# quantics_matched_budget.jl: quantics Gaussian states at matched gate budget
# ---------------------------------------------------------------------------------------------

const QUANTICS = (
    family="gaussian", rho=0.6, ordering=:serial,
    # (d, n values): d variables with n bits each, N = d n qubits.
    grid=[(d=2, n=3:5), (d=3, n=3:3)],
    seeds=[1, 2],
    simulation=(max_bond_dim=128, cutoff=1e-10),
    mpd=(layers=8, tolerance=1e-8, sweeps=0),
    # Budget is mpd.layers x (N - 1) gates for both arms.
    aspire=merge(SELECTOR_MATCHING, (max_depth=400,)),
)

# ---------------------------------------------------------------------------------------------
# advantage_decomposition.jl: decomposition of the advantage on a linear chain, in CNOTs
# ---------------------------------------------------------------------------------------------

const DECOMPOSITION = (
    N=12,
    targets=[
        (name="heisenberg_4x3", kind=:heisenberg, Lx=4, Ly=3, dmrg_maxdim=[4, 8, 16]),
        (name="tfim_3x4", kind=:tfim, Lx=3, Ly=4, h=2.0, dmrg_maxdim=[4, 8, 16]),
        (name="random_mps", kind=:random_gaussian, N=12, chimax=16),
    ],
    seeds=[17, 18],
    dmrg_sweeps=8, dmrg_cutoff=1e-12,
    simulation=(max_bond_dim=128, cutoff=1e-10),
    mpd=(layers=6, tolerance=1e-8),
    logical_cap=66,                  # gates per arm = the MPD's 6 layers x 11
    native_budget=198,               # CNOTs; 66 gates x 3
    pair_gate_cost=3, swap_cost=3,   # generic SU(4) = 3 CNOTs, SWAP = 3 CNOTs
    # Linear chain; routing cost c_ij = 3 (1 + s_ij) on the running layout.
    aspire=merge(SELECTOR_BLOSSOM, (score_jitter=1e-6, max_depth=4 * 66)),
    rhos=[0.0, 0.75],                # admission thresholds scanned
    batches=[:all],                  # layer widths scanned: all disjoint pairs, or 1
    # The cost-aware arm ranks on R_ij / c_ij with c_ij read from the initial layout; `true`
    # recomputes it on the current layout at every layer.
    cost_aware_tracks_layout=false,
)

# ---------------------------------------------------------------------------------------------
# Helpers shared by the scripts
# ---------------------------------------------------------------------------------------------

"Directory `experiments/out/<name>`, created on demand."
function outdir(name::AbstractString)
    dir = joinpath(@__DIR__, "out", name)
    mkpath(dir)
    return dir
end

"Write a table with column names `cols` and rows given as tuples or vectors."
function write_csv(path::AbstractString, cols, rows)
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((v isa AbstractFloat ? @sprintf("%.10g", v) : string(v) for v in r), ","))
        end
    end
    return path
end

"Write the settings of a run next to its results."
function write_json(path::AbstractString, obj)
    open(io -> JSON3.pretty(io, obj), path, "w")
    return path
end

"Run `f()` with the single truncation pair `sim = (max_bond_dim, cutoff)` of the experiment."
with_truncation(f, sim) = with_sim_config(f; maxdim=sim.max_bond_dim, cutoff=sim.cutoff)

geomean(x) = exp(mean(log.(max.(collect(Float64, x), 1e-16))))

"Amplitude infidelity floor used on logarithmic axes."
floor_eps(x) = max(x, 1e-15)

"""
    build_target(t, seed) -> MPS

Target state of the target description `t` (a named tuple with a `kind`). The seed fixes the DMRG
starting state or the random cores.
"""
function build_target(t, seed::Integer; dmrg_sweeps, dmrg_maxdim=get(t, :dmrg_maxdim, nothing),
        dmrg_cutoff=1e-12)
    rng = MersenneTwister(seed)
    psi = if t.kind === :tfim
        tfim_snake_state(t.Lx, t.Ly; J=get(t, :J, 1.0), h=t.h, nsweeps=dmrg_sweeps,
            maxdim=dmrg_maxdim, cutoff=dmrg_cutoff, rng=rng)
    elseif t.kind === :heisenberg
        heisenberg_snake_state(t.Lx, t.Ly; J=get(t, :J, 1.0), Delta=get(t, :Delta, 1.0),
            nsweeps=dmrg_sweeps, maxdim=dmrg_maxdim, cutoff=dmrg_cutoff, rng=rng)
    elseif t.kind === :random_gaussian
        random_gaussian_mps(t.N; chimax=get(t, :chimax, 64), rng=rng)
    else
        error("unknown target kind $(t.kind)")
    end
    return normalize!(ITensorMPS.dense(psi))
end

"Quantics Gaussian (or Weierstrass) target of the settings `s` for `d` variables of `n` bits."
function quantics_target(s, d::Int, n::Int, seed::Integer; family=s.family, sim=s.simulation,
        extra=Dict{String,Any}())
    spec = Dict{String,Any}("d" => d, "n" => n, "family" => family, "rho" => s.rho,
        "ordering" => String(s.ordering), "max_bond_dim" => sim.max_bond_dim,
        "cutoff" => sim.cutoff)
    merge!(spec, extra)
    return quantics_state_from_spec(spec, seed)
end

"""
    aspire_config(sel; max_depth, max_gates, hardware_graph, tie_break_seed, kwargs...) -> ASPIREConfig

`ASPIREConfig` of a selector description (`SELECTOR_MATCHING`, `SELECTOR_BLOSSOM`, ...). The keyword
arguments override single fields, for example `rho` or `tau` of an admission rule.
"""
function aspire_config(sel; max_depth::Int=400, max_gates::Int=typemax(Int), hardware_graph=nothing,
        tie_break_seed=nothing, kwargs...)
    fields = Dict{Symbol,Any}(
        :batch_mode => sel.batch_mode, :shortlist => sel.shortlist,
        :matching_max_edges => sel.matching_max_edges, :threshold => sel.tau,
        :relative_threshold => sel.rho, :cost_normalised => sel.cost_normalised,
        :pair_gate_cost => sel.pair_gate_cost, :accept_only => sel.accept_only,
        :min_objective_gain => sel.min_objective_gain, :max_depth => max_depth,
        :max_gates => max_gates, :hardware_graph => hardware_graph)
    sel.max_batch_size === nothing || (fields[:max_batch_size] = sel.max_batch_size)
    tie_break_seed === nothing || (fields[:tie_break_seed] = tie_break_seed)
    for (k, v) in kwargs
        fields[k] = v
    end
    return ASPIREConfig(; fields...)
end

"Pairs `(i, j)`, `i < j <= i + r`, of a chain; `r = 1` is the nearest-neighbour set."
radius_graph(N::Int, r::Int) = candidate_pairs(N; max_distance=r)

"""
    aspire_to_budget(psi; pairs, budget, tau, rho, kappa) -> (gates, eps, psi)

ASPIRE selector with greedy min-cut batching on the mutual information with thresholds
`tau`, `rho` and cross-pair bound `kappa`, run layer by layer until `budget` gates are applied (the
last layer is cut short). If a layer would be empty the floors are lifted to zero. Returns the
cumulative gate counts and the infidelities against the fixed |0...0> after every layer.
"""
function aspire_to_budget(psi::MPS; pairs, budget::Int, tau, rho, kappa)
    work = copy(psi)
    gates = Int[0]
    eps = Float64[zero_product_amplitude_infidelity(work)]
    done = 0
    while done < budget
        layer = parallel_disentangling_with_gate_log(work; pairs=pairs, max_depth=1,
            threshold=tau, relative_threshold=rho, couple_threshold=kappa)
        if isempty(layer.gate_log)
            layer = parallel_disentangling_with_gate_log(work; pairs=pairs, max_depth=1,
                threshold=0.0, relative_threshold=0.0, couple_threshold=kappa)
        end
        isempty(layer.gate_log) && break
        for g in first(layer.gate_log, budget - done)
            work = apply_two_qubit_unitary(work, g.i, g.j, g.U)
            done += 1
        end
        push!(gates, done)
        push!(eps, zero_product_amplitude_infidelity(work))
    end
    return gates, eps, work
end

"""
    gates_to_reach(gates, eps, target) -> Float64

Number of gates at which the curve `eps(gates)` first falls to `target`, by interpolation that is
linear in gates and logarithmic in eps. `NaN` if the curve does not reach `target`.
"""
function gates_to_reach(gates, eps, target)
    for k in 2:length(eps)
        if eps[k] <= target
            eps[k - 1] <= target && return Float64(gates[k - 1])
            t = (log(eps[k - 1]) - log(target)) / (log(eps[k - 1]) - log(max(eps[k], 1e-300)))
            return gates[k - 1] + t * (gates[k] - gates[k - 1])
        end
    end
    return NaN
end

"Interpolated value of the curve `eps(gates)` at `g` (log-linear in eps), `NaN` outside the range."
function eps_at(gates, eps, g)
    (g < first(gates) || g > last(gates)) && return NaN
    k = searchsortedfirst(gates, g)
    gates[k] == g && return eps[k]
    t = (g - gates[k - 1]) / (gates[k] - gates[k - 1])
    return exp((1 - t) * log(max(eps[k - 1], 1e-300)) + t * log(max(eps[k], 1e-300)))
end
