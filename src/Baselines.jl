"""
    Baselines

The matrix product disentangler (MPD) of Ran, the baseline against which ASPIRE is compared
(described in the accompanying paper), and the layered min-cut disentangler used for the connectivity study.
The MPD compiles a nearest-neighbour staircase layer by layer with the analytic disentangler of
`MPSCircuits`. Here it is wrapped as a circuit constructor and as a curve driver that records the
quantities of the comparison after every layer, with optional Procrustes optimisation of the
accumulated circuit.
"""
module Baselines

using ITensors
using ITensorMPS
using Random
import MPSCircuits

using ..SimConfig: global_cutoff, global_maxdim
using ..FidelityMetrics: amplitude_infidelity_from_squared
using ..Disentangling: ASPIRE_DEFAULT_THRESHOLD, ASPIRE_DEFAULT_RELATIVE_THRESHOLD,
    candidate_pairs, two_qubit_gate, apply_two_qubit_unitary, mi_disentangler,
    aspire_rank_threshold, mincut_batch, total_entanglement, total_correlation
using ..RDM: two_site_rdm, dense_two_site_matrix, quantum_mutual_information
using ..Selection: ASPIREConfig, aspire_step
using ..Saturation: SaturationRule, saturation_state
using ..TCount: t_count, TRASYN
using ..CircuitOptimization: prep_from_disentanglers, disentangle_with_prep_circuit,
    oall_sweep!, circuit_checkpoint, OPTIMIZER_MAXDIM, OPTIMIZER_CUTOFF

export mpd_circuit, ran_iter_oall_curve, mpd_checkpoints, aspire_checkpoints,
    aspire_layer_logged, zero_state_fidelity, random_gaussian_mps

"""
    zero_state_fidelity(work) -> Float64

Squared overlap `|<0...0|work>|^2` of the normalised state `work` with the all-up product state.
"""
function zero_state_fidelity(work::MPS)
    ref = MPS(siteinds(work), fill("Up", length(work)))
    return abs2(inner(ref, normalize!(copy(work))))
end

"""
    mpd_circuit(psi, layers; tolerance=1e-8, max_bond_dim=4096, cutoff=1e-12)
        -> Vector{MPSCircuits.UnitaryGate}

Preparation circuit of the matrix product disentangler with `layers` layers, without
optimisation: the analytic nearest-neighbour staircase compiled by
`MPSCircuits.compile_mps_circuit(..., DecomposeAllAnalytical())`. `tolerance` is the truncation
tolerance of the analytic disentangler (`1e-8` in the paper); the working state is truncated at
`max_bond_dim` and `cutoff`.
"""
function mpd_circuit(psi::MPS, layers::Int; tolerance::Float64=1e-8,
        max_bond_dim::Int=OPTIMIZER_MAXDIM, cutoff::Float64=OPTIMIZER_CUTOFF)
    return redirect_stdout(devnull) do
        MPSCircuits.compile_mps_circuit(ITensorMPS.dense(copy(psi)),
            MPSCircuits.DecomposeAllAnalytical(); n_layers_max=layers, tolerance=tolerance,
            max_bond_dim=max_bond_dim, working_cutoff=cutoff, backend=:cpu)
    end
end

"""
    random_gaussian_mps(N=12; chimax=64, rng=MersenneTwister(1)) -> MPS

Random `N`-site MPS built by `ITensorMPS.random_mps` with link dimension `chimax`, whose core
tensor entries are then replaced by independent complex Gaussians (real and imaginary parts of
unit variance), and normalised. Links wider than the Schmidt rank allowed by the cut,
`min(2^b, 2^(N-b))`, are redundant. The singular value spectra decay slowly, which makes the
state a hard target for shallow circuits (Rudolph et al., arXiv:2209.00595).
"""
function random_gaussian_mps(N::Int=12; chimax::Int=64, rng::AbstractRNG=MersenneTwister(1))
    s = siteinds("S=1/2", N)
    psi = random_mps(rng, s; linkdims=chimax)
    for j in 1:N
        is = inds(psi[j])
        psi[j] = ITensors.itensor(randn(rng, ComplexF64, dim.(is)...), is...)
    end
    return normalize!(psi)
end

"""
    aspire_layer_logged(psi; threshold=1e-3, relative_threshold=0.10, max_rank=nothing,
        couple_threshold=0.1, disentangler=mi_disentangler, score=quantum_mutual_information)
        -> (psi, gates)

One layer of the greedy min-cut selector of the connectivity study of the accompanying paper:
all pairs are scored by `score` on their two-site density matrix, a batch is formed by
[`mincut_batch`](@ref) with the pair mutual information as coupling (cross-pair bound
`couple_threshold`, `kappa = 0.1` in the paper) and the score floor
`max(threshold, relative_threshold * best score)`, and `disentangler` is applied to each pair.
Returns the updated state and the disentangling gates as `MPSCircuits.UnitaryGate`s in
application order, ready for [`prep_from_disentanglers`](@ref).
"""
function aspire_layer_logged(psi::MPS; threshold::Float64=ASPIRE_DEFAULT_THRESHOLD,
        relative_threshold::Float64=ASPIRE_DEFAULT_RELATIVE_THRESHOLD,
        max_rank::Union{Nothing,Int}=nothing,
        couple_threshold::Float64=0.1, disentangler=mi_disentangler,
        score=quantum_mutual_information)
    N = length(psi); s = siteinds(psi)
    pairs = candidate_pairs(N; adjacent=false)
    rhos = Dict(p => dense_two_site_matrix(two_site_rdm(psi, p[1], p[2]), s, p[1], p[2]) for p in pairs)
    rank_of = Dict(p => real(score(rhos[p])) for p in pairs)
    cache = Dict{Tuple{Int,Int},Float64}(p => real(quantum_mutual_information(rhos[p])) for p in pairs)
    couple = function (a, b)
        i, j = minmax(a, b)
        get!(cache, (i, j)) do
            real(quantum_mutual_information(dense_two_site_matrix(two_site_rdm(psi, i, j), s, i, j)))
        end
    end
    rank_threshold = aspire_rank_threshold(values(rank_of);
        threshold=threshold, relative_threshold=relative_threshold)
    batch = mincut_batch(pairs, rank_of; couple=couple,
        rank_threshold=rank_threshold, couple_threshold=couple_threshold,
        max_rank=max_rank)
    gates = MPSCircuits.UnitaryGate[]
    for (i, j) in batch
        U = disentangler(psi, i, j)
        push!(gates, MPSCircuits.UnitaryGate(two_qubit_gate(U, s[i], s[j])))
        psi = apply_two_qubit_unitary(psi, i, j, U)
        s = siteinds(psi)
    end
    return psi, gates
end

# One analytic MPD layer applied to the working state `work`. Returns the updated working state,
# the layer (or `nothing`) and a flag that is true once the state is a product state.
function _mpd_layer(work::MPS, tolerance::Float64, k::Int)
    return redirect_stdout(devnull) do
        MPSCircuits.generate_layer!(work; tolerance=tolerance, layer_number=k,
            max_bond_dim=global_maxdim(), working_cutoff=global_cutoff())
    end
end

"""
    ran_iter_oall_curve(psi; Lmax=20, sweeps=10, tolerance=1e-8, converged_tol=1e-8,
        t_error=1e-3, t_model=TRASYN, stop=nothing) -> NamedTuple

MPD curve with optional Procrustes optimisation, recorded layer by layer (`Iter[D_i O_all]` of
Rudolph et al.). In layer `k` the analytic MPD layer is generated from the residual of layer
`k - 1` (tolerance `tolerance`), prepended to the circuit, and the product-frame single-qubit
layer is rebuilt from the current residual and placed in front of the circuit. If `sweeps > 0`,
`sweeps` Procrustes sweeps ([`oall_sweep!`](@ref)) then refit the two-qubit gates against `psi`,
holding the frame fixed. The residual of the framed circuit is recorded.
`sweeps = 0` gives the unoptimised MPD curve. The paper's unoptimised curves use `sweeps=0` and
up to 24 layers for the largest system.

The truncation is the global one (`set_max_bond_dim!`, `set_cutoff!`). `Lmax` bounds the number
of layers; the loop ends earlier when the state becomes a product state (`actual_depth` is then
that layer) and `actual_depth` also records the first layer with residual entanglement below
`converged_tol`. `stop`, a [`Saturation.SaturationRule`](@ref), ends the run once the curve has
stopped paying, and makes `Lmax` a ceiling. `t_error` is the total synthesis error and `t_model`
the synthesis model of the `T`-gate count recorded per layer ([`TCount.t_count`](@ref)).

Returns the series `gates` (two-qubit gates), `depth`, entanglement `E`, total correlation `T`,
squared fidelity `fid` of the framed circuit against `|0...0>`, and `tcount`, each starting from
the zero-gate point, together with `actual_depth`, the final residual `final`, the full framed
`circuit` whose fidelity `fid` reports and the bare two-qubit `two_qubit_circuit`.
"""
function ran_iter_oall_curve(psi::MPS; Lmax::Int=20, sweeps::Int=10,
        tolerance::Float64=1e-8, converged_tol::Float64=1e-8,
        t_error::Float64=1e-3, t_model=TRASYN,
        stop::Union{Nothing,SaturationRule}=nothing)
    target = ITensorMPS.dense(copy(psi))
    work_for_layers = deepcopy(target)
    # `circuit` holds the two-qubit layers; `framed` is the preparation circuit with the
    # product-frame layer in front, against which every fidelity is measured.
    circuit = MPSCircuits.UnitaryGate[]
    framed = MPSCircuits.UnitaryGate[]
    final = target
    gates = Int[0]; depth = Int[0]
    E = Float64[total_entanglement(target)]; T = Float64[total_correlation(target)]
    fid = Float64[zero_state_fidelity(target)]
    tcount = Float64[0.0]
    converged_at = nothing
    N = length(target)

    for k in 1:Lmax
        work_for_layers, layer, flag = _mpd_layer(work_for_layers, tolerance, k)
        if flag || layer === nothing
            converged_at = min(something(converged_at, Lmax), k - 1)
            break
        end
        prepend!(circuit, layer)
        # The frame is rebuilt from the current residual at every layer and applied to a copy,
        # so that `circuit` stays a pure list of two-qubit gates: composing it into the circuit
        # once per layer would leave a stale single-qubit layer behind for every earlier layer.
        framed, _ = MPSCircuits.with_single_qubit_rotations(circuit, work_for_layers)
        nframe = length(framed) - length(circuit)
        oall_sweep!(target, framed; sweeps=sweeps,
                    gate_indices=(nframe + 1):length(framed))
        sweeps > 0 && copyto!(circuit, view(framed, (nframe + 1):length(framed)))

        final = disentangle_with_prep_circuit(target, framed)
        push!(gates, length(circuit)); push!(depth, k)
        push!(E, total_entanglement(final)); push!(T, total_correlation(final))
        push!(fid, zero_state_fidelity(final))
        push!(tcount, t_count(framed; total_error=t_error, model=t_model))
        if E[end] < converged_tol && converged_at === nothing
            converged_at = k
        end
        if stop !== nothing
            infid = amplitude_infidelity_from_squared.(fid)
            saturated, reason, g_first = saturation_state(gates, infid, stop; N=N)
            if saturated
                @info "MPD stopped: $reason" layer=k gates=gates[end] g_first
                converged_at = something(converged_at, k)
                break
            end
        end
    end

    return (gates=gates, depth=depth, E=E, T=T, fid=fid, tcount=tcount,
            actual_depth=something(converged_at, Lmax), final=final,
            circuit=deepcopy(framed), two_qubit_circuit=deepcopy(circuit))
end

# Number of two-qubit gates of a circuit; single-qubit frame gates are free.
_count_two_qubit(circuit) = count(g -> length(g.site_numbers) == 2, circuit)

"""
    mpd_checkpoints(psi; layers=6, sweeps=0, tolerance=1e-8) -> Vector{NamedTuple}

Framed MPD circuits after every layer, for noisy-circuit evaluation. Entry `k + 1` is
the [`circuit_checkpoint`](@ref) after `k` layers, each followed by `sweeps` Procrustes sweeps of
the two-qubit gates with the frame held fixed; entry 1 is the zero-gate checkpoint. The MPD
acts on MPS-adjacent pairs only; which of them are device couplers is determined downstream from the
site pairs of the circuits.
"""
function mpd_checkpoints(psi::MPS; layers::Int=6, sweeps::Int=0, tolerance::Float64=1e-8)
    target = ITensorMPS.dense(copy(psi))
    work = deepcopy(target)
    circuit = MPSCircuits.UnitaryGate[]
    records = [circuit_checkpoint(target, circuit, 0, 0)]
    for k in 1:layers
        work, layer, stop = _mpd_layer(work, tolerance, k)
        (stop || layer === nothing) && break
        prepend!(circuit, layer)
        framed, _ = MPSCircuits.with_single_qubit_rotations(circuit, work)
        nframe = length(framed) - length(circuit)
        oall_sweep!(target, framed; sweeps=sweeps, gate_indices=(nframe + 1):length(framed))
        sweeps > 0 && copyto!(circuit, view(framed, (nframe + 1):length(framed)))
        push!(records, circuit_checkpoint(target, framed, k, _count_two_qubit(framed)))
    end
    return records
end

"""
    aspire_checkpoints(psi, config; pairs=candidate_pairs(length(psi)), sweeps=0,
        gate_budget=nothing, checkpoint_interval=1) -> Vector{NamedTuple}

Framed ASPIRE circuits during construction, for noisy-circuit evaluation. Layers are selected by
`aspire_step` with `config`, restricted to the admissible `pairs` (for a device, its couplers,
so that every gate is native and costs one two-qubit gate with no routing), until `gate_budget`
gates, `config.max_depth` layers or an empty layer. After every `checkpoint_interval`-th layer
(every layer if `sweeps == 0`) and at the end, the circuit is refit with `sweeps` Procrustes sweeps
and recorded as a [`circuit_checkpoint`](@ref); entry 1 is the zero-gate checkpoint.
"""
function aspire_checkpoints(psi::MPS, config::ASPIREConfig;
        pairs=candidate_pairs(length(psi); adjacent=false), sweeps::Int=0,
        gate_budget::Union{Nothing,Integer}=nothing, checkpoint_interval::Int=1)
    target = ITensorMPS.dense(copy(psi))
    work = deepcopy(target)
    circuit = MPSCircuits.UnitaryGate[]
    interval = sweeps == 0 ? 1 : checkpoint_interval
    records = [circuit_checkpoint(target, circuit, 0, 0)]
    allowed = Set(Tuple(p) for p in pairs)
    last_step, last_recorded = 0, 0

    record! = function (step_index)
        oall_sweep!(target, circuit; sweeps=sweeps)
        push!(records, circuit_checkpoint(target, circuit, step_index, length(circuit)))
        last_recorded = step_index
    end

    for step_index in 1:config.max_depth
        gate_budget !== nothing && length(circuit) >= gate_budget && break
        remaining = gate_budget === nothing ? typemax(Int) : gate_budget - length(circuit)
        s = siteinds(work)
        step = aspire_step(work; config=config, pairs=pairs, layer=step_index,
            gate_limit=remaining)
        isempty(step.gates) && break
        work = step.psi
        layer = [MPSCircuits.UnitaryGate(two_qubit_gate(g.U, s[g.i], s[g.j]))
            for g in step.gates]
        @assert all(minmax(g.i, g.j) in allowed for g in step.gates)
        prepend!(circuit, prep_from_disentanglers(layer))
        last_step = step_index
        reached = gate_budget !== nothing && length(circuit) >= gate_budget
        (step_index % interval == 0 || reached) && record!(step_index)
    end
    last_step > last_recorded && record!(last_step)
    return records
end

end # module
