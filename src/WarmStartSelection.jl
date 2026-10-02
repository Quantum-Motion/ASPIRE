"""
    WarmStartSelection

Gate selection by the Procrustes score `P(i,j)` (warm-start Procrustes optimisation).
For the residual `|w> = C^dagger |psi_target>` of the current circuit `C`, `P(i,j)` is the largest
overlap with `|0...0>` that a single two-qubit gate on `(i,j)` can reach, which equals the norm of
the four amplitudes `v_ij = <0_rest|w>` obtained by projecting every other site onto `<0|`. A gate
that rotates `v_ij` onto `|00>` attains it, so ranking pairs by `P` ranks them by the fidelity
they can realise. The score is informative only once the residual has weight on `|0...0>`;
the schedule [`switch_schedule`](@ref) therefore selects by removable QMI for the first `k` layers
and by `P` afterwards.
"""
module WarmStartSelection

using ITensors
using ITensorMPS
using LinearAlgebra
import MPSCircuits

using ..Disentangling: candidate_pairs, two_qubit_gate, apply_two_qubit_unitary
using ..RDM: two_site_rdm, all_pairs_two_site_rdms, dense_two_site_matrix
using ..Selection: removable_mutual_information, eigenbasis_gate
using ..OptimizerCost: sweep_cost_model
using ..CircuitOptimization: prep_from_disentanglers, rudolph_infidelity,
    OPTIMIZER_MAXDIM, OPTIMIZER_CUTOFF

export procrustes_pair_scores, frame_optimal_gate, psel_layer, switch_schedule,
    protocol_arb_iter_oall, ARB_PROPOSALS

# `<0|` as an ITensor on the site index `s` (the reference `|0...0>` is the all-up state).
zero_bra(s::Index) = dag(ITensor(ComplexF64[i == 1 ? 1.0 : 0.0 for i in 1:dim(s)], s))

# `v_ij = <0_rest|w>` for every pair, as the 2x2 matrix `v[a,b] = <0..a..b..0|w>`. A right-to-left
# environment pass and, for each left site, one left-to-right accumulation: `O(N^2)`
# contractions of cost `O(chi^2)`.
function _pair_slices(w::MPS, pairs)
    N = length(w)
    s = siteinds(w)
    z = [zero_bra(s[k]) for k in 1:N]
    right = Vector{ITensor}(undef, N + 1)
    right[N + 1] = ITensor(ComplexF64(1))
    for k in N:-1:1
        right[k] = (w[k] * z[k]) * right[k + 1]
    end
    left = Vector{ITensor}(undef, N + 1)
    left[1] = ITensor(ComplexF64(1))
    for k in 1:N
        left[k + 1] = (left[k] * w[k]) * z[k]
    end

    wanted = Dict{Int,Set{Int}}()
    for (i, j) in pairs
        a, b = minmax(i, j)
        push!(get!(wanted, a, Set{Int}()), b)
    end

    out = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for i in sort(collect(keys(wanted)))
        partners = wanted[i]
        mid = left[i] * w[i]
        for j in (i + 1):N
            if j in partners
                tensor = (mid * w[j]) * right[j + 1]
                v = Matrix{ComplexF64}(undef, 2, 2)
                for a in 1:2, b in 1:2
                    v[a, b] = tensor[s[i] => a, s[j] => b]
                end
                out[(i, j)] = v
            end
            j < N && (mid = mid * (w[j] * z[j]))
        end
    end
    return out
end

# The 2x2 slice flattened in the basis order `|00>, |01>, |10>, |11>` of `two_qubit_gate`.
_slice_vector(v::AbstractMatrix) = ComplexF64[v[1, 1], v[1, 2], v[2, 1], v[2, 2]]

"""
    procrustes_pair_scores(w; pairs=candidate_pairs(length(w))) -> (score, gain, f0, slices)

Procrustes scores of the residual `w`. `score[p] = P(i,j) = ||v_ij||` is the amplitude fidelity
reachable by placing one optimised gate on `p = (i,j)`, `f0 = |<0...0|w>|` the current amplitude
fidelity and `gain[p] = score[p] - f0` the improvement. `slices[p]` is the 2x2 matrix `v_ij`.
"""
function procrustes_pair_scores(w::MPS; pairs=candidate_pairs(length(w); adjacent=false))
    slices = _pair_slices(w, pairs)
    score = Dict{Tuple{Int,Int},Float64}()
    f0 = 0.0
    for (p, v) in slices
        score[p] = norm(v)
        f0 = abs(v[1, 1])
    end
    gain = Dict(p => score[p] - f0 for p in keys(score))
    return (score=score, gain=gain, f0=f0, slices=slices)
end

"""
    frame_optimal_gate(v) -> Matrix{ComplexF64}

Unitary that rotates the slice `v` onto `|00>`: the Householder reflection taking `v / ||v||` to
`|00>` with the global phase fixed so that `<00|U|v>` is real and non-negative. It maximises
`|<00|U|v>|` over unitaries, which is the single-gate Procrustes solution for the support on which
`v` was measured. If `v` vanishes the identity is returned.
"""
function frame_optimal_gate(v::AbstractMatrix)
    u = _slice_vector(v)
    n = norm(u)
    n > 1e-14 || return Matrix{ComplexF64}(I, 4, 4)
    u ./= n
    e = ComplexF64[1, 0, 0, 0]
    phi = angle(dot(e, u))
    e_ph = e .* cis(phi)
    d = u - e_ph
    nd = norm(d)
    nd > 1e-12 || return Matrix{ComplexF64}(cis(-phi) * I, 4, 4)
    h = d ./ nd
    return cis(-phi) * (Matrix{ComplexF64}(I, 4, 4) - 2 * (h * h'))
end

"""
    psel_layer(w, remaining; select=:proc, gate=:proc, span_penalty=0.0, max_batch=typemax(Int),
        min_gain=1e-12, cutoff=1e-12, max_bond_dim=4096, pairs) -> (w', gates, info)

One layer of gate selection on the residual `w`, with at most `remaining` gates, as
`next_batch!(work, remaining)` of the optimisation drivers.

`select` orders the candidate pairs: `:proc` by the Procrustes gain `P(i,j) - f0`, `:qmi` by the
removable QMI `R_ij`. A layer is built greedily by taking pairs in this order and keeping each
one that is disjoint from those already kept, up to `max_batch` pairs (`max_batch = 1` is pure
greedy selection with rescoring after every gate), admitting only pairs whose ordering statistic
exceeds `min_gain` (`1e-12`).

`span_penalty` is the exponent `gamma` of the cost-aware ordering: the statistic of a pair is
divided by `(2|i-j| + 1)^gamma`, the number of elementary operations of a gate of span `|i-j|` in
one optimisation sweep. `gamma = 0` orders by the statistic itself; the paper uses
`gamma in {0, 1/4, 1/2}`. The penalty changes which pair is preferred, not whether a useful pair
is admitted.

`gate` is the unitary placed on each selected pair, fitted to the current state after the earlier
gates of the layer: `:proc` the optimal gate [`frame_optimal_gate`](@ref), `:mi` the eigenbasis
gate of the removable-QMI selector. The residual is truncated at `cutoff` and `max_bond_dim` when a gate is applied.

Returns the updated residual, the gates as `MPSCircuits.UnitaryGate`s in application order, and
`info = (chosen, gains)`: the selected pairs and the Procrustes gain `P - f0` of each gate when
it was placed.
"""
function psel_layer(w::MPS, remaining::Int;
        select::Symbol=:proc, gate::Symbol=:proc,
        span_penalty::Float64=0.0,
        cutoff::Real=OPTIMIZER_CUTOFF, max_bond_dim::Int=OPTIMIZER_MAXDIM,
        max_batch::Int=typemax(Int), min_gain::Float64=1e-12,
        pairs=candidate_pairs(length(w); adjacent=false))
    remaining <= 0 && return w, MPSCircuits.UnitaryGate[], (chosen=Tuple{Int,Int}[], gains=Float64[])
    select in (:proc, :qmi) || throw(ArgumentError("unknown select $select"))
    gate in (:proc, :mi) || throw(ArgumentError("unknown gate $gate"))

    s = siteinds(w)
    N = length(w)
    cand = sort!(unique(minmax(p[1], p[2]) for p in pairs))

    # Ordering statistic, divided by the cost penalty.
    span_cost(p) = 2.0 * (abs(p[2] - p[1]) + 1) - 1.0
    penalty(p) = span_penalty == 0.0 ? 1.0 : span_cost(p)^span_penalty
    stat = if select == :proc
        ps = procrustes_pair_scores(w; pairs=cand)
        ps.gain
    else
        rhos = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        for (p, rho) in all_pairs_two_site_rdms(w, cand)
            rhos[p] = Matrix{ComplexF64}(dense_two_site_matrix(rho, s, p[1], p[2]))
        end
        Dict(p => removable_mutual_information(rhos[p]) for p in cand)
    end
    scaled = Dict(p => stat[p] / penalty(p) for p in cand)
    order = sort(cand; by=p -> (-scaled[p], p))

    used = falses(N)
    limit = min(remaining, max_batch)
    batch = Tuple{Int,Int}[]
    for p in order
        length(batch) >= limit && break
        stat[p] > min_gain || continue
        (used[p[1]] || used[p[2]]) && continue
        push!(batch, p)
        used[p[1]] = used[p[2]] = true
    end

    gates = MPSCircuits.UnitaryGate[]
    gains = Float64[]
    work = w
    for p in batch
        # The slice is recomputed on the current work state: earlier gates of the layer act on
        # disjoint sites but still change <0_rest|w> through the environment.
        v = _pair_slices(work, [p])[p]
        U = if gate == :proc
            frame_optimal_gate(v)
        else
            sw = siteinds(work)
            rho = Matrix{ComplexF64}(dense_two_site_matrix(
                two_site_rdm(work, p[1], p[2]), sw, p[1], p[2]))
            eigenbasis_gate(rho)
        end
        push!(gains, norm(v) - abs(v[1, 1]))
        push!(gates, MPSCircuits.UnitaryGate(two_qubit_gate(U, siteinds(work)[p[1]], siteinds(work)[p[2]])))
        work = apply_two_qubit_unitary(work, p[1], p[2], U;
            cutoff=cutoff, maxdim=max_bond_dim)
    end
    return work, gates, (chosen=batch, gains=gains)
end

"Proposals `(select, gate)` compared by default in [`protocol_arb_iter_oall`](@ref)."
const ARB_PROPOSALS = [(:qmi, :mi), (:proc, :proc)]

_prop_label(pr) = "$(pr[1])/$(pr[2])"

"""
    switch_schedule(k; cold=(:qmi, :mi), warm=(:proc, :proc))

Proposal schedule `(layer, won) -> [proposal]` for [`protocol_arb_iter_oall`](@ref) that uses the
`cold` selector for the first `k` layers and the `warm` selector afterwards: removable-QMI
selection while the circuit is cold and Procrustes selection once it is warm. The number `k` of
warm-start layers is chosen per target.
"""
switch_schedule(k::Int; cold=(:qmi, :mi), warm=(:proc, :proc)) =
    (layer, won) -> [layer <= k ? cold : warm]

# Working state implied by the current circuit: the inverse circuit applied to the target.
_rebuild_work_at(mps_clean, circuit, cutoff, maxdim) = isempty(circuit) ? deepcopy(mps_clean) :
    MPSCircuits.apply_circuit(reverse(MPSCircuits.dagger.(circuit)), deepcopy(mps_clean);
        mixed_device=:coerce, conversion_precision=:preserve, cutoff=cutoff, maxdim=maxdim)

_twosite_count(C) = count(g -> length(g.site_numbers) == 2, C)

"""
    protocol_arb_iter_oall(psi; budget, sweeps=10, proposals=ARB_PROPOSALS, look_sweeps=1,
        optimise=true, cutoff=1e-12, max_bond_dim=4096, kwargs...) -> NamedTuple

Layered state preparation to `budget` two-qubit gates, with each layer chosen by [`psel_layer`](@ref)
and the whole circuit refit by `sweeps` Procrustes sweeps after every layer
(`optimise=true`). Gates are added in front of the circuit, so the working residual is the
inverse circuit applied to `psi`.

`proposals` lists candidate `(select, gate)` pairs, or is a function `(layer, won) -> list` (see
[`switch_schedule`](@ref)). With more than one proposal each is built for the layer, tried by
`look_sweeps` Procrustes sweeps on a throwaway copy of the circuit, and the one with the lowest
infidelity is kept; the sweeps spent on the trials are reported separately as `look_chi3_ops`.
With one proposal the layer is built directly. The remaining keywords go to `psel_layer`
(`max_batch`, `span_penalty`, `min_gain`).

Returns `(gates, infid, won, cum_chi3_ops, look_chi3_ops, circuits)`: after every layer the
cumulative two-qubit gate count, the infidelity ([`rudolph_infidelity`](@ref)), the winning
proposal, the cumulative `chi^3` operations of the committed sweeps and of the trials, and a copy
of the circuit. With `sweeps = 0, optimise = false` this is the unoptimised greedy selection used
for the converged-infidelity comparison.
"""
function protocol_arb_iter_oall(psi::MPS; budget::Int, sweeps::Int=10,
        proposals=ARB_PROPOSALS, look_sweeps::Int=1, optimise::Bool=true,
        cutoff::Real=OPTIMIZER_CUTOFF, max_bond_dim::Int=OPTIMIZER_MAXDIM, kwargs...)
    mps_clean = ITensorMPS.dense(copy(psi))
    work = deepcopy(mps_clean)
    circuit = MPSCircuits.UnitaryGate[]
    gates = Int[]; infid = Float64[]; won = String[]
    cum_chi3 = Float64[]; look_chi3 = Float64[]; circuits = Vector{MPSCircuits.UnitaryGate}[]
    chi3 = 0.0; look = 0.0
    layer = 0

    while _twosite_count(circuit) < budget
        remaining = budget - _twosite_count(circuit)
        layer += 1
        props = proposals isa Function ? proposals(layer, won) : proposals
        if length(props) == 1
            sel, gt = only(props)
            work, Dgates, _ = psel_layer(work, remaining; select=sel, gate=gt,
                cutoff=cutoff, max_bond_dim=max_bond_dim, kwargs...)
            isempty(Dgates) && break
            circuit = vcat(prep_from_disentanglers(Dgates), circuit)
            push!(won, _prop_label(only(props)))
        else
            best = nothing
            for pr in props
                sel, gt = pr
                w2, Dgates, _ = psel_layer(work, remaining; select=sel, gate=gt,
                    cutoff=cutoff, max_bond_dim=max_bond_dim, kwargs...)
                isempty(Dgates) && continue
                trial = vcat(prep_from_disentanglers(Dgates), deepcopy(circuit))
                probe = deepcopy(trial)
                tgt = ITensorMPS.dense(copy(psi))
                look += look_sweeps * sweep_cost_model(tgt, probe; cutoff=cutoff,
                    max_bond_dim=max_bond_dim).chi3_ops
                for _ in 1:look_sweeps
                    MPSCircuits.replace_gates!(tgt, probe, MPSCircuits.TelescopingEnvironment(),
                        1:length(probe); max_bond_dim=max_bond_dim, working_cutoff=cutoff)
                end
                score = rudolph_infidelity(probe, psi; cutoff=cutoff, max_bond_dim=max_bond_dim)
                if best === nothing || score < best.score
                    best = (score=score, work=w2, gates=Dgates, label=_prop_label(pr))
                end
            end
            best === nothing && break
            circuit = vcat(prep_from_disentanglers(best.gates), circuit)
            work = best.work
            push!(won, best.label)
        end

        if optimise
            cost = sweep_cost_model(mps_clean, circuit; cutoff=cutoff, max_bond_dim=max_bond_dim)
            for _ in 1:sweeps
                MPSCircuits.replace_gates!(mps_clean, circuit, MPSCircuits.TelescopingEnvironment(),
                    1:length(circuit); max_bond_dim=max_bond_dim, working_cutoff=cutoff)
                chi3 += cost.chi3_ops
            end
            work = _rebuild_work_at(mps_clean, circuit, cutoff, max_bond_dim)
        end
        push!(gates, _twosite_count(circuit))
        push!(infid, rudolph_infidelity(circuit, psi; cutoff=cutoff, max_bond_dim=max_bond_dim))
        push!(cum_chi3, chi3); push!(look_chi3, look)
        push!(circuits, deepcopy(circuit))
    end
    return (gates=gates, infid=infid, won=won, cum_chi3_ops=cum_chi3,
            look_chi3_ops=look_chi3, circuits=circuits)
end

end # module
