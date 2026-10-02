"""
    CircuitOptimization

Preparation circuits as lists of `MPSCircuits.UnitaryGate`s, their infidelity against a target
MPS, and the Procrustes optimisation. A disentangling sequence `D = [d_1, ..., d_m]`
(applied in order, driving `psi` towards `|0...0>`) corresponds to the preparation circuit
`C = reverse(dagger.(D))` with `C |0...0> = psi`. Optimisation sweeps re-fit the two-qubit gates
of an existing circuit by singular value decomposition of their environment tensors and change
neither the gate count nor the connectivity.
"""
module CircuitOptimization

using ITensors
using ITensorMPS
import MPSCircuits

using ..SimConfig: global_cutoff, global_maxdim
using ..Disentangling: DisentanglingGate, two_qubit_gate
using ..Selection: product_frame_fidelity
using ..OptimizerCost: sweep_cost_model

export prep_from_disentanglers, prep_circuit_from_gate_log, disentangle_with_prep_circuit,
    rudolph_infidelity, product_frame_squared_fidelity, framed_circuit, circuit_checkpoint,
    oall_sweep!, measurement_schedule, patience_converged, optimize_to_floor,
    OPTIMIZER_MAXDIM, OPTIMIZER_CUTOFF

"Default working bond dimension of the optimisation sweeps and of the infidelity evaluation."
const OPTIMIZER_MAXDIM = 4096

"Default SVD cutoff of the optimisation sweeps and of the infidelity evaluation."
const OPTIMIZER_CUTOFF = 1e-12

"""
    prep_from_disentanglers(D) -> Vector{UnitaryGate}

Preparation circuit `C` with `C|0...0> = psi` from the disentangling gates `D`, listed in the
order in which they were applied to `psi`: `C = reverse(dagger.(D))`.
"""
prep_from_disentanglers(D::Vector{MPSCircuits.UnitaryGate}) =
    reverse([MPSCircuits.dagger(g) for g in D])

"""
    prep_circuit_from_gate_log(gate_log, sites) -> Vector{UnitaryGate}

Preparation circuit of a selector run: the `DisentanglingGate`s of `gate_log` (as returned by
`aspire`, in application order) become unitary gates on the physical indices `sites`, and the
sequence is inverted with [`prep_from_disentanglers`](@ref).
"""
function prep_circuit_from_gate_log(gate_log::AbstractVector{<:DisentanglingGate}, sites)
    D = MPSCircuits.UnitaryGate[MPSCircuits.UnitaryGate(two_qubit_gate(g.U, sites[g.i], sites[g.j]))
        for g in gate_log]
    return prep_from_disentanglers(D)
end

"""
    disentangle_with_prep_circuit(psi, circuit) -> MPS

Normalised state obtained by applying the inverse of the preparation `circuit` to `psi`, truncated
at the global cutoff and bond dimension. It equals `|0...0>` when `circuit` prepares `psi` exactly.
"""
function disentangle_with_prep_circuit(psi::MPS, circuit::Vector{MPSCircuits.UnitaryGate})
    work = ITensorMPS.dense(copy(psi))
    for gate in reverse(circuit)
        work = MPSCircuits.apply_gate(MPSCircuits.dagger(gate), work;
            mixed_device=:coerce, cutoff=global_cutoff(), maxdim=global_maxdim())
    end
    return ITensorMPS.normalize!(work)
end

"""
    rudolph_infidelity(C, psi; cutoff=1e-12, max_bond_dim=4096) -> Float64

Infidelity `1 - f` of the preparation circuit `C` against `psi`, with
`f = |<0...0|C^dagger|psi>|` the overlap of `C|0...0>` with the target, evaluated with the
inverse circuit applied to `psi` at the given truncation. The overlap is taken against the fixed
state `|0...0>`: no free single-qubit frame is credited, so the circuit has to contain it. An empty
circuit has infidelity 1.
"""
function rudolph_infidelity(C::Vector{MPSCircuits.UnitaryGate}, psi::MPS;
        cutoff::Real=OPTIMIZER_CUTOFF, max_bond_dim::Int=OPTIMIZER_MAXDIM)
    isempty(C) && return 1.0
    f2 = real(MPSCircuits.evaluate_circuit_fidelity(C, psi; cutoff=cutoff, max_bond_dim=max_bond_dim))
    return 1.0 - sqrt(clamp(f2, 0.0, 1.0))
end

"""
    product_frame_squared_fidelity(target, circuit) -> Float64

Squared fidelity of `circuit` against `target` when a free layer of single-qubit rotations is
credited in front of it. The inverse circuit is applied to `target`, the residual is truncated to
bond dimension one, and the single-qubit gates that carry `|0...0>` into that product state are
prepended to `circuit`. The frame is therefore the bond-dimension-one truncation of the residual,
not the optimal product state: the returned fidelity is a lower bound on the squared
fidelity maximised over product frames, and `1 - sqrt` of it an upper bound on `epsilon_*`.
For an empty circuit the frame is that of `target` itself.
"""
function product_frame_squared_fidelity(
        target::MPS, circuit::Vector{MPSCircuits.UnitaryGate})
    isempty(circuit) &&
        return clamp(product_frame_fidelity(ITensorMPS.dense(copy(target))), 0.0, 1.0)
    disentangled = disentangle_with_prep_circuit(target, circuit)
    product = ITensorMPS.truncate(disentangled; maxdim=1)
    frame_layer = [MPSCircuits.product_state_site_gate(product, n)
        for n in 1:length(product)]
    full = vcat(frame_layer, circuit)
    fidelity = real(MPSCircuits.evaluate_circuit_fidelity(full, target;
        cutoff=global_cutoff(), max_bond_dim=global_maxdim()))
    return clamp(fidelity, 0.0, 1.0)
end

"""
    framed_circuit(target, circuit) -> Vector{UnitaryGate}

`circuit` preceded by the single-qubit layer that carries `|0...0>` into the bond-dimension-one
truncation of the residual `circuit^dagger |target>`. Its fidelity is
[`product_frame_squared_fidelity`](@ref). The empty circuit is framed like any other, so a
zero-gate circuit is the best product approximation to `target`.
"""
function framed_circuit(target::MPS, circuit::Vector{MPSCircuits.UnitaryGate})
    disentangled = isempty(circuit) ? ITensorMPS.dense(copy(target)) :
        disentangle_with_prep_circuit(target, circuit)
    product = ITensorMPS.truncate(disentangled; maxdim=1)
    frame = MPSCircuits.UnitaryGate[MPSCircuits.product_state_site_gate(product, n)
        for n in 1:length(product)]
    return vcat(frame, circuit)
end

"""
    circuit_checkpoint(target, circuit, layer, two_qubit_gates) -> NamedTuple

Record of a circuit under construction: the framed circuit ([`framed_circuit`](@ref)), the layer
index, the number of two-qubit gates and the squared fidelity of the framed circuit against
`target`, as `(layer, two_qubit_gates, squared_fidelity, circuit)`.
"""
function circuit_checkpoint(target::MPS, circuit::Vector{MPSCircuits.UnitaryGate},
        layer::Int, two_qubit_gates::Int)
    full = framed_circuit(target, circuit)
    squared = clamp(real(MPSCircuits.evaluate_circuit_fidelity(full, target;
        cutoff=global_cutoff(), max_bond_dim=global_maxdim())), 0.0, 1.0)
    return (layer=layer, two_qubit_gates=two_qubit_gates, squared_fidelity=squared,
        circuit=full)
end

"""
    oall_sweep!(target, circuit; sweeps=10, cost=false, gate_indices=nothing) -> Float64

Run `sweeps` Procrustes sweeps over the gates of `circuit` against `target`, in
place. Each sweep visits the gates in circuit order and replaces each by the unitary that
maximises `|tr(U' F)|` for its environment `F`, computed with telescoping bra and ket
chains truncated at the global bond dimension and cutoff. `gate_indices` restricts the update to a
subset of the gates while the rest of `circuit` stays in the environment; this holds a leading
single-qubit frame fixed while the two-qubit gates are refined.

Returns the classical cost of the sweeps in `chi^3` operations (the maximum-bond-dimension count
of `OptimizerCost.sweep_cost_model`) if `cost=true`, and `0.0` otherwise; evaluating it costs one
replay of the bra chain.
"""
function oall_sweep!(target::MPS, circuit::Vector{MPSCircuits.UnitaryGate};
        sweeps::Int=10, cost::Bool=false,
        gate_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    (isempty(circuit) || sweeps <= 0) && return 0.0
    per_sweep = cost ? sweep_cost_model(target, circuit; cutoff=global_cutoff(),
                                        max_bond_dim=global_maxdim()).chi3_ops : 0.0
    idx = gate_indices === nothing ? (1:length(circuit)) : gate_indices
    for _ in 1:sweeps
        MPSCircuits.replace_gates!(target, circuit, MPSCircuits.TelescopingEnvironment(), idx;
            max_bond_dim=global_maxdim(), working_cutoff=global_cutoff())
    end
    return sweeps * per_sweep
end

"""
    measurement_schedule(n; growth=1.15) -> Vector{Int}

Sweep counts at which the infidelity is measured during an optimisation of up to `n` sweeps:
`0`, then a geometric progression `k -> max(k + 1, ceil(growth * k))` from 1, ending exactly
at `n`. Measuring after every sweep would cost as much as the sweeps themselves.
"""
function measurement_schedule(n::Int; growth::Float64=1.15)
    n <= 0 && return Int[0]
    growth > 1 || throw(ArgumentError("growth must exceed 1, got $growth"))
    pts = Int[0]
    k = 1
    while k <= n
        push!(pts, k)
        k = max(k + 1, ceil(Int, k * growth))
    end
    pts[end] == n || push!(pts, n)
    return pts
end

# One measurement interval of the trace `ys` bought less than a fraction `rtol` of the
# infidelity at its start, per sweep of the interval (`gap` sweeps).
_converged(ys, rtol, gap::Int=1) = rtol !== nothing && length(ys) > 1 &&
    (ys[end-1] - ys[end]) <= max(gap, 1) * rtol * max(ys[end-1], eps())

"""
    patience_converged(ys, at, rtol, patience=1) -> Bool

Convergence rule of the optimisation: with the infidelity `ys[k]` measured after `at[k]` sweeps,
the run has converged when each of the last `patience` measurement intervals lowered the
infidelity by less than a fraction `rtol` of its value at the start of the interval, per sweep of
the interval. This is the rule with `delta = rtol` and a patience of three intervals used for
the converged infidelities of the accompanying paper (`rtol = 1e-7`, `patience = 3`). Long-span circuits
plateau and then escape, so requiring several consecutive flat intervals avoids stopping on a
plateau. `rtol = nothing` never converges.
"""
function patience_converged(ys, at, rtol, patience::Int=1)
    rtol === nothing && return false
    patience <= 1 &&
        return _converged(ys, rtol, length(at) > 1 ? max(at[end] - at[end-1], 1) : 1)
    length(ys) > patience || return false
    for j in 0:(patience - 1)
        i = length(ys) - j
        g = max(at[i] - at[i-1], 1)
        (ys[i-1] - ys[i]) <= g * rtol * max(ys[i-1], eps()) || return false
    end
    return true
end

"""
    optimize_to_floor(psi, circuit; rtol=1e-7, cap=30000, patience=3, max_bond_dim=4096,
        cutoff=1e-12, trace_growth=1.15, bond_resolved=false) -> NamedTuple

Procrustes-sweep the whole `circuit` against `psi` until [`patience_converged`](@ref) holds or
`cap` sweeps have run. The infidelity [`rudolph_infidelity`](@ref) is measured on the schedule of
[`measurement_schedule`](@ref) with growth `trace_growth`. `rtol = nothing` disables the
convergence test and runs exactly `cap` sweeps. The sweeps and all measurements run at the working
truncation `max_bond_dim` and `cutoff`.

Returns `(infid0, floor, floor_exact, exact_maxdim, sweeps, capped, cost, cost_maxchi, cost_bond,
bond_resolved, seconds, gates, span, trace, trace_at)`: the infidelity before and after the
sweeps, the final infidelity re-evaluated at up to four times the working bond dimension, the
number of sweeps and whether the cap stopped the run (`floor` is then only an upper bound on the
converged infidelity), the classical cost of the sweeps in `chi^3` operations under each cost model
([`OptimizerCost`](@ref); `cost` follows `bond_resolved`), the wall time, the gate count and mean
span, and the measured trace with the sweep index of each entry.
"""
function optimize_to_floor(psi::MPS, circuit; rtol::Union{Nothing,Float64}=1e-7, cap::Int=30000,
        max_bond_dim::Int=OPTIMIZER_MAXDIM, cutoff::Float64=OPTIMIZER_CUTOFF,
        trace_growth::Float64=1.15, bond_resolved::Bool=false, patience::Int=3)
    target = ITensorMPS.dense(copy(psi))
    C = deepcopy(circuit)
    m = sweep_cost_model(target, C; cutoff=cutoff, max_bond_dim=max_bond_dim,
                         bond_resolved=bond_resolved)
    measure(circ) = rudolph_infidelity(circ, psi; cutoff=cutoff, max_bond_dim=max_bond_dim)
    sched = Set(measurement_schedule(cap; growth=trace_growth))
    ys = Float64[measure(C)]
    at = Int[0]
    n = 0
    t = @elapsed for k in 1:cap
        MPSCircuits.replace_gates!(target, C, MPSCircuits.TelescopingEnvironment(),
            1:length(C); max_bond_dim=max_bond_dim, working_cutoff=cutoff)
        n = k
        if k in sched
            push!(ys, measure(C)); push!(at, k)
            patience_converged(ys, at, rtol, patience) && break
        end
    end
    if at[end] != n
        push!(ys, measure(C)); push!(at, n)
    end
    # A second reading at higher truncation shows whether the floor is limited by truncation.
    exact_maxdim = min(OPTIMIZER_MAXDIM, 4 * max_bond_dim)
    exact = rudolph_infidelity(C, psi; cutoff=cutoff, max_bond_dim=exact_maxdim)
    return (infid0=ys[1], floor=ys[end], floor_exact=exact, exact_maxdim=exact_maxdim,
            sweeps=n, capped=(n == cap),
            cost=n * m.chi3_ops, cost_maxchi=n * m.chi3_ops_maxchi,
            cost_bond=n * m.chi3_ops_bond, bond_resolved=bond_resolved,
            seconds=t, gates=m.ngates, span=m.mean_span,
            trace=ys, trace_at=at)
end

end # module
