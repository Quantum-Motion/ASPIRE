"""
    EnergyProcrustes

Energy refinement of a state-preparation circuit. Given a Hamiltonian MPO `H`
and a circuit `C` of two-qubit unitaries, the gates are updated one at a time to
lower the energy `E[C] = <0|C† H C|0>`, with every environment and energy
evaluated as a truncated MPS computation.
"""
module EnergyProcrustes

using ITensors
using ITensorMPS
using LinearAlgebra
import MPSCircuits

using ..ModelStates: energy
using ..SimConfig: global_cutoff, global_maxdim

export optimize_energy_gate_mps!, energy_mps_coordinate_refine

# A candidate gate is accepted when the energy does not rise by more than this.
const _ACCEPT_TOL = 1e-12

function _check_gate(circuit, gate_index::Int)
    1 <= gate_index <= length(circuit) ||
        throw(ArgumentError("gate_index must be between 1 and $(length(circuit)); got $gate_index."))
    gate = circuit[gate_index]
    length(gate.site_numbers) == 2 ||
        throw(ArgumentError("gate $gate_index must be a two-qubit gate; it acts on $(gate.site_numbers)."))
    all(dim(i) == 2 for i in gate.site_indices) ||
        throw(ArgumentError("gate $gate_index must act on qubit indices; got dimensions $(dim.(gate.site_indices))."))
    return nothing
end

# The 4x4 matrix of a two-qubit gate, rows indexed by the output legs.
function _gate_matrix(gate)
    return reshape(
        ITensors.array(
            MPSCircuits.tensor(gate),
            ITensors.prime.(gate.site_indices)...,
            gate.site_indices...,
        ),
        4,
        4,
    )
end

# The gate with matrix `matrix` acting on the same sites and indices as `gate`.
function _physical_gate_from_matrix(matrix::AbstractMatrix, gate::MPSCircuits.UnitaryGate)
    array = reshape(ComplexF64.(matrix), 2, 2, 2, 2)
    tensor = ITensors.itensor(
        array,
        ITensors.prime.(gate.site_indices)...,
        gate.site_indices...,
    )
    return MPSCircuits.UnitaryGate(tensor, gate.site_numbers, gate.site_indices)
end

# Unitary polar factor of (1 - t) U + t V, where U, V are the matrices of
# `current` and `proposal` and t = `fraction`. At t = 1 this is the unitary
# nearest to V, i.e. V itself when V is unitary.
function _polar_interpolate(current, proposal, fraction::Float64)
    0 < fraction <= 1 ||
        throw(ArgumentError("fraction must lie in (0, 1]."))
    U = _gate_matrix(current)
    V = _gate_matrix(proposal)
    factorization = svd((1 - fraction) * U + fraction * V)
    return _physical_gate_from_matrix(
        factorization.U * factorization.Vt,
        current,
    )
end

function _zero_state_for(H::MPO)
    sites = [ITensorMPS.siteind(H, n; plev=0) for n in 1:length(H)]
    return MPS(sites, fill("0", length(sites)))
end

function _apply_circuit_gate(gate, state; cutoff::Float64, maxdim::Int)
    return MPSCircuits.apply_gate(
        gate,
        state;
        mixed_device=:coerce,
        cutoff=cutoff,
        maxdim=maxdim,
    )
end

# The state C|0> obtained by applying the gates of `circuit` in order to |0...0>.
function _prepared_state(H::MPO, circuit; cutoff::Float64, maxdim::Int)
    state = _zero_state_for(H)
    for gate in circuit
        state = _apply_circuit_gate(gate, state; cutoff=cutoff, maxdim=maxdim)
    end
    return state
end

"""
    optimize_energy_gate_mps!(H, circuit, gate_index; initial_fraction=1.0,
                              backtrack=0.5, max_backtracks=8,
                              cutoff=global_cutoff(), maxdim=global_maxdim())

Update the two-qubit gate `circuit[gate_index]` to lower the energy
`E[C] = <0|C† H C|0>` with all other gates held fixed, and return a named tuple
`(gate_index, before, after, accepted, fraction, tested_energies, environment)`.
The circuit is modified in place.

Write `C = S G P`, with `P` the gates before the selected gate `G` and `S`
the gates after it, and let `|ψ_P> = P|0>`. With the other gates fixed, the
energy as a function of `G` is `E(G) = <ψ_P| G† S† H S G |ψ_P>`, which is
quadratic in `G`. Replacing the conjugate copy of `G` by its current value `G₀`
gives the linear form `Re tr(Γ G)`, where `Γ` is the four-leg environment
obtained by contracting `S† H C|0>` (the full state with `H` applied and the
suffix gates daggered in reverse order) with `|ψ_P>` over all sites outside the
support of `G`, computed by `MPSCircuits.partial_inner`. The proposal `V` is the
unitary minimising this form, the polar (Procrustes) factor of `-Γ`.

The step from the current matrix `U` to `V` is the polar interpolation
`G(t) = polar((1 - t) U + t V)`. A backtracking line search starts at
`t = initial_fraction` and multiplies `t` by `backtrack` after each rejection.
A candidate is accepted when the energy of the full prepared state,
`<0|C(t)† H C(t)|0>`, does not exceed the current energy by more than `1e-12`.
After `max_backtracks` rejections the original gate is restored and
`accepted` is false. States and energies are truncated MPS computations
governed by `cutoff` and `maxdim`.
"""
function optimize_energy_gate_mps!(
    H::MPO,
    circuit::Vector{MPSCircuits.UnitaryGate},
    gate_index::Integer;
    initial_fraction::Float64=1.0,
    backtrack::Float64=0.5,
    max_backtracks::Int=8,
    cutoff::Float64=global_cutoff(),
    maxdim::Int=global_maxdim(),
)
    isempty(circuit) && throw(ArgumentError("circuit cannot be empty."))
    k = Int(gate_index)
    _check_gate(circuit, k)
    0 < initial_fraction <= 1 ||
        throw(ArgumentError("initial_fraction must lie in (0, 1]."))
    0 < backtrack < 1 ||
        throw(ArgumentError("backtrack must lie in (0, 1)."))
    max_backtracks >= 1 ||
        throw(ArgumentError("max_backtracks must be >= 1."))

    prefix = _zero_state_for(H)
    for j in 1:(k - 1)
        prefix = _apply_circuit_gate(circuit[j], prefix; cutoff=cutoff, maxdim=maxdim)
    end

    full_state = copy(prefix)
    for j in k:length(circuit)
        full_state = _apply_circuit_gate(circuit[j], full_state; cutoff=cutoff, maxdim=maxdim)
    end
    current_energy = energy(full_state, H)

    # H C|0> pulled back through the suffix gates, last gate first.
    bra_energy = apply(H, full_state; cutoff=cutoff, maxdim=maxdim)
    for j in length(circuit):-1:(k + 1)
        bra_energy = _apply_circuit_gate(
            MPSCircuits.dagger(circuit[j]),
            bra_energy;
            cutoff=cutoff,
            maxdim=maxdim,
        )
    end
    environment = MPSCircuits.partial_inner(
        bra_energy,
        prefix,
        circuit[k].site_numbers...,
    )
    proposal = MPSCircuits.new_optimal_gate(
        -environment,
        circuit[k].site_indices,
    )

    original = circuit[k]
    fraction = initial_fraction
    accepted = false
    candidate_energy = current_energy
    tested_energies = Float64[]
    for _ in 1:max_backtracks
        circuit[k] = _polar_interpolate(original, proposal, fraction)
        candidate_state = _prepared_state(H, circuit; cutoff=cutoff, maxdim=maxdim)
        candidate_energy = energy(candidate_state, H)
        push!(tested_energies, candidate_energy)
        if candidate_energy <= current_energy + _ACCEPT_TOL
            accepted = true
            break
        end
        fraction *= backtrack
    end
    accepted || (circuit[k] = original)

    return (
        gate_index=k,
        before=current_energy,
        after=accepted ? candidate_energy : current_energy,
        accepted=accepted,
        fraction=accepted ? fraction : 0.0,
        tested_energies=tested_energies,
        environment=environment,
    )
end

"""
    energy_mps_coordinate_refine(H, circuit; n_sweeps=4, converge_rtol=0.0,
                                 initial_fraction=1.0, backtrack=0.5,
                                 max_backtracks=8, cutoff=global_cutoff(),
                                 maxdim=global_maxdim())

Lower the energy `<0|C† H C|0>` of a copy of `circuit` by sweeping
[`optimize_energy_gate_mps!`](@ref) over its gates. Sweep `s` visits the gates in
circuit order for odd `s` and in reverse order for even `s`; the environment of
each gate is rebuilt after the previous update, so later gates see earlier ones.
Because every update is accepted only if the energy does not rise, the energy
is non-increasing along the sweeps.

The line-search keywords are passed to each gate update. If `converge_rtol > 0`
the refinement stops after the first sweep whose energy decrease is at most
`converge_rtol` times the previous energy in magnitude.

Returns `(circuit, energies, sweeps, stopped_early)`. `energies[s + 1]` is the
energy after sweep `s` and `energies[1]` that of the input. Each entry of
`sweeps` holds the gate `order`, the per-gate `updates`, the energy `after` the
sweep and the number of `accepted` updates. The input circuit is not modified.
"""
function energy_mps_coordinate_refine(
    H::MPO,
    circuit::Vector{MPSCircuits.UnitaryGate};
    n_sweeps::Int=4,
    converge_rtol::Float64=0.0,
    initial_fraction::Float64=1.0,
    backtrack::Float64=0.5,
    max_backtracks::Int=8,
    cutoff::Float64=global_cutoff(),
    maxdim::Int=global_maxdim(),
)
    n_sweeps >= 0 || throw(ArgumentError("n_sweeps must be nonnegative."))
    converge_rtol >= 0 ||
        throw(ArgumentError("converge_rtol must be nonnegative."))
    refined = deepcopy(circuit)
    initial_energy = energy(
        _prepared_state(H, refined; cutoff=cutoff, maxdim=maxdim),
        H,
    )
    energies = Float64[initial_energy]
    sweeps = Any[]
    for sweep in 1:n_sweeps
        order = collect(1:length(refined))
        iseven(sweep) && reverse!(order)
        updates = [
            optimize_energy_gate_mps!(
                H,
                refined,
                k;
                initial_fraction=initial_fraction,
                backtrack=backtrack,
                max_backtracks=max_backtracks,
                cutoff=cutoff,
                maxdim=maxdim,
            ) for k in order
        ]
        final_energy = energy(
            _prepared_state(H, refined; cutoff=cutoff, maxdim=maxdim),
            H,
        )
        push!(
            sweeps,
            (
                order=order,
                updates=updates,
                after=final_energy,
                accepted=count(update -> update.accepted, updates),
            ),
        )
        push!(energies, final_energy)
        if converge_rtol > 0
            previous = energies[end - 1]
            gain = previous - final_energy
            if gain <= converge_rtol * max(abs(previous), eps())
                return (
                    circuit=refined,
                    energies=energies,
                    sweeps=sweeps,
                    stopped_early=true,
                )
            end
        end
    end
    return (circuit=refined, energies=energies, sweeps=sweeps, stopped_early=false)
end

end # module
