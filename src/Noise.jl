"""
    Noise

Global depolarising noise model for state-preparation circuits and the
routing charge applied to the MPD. After every native two-qubit gate the whole register passes
through the channel `rho -> (1 - L) rho + L 1/2^N` with `L = 16p/15`, the weight that a two-qubit
depolarising channel of strength `p` places on the maximally mixed state of its support. This
channel commutes with every gate, so a circuit of `M` native gates outputs, with weight
`q = (1 - L)^M`, the noiseless state and otherwise the maximally mixed state. The squared
fidelity with the target is therefore

    F = q F0 + (1 - q) / 2^N,        F0 = noiseless squared fidelity,

and the plotted infidelity is `1 - sqrt(F)`. Single-qubit gates are noiseless.
"""
module Noise

export global_depolarizing_strength, global_depolarizing_squared_fidelity,
    global_depolarizing_infidelity, native_gate_count,
    MPD_SWAPS_PER_GATE, MPD_ROUTING_OVERHEAD

"""
    global_depolarizing_strength(p) -> Float64

Weight `16p/15` removed from the identity by one two-qubit depolarising channel of strength `p`.
"""
global_depolarizing_strength(p::Real) = 16.0 * p / 15.0

"""
    global_depolarizing_squared_fidelity(F0, native_gates, p, N) -> Float64

Squared fidelity `F = q F0 + (1 - q) 2^-N` of an `N`-qubit register prepared by a circuit of
`native_gates` two-qubit gates, each followed by depolarising noise of strength `p`, where `F0` is
the noiseless squared fidelity and `q = (1 - 16p/15)^native_gates`.
"""
function global_depolarizing_squared_fidelity(F0::Real, native_gates::Real, p::Real, N::Integer)
    q = (1.0 - global_depolarizing_strength(p))^native_gates
    return q * F0 + (1.0 - q) / 2.0^N
end

"""
    global_depolarizing_infidelity(F0, native_gates, p, N) -> Float64

Amplitude infidelity `1 - sqrt(F)` for `F = global_depolarizing_squared_fidelity(...)`.
"""
global_depolarizing_infidelity(F0::Real, native_gates::Real, p::Real, N::Integer) =
    1.0 - sqrt(clamp(global_depolarizing_squared_fidelity(F0, native_gates, p, N), 0.0, 1.0))

"SWAPs per logical gate charged to the MPD in the heavy-hex comparison: `2/5`."
const MPD_SWAPS_PER_GATE = 2 / 5

"""
Native gates per logical MPD gate on heavy-hex, `1 + 2/5 = 1.4`. The MPS ordering of a heavy-hex
lattice leaves at least `J` jumps between chain-adjacent qubits that are not coupled
(see the accompanying paper), with `J / (n - 1) -> 1/5` for large lattices. Each jump costs one SWAP out and one
back, and a SWAP is priced like the emitted two-qubit gate it routes, since both are three CNOTs,
so an MPD layer of `n - 1` gates costs `1 + 2J/(n - 1) -> 1.4` native gates per gate.
"""
const MPD_ROUTING_OVERHEAD = 1.0 + MPD_SWAPS_PER_GATE

"""
    native_gate_count(logical_gates; swaps_per_gate=0.0, swap_gate_cost=1.0) -> Int

Native two-qubit gates charged for `logical_gates` emitted gates when each costs `1` plus
`swap_gate_cost * swaps_per_gate` for its routing. The sum is accumulated gate by gate and rounded
once at the end, as an average overhead is fractional. The default charges no routing, as for
ASPIRE restricted to device couplers; `swaps_per_gate = MPD_SWAPS_PER_GATE` charges the MPD the
`MPD_ROUTING_OVERHEAD = 1.4` native gates per logical gate of the heavy-hex comparison.
"""
function native_gate_count(logical_gates::Integer; swaps_per_gate::Real=0.0,
        swap_gate_cost::Real=1.0)
    per_gate = 1.0 + swap_gate_cost * swaps_per_gate
    total = 0.0
    for _ in 1:logical_gates
        total += per_gate
    end
    return round(Int, total)
end

end # module
