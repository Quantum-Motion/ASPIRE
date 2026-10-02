"""
    ToyStates

Small analytically defined states used to illustrate the effect of connectivity: a GHZ block,
two Bell pairs coupled across their inner bond, and the one-dimensional cluster state.
"""
module ToyStates

using ITensors
using ITensorMPS
using LinearAlgebra

using ..Disentangling: apply_two_qubit_unitary
using ..ModelStates: apply_one_qubit_unitary

export ghz_block_state, coupled_bell_pairs, cluster_state

const _HADAMARD = (1 / sqrt(2)) * ComplexF64[1 1; 1 -1]
const _CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
const _CZ = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]

"""
    cluster_state(N) -> MPS

One-dimensional cluster state on `N` qubits: `|+>^N` with a CZ gate on every neighbouring pair.
Every stabiliser generator `Z_{i-1} X_i Z_{i+1}` has weight three, so all two-site marginals
away from the ends are maximally mixed and `R_ij = 0` for every pair except the end pairs
`(1,2)` and `(N-1,N)`, while the total correlation is `T = N` bits. It is the extreme case for
pair-only selection (see the accompanying paper).
"""
function cluster_state(N::Int)
    N >= 1 || throw(ArgumentError("N must be at least one, got $N"))
    s = siteinds("S=1/2", N)
    psi = MPS(ComplexF64, s, fill("Up", N))
    for i in 1:N
        psi = apply_one_qubit_unitary(psi, i, _HADAMARD)
    end
    for i in 1:(N - 1)
        psi = apply_two_qubit_unitary(psi, i, i + 1, _CZ)
    end
    return normalize!(psi)
end

"""
    ghz_block_state(N, a, b) -> MPS

GHZ state on the contiguous sites `a..b` of an otherwise all-up chain of `N` qubits, a
multipartite block whose two-site marginals are separable.
"""
function ghz_block_state(N::Int, a::Int, b::Int)
    s = siteinds("S=1/2", N)
    psi = MPS(ComplexF64, s, fill("Up", N))
    I2 = ComplexF64[1 0; 0 1]
    psi = apply_two_qubit_unitary(psi, a, a + 1, kron(_HADAMARD, I2))
    for i in a:(b - 1)
        psi = apply_two_qubit_unitary(psi, i, i + 1, _CNOT)
    end
    return normalize!(psi)
end

"""
    coupled_bell_pairs(lambda) -> MPS

Bell pairs on the sites `(1,2)` and `(3,4)` of a four-qubit chain, coupled across the inner bond
by `exp(-i lambda X_2 X_3)`. `lambda = 0` leaves two decoupled Bell pairs.
"""
function coupled_bell_pairs(lambda::Real)
    s = siteinds("S=1/2", 4)
    psi = MPS(ComplexF64, s, fill("Up", 4))
    I2 = ComplexF64[1 0; 0 1]
    XX = kron(ComplexF64[0 1; 1 0], ComplexF64[0 1; 1 0])
    coupler = cos(lambda) * Matrix{ComplexF64}(I, 4, 4) - im * sin(lambda) * XX
    psi = apply_two_qubit_unitary(psi, 1, 2, kron(_HADAMARD, I2))
    psi = apply_two_qubit_unitary(psi, 1, 2, _CNOT)
    psi = apply_two_qubit_unitary(psi, 3, 4, kron(_HADAMARD, I2))
    psi = apply_two_qubit_unitary(psi, 3, 4, _CNOT)
    psi = apply_two_qubit_unitary(psi, 2, 3, coupler)
    return normalize!(psi)
end

end # module
