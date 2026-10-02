"""
    TCount

Fault-tolerant cost of a preparation circuit: the number of `T` gates needed to
realise it in Clifford+`T`, following arXiv:2603.11156. Clifford gates (CNOT,
SWAP, `H`, `S`, Pauli frame) are free, so the cost is set by the non-Clifford
one-qubit rotations that remain after the circuit is decomposed and fused, and
by a per-rotation synthesis model ([`GRIDSYNTH`](@ref), [`TRASYN`](@ref)).
"""
module TCount

using ITensors
using LinearAlgebra
import MPSCircuits

export GRIDSYNTH, TRASYN, t_count, rotation_count

# Absolute tolerance for recognising a rotation angle as a multiple of pi/2 or
# pi/4. It is tight on purpose: a nearly Clifford rotation is not Clifford.
const ANGLE_ATOL = 1e-9

# Operator-Schmidt tolerance below which a two-qubit gate counts as a product.
const PRODUCT_ATOL = 1e-10

# Tolerance for recognising a two-qubit unitary as a Clifford gate.
const CLIFFORD_ATOL = 1e-9

_near_multiple(theta::Real, unit::Real) =
    abs(rem(float(theta), float(unit), RoundNearest)) <= ANGLE_ATOL

"""
    _angle_class(theta) -> Symbol

Cost class of a rotation by `theta` about a fixed axis. `:clifford` if `theta` is
a multiple of `pi/2` (free), `:exact_t` if it is a multiple of `pi/4` but not of
`pi/2` (exactly one `T` gate, no synthesis error), and `:approx` otherwise. The
class does not depend on the axis, since `Ry(theta)` is `Rz(theta)` conjugated by
Cliffords.
"""
function _angle_class(theta::Real)
    _near_multiple(theta, pi / 2) && return :clifford
    _near_multiple(theta, pi / 4) && return :exact_t
    return :approx
end

"""
    RotationSynthesis(name, slope, intercept, per_block)

Model for the `T` cost of approximating a rotation to operator-norm precision
`epsilon`,

    T(epsilon) = slope * log2(1/epsilon) + intercept.

`per_block = false` charges one synthesis call per non-Clifford rotation, so a
general one-qubit block costs three calls. `per_block = true` charges one call
for each fused one-qubit block, regardless of the number of rotations in its
`ZYZ` Euler decomposition. Rotations that are odd multiples of `pi/4` are exact
Clifford+`T` operators and cost one `T` each under either model.
"""
struct RotationSynthesis
    name::Symbol
    slope::Float64
    intercept::Float64
    per_block::Bool
end

"""
Ross-Selinger ancilla-free `Rz` synthesis (arXiv:1403.2975), one call per
rotation, with cost `T = 3.054 log2(1/epsilon) + 1.83`. The slope and intercept
are a least-squares fit to the mean `T` count of `pygridsynth` 2.0.0 over 300
uniform random angles at each of `epsilon = 1e-3, ..., 1e-10`; the slope agrees
with the asymptotic `3 log2(1/epsilon)` of arXiv:1403.2975. The `T` count of an
individual angle scatters about this mean by about 3% at `epsilon = 1e-6`.
"""
const GRIDSYNTH = RotationSynthesis(:gridsynth, 3.054, 1.83, false)

"""
Direct synthesis of an arbitrary one-qubit unitary as a single `U3` gate
(`trasyn`, arXiv:2503.15843; the specialized `U3` synthesis of arXiv:2603.11156),
one call per fused one-qubit block. The per-call cost is taken equal to that of
[`GRIDSYNTH`](@ref), `T = 3.054 log2(1/epsilon) + 1.83`, which reproduces the
threefold per-block reduction reported in arXiv:2603.11156 and is the
conservative end of the up to 3.5-fold reduction claimed in arXiv:2503.15843.
"""
const TRASYN = RotationSynthesis(:trasyn, 3.054, 1.83, true)

# `T` gates spent on one synthesis call at precision `epsilon`. Not rounded, so
# that a circuit total does not accumulate a rounding bias per call.
function _synthesis_t_gates(model::RotationSynthesis, epsilon::Real)
    0 < epsilon < 1 || throw(ArgumentError(
        "epsilon must lie in (0, 1), got $epsilon"))
    return model.slope * log2(1 / epsilon) + model.intercept
end

# Synthesis calls and exactly synthesizable rotations needed by one fused block,
# given as its list of non-Clifford angles. Under a per-block model the whole
# block is one call if any angle needs approximating, and is exact otherwise.
function _block_synthesis(angles::Vector{Float64}, model::RotationSynthesis)
    approx = count(a -> _angle_class(a) === :approx, angles)
    exact = length(angles) - approx
    model.per_block || return (calls=approx, exact_t=exact)
    return approx > 0 ? (calls=1, exact_t=0) : (calls=0, exact_t=exact)
end

_rz(theta::Real) = ComplexF64[exp(-im * theta / 2) 0; 0 exp(im * theta / 2)]

function _ry(theta::Real)
    c, s = cos(theta / 2), sin(theta / 2)
    return ComplexF64[c -s; s c]
end

const _EYE2 = ComplexF64[1 0; 0 1]

# 2x2 matrix of a one-qubit gate, as (row, column) = (out, in).
function _matrix_1q(gate)
    site = gate.site_indices[1]
    return Matrix{ComplexF64}(ITensors.array(gate.tensor, ITensors.prime(site), site))
end

# 4x4 matrix of a two-qubit gate. Index ordering follows `MPSCircuits.KAKGateSU4`:
# the tensor is read as (s2', s1', s2, s1) and reshaped column-major, so
# `site_indices[1]` is the high-order factor of a `kron` and `site_indices[2]`
# the low-order one.
function _matrix_2q(gate)
    s1, s2 = gate.site_indices[1], gate.site_indices[2]
    array = ITensors.array(gate.tensor, ITensors.prime(s2), ITensors.prime(s1), s2, s1)
    return Matrix{ComplexF64}(reshape(array, (4, 4)))
end

const _PAULI_I = ComplexF64[1 0; 0 1]
const _PAULI_X = ComplexF64[0 1; 1 0]
const _PAULI_Y = ComplexF64[0 -im; im 0]
const _PAULI_Z = ComplexF64[1 0; 0 -1]
const _PAULI_1Q = (_PAULI_I, _PAULI_X, _PAULI_Y, _PAULI_Z)
const _PAULI_2Q = [kron(A, B) for A in _PAULI_1Q for B in _PAULI_1Q]
# Generators of the two-qubit Pauli group. A unitary is Clifford exactly when it
# maps all four back into the group.
const _CLIFFORD_GENERATORS = (
    kron(_PAULI_X, _PAULI_I), kron(_PAULI_Z, _PAULI_I),
    kron(_PAULI_I, _PAULI_X), kron(_PAULI_I, _PAULI_Z),
)

"""
    _is_clifford_gate(M) -> Bool

Whether the two-qubit unitary `M` is Clifford, i.e. `M P M'` is a signed Pauli
string for every Pauli `P`. It suffices to check the four generators of the Pauli
group, since conjugation is a homomorphism. The image `Q` of a generator is a
Pauli string exactly when one of its 16 Pauli coefficients `tr(S' Q)/4` has
modulus one.

This test is needed because the KAK decomposition of a degenerate gate such as a
CNOT is only fixed up to a rotation inside each eigenspace, and can return
non-Clifford local factors that cancel only when the gate is reassembled.
"""
function _is_clifford_gate(M::AbstractMatrix)
    for G in _CLIFFORD_GENERATORS
        image = M * G * M'
        best = 0.0
        for P in _PAULI_2Q
            best = max(best, abs(tr(P' * image)) / 4)
            best >= 1 - CLIFFORD_ATOL && break
        end
        best >= 1 - CLIFFORD_ATOL || return false
    end
    return true
end

"""
    _product_factors(M) -> Union{Nothing,Tuple{Matrix,Matrix}}

Split `M = kron(A, B)` when the operator-Schmidt rank of `M` is one, otherwise
return `nothing`. `A` acts on the high-order qubit and `B` on the low-order one.
The reshuffled matrix `R[(r1,c1),(r2,c2)] = M[(r1,r2),(c1,c2)]` equals
`vec(A) vec(B)^T` exactly when `M` is a product, so the test is a rank test on
`R`. The factors are read off the leading singular vectors and verified against
`M`.
"""
function _product_factors(M::AbstractMatrix)
    R = Matrix{ComplexF64}(undef, 4, 4)
    for r1 in 0:1, c1 in 0:1, r2 in 0:1, c2 in 0:1
        R[1 + r1 + 2 * c1, 1 + r2 + 2 * c2] = M[1 + r2 + 2 * r1, 1 + c2 + 2 * c1]
    end
    factorization = svd(R)
    factorization.S[2] <= PRODUCT_ATOL * max(factorization.S[1], 1.0) || return nothing
    scale = sqrt(factorization.S[1])
    A = reshape(factorization.U[:, 1] .* scale, 2, 2)
    B = reshape(conj.(factorization.V[:, 1]) .* scale, 2, 2)
    norm(kron(A, B) - M) <= sqrt(PRODUCT_ATOL) || return nothing
    return (A, B)
end

# A wire carries a pending one-qubit unitary that keeps absorbing gates until a
# CNOT closes it. `fresh` means that no entangling gate has touched the wire yet,
# so it still holds |0> in a product with the rest of the register.
mutable struct _Wire
    pending::Matrix{ComplexF64}
    fresh::Bool
end

_Wire(fresh::Bool) = _Wire(copy(_EYE2), fresh)

_push!(wire::_Wire, U::AbstractMatrix) = (wire.pending = U * wire.pending)

_is_identity(U::AbstractMatrix) = isapprox(U, _EYE2; atol=1e-14)

"""
    _block_angles(U; boundary) -> Vector{Float64}

Non-Clifford rotation angles of the fused one-qubit unitary `U`.
`MPSCircuits.zyz_decomposition` returns `(t1, t2, t3)` with
`U = Rz(t3) Ry(t2) Rz(t1)` up to a global phase, so `Rz(t1)` acts first. It
collapses gimbal-locked cases to `t1 = 0`, so a diagonal `U` costs one rotation.
Angles that are multiples of `pi/2` are dropped.

With `boundary = true` the input state is |0>: `Rz(t1)` is a global phase and is
dropped, and if `Ry(t2)` leaves the wire on a pole (`t2` a multiple of `pi`) then
`Rz(t3)` is also a phase and the block is free.
"""
function _block_angles(U::AbstractMatrix; boundary::Bool)
    t1, t2, t3 = MPSCircuits.zyz_decomposition(U)
    kept = Float64[]
    if boundary
        _near_multiple(t2, pi) && return kept
        for theta in (t2, t3)
            _angle_class(theta) === :clifford || push!(kept, theta)
        end
        return kept
    end
    for theta in (t1, t2, t3)
        _angle_class(theta) === :clifford || push!(kept, theta)
    end
    return kept
end

# Close the pending block on `wire` before a CNOT: return its non-Clifford angles
# (empty if the block is free) and reset the wire to the identity.
function _flush!(wire::_Wire, use_boundary::Bool)
    boundary = use_boundary && wire.fresh
    angles = Float64[]
    if !(_is_identity(wire.pending) && !boundary)
        angles = _block_angles(wire.pending; boundary=boundary)
    end
    wire.pending = copy(_EYE2)
    return angles
end

"""
    _rotation_blocks(circuit; initial_state=:zero) -> Vector{Vector{Float64}}

Decompose `circuit` (a list of one- and two-qubit `MPSCircuits.UnitaryGate`s) into
fused one-qubit blocks, each given by its non-Clifford rotation angles. Blocks
with no such angles are omitted.

Each one-qubit gate is multiplied into the pending unitary of its wire. Each
two-qubit gate is decomposed by a Cartan KAK decomposition into two incoming
one-qubit unitaries, a core of three CNOTs interleaved with three rotations, and
two outgoing one-qubit unitaries. The CNOTs are the only barriers: the outgoing
unitaries of one gate stay pending and fuse with the incoming ones of the next
gate on the same wire, so a chain of generic SU(4) gates costs 9 rotations per
gate rather than the 15 of the unfused decomposition (arXiv:2603.11156).
Two special cases are free: a gate of operator-Schmidt rank one is a product of
one-qubit gates and raises no barrier, and a Clifford gate carries no rotations
but is still a barrier, because a rotation pushed through an entangling Clifford
becomes a two-qubit Pauli rotation and cannot be merged with its neighbours.

With `initial_state = :zero` the circuit acts on |0...0>, so the leading `Rz` of
the first block on each wire is a global phase (see `_block_angles`). With
`initial_state = :arbitrary` this saving is forfeited.
"""
function _rotation_blocks(circuit; initial_state::Symbol=:zero)
    initial_state in (:zero, :arbitrary) || throw(ArgumentError(
        "initial_state must be :zero or :arbitrary, got `$initial_state`"))
    use_boundary = initial_state === :zero

    nqubits = isempty(circuit) ? 0 :
        maximum(maximum(gate.site_numbers) for gate in circuit)
    wires = Dict(q => _Wire(use_boundary) for q in 1:nqubits)
    blocks = Vector{Vector{Float64}}()

    function emit!(q)
        angles = _flush!(wires[q], use_boundary)
        isempty(angles) || push!(blocks, angles)
        wires[q].fresh = false
        return nothing
    end

    for gate in circuit
        hasproperty(gate, :site_numbers) || throw(ArgumentError(
            "gates must carry their site numbers and indices " *
            "(e.g. `MPSCircuits.UnitaryGate`), got a $(typeof(gate))"))
        sites = gate.site_numbers
        if length(sites) == 1
            _push!(wires[sites[1]], _matrix_1q(gate))
            continue
        end
        length(sites) == 2 || throw(ArgumentError(
            "expected a one- or two-qubit gate, got support $(sites)"))

        q1, q2 = sites[1], sites[2]
        M = _matrix_2q(gate)

        factors = _product_factors(M)
        if factors !== nothing
            A, B = factors
            _push!(wires[q1], A)
            _push!(wires[q2], B)
            continue
        end

        if _is_clifford_gate(M)
            emit!(q1); emit!(q2)
            continue
        end

        kak = MPSCircuits.KAKGateSU4(MPSCircuits.UnitaryGate(
            gate.tensor, gate.site_numbers, gate.site_indices))

        # Incoming one-qubit unitaries fuse with what the wires already hold.
        _push!(wires[q1], _matrix_1q(kak.A_R1))
        _push!(wires[q2], _matrix_1q(kak.A_R2))
        # The three CNOTs of the core, with its rotations between them.
        emit!(q1); emit!(q2)
        _push!(wires[q1], _rz(-2 * kak.core.gamma + pi / 2))
        _push!(wires[q2], _ry(-pi / 2 + 2 * kak.core.alpha))
        emit!(q1); emit!(q2)
        _push!(wires[q2], _ry(-2 * kak.core.beta + pi / 2))
        emit!(q1); emit!(q2)
        # Outgoing one-qubit unitaries stay pending for the next gate.
        _push!(wires[q1], _matrix_1q(kak.A_L1))
        _push!(wires[q2], _matrix_1q(kak.A_L2))
    end

    # Close every wire: the final blocks have no successor to fuse into.
    for q in 1:nqubits
        angles = _flush!(wires[q], use_boundary)
        isempty(angles) || push!(blocks, angles)
    end
    return blocks
end

"""
    rotation_count(circuit; initial_state=:zero) -> Int

Number of non-Clifford single-qubit rotations in `circuit` after KAK
decomposition, fusion of one-qubit runs and removal of Clifford rotations. This
count does not depend on the synthesis precision. For a chain of generic SU(4)
gates it is 9 per gate plus the boundary blocks of the last gate on each wire
(arXiv:2603.11156). See `_rotation_blocks` for the decomposition and
for `initial_state`.
"""
rotation_count(circuit; kwargs...) =
    sum(length, _rotation_blocks(circuit; kwargs...); init=0)

"""
    t_count(circuit; epsilon=nothing, total_error=nothing, model=TRASYN,
        initial_state=:zero) -> Float64

Number of `T` gates needed to realise `circuit` in Clifford+`T`. The circuit is
reduced to fused one-qubit blocks (see `_rotation_blocks`) and each block is
charged according to `model`, a [`RotationSynthesis`](@ref): with `c` synthesis
calls at precision `epsilon` and `x` exactly synthesizable rotations,

    T = c * (slope * log2(1/epsilon) + intercept) + x.

Exactly one of `epsilon` and `total_error` must be given. `epsilon` is the
precision of each synthesis call. `total_error` is a budget for the whole
circuit, split evenly over the calls, `epsilon = total_error / c`; by the union
bound the `c` approximations then compose to an operator-norm error of at most
`total_error`. The result is not rounded.
"""
function t_count(circuit; epsilon=nothing, total_error=nothing,
        model::RotationSynthesis=TRASYN, initial_state::Symbol=:zero)
    (epsilon === nothing) == (total_error === nothing) && throw(ArgumentError(
        "pass exactly one of `epsilon` (per synthesis call) or `total_error` " *
        "(budget for the whole circuit)"))
    blocks = _rotation_blocks(circuit; initial_state=initial_state)
    calls = 0
    exact = 0
    for angles in blocks
        use = _block_synthesis(angles, model)
        calls += use.calls
        exact += use.exact_t
    end
    if epsilon !== nothing
        eps = float(epsilon)
    else
        total_error > 0 || throw(ArgumentError(
            "total_error must be positive, got $total_error"))
        # With nothing to approximate the count is independent of epsilon.
        eps = calls == 0 ? float(total_error) : total_error / calls
    end
    return calls * _synthesis_t_gates(model, eps) + exact
end

end # module
