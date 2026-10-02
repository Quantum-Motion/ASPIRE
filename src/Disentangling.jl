"""
    Disentangling

Two-qubit gates on an MPS and the closed-form disentangler used throughout the package: the
unitary that rotates a pair into the eigenbasis of its reduced density matrix. Also provides the
global entanglement and total-correlation measures, the candidate-pair enumeration, and the
greedy min-cut layer builder `mincut_batch` with the layered driver
`parallel_disentangling_with_gate_log`.
"""
module Disentangling

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

using ..SimConfig: global_cutoff, global_maxdim
using ..EntanglementBasics: bond_entropy
using ..RDM: two_site_rdm, dense_two_site_matrix, von_neumann_entropy,
    quantum_mutual_information

export total_entanglement, candidate_pairs,
    two_qubit_gate, apply_two_qubit_unitary, apply_two_qubit_unitary_tracked,
    eig_disentangler, mi_disentangler, disentangle_pair,
    haar_unitary, random_circuit_state,
    total_correlation,
    ASPIRE_DEFAULT_THRESHOLD, ASPIRE_DEFAULT_RELATIVE_THRESHOLD,
    aspire_rank_threshold, mincut_batch,
    DisentanglingGate, bond_dimension_summary, compressed_mps,
    parallel_disentangling_with_gate_log

# Default admission floors of the min-cut layer builder: an absolute floor on the pair score
# that keeps near-product states from chasing numerical dust, and a floor relative to the best
# score of the layer that removes the low-score tail of otherwise useful layers.
const ASPIRE_DEFAULT_THRESHOLD = 1e-3
const ASPIRE_DEFAULT_RELATIVE_THRESHOLD = 0.10

# Flat index into a 4x4 two-qubit operator for the basis state |a b>, first qubit most
# significant. Matches `RDM.dense_two_site_matrix`.
two_qubit_index(a::Int, b::Int) = 2 * (a - 1) + b

"""
    total_entanglement(psi) -> Float64

Sum over all bonds of the von Neumann entanglement entropy (in bits) of the MPS `psi`.

All bonds are read off a single left-to-right sweep of the orthogonality centre, which costs
`O(N chi^3)`; evaluating `bond_entropy` bond by bond would re-orthogonalise the chain every time.
"""
function total_entanglement(psi::MPS)
    N = length(psi)
    N < 2 && return 0.0
    phi = orthogonalize(psi, 1)
    total = 0.0
    for b in 1:(N - 1)
        b > 1 && orthogonalize!(phi, b)
        total += bond_entropy(phi, b)
    end
    return total
end

"""
    candidate_pairs(N; adjacent=false, max_distance=nothing) -> Vector{Tuple{Int,Int}}

Unordered site pairs `(i, j)`, `i < j`, of an `N`-site chain that a selector may act on.
The default is all pairs (all-to-all connectivity). `max_distance=d` keeps only pairs with
`j - i <= d`, so `d = 1` is the nearest-neighbour set and `d = N - 1` is all pairs; `adjacent=true`
is shorthand for `max_distance=1`. Non-adjacent pairs are applied through the swap network of
`ITensorMPS.apply`.
"""
function candidate_pairs(N::Int; adjacent::Bool=false,
        max_distance::Union{Nothing,Int}=nothing)
    N >= 0 || throw(ArgumentError("N must be non-negative."))
    max_distance !== nothing && max_distance < 1 &&
        throw(ArgumentError("max_distance must be at least 1."))
    distance = adjacent ? 1 : something(max_distance, max(N - 1, 0))
    adjacent && max_distance !== nothing && max_distance != 1 &&
        throw(ArgumentError("adjacent=true is only compatible with max_distance=1."))
    return [(i, j) for i in 1:N for j in (i + 1):min(N, i + distance)]
end

"""
    two_qubit_gate(U, si, sj) -> ITensor

Two-site gate tensor of the 4x4 matrix `U` acting on the physical indices `si` and `sj`. The
index `si` is the most significant qubit: the gate maps `|c d>` to `sum_ab U[(a,b),(c,d)] |a b>`,
the ordering of `RDM.dense_two_site_matrix`.
"""
function two_qubit_gate(U::AbstractMatrix, si::Index, sj::Index)
    size(U) == (4, 4) || throw(ArgumentError("Expected a 4x4 unitary."))
    G = ITensor(ComplexF64, prime(si), prime(sj), si, sj)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        G[prime(si) => a, prime(sj) => b, si => c, sj => d] =
            U[two_qubit_index(a, b), two_qubit_index(c, d)]
    end
    return G
end

"""
    apply_two_qubit_unitary(psi, i, j, U; cutoff=global_cutoff(), maxdim=global_maxdim()) -> MPS

Apply the 4x4 unitary `U` to sites `i` and `j` of `psi` and return the renormalised state.
Adjacent sites act on one bond; non-adjacent sites go through the swap network of
`ITensorMPS.apply`. The state is truncated at `cutoff` and `maxdim`.
"""
function apply_two_qubit_unitary(psi::MPS, i::Int, j::Int, U::AbstractMatrix;
        cutoff::Real=global_cutoff(), maxdim::Integer=global_maxdim())
    return apply_two_qubit_unitary_tracked(psi, i, j, U;
        cutoff=cutoff, maxdim=maxdim, track=false).psi
end

"""
    apply_two_qubit_unitary_tracked(psi, i, j, U; cutoff, maxdim, track=true) -> (psi, discarded)

As [`apply_two_qubit_unitary`](@ref), and also report the weight discarded by the truncation.
A unitary preserves the norm, so on a normalised input the norm lost to the SVD truncation is
the discarded weight, `discarded = 1 - ||phi||^2`. With `track=false` the norm is not evaluated
and `discarded` is `NaN`.
"""
function apply_two_qubit_unitary_tracked(psi::MPS, i::Int, j::Int,
        U::AbstractMatrix; cutoff::Real=global_cutoff(),
        maxdim::Integer=global_maxdim(), track::Bool=true)
    cutoff >= 0 || throw(ArgumentError("cutoff must be non-negative, got $cutoff"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive, got $maxdim"))
    s = siteinds(psi)
    phi = apply(two_qubit_gate(U, s[i], s[j]), psi; cutoff=Float64(cutoff), maxdim=Int(maxdim))
    discarded = NaN
    if track
        before = real(norm(psi))
        after = real(norm(phi))
        discarded = before <= 0 ? NaN : clamp(1 - (after / before)^2, 0.0, 1.0)
    end
    return (psi=normalize!(phi), discarded=discarded)
end

"""
    eig_disentangler(psi, i, j) -> Matrix{ComplexF64}

Closed-form two-site disentangler of Hyatt, Garrison and Bauer (Phys. Rev. B 95, 035117,
App. B.2): the unitary that minimises the mutual information `I(i:j)` of the pair.

A two-site unitary leaves `S(rho_ij)` unchanged, so minimising `I(i:j) = S(rho_i) + S(rho_j) -
S(rho_ij)` means minimising `S(rho_i) + S(rho_j)`. With the eigendecomposition
`rho_ij = V diag(lambda) V'`, `lambda` in descending order, the minimiser is `U = V'`, which
assigns the eigenvectors to the computational basis in descending eigenvalue order. Returns `U`
in the basis of `RDM.dense_two_site_matrix`. `mi_disentangler` is an alias.
"""
function eig_disentangler(psi::MPS, i::Int, j::Int)
    s = siteinds(psi)
    rho = dense_two_site_matrix(two_site_rdm(psi, i, j), s, i, j)
    F = eigen(Hermitian((rho + rho') / 2))
    order = sortperm(F.values; rev=true)
    return Matrix(F.vectors[:, order]')
end

const mi_disentangler = eig_disentangler

"""
    disentangle_pair(psi, i, j; disentangler=eig_disentangler) -> MPS

Apply `disentangler(psi, i, j)`, a 4x4 unitary, to the pair `(i, j)` of `psi`.
"""
function disentangle_pair(psi::MPS, i::Int, j::Int; disentangler=eig_disentangler)
    U = disentangler(psi, i, j)
    return apply_two_qubit_unitary(psi, i, j, U)
end

"""
    haar_unitary(n, rng) -> Matrix{ComplexF64}

Haar-random `n x n` unitary, drawn from the QR decomposition of a complex Gaussian matrix with
the phases of the diagonal of `R` removed.
"""
function haar_unitary(n::Int, rng::AbstractRNG)
    z = (randn(rng, ComplexF64, n, n)) / sqrt(2)
    F = qr(z)
    d = diag(F.R)
    ph = d ./ abs.(d)
    return Matrix(F.Q) * Diagonal(ph)
end

"""
    random_circuit_state(sites; gates=2*length(sites), adjacent=true, rng) -> MPS

Apply `gates` Haar-random two-qubit gates to uniformly chosen pairs of the all-up product state.
With `adjacent=true` the pairs are nearest neighbours; with `adjacent=false` they are drawn from
all pairs, which plants long-range entanglement. Used to generate test states.
"""
function random_circuit_state(sites; gates::Int=2 * length(sites),
        adjacent::Bool=true, rng::AbstractRNG=Random.default_rng())
    N = length(sites)
    N >= 2 || throw(ArgumentError("Need at least 2 sites."))
    bonds = candidate_pairs(N; adjacent=adjacent)
    psi = MPS(ComplexF64, sites, fill("Up", N))
    for _ in 1:gates
        i, j = rand(rng, bonds)
        psi = apply_two_qubit_unitary(psi, i, j, haar_unitary(4, rng))
    end
    return normalize!(psi)
end

"""
    total_correlation(psi) -> Float64

Total correlation `T = sum_i S(rho_i) - S(rho)` of a pure MPS in bits. For a pure state
`S(rho) = 0`, so `T` is the sum of the single-site entropies and vanishes exactly on product
states. It is the quantity driven to zero by the disentangling protocols (Hyatt et al.), and the quantity whose decrease `Delta T` under a gate equals the removable
correlation `R_ij`. All sites are read off one sweep of the orthogonality centre.
"""
function total_correlation(psi::MPS)
    N = length(psi)
    N == 0 && return 0.0
    phi = orthogonalize(psi, 1)
    s = siteinds(phi)
    total = 0.0
    for i in 1:N
        i > 1 && orthogonalize!(phi, i)
        A = phi[i]
        rho = A * dag(prime(A, s[i]))
        total += von_neumann_entropy(Matrix(Array(rho, prime(s[i]), s[i])))
    end
    return total
end

"""
    aspire_rank_threshold(rank_values; threshold=1e-6, relative_threshold=0.0) -> Float64

Score floor of a layer: `max(threshold, relative_threshold * maximum(rank_values))`.
"""
function aspire_rank_threshold(rank_values;
        threshold::Real=1e-6, relative_threshold::Real=0.0)
    threshold < 0 && throw(ArgumentError("threshold must be non-negative."))
    relative_threshold < 0 && throw(ArgumentError("relative_threshold must be non-negative."))
    vals = collect(Float64, rank_values)
    isempty(vals) && return Float64(threshold)
    return max(Float64(threshold), Float64(relative_threshold) * maximum(vals))
end

"""
    mincut_batch(pairs, rank_of; couple_of=rank_of, couple, rank_threshold, couple_threshold,
                 max_rank=nothing) -> Vector{Tuple{Int,Int}}

Greedy layer of pairs to disentangle in parallel. Pairs are visited in descending order of the
score `rank_of[p]`, stopping below `rank_threshold`. A pair `(i, j)` joins the batch if it is
disjoint from the pairs already chosen and decoupled from them: `couple(a, b)`, the coupling
between a qubit of the candidate and a qubit of a chosen pair, must not exceed `couple_threshold`
for any of the four combinations.

`rank_of` maps each candidate pair to its selection score. `couple(a, b)` is evaluated on
arbitrary qubit pairs, not only candidates; by default it is read from `couple_of`, which is
valid only when `couple_of` has an entry for every qubit pair. `max_rank` restricts the pool
to the `max_rank` best-ranked pairs before the disjointness and coupling tests.
"""
function mincut_batch(pairs, rank_of::Dict; couple_of::Dict=rank_of,
        couple::Function=(a, b) -> couple_of[(min(a, b), max(a, b))],
        rank_threshold::Float64, couple_threshold::Float64,
        max_rank::Union{Nothing,Int}=nothing)
    ranked = sort(collect(pairs); by=p -> rank_of[p], rev=true)
    if max_rank !== nothing
        max_rank < 0 && throw(ArgumentError("max_rank must be non-negative or `nothing`."))
        ranked = first(ranked, min(max_rank, length(ranked)))
    end
    batch = Tuple{Int,Int}[]
    used = Set{Int}()
    for (i, j) in ranked
        rank_of[(i, j)] < rank_threshold && break
        (i in used || j in used) && continue
        coupled = any(batch) do p
            max(couple(i, p[1]), couple(i, p[2]), couple(j, p[1]), couple(j, p[2])) > couple_threshold
        end
        coupled && continue
        push!(batch, (i, j)); push!(used, i); push!(used, j)
    end
    return batch
end

# Memoised mutual-information coupling `(a, b) -> I(a:b)` for the qubits of `psi`, seeded with
# the values already computed for the candidate pairs. Misses are filled by contracting the
# two-site reduced density matrix on demand.
function _coupling_fn(psi::MPS, s, seed::Dict)
    cache = Dict{Tuple{Int,Int},Float64}(seed)
    return function (a, b)
        i, j = minmax(a, b)
        get!(cache, (i, j)) do
            real(quantum_mutual_information(dense_two_site_matrix(two_site_rdm(psi, i, j), s, i, j)))
        end
    end
end

"""
    DisentanglingGate(i, j, U, layer)

A two-qubit unitary `U` applied to sites `(i, j)` in layer `layer` of a disentangling run.
"""
struct DisentanglingGate
    i::Int
    j::Int
    U::Matrix{ComplexF64}
    layer::Int
end

# Link dimensions between neighbouring MPS tensors.
function _linkdims(psi::MPS)
    N = length(psi)
    N <= 1 && return Int[]
    return [dim(commonind(psi[b], psi[b + 1])) for b in 1:(N - 1)]
end

"""
    bond_dimension_summary(psi) -> (dims, maxdim, sumdim)

Link dimensions of `psi` with their maximum and sum.
"""
function bond_dimension_summary(psi::MPS)
    dims = _linkdims(psi)
    maxdim = isempty(dims) ? 1 : maximum(dims)
    return (dims=dims, maxdim=maxdim, sumdim=sum(dims))
end

"""
    compressed_mps(psi; cutoff, maxdim=nothing) -> MPS

Normalised copy of `psi` truncated at the SVD `cutoff` (and `maxdim` if given).
"""
function compressed_mps(psi::MPS; cutoff::Real, maxdim::Union{Nothing,Int}=nothing)
    phi = copy(psi)
    if isnothing(maxdim)
        truncate!(phi; cutoff=Float64(cutoff))
    else
        truncate!(phi; cutoff=Float64(cutoff), maxdim=maxdim)
    end
    return normalize!(phi)
end

"""
    parallel_disentangling_with_gate_log(psi; threshold, relative_threshold, max_rank,
        couple_threshold=0.1, max_depth=80, pairs, disentangler=mi_disentangler,
        score=quantum_mutual_information) -> NamedTuple

Layered min-cut disentangling. In each layer every pair in `pairs` is scored by `score` on its
two-site reduced density matrix, a batch is selected by [`mincut_batch`](@ref) with the pair
mutual information as coupling, and `disentangler` is applied to each pair of the batch. The
run stops after `max_depth` layers or when a layer is empty.

The score floor is `max(threshold, relative_threshold * best_score)` per layer
([`aspire_rank_threshold`](@ref)); `max_rank` restricts each layer to the best-ranked pairs.

Returns `(psi, depth, gates, total_correlation, gate_counts, gate_log)`: the final state,
the number of layers and gates, the trace of `T` and of the cumulative gate count after each
layer (both starting from the input), and the applied gates as `DisentanglingGate`s.
"""
function parallel_disentangling_with_gate_log(psi::MPS;
        threshold::Float64=ASPIRE_DEFAULT_THRESHOLD,
        relative_threshold::Float64=ASPIRE_DEFAULT_RELATIVE_THRESHOLD,
        max_rank::Union{Nothing,Int}=nothing,
        couple_threshold::Float64=0.1, max_depth::Int=80,
        pairs=candidate_pairs(length(psi); adjacent=false), disentangler=mi_disentangler,
        score=quantum_mutual_information)
    phi = copy(psi)
    s = siteinds(phi)
    trace = Float64[total_correlation(phi)]
    gate_counts = Int[0]
    gate_log = DisentanglingGate[]
    gates = 0
    depth = 0

    for layer in 1:max_depth
        rhos = Dict(p => dense_two_site_matrix(two_site_rdm(phi, p[1], p[2]), s, p[1], p[2]) for p in pairs)
        rank_of = Dict(p => real(score(rhos[p])) for p in pairs)
        couple_of = Dict(p => real(quantum_mutual_information(rhos[p])) for p in pairs)
        rank_threshold = aspire_rank_threshold(values(rank_of);
            threshold=threshold, relative_threshold=relative_threshold)
        batch = mincut_batch(pairs, rank_of; couple=_coupling_fn(phi, s, couple_of),
            rank_threshold=rank_threshold, couple_threshold=couple_threshold,
            max_rank=max_rank)
        isempty(batch) && break

        for (i, j) in batch
            U = Matrix{ComplexF64}(disentangler(phi, i, j))
            push!(gate_log, DisentanglingGate(i, j, U, layer))
            phi = apply_two_qubit_unitary(phi, i, j, U)
        end

        s = siteinds(phi)
        gates += length(batch)
        depth += 1
        push!(trace, total_correlation(phi))
        push!(gate_counts, gates)
    end

    return (psi=phi, depth=depth, gates=gates, total_correlation=trace,
        gate_counts=gate_counts, gate_log=gate_log)
end

end # module
