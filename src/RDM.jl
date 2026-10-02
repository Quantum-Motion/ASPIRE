"""
    RDM

Two-site reduced density matrices of an MPS and the entropies computed from them. The pair
tensors are obtained by transfer-matrix sweeps in mixed-canonical form; `dense_two_site_matrix`
converts them to 4x4 matrices on which `von_neumann_entropy` and `quantum_mutual_information`
act.
"""
module RDM
using ITensors
using ITensorMPS
using LinearAlgebra

export two_site_rdm, two_site_rdms, all_pairs_two_site_rdms, dense_two_site_matrix,
    von_neumann_entropy, quantum_mutual_information

# Eigenvalues below this magnitude are treated as zero when forming entropies.
const EIG_TOL = 1e-14

# Flat index into a 4x4 two-qubit operator for the basis state |a b>, site i (the `a` qubit)
# most significant. Matches the ordering of `dense_two_site_matrix`.
two_qubit_index(a::Int, b::Int) = 2 * (a - 1) + b

"""
    two_site_rdm(psi, i, j) -> ITensor

Two-site reduced density matrix of sites `i < j` of `psi`, with open indices `s[i]`, `s[i]'`,
`s[j]`, `s[j]'`.

The MPS is put into mixed-canonical form with the orthogonality centre at `i`, so every site
outside `[i, j]` contracts to the identity. The transfer matrix is then swept from `i` to `j`,
tracing the physical index of each intermediate site. The cost is `O((j - i) chi^3 d)` and no
tensor with more than two open physical indices is formed.
"""
function two_site_rdm(psi::MPS, i::Int64, j::Int64)
    i < j || throw(ArgumentError("Expected i < j."))
    N = length(psi)
    (1 <= i < j <= N) || throw(ArgumentError("Site indices out of bounds."))

    psi = orthogonalize(psi, i)
    s = siteinds(psi)

    # Links internal to the kept region [i, j]: l_i, ..., l_{j-1}. These get
    # primed on the bra so ket/bra stay distinct; the boundary links l_{i-1}
    # and l_j are left unprimed so they contract as the identity (guaranteed by
    # the left-/right-canonical sites outside [i, j]).
    internal_links = [linkind(psi, k) for k in i:(j - 1)]

    bra_of = function (k)
        A = dag(psi[k])
        for l in internal_links
            if hasind(psi[k], l)
                A = prime(A, l)
            end
        end
        # Keep the physical index open only on the two sites of interest; for
        # intermediate sites the unprimed site index contracts and is traced.
        if k == i || k == j
            A = prime(A, s[k])
        end
        return A
    end

    rho = psi[i] * bra_of(i)
    for k in (i + 1):j
        rho *= psi[k]
        rho *= bra_of(k)
    end

    return rho
end

"""
    two_site_rdms(psi, i, js) -> Dict{Int,ITensor}

Reduced density matrices of the pairs `(i, j)`, `j` in `js` (all `> i`), from one left-to-right
sweep out of site `i`.

Calling [`two_site_rdm`](@ref) once per `j` re-canonicalises the chain every time, at
`O(N chi^3)` each. Here the chain is orthogonalised once and the partially traced environment
(open indices `s[i]`, `s[i]'` and the bond `l_k`, `l_k'`) is carried to the right, so each
additional `j` costs `O(chi^3)`. Each returned tensor equals `two_site_rdm(psi, i, j)`. `js` need
not be sorted and may skip sites; the sweep stops at `maximum(js)`.
"""
function two_site_rdms(psi::MPS, i::Int, js)
    N = length(psi)
    wanted = sort!(unique(collect(Int, js)))
    isempty(wanted) && return Dict{Int,ITensor}()
    (1 <= i < N) || throw(ArgumentError("Expected 1 <= i < N, got i=$i, N=$N."))
    wanted[1] > i || throw(ArgumentError("Every j must exceed i=$i, got $(wanted[1])."))
    wanted[end] <= N || throw(ArgumentError("Site index $(wanted[end]) out of bounds."))

    phi = orthogonalize(psi, i)
    s = siteinds(phi)
    jmax = wanted[end]
    targets = Set(wanted)
    out = Dict{Int,ITensor}()

    # Bra at site i: the right bond is internal to every pair (i, j) with j > i, so it
    # is primed; the left bond is not and contracts as the identity against the
    # left-canonical tail. The physical index stays open on both sides of the pair.
    bra_i = prime(prime(dag(phi[i]), linkind(phi, i)), s[i])
    E = phi[i] * bra_i                     # open: s[i], s[i]', l_i, l_i'

    for k in (i + 1):jmax
        l_left = linkind(phi, k - 1)       # internal to the pair -> primed on the bra
        l_right = k < N ? linkind(phi, k) : nothing
        if k in targets
            # Second site of the pair: physical index open, right bond left unprimed so
            # it traces against the right-canonical tail.
            bra = prime(prime(dag(phi[k]), l_left), s[k])
            out[k] = E * phi[k] * bra
        end
        if k < jmax
            # Intermediate site: physical index traced, both bonds internal -> primed.
            bra = prime(dag(phi[k]), l_left)
            l_right === nothing || (bra = prime(bra, l_right))
            E = E * phi[k] * bra
        end
    end

    return out
end

"""
    all_pairs_two_site_rdms(psi, pairs) -> Dict{Tuple{Int,Int},ITensor}

[`two_site_rdm`](@ref) for every `(i, j)` in `pairs`, grouped by `i` so that each distinct `i`
costs one [`two_site_rdms`](@ref) sweep. This is the all-pairs scan performed once per layer by
the selectors; the grouping reduces its cost from `O(N^3 chi^3)` to `O(N^2 chi^3)`.
"""
function all_pairs_two_site_rdms(psi::MPS, pairs)
    by_i = Dict{Int,Vector{Int}}()
    for pair in pairs
        i, j = minmax(Int(pair[1]), Int(pair[2]))
        i == j && throw(ArgumentError("A pair must join distinct sites, got ($i, $j)."))
        push!(get!(by_i, i, Int[]), j)
    end
    out = Dict{Tuple{Int,Int},ITensor}()
    for (i, js) in by_i
        for (j, rho) in two_site_rdms(psi, i, js)
            out[(i, j)] = rho
        end
    end
    return out
end

"""
    dense_two_site_matrix(rho, sites, i, j) -> Matrix{ComplexF64}

Dense 4x4 matrix of a two-site reduced density matrix tensor. Rows and columns are ordered
`(1,1), (1,2), (2,1), (2,2)` over `(site i, site j)`, so site `i` is the most significant qubit.
"""
function dense_two_site_matrix(rho::ITensor, sites, i::Int, j::Int)
    mat = zeros(ComplexF64, 4, 4)
    basis = ((1, 1), (1, 2), (2, 1), (2, 2))
    for row in 1:4
        a, b = basis[row]
        for col in 1:4
            c, d = basis[col]
            mat[row, col] = rho[
                sites[i] => a,
                sites[j] => b,
                prime(sites[i]) => c,
                prime(sites[j]) => d,
            ]
        end
    end
    return mat
end

"Return a matrix unchanged when the reduced density matrix is already dense."
dense_two_site_matrix(rho::AbstractMatrix) = rho

# Hermitian eigenvalues of a density matrix, symmetrised first to remove numerical
# non-Hermiticity from the tensor contraction.
hermitian_eigvals(rho::AbstractMatrix) = eigvals(Hermitian((rho + rho') / 2))

"""
    von_neumann_entropy(rho) -> Float64

Von Neumann entropy `-tr(rho log2 rho)` in bits of a density matrix.
"""
function von_neumann_entropy(rho::AbstractMatrix)
    s = 0.0
    for lambda in hermitian_eigvals(rho)
        p = real(lambda)
        if p > EIG_TOL
            s -= p * log2(p)
        end
    end
    return s
end

"""
    partial_trace_two_qubit(rho, keep) -> Matrix

Reduced density matrix of qubit `keep` (1 or 2) of a 4x4 two-qubit density matrix.
"""
function partial_trace_two_qubit(rho::AbstractMatrix, keep::Int)
    keep in (1, 2) || throw(ArgumentError("keep must be 1 or 2"))
    size(rho) == (4, 4) || throw(ArgumentError("Expected a 4x4 matrix."))
    out = zeros(eltype(rho), 2, 2)
    if keep == 1
        for a in 1:2, ap in 1:2, b in 1:2
            out[a, ap] += rho[two_qubit_index(a, b), two_qubit_index(ap, b)]
        end
    else
        for b in 1:2, bp in 1:2, a in 1:2
            out[b, bp] += rho[two_qubit_index(a, b), two_qubit_index(a, bp)]
        end
    end
    return out
end

"""
    quantum_mutual_information(rho) -> Float64

Quantum mutual information `I(A:B) = S(A) + S(B) - S(AB)` in bits of a two-qubit density matrix.
"""
function quantum_mutual_information(rho::AbstractMatrix)
    ra = partial_trace_two_qubit(rho, 1)
    rb = partial_trace_two_qubit(rho, 2)
    return von_neumann_entropy(ra) + von_neumann_entropy(rb) - von_neumann_entropy(rho)
end

end # module
