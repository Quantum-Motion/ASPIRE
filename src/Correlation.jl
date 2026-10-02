"""
    Correlation

Entropic quantities of an MPS used by ASPIRE: single-site entropies and the total
correlation T (bits), the certificate quantity Lambda and the infidelity bounds
built from it (the certified bound), reduced density matrices on small supports,
and the removable correlation R_S. A dense k-site
reduced density matrix is a d^k x d^k matrix whose first site is the most
significant index, as in `RDM.dense_two_site_matrix`.
"""
module Correlation

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

using ..RDM: two_site_rdm, dense_two_site_matrix, von_neumann_entropy,
    quantum_mutual_information

export site_entropies, total_correlation_support, register_total_correlation
export lambda_certificate, infidelity_upper_bound, certified_infidelity_bound
export reduced_density_matrix, dense_reduced_density_matrix, marginal
export removable_correlation_pair, removable_correlation,
    removable_correlation_symmetric

const EIG_FLOOR = 1e-12
const PURITY_TOL = 1e-10

"Shannon entropy in bits of a (possibly unnormalised) spectrum; entries below `EIG_FLOOR` are dropped."
function spectrum_entropy(values)
    total = 0.0
    for v in values
        p = real(v)
        p > EIG_FLOOR || continue
        total -= p * log2(p)
    end
    return total
end

hermitian_spectrum(rho::AbstractMatrix) =
    real.(eigvals(Hermitian((Matrix(rho) + Matrix(rho)') / 2)))

"""
    site_entropies(psi) -> Vector{Float64}

Single-site von Neumann entropies `S(rho^a)` of an MPS, in bits. One
left-to-right canonicalisation sweep, cost `O(N chi^3)`; the state vector is
never formed.
"""
function site_entropies(psi::MPS)
    N = length(psi)
    N == 0 && return Float64[]
    phi = orthogonalize(psi, 1)
    s = siteinds(phi)
    out = Vector{Float64}(undef, N)
    for i in 1:N
        i > 1 && orthogonalize!(phi, i)
        A = phi[i]
        rho = A * dag(prime(A, s[i]))
        out[i] = von_neumann_entropy(
            Matrix(Array(rho, prime(s[i]), s[i])))
    end
    return out
end

"""
    register_total_correlation(psi; entropies=nothing) -> Float64

Total correlation `T = sum_a S(rho^a) - S(rho^full)` of the full register, in
bits. An MPS is pure, so `S(rho^full) = 0` provided it is normalised; this is
checked by [`assert_pure`](@ref). Precomputed `entropies` may be passed to avoid
a second sweep.
"""
function register_total_correlation(psi::MPS; entropies=nothing)
    assert_pure(psi)
    return sum(entropies === nothing ? site_entropies(psi) : entropies)
end

"""
    assert_pure(psi; atol=1e-10)

Throw an `ArgumentError` unless `<psi|psi> = 1` to within `atol`.
"""
function assert_pure(psi::MPS; atol::Real=PURITY_TOL)
    n2 = real(inner(psi, psi))
    abs(n2 - 1) <= atol || throw(ArgumentError(
        "the S(rho^full)=0 shortcut needs a normalised MPS; <psi|psi> = $n2"))
    return true
end

"""
    total_correlation_support(psi, S=nothing) -> Float64

Total correlation `T_S = sum_{a in S} S(rho^a) - S(rho^S)` in bits for a subset
`S` of sites. `S === nothing` or the full register uses the pure-state shortcut
of [`register_total_correlation`](@ref); otherwise the `|S|`-site marginal is
formed densely, which costs `d^{2|S|}` memory and is intended for `|S| <= 6`.
"""
function total_correlation_support(psi::MPS, S=nothing)
    S === nothing && return register_total_correlation(psi)
    support = sort!(unique(collect(Int, S)))
    length(support) == length(psi) && support == collect(1:length(psi)) &&
        return register_total_correlation(psi)
    rho = dense_reduced_density_matrix(psi, support)
    return spectrum_total_correlation(rho, length(support))
end

"`T_S` from a dense `d^n x d^n` marginal alone."
function spectrum_total_correlation(rho_S::AbstractMatrix, n_sites::Int; d::Int=2)
    total = 0.0
    for a in 1:n_sites
        total += von_neumann_entropy(marginal(rho_S, n_sites, a; d=d))
    end
    return total - spectrum_entropy(hermitian_spectrum(rho_S))
end

"""
    lambda_certificate(psi) -> Float64

Certificate quantity `Lambda = sum_a (1 - lambda_a)` in the notation of the certified
infidelity bound, where `lambda_a` is the largest eigenvalue of the single-site marginal
`rho^a`. It is built from the same single-site marginals as `T` and satisfies
`Lambda <= T/2`: for `lambda_a >= 1/2`, binning the non-maximal eigenvalues gives
`S(rho^a) >= h2(1 - lambda_a) >= 2 (1 - lambda_a)`, and for `lambda_a < 1/2`
(possible only for `d > 2`) `S(rho^a) >= -log2(lambda_a) >= 2 (1 - lambda_a)`.
Summing over `a` gives `2 Lambda <= T`. The infidelity bounds that follow from
`Lambda` are [`certified_infidelity_bound`](@ref) and
[`infidelity_upper_bound`](@ref).
"""
function lambda_certificate(psi::MPS)
    N = length(psi)
    N == 0 && return 0.0
    phi = orthogonalize(psi, 1)
    s = siteinds(phi)
    total = 0.0
    for i in 1:N
        i > 1 && orthogonalize!(phi, i)
        A = phi[i]
        rho = A * dag(prime(A, s[i]))
        M = Matrix(Array(rho, prime(s[i]), s[i]))
        lambda_max = maximum(hermitian_spectrum(M))
        total += 1 - clamp(lambda_max, 0.0, 1.0)
    end
    return total
end

"""
    infidelity_upper_bound(state_or_lambda) -> Float64

The bound `1 - sqrt(1 - Lambda)` on the product-frame infidelity `eps_*`
for `Lambda <= 1`, and `1.0` otherwise. It is weaker than the
certified bound [`certified_infidelity_bound`](@ref), because
`tildeLambda <= Lambda`; it is the bound used by the stopping rule of the greedy
selector. Accepts an MPS or a value of `Lambda`.
"""
infidelity_upper_bound(Lambda::Real) =
    Lambda <= 1 ? 1 - sqrt(1 - Float64(Lambda)) : 1.0
infidelity_upper_bound(psi::MPS) = infidelity_upper_bound(lambda_certificate(psi))

"""
    certified_infidelity_bound(state_or_lambda) -> Float64

Certified infidelity bound: with `tildeLambda = Lambda / (2 - Lambda)`,

    eps_* <= 1 - sqrt(1 - tildeLambda) <= tildeLambda <= Lambda <= T/2,

valid for `Lambda <= 1`; the function returns `1.0` when `Lambda > 1`. For small
`Lambda` the bound is `Lambda/4 + O(Lambda^2)`. Accepts an MPS or a value of
`Lambda`, and is never larger than [`infidelity_upper_bound`](@ref).
"""
function certified_infidelity_bound(Lambda::Real)
    Lambda <= 1 || return 1.0
    tilde = Float64(Lambda) / (2 - Float64(Lambda))
    return 1 - sqrt(1 - tilde)
end
certified_infidelity_bound(psi::MPS) = certified_infidelity_bound(lambda_certificate(psi))


"""
    reduced_density_matrix(psi, support) -> ITensor

Marginal of an MPS on an arbitrary set of sites, as an ITensor with open indices
`s[a]` and `s[a]'` for every `a` in `support`. The MPS is canonicalised at the
leftmost site of the support and contracted to the rightmost, leaving the
physical index open on support sites and tracing it elsewhere. Cost
`O((max(S) - min(S)) chi^3 d^{|S|})`.
"""
function reduced_density_matrix(psi::MPS, support)
    sites_kept = sort!(unique(collect(Int, support)))
    isempty(sites_kept) && throw(ArgumentError("support must be non-empty"))
    N = length(psi)
    (1 <= sites_kept[1] && sites_kept[end] <= N) ||
        throw(ArgumentError("support $sites_kept is outside 1:$N"))

    i, j = sites_kept[1], sites_kept[end]
    phi = orthogonalize(psi, i)
    s = siteinds(phi)
    keep = Set(sites_kept)
    internal_links = [linkind(phi, k) for k in i:(j - 1)]

    bra_of = function (k)
        A = dag(phi[k])
        for l in internal_links
            hasind(phi[k], l) && (A = prime(A, l))
        end
        k in keep && (A = prime(A, s[k]))
        return A
    end

    rho = phi[i] * bra_of(i)
    for k in (i + 1):j
        rho *= phi[k]
        rho *= bra_of(k)
    end
    return rho
end

"""
    dense_reduced_density_matrix(psi, support) -> Matrix{ComplexF64}

[`reduced_density_matrix`](@ref) flattened to a `d^k x d^k` matrix with the first
site of the support the most significant index.
"""
function dense_reduced_density_matrix(psi::MPS, support)
    sites_kept = sort!(unique(collect(Int, support)))
    length(sites_kept) == 2 && return Matrix{ComplexF64}(dense_two_site_matrix(
        two_site_rdm(psi, sites_kept[1], sites_kept[2]), siteinds(psi),
        sites_kept[1], sites_kept[2]))
    rho = reduced_density_matrix(psi, sites_kept)
    s = siteinds(psi)
    ket = [s[a] for a in sites_kept]
    bra = [prime(s[a]) for a in sites_kept]
    # Array(..., ket..., bra...) yields A[k1,...,kn, b1,...,bn]; reshaping a
    # column-major array to (rows, cols) needs the row index varying fastest, so
    # the ket block is reversed to make site 1 the most significant.
    arr = Array(rho, reverse(ket)..., reverse(bra)...)
    dtot = prod(dim.(ket))
    return Matrix{ComplexF64}(reshape(arr, dtot, dtot))
end

"""
    marginal(rho, n_sites, keep; d=2) -> Matrix{ComplexF64}

Partial trace of a dense `d^n x d^n` operator onto the sites in `keep` (an `Int`
or an iterable), with site 1 the most significant index.
"""
function marginal(rho::AbstractMatrix, n_sites::Int, keep; d::Int=2)
    kept = sort!(unique(collect(Int, keep isa Integer ? [keep] : keep)))
    all(1 .<= kept .<= n_sites) ||
        throw(ArgumentError("keep=$kept is outside 1:$n_sites"))
    size(rho, 1) == d^n_sites ||
        throw(ArgumentError("expected a $(d^n_sites)-dimensional operator"))
    length(kept) == n_sites && return Matrix{ComplexF64}(rho)

    # Column-major reshape of a (d^n, d^n) matrix whose site 1 is most
    # significant puts site n on axis 1: ket axis of site a is n - a + 1, bra
    # axis is that plus n. Permute the kept axes to the front (reversed, so the
    # first kept site stays most significant after flattening), pair up the
    # traced axes at the back, and sum the diagonal.
    A = reshape(Matrix{ComplexF64}(rho), ntuple(_ -> d, 2 * n_sites)...)
    ket(a) = n_sites - a + 1
    bra(a) = 2 * n_sites - a + 1
    traced = collect(setdiff(1:n_sites, kept))
    perm = vcat([ket(a) for a in reverse(kept)], [bra(a) for a in reverse(kept)],
        [ket(a) for a in traced], [bra(a) for a in traced])
    P = permutedims(A, perm)
    m = d^length(kept)
    t = d^length(traced)
    Q = reshape(P, m, m, t, t)
    out = zeros(ComplexF64, m, m)
    for k in 1:t
        @views out .+= Q[:, :, k, k]
    end
    return out
end

"Sum of the single-site von Neumann entropies of a dense `n`-site operator, in bits."
function marginal_entropy_sum(rho::AbstractMatrix, n_sites::Int; d::Int=2)
    total = 0.0
    for a in 1:n_sites
        total += von_neumann_entropy(marginal(rho, n_sites, a; d=d))
    end
    return total
end

"""
    removable_correlation_pair(rho_ij; restarts=8, tol=1e-10, rng) -> (R_ij, U)

Removable QMI `R_ij` of a two-qubit marginal, in bits, together with
the `4x4` unitary attaining it.

The optimal gate is the rotation of the eigenbasis of `rho_ij`,
ordered by decreasing eigenvalue `p1 >= p2 >= p3 >= p4`, onto
`|00>, |01>, |10>, |11>`. It leaves marginals with spectra `{p1+p2, p3+p4}` and
`{p1+p3, p2+p4}`, and `R_ij = I(i:j)` exactly when `p1 p4 = p2 p3`; this
saturation case returns the eigenbasis gate directly. Otherwise the eigenbasis
gate seeds a descent of `S(rho'^i) + S(rho'^j)` over `U(4)`, with `restarts`
additional Haar-random starts, and the best gate found is returned. The result
is checked against `0 <= R_ij <= I(i:j)`.
"""
function removable_correlation_pair(rho_ij::AbstractMatrix;
        restarts::Int=8, tol::Real=1e-10, rng::AbstractRNG=MersenneTwister(0))
    size(rho_ij) == (4, 4) ||
        throw(ArgumentError("expected a 4x4 two-qubit density matrix"))
    rho = Matrix{ComplexF64}(Hermitian((rho_ij + rho_ij') / 2))
    F = eigen(Hermitian(rho))
    order = sortperm(real.(F.values); rev=true)
    p = clamp.(real.(F.values[order]), 0.0, 1.0)
    U_eig = Matrix{ComplexF64}(F.vectors[:, order]')

    base = marginal_entropy_sum(rho, 2)
    qmi = max(0.0, real(quantum_mutual_information(rho)))

    if abs(p[1] * p[4] - p[2] * p[3]) < tol
        # Saturation: the eigenbasis rotation removes the whole mutual information.
        R = base - marginal_entropy_sum(U_eig * rho * U_eig', 2)
        _assert_pair_bounds(R, qmi)
        return (max(0.0, min(R, qmi)), U_eig)
    end

    best_U = U_eig
    best_value = marginal_entropy_sum(U_eig * rho * U_eig', 2)
    for start in 0:restarts
        U0 = start == 0 ? U_eig : haar_unitary(4, rng)
        U, value = _minimise_marginal_entropy(rho, 2, U0)
        if value < best_value - 1e-14
            best_value = value
            best_U = U
        end
    end
    R = base - best_value
    _assert_pair_bounds(R, qmi)
    return (max(0.0, min(R, qmi)), best_U)
end

"Weight operator commutator `norm([rho_ij, n_i + n_j], Inf)` on the basis `|00>, |01>, |10>, |11>`."
function _weight_commutator_defect(rho_ij::AbstractMatrix)
    N = Diagonal(ComplexF64[0, 1, 1, 2])
    return opnorm(rho_ij * N - N * rho_ij, Inf)
end

"""
    removable_correlation_symmetric(rho_ij; tol=1e-10, restarts=3, rng) -> (R_sym, U, exact)

Removable QMI of a two-qubit marginal under weight-preserving (`Sz`-conserving)
gates only, in bits, with the unitary attaining it. Such a gate is block
diagonal in total weight: a phase on `|00>`, a `U(2)` rotation on the weight-one
block `{|01>, |10>}`, and a phase on `|11>`.

If `rho_ij` commutes with the weight operator to within `tol`, the optimum is
explicit. With `a = <00|rho|00>` and `b1 >= b2` the eigenvalues of the
weight-one block, the rotation diagonalising that block with the larger
eigenvalue on `|01>` gives

    R_sym = S(rho_i) + S(rho_j) - h2(a + b1) - h2(a + b2),

with `h2` the binary entropy in bits, and `exact = true`. Otherwise the
objective is minimised numerically over the six real parameters of the
weight-preserving group, starting from that rotation and `restarts` random
members of the group, and `exact = false`. Always `R_sym <= R_ij`, since the
weight-preserving gates form a subgroup of `U(4)`.
"""
function removable_correlation_symmetric(rho_ij::AbstractMatrix;
        tol::Real=1e-10, restarts::Int=3, rng::AbstractRNG=MersenneTwister(0))
    size(rho_ij) == (4, 4) ||
        throw(ArgumentError("expected a 4x4 two-qubit density matrix"))
    rho = Matrix{ComplexF64}(Hermitian((rho_ij + rho_ij') / 2))
    base = marginal_entropy_sum(rho, 2)
    qmi = max(0.0, real(quantum_mutual_information(rho)))
    defect = _weight_commutator_defect(rho)

    if defect <= tol
        a = real(rho[1, 1])
        block = Matrix{ComplexF64}(Hermitian(rho[2:3, 2:3]))
        b = sort(clamp.(real.(eigvals(Hermitian(block))), 0.0, 1.0); rev=true)
        U = _weight_block_givens(rho)
        R = base - (_binary_entropy(a + b[1]) + _binary_entropy(a + b[2]))
        _assert_pair_bounds(R, qmi)
        return (max(0.0, min(R, qmi)), U, true)
    end

    # Seed with the rotation that diagonalises the weight-one block; off the
    # symmetric sector it is not optimal but lies in the right neighbourhood.
    givens = _weight_block_givens(rho)
    best_U = givens
    best_value = _pair_marginal_entropy_sum(givens * rho * givens')
    for start in 0:restarts
        U0 = start == 0 ? givens :
            _weight_preserving_unitary(haar_unitary(2, rng), rng)
        U, value = _minimise_marginal_entropy_symmetric(rho, U0)
        if value < best_value - 1e-14
            best_value = value
            best_U = U
        end
    end
    R = base - best_value
    _assert_pair_bounds(R, qmi)
    return (max(0.0, min(R, qmi)), best_U, false)
end

"""
    _pair_marginal_entropy_sum(rho) -> Float64

`S(rho_i) + S(rho_j)` for a `4x4` two-qubit operator, written out for the
weight-preserving search, which evaluates it many times per pair. The
eigenvalues of a Hermitian `[p c; c' q]` are `(p+q)/2 +- sqrt(((p-q)/2)^2 + |c|^2)`.
"""
function _pair_marginal_entropy_sum(rho::AbstractMatrix)
    # rho_i: trace out the second qubit; rho_j: trace out the first.
    pi0 = real(rho[1, 1] + rho[2, 2]); pi1 = real(rho[3, 3] + rho[4, 4])
    ci = rho[1, 3] + rho[2, 4]
    pj0 = real(rho[1, 1] + rho[3, 3]); pj1 = real(rho[2, 2] + rho[4, 4])
    cj = rho[1, 2] + rho[3, 4]
    return _qubit_entropy(pi0, pi1, ci) + _qubit_entropy(pj0, pj1, cj)
end

"Von Neumann entropy in bits of the Hermitian 2x2 `[p c; c' q]`."
function _qubit_entropy(p::Real, q::Real, c::Number)
    half = (p + q) / 2
    radius = sqrt(max(0.0, ((p - q) / 2)^2 + abs2(c)))
    return _spectrum_entropy2(half + radius) + _spectrum_entropy2(half - radius)
end

"`-lambda log2 lambda`, clamped, with the `0 log 0 = 0` endpoint."
function _spectrum_entropy2(lambda::Real)
    l = clamp(Float64(lambda), 0.0, 1.0)
    return l <= 0 ? 0.0 : -l * log2(l)
end

"Binary entropy in bits, with the `0 log 0 = 0` endpoints handled."
function _binary_entropy(p::Real)
    q = clamp(Float64(p), 0.0, 1.0)
    (q <= 0 || q >= 1) && return 0.0
    return -(q * log2(q) + (1 - q) * log2(1 - q))
end

"""
    _weight_block_givens(rho) -> Matrix{ComplexF64}

Weight-preserving gate that diagonalises the weight-one block of `rho`, with the
larger eigenvalue assigned to `|01>`. This assignment is optimal among all
weight-preserving gates: the achievable weight-one diagonal entries at fixed sum
lie between the two block eigenvalues (Schur-Horn), and a sum of two concave
binary entropies on that interval is minimised at an endpoint.
"""
function _weight_block_givens(rho::AbstractMatrix)
    F = eigen(Hermitian(Matrix{ComplexF64}(rho[2:3, 2:3])))
    order = sortperm(real.(F.values); rev=true)
    U = Matrix{ComplexF64}(I, 4, 4)
    U[2:3, 2:3] = Matrix{ComplexF64}(F.vectors[:, order]')
    return U
end

"Embed a `U(2)` weight-one block and two phases as a `4x4` weight-preserving gate."
function _weight_preserving_unitary(V::AbstractMatrix, rng::AbstractRNG)
    U = Matrix{ComplexF64}(I, 4, 4)
    U[1, 1] = cis(2pi * rand(rng))
    U[4, 4] = cis(2pi * rand(rng))
    U[2:3, 2:3] = V
    return U
end

"""
    _minimise_marginal_entropy_symmetric(rho, U0) -> (U, value)

Minimise `S(rho'^i) + S(rho'^j)` over the weight-preserving group by coordinate
search on its six real parameters, starting from `U0`.
"""
function _minimise_marginal_entropy_symmetric(rho::AbstractMatrix,
        U0::AbstractMatrix; iterations::Int=150, tol::Real=1e-13)
    params = _weight_preserving_parameters(U0)
    function objective(x)
        U = _weight_preserving_from_parameters(x)
        return _pair_marginal_entropy_sum(U * rho * U')
    end
    value = objective(params)
    step = 0.4
    for _ in 1:iterations
        improved = false
        for k in eachindex(params)
            for sign in (1, -1)
                trial = copy(params)
                trial[k] += sign * step
                trial_value = objective(trial)
                if trial_value < value - tol
                    params = trial
                    value = trial_value
                    improved = true
                    break
                end
            end
        end
        improved || (step /= 2)
        step < 1e-10 && break
    end
    return (_weight_preserving_from_parameters(params), value)
end

"""
    _weight_preserving_from_parameters(x) -> Matrix{ComplexF64}

Weight-preserving gate from six real parameters: the outer phases `phi0, phi3`,
and the `U(2)` block `exp(i chi)` times an `SU(2)` element with angles
`theta, alpha, beta`.
"""
function _weight_preserving_from_parameters(x::AbstractVector)
    phi0, phi3, chi, theta, alpha, beta = x
    V = cis(chi) * ComplexF64[
        cos(theta)*cis(alpha)   sin(theta)*cis(beta);
        -sin(theta)*cis(-beta)  cos(theta)*cis(-alpha)]
    U = Matrix{ComplexF64}(I, 4, 4)
    U[1, 1] = cis(phi0)
    U[4, 4] = cis(phi3)
    U[2:3, 2:3] = V
    return U
end

"Parameters `(phi0, phi3, chi, theta, alpha, beta)` of a weight-preserving `U0`, used to seed the search."
function _weight_preserving_parameters(U0::AbstractMatrix)
    phi0 = angle(U0[1, 1] == 0 ? one(ComplexF64) : U0[1, 1])
    phi3 = angle(U0[4, 4] == 0 ? one(ComplexF64) : U0[4, 4])
    V = U0[2:3, 2:3]
    chi = angle(det(V)) / 2
    W = V / cis(chi)
    theta = atan(abs(W[1, 2]), abs(W[1, 1]))
    alpha = angle(W[1, 1] == 0 ? one(ComplexF64) : W[1, 1])
    beta = angle(W[1, 2] == 0 ? one(ComplexF64) : W[1, 2])
    return Float64[phi0, phi3, chi, theta, alpha, beta]
end

function _assert_pair_bounds(R::Real, qmi::Real)
    (-1e-8 <= R <= qmi + 1e-8) || error(
        "removable correlation $R violates 0 <= R <= I(i:j) = $qmi")
    return nothing
end

"""
    removable_correlation(rho_S, n_sites; d=2, restarts=20, rng) -> (R_S, U)

Removable correlation `R_S` of a dense `n`-site marginal over the full
`U(d^n)`, with the unitary attaining it. For two qubits this is
[`removable_correlation_pair`](@ref). A flat spectrum with `d^m` equal non-zero
entries (a stabiliser marginal) factorises and gives `R_S = T_S` from the
eigenbasis rotation. Otherwise the objective `sum_a S(rho'^a)` is minimised by
Riemannian descent from the eigenbasis rotation and `restarts` Haar starts. The
result is checked against `0 <= R_S <= T_S`.
"""
function removable_correlation(rho_S::AbstractMatrix, n_sites::Int;
        d::Int=2, restarts::Int=20, rng::AbstractRNG=MersenneTwister(0))
    n_sites == 2 && d == 2 &&
        return removable_correlation_pair(rho_S; restarts=min(restarts, 8), rng=rng)
    dim = d^n_sites
    size(rho_S) == (dim, dim) ||
        throw(ArgumentError("expected a $dim x $dim marginal"))
    rho = Matrix{ComplexF64}(Hermitian((rho_S + rho_S') / 2))
    base = marginal_entropy_sum(rho, n_sites; d=d)
    ceiling = spectrum_total_correlation(rho, n_sites; d=d)

    F = eigen(Hermitian(rho))
    if _flat_spectrum(real.(F.values), d)
        order = sortperm(real.(F.values); rev=true)
        return (max(0.0, ceiling), Matrix{ComplexF64}(F.vectors[:, order]'))
    end
    order = sortperm(real.(F.values); rev=true)
    U_eig = Matrix{ComplexF64}(F.vectors[:, order]')

    best_U = Matrix{ComplexF64}(I, dim, dim)
    best_value = base
    for start in 0:restarts
        U0 = start == 0 ? U_eig : haar_unitary(dim, rng)
        U, value = _minimise_marginal_entropy(rho, n_sites, U0; d=d)
        if value < best_value - 1e-14
            best_value = value
            best_U = U
        end
    end
    R = base - best_value
    (-1e-8 <= R <= ceiling + 1e-8) || error(
        "R_S = $R violates 0 <= R_S <= T_S = $ceiling")
    return (max(0.0, min(R, ceiling)), best_U)
end

"Objective `sum_a S(rho'^a)` of the rotated operator `rho' = U rho U'`."
function _objective(rho::AbstractMatrix, n_sites::Int, U::AbstractMatrix; d::Int=2)
    return marginal_entropy_sum(U * rho * U', n_sites; d=d)
end

"log2 of a Hermitian marginal with a floor on its spectrum."
function _log2_floor(sigma::AbstractMatrix)
    F = eigen(Hermitian((sigma + sigma') / 2))
    vals = log2.(max.(real.(F.values), EIG_FLOOR))
    return F.vectors * Diagonal(vals) * F.vectors'
end

"Embed a single-site operator acting on `site` into the full `d^n` space."
function _embed_site(op::AbstractMatrix, n_sites::Int, site::Int; d::Int=2)
    left = d^(site - 1)
    right = d^(n_sites - site)
    return kron(Matrix{ComplexF64}(I, left, left), Matrix{ComplexF64}(op),
        Matrix{ComplexF64}(I, right, right))
end

"Operator `G = sum_a (log2 rho'^a) (x) 1` of the entropy gradient at `rho'`."
function _gradient_operator(rho_rot::AbstractMatrix, n_sites::Int; d::Int=2)
    dim = d^n_sites
    G = zeros(ComplexF64, dim, dim)
    for a in 1:n_sites
        G .+= _embed_site(_log2_floor(marginal(rho_rot, n_sites, a; d=d)),
            n_sites, a; d=d)
    end
    return G
end

"""
    _minimise_marginal_entropy(rho, n_sites, U0; d=2, iterations=200, tol=1e-12) -> (U, value)

Riemannian descent on `U(d^n)` of `sum_a S(rho'^a)`, `rho' = U rho U'`, from
`U0`. Since `df = -2 Re Tr[dU' G U rho]`, the Euclidean gradient is `-G U rho` and
the Riemannian gradient is its skew part; a backtracking line search keeps the
objective monotone.
"""
function _minimise_marginal_entropy(rho::AbstractMatrix, n_sites::Int,
        U0::AbstractMatrix; d::Int=2, iterations::Int=200, tol::Real=1e-12)
    U = Matrix{ComplexF64}(U0)
    value = _objective(rho, n_sites, U; d=d)
    step = 1.0
    for _ in 1:iterations
        rot = U * rho * U'
        G = _gradient_operator(rot, n_sites; d=d)
        E = -G * U * rho                   # Euclidean gradient direction
        A = E * U'
        Omega = A - A'                     # Riemannian (skew) gradient
        gnorm = norm(Omega)
        gnorm < tol && break
        improved = false
        for _ in 1:30
            candidate = exp(-step * Omega) * U
            candidate_value = _objective(rho, n_sites, candidate; d=d)
            if candidate_value < value - 1e-15
                U = candidate
                value = candidate_value
                step *= 1.5
                improved = true
                break
            end
            step /= 2
            step < 1e-14 && break
        end
        improved || break
    end
    return U, value
end

"""
    _flat_spectrum(values, d; tol=1e-9) -> Bool

True when the non-zero entries of `values` are all equal and their number is a
power of `d`, the signature of a stabiliser marginal, whose spectrum factorises
so that `R_S = T_S`.
"""
function _flat_spectrum(values, d::Int; tol::Real=1e-9)
    live = [v for v in values if v > tol]
    isempty(live) && return false
    maximum(live) - minimum(live) <= tol || return false
    rank = length(live)
    power = round(Int, log(d, rank))
    d^power == rank || return false
    return abs(maximum(live) - 1 / rank) <= tol
end

"Haar-random unitary of dimension `n` (internal; `Disentangling` exports its own)."
function haar_unitary(n::Int, rng::AbstractRNG)
    Z = (randn(rng, n, n) + im * randn(rng, n, n)) / sqrt(2)
    Q, R = qr(Z)
    return Matrix{ComplexF64}(Q) * Diagonal(sign.(diag(R)))
end

end # module
