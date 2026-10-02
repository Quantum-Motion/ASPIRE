"""
    Selection

The ASPIRE gate-placement protocol (see the accompanying paper). Each layer scores every admissible pair by its removable quantum mutual information
`R_ij`, optionally per native two-qubit gate, admits the pairs that pass the thresholds
`tau` and `rho`, selects a set of disjoint pairs by maximum-weight matching, applies the
eigenbasis gate of each pair to the working state and repeats. The inverses of the applied gates,
in reverse order, form the preparation circuit.
"""
module Selection

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

using ..SimConfig: global_cutoff, global_maxdim
using ..FidelityMetrics:
    squared_fidelity, amplitude_fidelity_from_squared,
    amplitude_infidelity_from_squared
using ..RDM:
    two_site_rdm, all_pairs_two_site_rdms, dense_two_site_matrix, partial_trace_two_qubit,
    von_neumann_entropy, quantum_mutual_information
using ..Disentangling:
    ASPIRE_DEFAULT_THRESHOLD, ASPIRE_DEFAULT_RELATIVE_THRESHOLD,
    DisentanglingGate, candidate_pairs, apply_two_qubit_unitary_tracked,
    total_entanglement, total_correlation, aspire_rank_threshold, mincut_batch
using ..Matching: maximum_weight_matching
using ..HardwareCost: graph_distances, graph_routing_tables, gate_cost,
    ROUTING_CONVENTIONS, DEFAULT_ROUTING_CONVENTION, SWAP_GATE_COST,
    RoutingLayout, layout_distance, routing_swaps, routed_layer_cost!
using ..Correlation: lambda_certificate, infidelity_upper_bound

export ASPIREPairFeatures, ASPIREConfig,
    pair_spectrum_features, removable_mutual_information, removable_lambda, eigenbasis_gate,
    pair_native_cost, cost_normalised_transform, select_layer,
    maximum_weight_pair_matching, maximum_weight_cross_qmi_matching,
    zero_product_fidelity, product_frame_fidelity,
    zero_product_amplitude_infidelity, product_frame_amplitude_infidelity,
    aspire_step, aspire_support_step, aspire

const _SCORE_TOL = 1e-12

"""
    ASPIREPairFeatures

Quantities derived from a two-qubit reduced density matrix `rho_ij`:

- `qmi`: the mutual information `I(i:j)`;
- `removable_qmi`: the removable mutual information `R_ij`, the exact decrease of
  `S(rho_i) + S(rho_j)` under the eigenbasis gate;
- `actionability = R_ij / I(i:j)` and `residual_qmi = I(i:j) - R_ij`;
- `purity`: `tr(rho_ij^2)`;
- `removable_lambda`: the decrease of the certificate `Lambda` under the same gate.
"""
struct ASPIREPairFeatures
    qmi::Float64
    removable_qmi::Float64
    actionability::Float64
    residual_qmi::Float64
    purity::Float64
    removable_lambda::Float64
end

"""
    ASPIREConfig(; ...)

Settings of the ASPIRE selector. The symbols of the accompanying paper are given in brackets.

Admission and score:
- `cost_normalised`: score pairs by `R_ij / c_ij`, bits per native gate, instead of `R_ij`;
  `threshold` is then `tau` in bits per native two-qubit gate.
- `threshold` [tau]: absolute score floor (default `1e-3`).
- `relative_threshold` [rho]: a pair must score at least `rho` times the best score of the
  layer (default `0.10`). The paper's runs use `tau = rho = 0` unless stated.
- `max_rank`: keep only the `max_rank` best-scoring pairs.
- `score_mode`: `:qmi` scores by `R_ij`, `:lambda` by the decrease of `Lambda`.

Layer selection:
- `batch_mode`: how the layer is chosen from the admitted pairs. `:matching` is the exact
  maximum-weight matching over the `shortlist` best pairs (branch and bound), `:blossom` the
  exact matching over all admitted pairs (Edmonds' blossom algorithm), `:mincut` the greedy
  rule of descending score with cross-pair coupling at most `couple_threshold`, and
  `:constrained_matching` the exact optimum under the same coupling constraint. `:sequential`
  tries the `shortlist` best pairs one at a time, keeping the best (objective) gate.
- `shortlist` [n_c]: number of highest-scoring admitted pairs handed to the matching (32 in
  the paper).
- `matching_max_edges`: guard for the branch-and-bound matching; an error is raised if more
  positive-weight edges than this are passed in (48 in the paper's runs).
- `max_batch_size` [m]: cap on the number of gates per layer.
- `couple_threshold` [kappa]: cross-pair mutual-information bound of `:mincut` and
  `:constrained_matching` (0.1 bits in the paper).
- `max_depth`, `max_gates`: limits on layers and on logical gates.

Cost model:
- `hardware_graph`: coupling graph (list of edges) against which routing is charged; `nothing`
  means all-to-all connectivity.
- `routing_convention`: `:running` charges each pair at its distance in the live qubit layout
  and keeps the routing permutation (cost `c_ij = 3(1 + s_ij)` CNOTs for `s_ij` SWAPs).
- `pair_gate_cost`: native two-qubit gates of one emitted pair gate (3 for a generic SU(4)).

Acceptance of a proposed gate:
- `accept_only`, `min_objective_gain`, `entanglement_weight`: a gate is applied only if its
  objective `dT/T + entanglement_weight * dE/E` reaches `min_objective_gain` and does not
  increase `T`.
- `noise_rate` [p]: if set, additionally require `Delta Lambda > p c_ij` (noise-aware admission).
- `target_infidelity`: stop once the certificate bound `1 - sqrt(1 - Lambda)` is below it.

Tie handling and diagnostics:
- `tie_break_seed`: seed that randomises the choice among exactly tied matchings.
- `matching_weight_tolerance`: weights closer than this count as tied.
- `score_jitter`: relative noise added to the scores (requires `tie_break_seed`).
- `track_truncation`: record the weight discarded by the truncation of each applied gate.
"""
Base.@kwdef struct ASPIREConfig
    threshold::Float64 = ASPIRE_DEFAULT_THRESHOLD
    relative_threshold::Float64 = ASPIRE_DEFAULT_RELATIVE_THRESHOLD
    couple_threshold::Float64 = 0.1
    max_depth::Int = 20
    max_gates::Int = typemax(Int)
    max_rank::Union{Nothing,Int} = nothing
    shortlist::Int = 32
    max_batch_size::Int = typemax(Int)
    batch_mode::Symbol = :matching       # :matching, :blossom, :mincut, :constrained_matching, :sequential
    cost_normalised::Bool = true
    hardware_graph::Union{Nothing,Vector{Tuple{Int,Int}}} = nothing
    routing_convention::Symbol = DEFAULT_ROUTING_CONVENTION
    pair_gate_cost::Int = 3
    matching_max_edges::Int = 48
    entanglement_weight::Float64 = 0.25
    min_objective_gain::Float64 = 1e-10
    accept_only::Bool = true
    tie_break_seed::Union{Nothing,Int} = nothing
    matching_weight_tolerance::Float64 = _SCORE_TOL
    track_truncation::Bool = false
    score_mode::Symbol = :qmi            # :qmi, :lambda
    target_infidelity::Union{Nothing,Float64} = nothing
    noise_rate::Union{Nothing,Float64} = nothing
    score_jitter::Float64 = 0.0
end

function _validate_config(config::ASPIREConfig)
    config.threshold >= 0 || throw(ArgumentError("threshold must be non-negative."))
    config.relative_threshold >= 0 || throw(ArgumentError("relative_threshold must be non-negative."))
    config.couple_threshold >= 0 || throw(ArgumentError("couple_threshold must be non-negative."))
    config.max_depth >= 0 || throw(ArgumentError("max_depth must be non-negative."))
    config.max_gates >= 0 || throw(ArgumentError("max_gates must be non-negative."))
    (config.max_rank === nothing || config.max_rank >= 1) ||
        throw(ArgumentError("max_rank must be nothing or at least one."))
    config.shortlist >= 1 || throw(ArgumentError("shortlist must be at least one."))
    config.max_batch_size >= 1 || throw(ArgumentError("max_batch_size must be at least one."))
    config.batch_mode in (:blossom, :mincut, :matching, :constrained_matching, :sequential) ||
        throw(ArgumentError(
            "batch_mode must be :matching, :blossom, :mincut, :constrained_matching, " *
            "or :sequential."))
    config.pair_gate_cost >= 1 ||
        throw(ArgumentError("pair_gate_cost must be at least one."))
    haskey(ROUTING_CONVENTIONS, config.routing_convention) ||
        throw(ArgumentError("unknown routing convention `$(config.routing_convention)`."))
    config.matching_max_edges >= 1 ||
        throw(ArgumentError("matching_max_edges must be at least one."))
    config.score_mode in (:qmi, :lambda) ||
        throw(ArgumentError("score_mode must be :qmi or :lambda."))
    (config.target_infidelity === nothing || config.target_infidelity > 0) ||
        throw(ArgumentError("target_infidelity must be nothing or positive."))
    config.matching_weight_tolerance >= 0 ||
        throw(ArgumentError("matching_weight_tolerance must be non-negative."))
    (config.noise_rate === nothing || config.noise_rate >= 0) ||
        throw(ArgumentError("noise_rate must be nothing or non-negative."))
    config.score_jitter >= 0 ||
        throw(ArgumentError("score_jitter must be non-negative."))
    (config.score_jitter == 0 || config.tie_break_seed !== nothing) ||
        throw(ArgumentError(
            "score_jitter requires tie_break_seed so draws are reproducible."))
    config.entanglement_weight >= 0 ||
        throw(ArgumentError("entanglement_weight must be non-negative."))
    return config
end

# Fix the global phase of a unit vector by anchoring on the lowest-index component whose
# modulus is within `rtol` of the largest. An anchor chosen by an absolute threshold would
# jump under machine-precision perturbations, and exact ties are common (a Bell state has four
# components of equal modulus).
function _fix_phase(v::AbstractVector; rtol::Float64=1e-10)
    a = abs.(v)
    m = maximum(a)
    m <= 1e-14 && return Vector{ComplexF64}(v)
    k = findfirst(x -> x >= (1 - rtol) * m, a)
    return Vector{ComplexF64}(v .* (a[k] / v[k]))
end

# Eigenbasis of a two-qubit density matrix as a function of its eigenspaces rather than of the
# eigenvectors returned by LAPACK.
#
# The score `R_ij` depends only on the spectrum, but the eigenbasis inside a degenerate block is
# arbitrary, and symmetric targets (the rung singlets of a Heisenberg ladder) have exactly
# degenerate pair marginals. Eigenvalues within relative tolerance `tol` form a block. A block
# of size one fixes its eigenvector up to a phase, which `_fix_phase` fixes. For a block of size
# k > 1 the returned vectors are discarded: the eigenprojector P = V_block V_block' is applied
# to the computational basis vectors in the order |00>, |01>, |10>, |11>, the images are
# Gram-Schmidt orthonormalised, and the first k independent ones are kept with phases fixed as
# above. The gate is then a function of the eigenspace alone, so runs agree across processes
# and BLAS versions. Every basis of a degenerate block is an equally optimal disentangler.
function _canonical_eigenbasis(V::AbstractMatrix, p::AbstractVector;
                               tol::Float64=1e-9)
    W = Matrix{ComplexF64}(V)
    dim = size(W, 1)
    start = 1
    while start <= length(p)
        stop = start
        while stop < length(p) &&
              abs(p[stop + 1] - p[start]) <= tol * max(1.0, abs(p[start]))
            stop += 1
        end
        if stop == start
            W[:, start] = _fix_phase(W[:, start])
        else
            block = W[:, start:stop]
            proj = block * block'
            cols = Vector{Vector{ComplexF64}}()
            for e in 1:dim
                u = Vector{ComplexF64}(proj[:, e])
                for c in cols
                    u -= dot(c, u) * c
                end
                nrm = norm(u)
                if nrm > 1e-8
                    push!(cols, _fix_phase(u ./ nrm))
                end
                length(cols) == stop - start + 1 && break
            end
            if length(cols) == stop - start + 1
                for (k, c) in enumerate(cols)
                    W[:, start + k - 1] = c
                end
            end
        end
        start = stop + 1
    end
    return W
end

"""
    eigenbasis_gate(rho) -> Matrix{ComplexF64}

Optimal gate `U*` for the two-qubit density matrix `rho`: the eigenvectors of `rho`,
ordered by decreasing eigenvalue, are mapped onto `|00>, |01>, |10>, |11>`. The
choice of eigenvectors inside a degenerate eigenvalue block is made canonically.
"""
function eigenbasis_gate(rho::AbstractMatrix)
    F = eigen(Hermitian((rho + rho') / 2))
    order = sortperm(F.values; rev=true)
    p = clamp.(real.(F.values[order]), 0.0, 1.0)
    total = sum(p)
    total > _SCORE_TOL && (p = p ./ total)
    V = _canonical_eigenbasis(F.vectors[:, order], p)
    return Matrix{ComplexF64}(V')
end

function _single_site_entropy_sum(rho::AbstractMatrix)
    return von_neumann_entropy(partial_trace_two_qubit(rho, 1)) +
           von_neumann_entropy(partial_trace_two_qubit(rho, 2))
end

# sum_a (1 - lambda_max(rho^a)) over the two sites of a two-qubit density matrix: the
# contribution of the pair to the certificate quantity Lambda.
function _single_site_lambda_sum(rho::AbstractMatrix)
    total = 0.0
    for site in 1:2
        sigma = partial_trace_two_qubit(rho, site)
        vals = real.(eigvals(Hermitian((sigma + sigma') / 2)))
        total += 1 - clamp(maximum(vals), 0.0, 1.0)
    end
    return total
end

"""
    removable_lambda(rho) -> Float64

Decrease of the certificate quantity `Lambda = sum_a (1 - lambda_max(rho^a))` in the
two affected sites under the eigenbasis gate of the two-qubit density matrix `rho`, the same gate
that [`removable_mutual_information`](@ref) scores.
"""
function removable_lambda(rho::AbstractMatrix)
    U = eigenbasis_gate(rho)
    rotated = U * rho * U'
    return max(0.0, _single_site_lambda_sum(rho) - _single_site_lambda_sum(rotated))
end

"""
    removable_mutual_information(rho) -> Float64

Removable quantum mutual information `R_ij` of the two-qubit density matrix `rho`:
the decrease of `S(rho_i) + S(rho_j)` under the eigenbasis gate `U*`. A two-qubit
unitary leaves every other single-site marginal unchanged, so this is also the exact decrease
of the total correlation `T` of the full state under that gate. In terms of the eigenvalues
`p_1 >= ... >= p_4` of `rho`, `R_ij = h(mu_i) + h(mu_j) - h(p_2 + p_4) - h(p_3 + p_4)`
(the two-qubit closed form), with `h` the binary entropy and `mu_a = 1 - lambda_max(rho^a)`.
"""
function removable_mutual_information(rho::AbstractMatrix)
    U = eigenbasis_gate(rho)
    rotated = U * rho * U'
    return max(0.0, _single_site_entropy_sum(rho) - _single_site_entropy_sum(rotated))
end

"""
    pair_spectrum_features(rho) -> ASPIREPairFeatures

Selection features of a dense two-site reduced density matrix.
"""
function pair_spectrum_features(rho::AbstractMatrix)
    size(rho) == (4, 4) || throw(ArgumentError("Expected a 4x4 two-qubit density matrix."))
    qmi = max(0.0, real(quantum_mutual_information(rho)))
    removable = removable_mutual_information(rho)
    actionability = qmi <= _SCORE_TOL ? 0.0 : clamp(removable / qmi, 0.0, 1.0)
    purity = clamp(real(tr(rho * rho)), 0.0, 1.0)
    return ASPIREPairFeatures(qmi, removable, actionability, max(0.0, qmi - removable),
        purity, removable_lambda(rho))
end

"""
    zero_product_fidelity(psi) -> Float64

Squared overlap of `psi` with the all-up product state `|0...0>`.
"""
function zero_product_fidelity(psi::MPS)
    ref = MPS(siteinds(psi), fill("Up", length(psi)))
    return squared_fidelity(psi, ref)
end

"Amplitude infidelity `1 - f` against `|0...0>`."
zero_product_amplitude_infidelity(psi::MPS) =
    amplitude_infidelity_from_squared(zero_product_fidelity(psi))

function _product_approximation(psi::MPS)
    product = copy(psi)
    truncate!(product; maxdim=1, cutoff=0.0)
    return normalize!(product)
end

"""
    product_frame_fidelity(psi) -> Float64

Squared overlap of `psi` with its bond-dimension-one truncation. Rotating the truncated product
state onto `|0...0>` is the free single-qubit layer of the circuit, so this is the fidelity
credited with a product frame. The truncation is a particular product state; the best product
state, the maximiser over product states, has overlap at least as large, so
`product_frame_amplitude_infidelity` is an upper bound on the infidelity `epsilon_*` of
the optimal product frame.
"""
function product_frame_fidelity(psi::MPS)
    product = _product_approximation(psi)
    return squared_fidelity(psi, product)
end

"Amplitude infidelity `1 - f` against the bond-dimension-one truncation of `psi`."
product_frame_amplitude_infidelity(psi::MPS) =
    amplitude_infidelity_from_squared(product_frame_fidelity(psi))

# Reduced density matrices and selection features of all candidate pairs, grouped by left site
# so that each distinct i costs one canonicalisation.
function _pair_data(psi::MPS, pairs)
    s = siteinds(psi)
    rhos = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    features = Dict{Tuple{Int,Int},ASPIREPairFeatures}()
    for (pair, rho_tensor) in all_pairs_two_site_rdms(psi, pairs)
        i, j = pair
        rho = Matrix{ComplexF64}(dense_two_site_matrix(rho_tensor, s, i, j))
        rhos[pair] = rho
        features[pair] = pair_spectrum_features(rho)
    end
    return rhos, features
end

# Memoised pair mutual information I(a:b), seeded with the already computed candidate values.
function _coupling_function(psi::MPS, seed::Dict{Tuple{Int,Int},Float64})
    s = siteinds(psi)
    cache = copy(seed)
    return function (a, b)
        i, j = minmax(a, b)
        get!(cache, (i, j)) do
            rho = dense_two_site_matrix(two_site_rdm(psi, i, j), s, i, j)
            max(0.0, real(quantum_mutual_information(rho)))
        end
    end
end

function _lexicographically_less(left, right)
    for (a, b) in zip(left, right)
        a == b || return a < b
    end
    return length(left) < length(right)
end

# Positive-weight edges of the candidate set, canonicalised and sorted by descending weight.
function _weighted_edges(pairs, weight_of::AbstractDict, max_edges::Int)
    max_edges >= 1 || throw(ArgumentError("max_edges must be at least one."))

    seen = Set{Tuple{Int,Int}}()
    edges = NamedTuple{(:pair, :weight),Tuple{Tuple{Int,Int},Float64}}[]
    for raw_pair in pairs
        i, j = raw_pair
        i == j && throw(ArgumentError("matching pairs must join distinct sites"))
        pair = minmax(Int(i), Int(j))
        pair in seen && continue
        push!(seen, pair)
        raw_weight = if haskey(weight_of, raw_pair)
            weight_of[raw_pair]
        elseif haskey(weight_of, pair)
            weight_of[pair]
        else
            throw(KeyError(raw_pair))
        end
        weight = Float64(raw_weight)
        isfinite(weight) ||
            throw(DomainError(raw_weight, "matching weights must be finite"))
        weight > _SCORE_TOL && push!(edges, (pair=pair, weight=weight))
    end
    length(edges) <= max_edges || throw(ArgumentError(
        "exact matching received $(length(edges)) positive edges, exceeding " *
        "max_edges=$max_edges; reduce `shortlist` or raise `matching_max_edges` explicitly"))
    sort!(edges; by=edge -> (-edge.weight, edge.pair[1], edge.pair[2]))
    return edges
end

# Branch and bound over subsets of the weighted edges. `interaction(e, f)` is `Inf` for an
# incompatible pair of edges and otherwise a non-negative structural cost that is minimised
# among subsets of equal total weight.
#
# Subsets whose weights agree to within `weight_tolerance` are tied and are ordered by the
# interaction cost, then by `tie_order`: the lexicographic order of the pairs by default, or a
# random priority when `rng` is given, which samples uniformly among the tied optima. Symmetry
# related bonds carry equal scores in exact arithmetic, but their computed values differ by the
# truncation error, so a tolerance at the truncation scale is needed to treat them as tied.
# Only the comparison changes; the edges stay in descending-weight order, which the upper bound
# of the search relies on.
function _maximum_weight_compatible_pairs(pairs, weight_of::AbstractDict,
        interaction::Function; max_pairs::Int=typemax(Int), max_edges::Int=24,
        rng::Union{Nothing,AbstractRNG}=nothing,
        weight_tolerance::Float64=_SCORE_TOL)
    max_pairs >= 0 || throw(ArgumentError("max_pairs must be non-negative."))
    weight_tolerance >= 0 ||
        throw(ArgumentError("weight_tolerance must be non-negative."))
    max_pairs == 0 && return Tuple{Int,Int}[]
    edges = _weighted_edges(pairs, weight_of, max_edges)
    isempty(edges) && return Tuple{Int,Int}[]

    tie_rank = rng === nothing ? nothing :
        Dict(edge.pair => rank
             for (edge, rank) in zip(edges, randperm(rng, length(edges))))
    tie_order(candidate) = tie_rank === nothing ? candidate :
        sort!([tie_rank[pair] for pair in candidate])

    pair_interaction = fill(0.0, length(edges), length(edges))
    for i in eachindex(edges), j in 1:(i - 1)
        value = Float64(interaction(edges[i].pair, edges[j].pair))
        isnan(value) && throw(DomainError(value, "pair interaction must not be NaN"))
        value >= 0 ||
            throw(DomainError(value, "pair interaction must be non-negative"))
        pair_interaction[i, j] = pair_interaction[j, i] = value
    end

    best_weight = Ref(-Inf)
    best_interaction = Ref(Inf)
    best_pairs = Ref(Tuple{Int,Int}[])
    selected = Int[]

    function consider!(weight, interaction_cost)
        candidate = sort([edges[i].pair for i in selected])
        if weight > best_weight[] + weight_tolerance ||
                (weight >= best_weight[] - weight_tolerance &&
                 (interaction_cost < best_interaction[] - _SCORE_TOL ||
                  (isapprox(interaction_cost, best_interaction[];
                       atol=_SCORE_TOL, rtol=0) &&
                   _lexicographically_less(tie_order(candidate),
                       tie_order(best_pairs[])))))
            # The reference weight stays at the highest value seen, so that a tied candidate
            # winning on tie-break with marginally lower weight cannot let such margins
            # accumulate into a worse matching.
            best_weight[] = max(best_weight[], weight)
            best_interaction[] = interaction_cost
            best_pairs[] = candidate
        end
    end

    function search!(index::Int, weight::Float64, interaction_cost::Float64)
        slots = max_pairs - length(selected)
        if index > length(edges) || slots == 0
            consider!(weight, interaction_cost)
            return
        end

        # Optimistic remaining weight, ignoring vertex conflicts. The pruning slack equals
        # the comparison tolerance so that tied branches are not cut before their tie-break.
        upper = weight
        remaining_edges = length(edges) - index + 1
        last_index = slots >= remaining_edges ? length(edges) : index + slots - 1
        for k in index:last_index
            upper += edges[k].weight
        end
        upper + weight_tolerance < best_weight[] && return

        added_interaction = 0.0
        compatible = true
        for chosen in selected
            value = pair_interaction[index, chosen]
            if !isfinite(value)
                compatible = false
                break
            end
            added_interaction += value
        end
        if compatible
            push!(selected, index)
            search!(index + 1, weight + edges[index].weight,
                interaction_cost + added_interaction)
            pop!(selected)
        end
        search!(index + 1, weight, interaction_cost)
    end

    search!(1, 0.0, 0.0)
    return best_pairs[]
end

"""
    maximum_weight_pair_matching(pairs, weight_of; max_pairs=typemax(Int), max_edges=24,
        rng=nothing, weight_tolerance=1e-12) -> Vector{Tuple{Int,Int}}

Exact maximum-weight set of vertex-disjoint pairs among the candidate `pairs`, with weights
`weight_of[pair]`, found by branch and bound. Only finite, strictly positive weights enter;
reversed and duplicate pairs are identified. `max_edges` bounds the number of positive edges
accepted, since the search is exponential in the worst case, and `max_pairs` bounds the size of
the matching.

Ties in the total weight (to within `weight_tolerance`) are broken lexicographically by site
label, or, when an `rng` is given, uniformly at random among the tied optima.
"""
function maximum_weight_pair_matching(pairs, weight_of::AbstractDict;
        max_pairs::Int=typemax(Int), max_edges::Int=24,
        rng::Union{Nothing,AbstractRNG}=nothing,
        weight_tolerance::Float64=_SCORE_TOL)
    interaction = function (left, right)
        all(a != b for a in left for b in right) ? 0.0 : Inf
    end
    return _maximum_weight_compatible_pairs(pairs, weight_of, interaction;
        max_pairs=max_pairs, max_edges=max_edges, rng=rng,
        weight_tolerance=weight_tolerance)
end

"""
    maximum_weight_cross_qmi_matching(pairs, weight_of; couple, couple_threshold,
        max_pairs=typemax(Int), max_edges=24, rng=nothing, weight_tolerance=1e-12)

Exact maximum-weight subset of `pairs` such that the pairs are vertex-disjoint and the cross
coupling `couple(a, b)` between the endpoints of any two selected pairs is at most
`couple_threshold`: the constraint of [`mincut_batch`](@ref), optimised exactly instead of
greedily. Among subsets of equal weight the sum of the maximal cross couplings is minimised,
then the tie-break of [`maximum_weight_pair_matching`](@ref) applies.
"""
function maximum_weight_cross_qmi_matching(pairs, weight_of::AbstractDict;
        couple::Function, couple_threshold::Real,
        max_pairs::Int=typemax(Int), max_edges::Int=24,
        rng::Union{Nothing,AbstractRNG}=nothing,
        weight_tolerance::Float64=_SCORE_TOL)
    couple_threshold >= 0 ||
        throw(ArgumentError("couple_threshold must be non-negative."))
    interaction = function (left, right)
        all(a != b for a in left for b in right) || return Inf
        cross_qmi = maximum(couple(a, b) for a in left for b in right)
        isfinite(cross_qmi) ||
            throw(DomainError(cross_qmi, "cross-pair coupling must be finite"))
        cross_qmi >= 0 ||
            throw(DomainError(cross_qmi, "cross-pair coupling must be non-negative"))
        return cross_qmi <= couple_threshold ? cross_qmi : Inf
    end
    return _maximum_weight_compatible_pairs(pairs, weight_of, interaction;
        max_pairs=max_pairs, max_edges=max_edges, rng=rng,
        weight_tolerance=weight_tolerance)
end

"""
    select_layer(candidates) -> Vector{Tuple{Int,Int}}

Layer selection: the maximum-weight set of vertex-disjoint pairs under the weights in
`candidates`, a map `(i, j) => weight`. Disjoint supports contribute additively to the decrease
of `T` with no cross term, so the layer is a maximum-weight matching, found
exactly in `O(N^3)` by Edmonds' blossom algorithm ([`Matching.maximum_weight_matching`](@ref)).
Non-positive weights are dropped.
"""
function select_layer(candidates::AbstractDict)
    return maximum_weight_matching(
        ((p[1], p[2], w) for (p, w) in candidates if w > 0))
end

"""
    pair_native_cost(pair, config; N, distances=nothing, layout=nothing) -> Int

Native two-qubit gate cost `c_ij` of a gate on `pair`, including routing. Without a hardware
graph every pair costs `config.pair_gate_cost`. With one, the pair is charged
`pair_gate_cost + 3 s_ij` under `:running`, with `s_ij = d - 1` the SWAPs needed at the distance
`d` read from the live `layout` (the pristine graph if no layout is given), and the cost of the
chosen convention otherwise.
"""
function pair_native_cost(pair, config::ASPIREConfig; N::Int,
        distances::Union{Nothing,AbstractMatrix}=nothing,
        layout::Union{Nothing,RoutingLayout}=nothing)
    config.hardware_graph === nothing && return config.pair_gate_cost
    D = distances === nothing ?
        graph_distances(N, config.hardware_graph) : distances
    i, j = Int(pair[1]), Int(pair[2])
    running = config.routing_convention === :running
    d = (running && layout !== nothing) ? layout_distance(layout, D, i, j) :
        D[i, j]
    d >= 1 || throw(ArgumentError(
        "pair $pair is unreachable on the hardware graph, so it has no cost"))
    swaps = routing_swaps(d)
    running && return config.pair_gate_cost + SWAP_GATE_COST * swaps
    return config.pair_gate_cost +
           (gate_cost(d; convention=config.routing_convention) - 1)
end

# Shortest-path distances and predecessors of the hardware graph, or `nothing` without one.
_routing_tables(config::ASPIREConfig, N::Int) =
    config.hardware_graph === nothing ? nothing :
    graph_routing_tables(N, config.hardware_graph)

# Carry the routing permutation forward through one layer. Under `:running` the gates are
# charged in emission order, each against the layout its predecessors left behind, and the
# permutation induced by their SWAPs is kept; other conventions leave the layout untouched.
function _advance_layout!(layout::Union{Nothing,RoutingLayout}, routing,
        gates, config::ASPIREConfig)
    (layout === nothing || config.routing_convention !== :running) && return layout
    routed_layer_cost!(layout, routing, ((g.i, g.j) for g in gates);
        gate_cost=config.pair_gate_cost)
    return layout
end

"""
    cost_normalised_transform(config, N; layout=nothing) -> Function or nothing

The map `(pair, score) -> score / c_ij` that turns the selection score into correlation removed
per native gate. `nothing` if `config.cost_normalised` is false. Under `:running`, `layout` is
the live routing permutation that determines `c_ij`.
"""
function cost_normalised_transform(config::ASPIREConfig, N::Int;
        layout::Union{Nothing,RoutingLayout}=nothing)
    config.cost_normalised || return nothing
    D = config.hardware_graph === nothing ? nothing :
        graph_distances(N, config.hardware_graph)
    return (pair, score) ->
        score / pair_native_cost(pair, config; N=N, distances=D, layout=layout)
end

# Scores with relative noise of size `config.score_jitter` added, seeded by (tie_break_seed,
# layer). Symmetry-related bonds have equal scores in exact arithmetic but differ by the
# truncation error, and the matching would otherwise resolve this degeneracy
# deterministically by numerical noise; the jitter samples the near-degenerate choices.
function _jittered_ranks(rank_of::AbstractDict, config::ASPIREConfig, layer::Int)
    config.score_jitter > 0 || return rank_of
    rng = MersenneTwister(hash((config.tie_break_seed, layer, :score_jitter)))
    return Dict(pair => value * (1 + config.score_jitter * (2 * rand(rng) - 1))
        for (pair, value) in rank_of)
end

# The gate proposed for a pair on the current working state, with its predicted and realised
# effect: `dT` and `dLambda` are the decreases of T and Lambda predicted from the two-qubit
# density matrix, `actual_dT` and `dE` the realised decreases of T and of the total
# entanglement after applying the gate to the full state.
function _trial_pair(work::MPS, pair, config::ASPIREConfig)
    s = siteinds(work)
    rho = Matrix{ComplexF64}(dense_two_site_matrix(
        two_site_rdm(work, pair[1], pair[2]), s, pair[1], pair[2]))
    features = pair_spectrum_features(rho)
    T0 = total_correlation(work)
    E0 = total_entanglement(work)

    U = eigenbasis_gate(rho)
    rotated = U * rho * U'
    dT = _single_site_entropy_sum(rho) - _single_site_entropy_sum(rotated)
    dLambda = _single_site_lambda_sum(rho) - _single_site_lambda_sum(rotated)
    applied = apply_two_qubit_unitary_tracked(work, pair..., U;
        track=config.track_truncation)
    trial = applied.psi
    actual_dT = T0 - total_correlation(trial)
    dE = E0 - total_entanglement(trial)
    objective = dT / max(T0, _SCORE_TOL) +
                config.entanglement_weight * dE / max(E0, _SCORE_TOL)
    return (pair=pair, U=Matrix{ComplexF64}(U), trial=trial, features=features,
        dT=dT, dLambda=dLambda, actual_dT=actual_dT, dE=dE, objective=objective,
        discarded=applied.discarded)
end

# Whether a proposed gate is applied. The noise-aware test requires the certified decrease of
# Lambda to exceed the error injected by the `pair_cost(pair)` native gates, and is applied in
# addition to the objective test.
function _accepted(proposal, config::ASPIREConfig; pair_cost=nothing)
    if config.noise_rate !== nothing
        c = pair_cost === nothing ? Float64(config.pair_gate_cost) :
            Float64(pair_cost(proposal.pair))
        proposal.dLambda > config.noise_rate * c || return false
    end
    proposal.objective >= config.min_objective_gain || return false
    !config.accept_only && return true
    return proposal.dT >= -1e-10 &&
        (proposal.dT > config.min_objective_gain || proposal.dE > config.min_objective_gain)
end

# Per-gate diagnostics row.
_diagnostic(trial, layer::Int, kept::Bool, mode::Symbol) =
    (layer=layer, pair=trial.pair,
        qmi=trial.features.qmi, removable_qmi=trial.features.removable_qmi,
        actionability=trial.features.actionability,
        residual_qmi=trial.features.residual_qmi, purity=trial.features.purity,
        dT=trial.dT, dLambda=trial.dLambda, actual_dT=trial.actual_dT,
        dE=trial.dE, objective=trial.objective, discarded=trial.discarded,
        kept=kept, batch_mode=mode)

# Base score each pair is ranked on, before cost normalisation and jitter: R_ij under `:qmi`,
# the decrease of Lambda under `:lambda`.
function _selection_scores(config::ASPIREConfig, pairs, features)
    field = config.score_mode === :lambda ? :removable_lambda : :removable_qmi
    return Dict(p => Float64(getfield(features[p], field)) for p in pairs)
end

"""
    aspire_step(psi; config=ASPIREConfig(), pairs=candidate_pairs(length(psi)), layer=1,
        gate_limit=typemax(Int), rank_transform=nothing, layout=nothing) -> NamedTuple

One iteration of the main loop. All pairs in `pairs` are scored; scores are
divided by `c_ij` if `config.cost_normalised` (with `c_ij` read from the live routing `layout`),
then composed with the optional `rank_transform(pair, score)`. Pairs passing the thresholds are
shortlisted, a layer is selected according to `config.batch_mode`, and each selected gate is
applied to the working state if accepted.

Returns `(psi, gates, diagnostics, batch_mode, proposed, rank_threshold, scores)`: the updated
state, the applied `DisentanglingGate`s, one diagnostics row per attempted gate, the resolved
batch mode, the pairs proposed by the selection, the score floor of the layer and the scores.
`gate_limit` caps the number of gates applied.
"""
function aspire_step(psi::MPS; config::ASPIREConfig=ASPIREConfig(),
        pairs=candidate_pairs(length(psi); adjacent=false), layer::Int=1,
        gate_limit::Int=typemax(Int), rank_transform=nothing,
        layout::Union{Nothing,RoutingLayout}=nothing)
    _validate_config(config)
    gate_limit >= 0 || throw(ArgumentError("gate_limit must be non-negative."))
    candidate_set = collect(Tuple{Int,Int}, pairs)
    isempty(candidate_set) && return (psi=copy(psi), gates=DisentanglingGate[],
        diagnostics=NamedTuple[], batch_mode=:sequential, proposed=Tuple{Int,Int}[],
        rank_threshold=NaN, scores=Dict{Tuple{Int,Int},Float64}())

    rhos, features = _pair_data(psi, candidate_set)
    base = _selection_scores(config, candidate_set, features)
    cost_transform = cost_normalised_transform(config, length(psi); layout=layout)
    if cost_transform !== nothing
        base = Dict(p => Float64(cost_transform(p, base[p])) for p in candidate_set)
    end
    if rank_transform !== nothing
        base = Dict(p => Float64(rank_transform(p, base[p])) for p in candidate_set)
    end
    rank_of = _jittered_ranks(base, config, layer)
    qmi_of = Dict(p => features[p].qmi for p in candidate_set)
    ranked = sort(candidate_set; by=p -> rank_of[p], rev=true)
    rank_threshold = aspire_rank_threshold(values(rank_of);
        threshold=config.threshold, relative_threshold=config.relative_threshold)
    eligible_full = [p for p in ranked if rank_of[p] >= rank_threshold]
    if config.max_rank !== nothing
        eligible_full = first(eligible_full, min(config.max_rank, length(eligible_full)))
    end
    eligible = first(eligible_full, min(config.shortlist, length(eligible_full)))
    isempty(eligible) && return (psi=copy(psi), gates=DisentanglingGate[],
        diagnostics=NamedTuple[], batch_mode=:sequential, proposed=Tuple{Int,Int}[],
        rank_threshold=rank_threshold, scores=rank_of)

    mode = config.batch_mode
    # Successive layers draw independent tie-breaks.
    tie_rng = config.tie_break_seed === nothing ? nothing :
        MersenneTwister(hash((config.tie_break_seed, layer)))
    proposed = if mode == :sequential
        eligible
    elseif mode == :blossom
        # Exact over all admitted pairs rather than the shortlist: the blossom algorithm is
        # polynomial, so truncating the candidate set would only lose optimality.
        selection = select_layer(Dict(p => rank_of[p] for p in eligible_full))
        limit = min(config.max_batch_size, gate_limit)
        limit >= length(selection) ? selection :
            first(sort(selection; by=p -> rank_of[p], rev=true), limit)
    elseif mode == :matching
        maximum_weight_pair_matching(eligible, rank_of;
            max_pairs=min(config.max_batch_size, gate_limit),
            max_edges=config.matching_max_edges, rng=tie_rng,
            weight_tolerance=config.matching_weight_tolerance)
    elseif mode == :constrained_matching
        maximum_weight_cross_qmi_matching(eligible, rank_of;
            couple=_coupling_function(psi, qmi_of),
            couple_threshold=config.couple_threshold,
            max_pairs=min(config.max_batch_size, gate_limit),
            max_edges=config.matching_max_edges, rng=tie_rng,
            weight_tolerance=config.matching_weight_tolerance)
    else
        mincut_batch(eligible, rank_of;
            couple=_coupling_function(psi, qmi_of), rank_threshold=rank_threshold,
            couple_threshold=config.couple_threshold,
            max_rank=min(config.max_batch_size, gate_limit))
    end

    work = copy(psi)
    gates = DisentanglingGate[]
    diagnostics = NamedTuple[]
    # Native cost c_ij of a pair for the noise-aware admission rule.
    D_admit = config.noise_rate === nothing || config.hardware_graph === nothing ? nothing :
        graph_distances(length(psi), config.hardware_graph)
    admit_cost = config.noise_rate === nothing ? nothing :
        pair -> pair_native_cost(pair, config; N=length(psi), distances=D_admit,
            layout=layout)

    if gate_limit == 0
        return (psi=work, gates=gates, diagnostics=diagnostics,
            batch_mode=mode, proposed=proposed, rank_threshold=rank_threshold,
            scores=rank_of)
    elseif mode == :sequential
        # Evaluate the shortlisted pairs on the same state and apply only the accepted gate
        # of largest objective.
        trials = [_trial_pair(work, pair, config) for pair in proposed]
        admissible = findall(trial -> _accepted(trial, config; pair_cost=admit_cost), trials)
        best_index = isempty(admissible) ? 0 : admissible[
            argmax(trials[idx].objective for idx in admissible)]
        for (idx, trial) in enumerate(trials)
            keep = idx == best_index
            push!(diagnostics, _diagnostic(trial, layer, keep, mode))
            if keep
                push!(gates, DisentanglingGate(trial.pair..., trial.U, layer))
                work = trial.trial
            end
        end
    else
        limit = min(length(proposed), config.max_batch_size, gate_limit)
        for pair in first(proposed, limit)
            trial = _trial_pair(work, pair, config)
            keep = _accepted(trial, config; pair_cost=admit_cost)
            push!(diagnostics, _diagnostic(trial, layer, keep, mode))
            if keep
                push!(gates, DisentanglingGate(trial.pair..., trial.U, layer))
                work = trial.trial
            end
        end
    end

    return (psi=work, gates=gates, diagnostics=diagnostics,
        batch_mode=mode, proposed=proposed, rank_threshold=rank_threshold,
        scores=rank_of)
end

"""
    aspire_support_step(psi, supports; config=ASPIREConfig(), layer=1,
        gate_limit=typemax(Int)) -> NamedTuple

Apply the eigenbasis gate to each of an explicitly given ordered list of supports, skipping the
ranking and selection of [`aspire_step`](@ref) but keeping its acceptance rule. Used to run a
prescribed gate pattern, such as the staircase of the matrix product disentangler, with ASPIRE
gates. Returns `(psi, gates, diagnostics, batch_mode, proposed)`.
"""
function aspire_support_step(psi::MPS, supports;
        config::ASPIREConfig=ASPIREConfig(), layer::Int=1,
        gate_limit::Int=typemax(Int))
    _validate_config(config)
    gate_limit >= 0 || throw(ArgumentError("gate_limit must be non-negative."))
    proposed = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for raw_pair in supports
        i, j = minmax(Int(raw_pair[1]), Int(raw_pair[2]))
        i == j && throw(ArgumentError("fixed supports must join distinct sites"))
        (i, j) in seen && continue
        push!(seen, (i, j))
        push!(proposed, (i, j))
    end

    work = copy(psi)
    gates = DisentanglingGate[]
    diagnostics = NamedTuple[]
    for pair in first(proposed, min(length(proposed), gate_limit))
        trial = _trial_pair(work, pair, config)
        keep = _accepted(trial, config)
        push!(diagnostics, _diagnostic(trial, layer, keep, :fixed_support))
        if keep
            push!(gates, DisentanglingGate(trial.pair..., trial.U, layer))
            work = trial.trial
        end
    end
    return (psi=work, gates=gates, diagnostics=diagnostics,
        batch_mode=:fixed_support, proposed=proposed)
end

"""
    aspire(psi; config=ASPIREConfig(), pairs=candidate_pairs(length(psi))) -> NamedTuple

Run ASPIRE on `psi` for up to `config.max_depth` layers. Stops early when a layer
applies no gate, when `config.max_gates` is reached, or when the certificate bound
`1 - sqrt(1 - Lambda)` falls below `config.target_infidelity`.

The returned named tuple holds the final state `psi` (also as `final`), and series with one entry
per layer, starting from the input: cumulative `gates`, `depth`, entanglement `E`, total
correlation `T`, certificate `lambda`, the squared fidelity and amplitude infidelity against
`|0...0>` (`zero_product_*`) and against the bond-dimension-one truncation
(`product_frame_*`, see [`product_frame_fidelity`](@ref)). Also `gate_log` (the applied
`DisentanglingGate`s in order), `diagnostics`, the batch `modes`, the `config`, and
`certified_at`, the layer at which the target infidelity was certified, or `nothing`.
"""
function aspire(psi::MPS; config::ASPIREConfig=ASPIREConfig(),
        pairs=candidate_pairs(length(psi); adjacent=false))
    _validate_config(config)
    work = copy(psi)
    N = length(psi)
    # The routing permutation is threaded through the layers: under `:running` every layer is
    # both scored and charged against the layout its predecessors left behind.
    routing = _routing_tables(config, N)
    layout = routing === nothing ? nothing : RoutingLayout(N)
    gate_log = DisentanglingGate[]
    diagnostics = NamedTuple[]
    modes = Symbol[]
    gates = Int[0]
    depth = Int[0]
    E = Float64[total_entanglement(work)]
    T = Float64[total_correlation(work)]
    lambda = Float64[lambda_certificate(work)]
    fid = Float64[zero_product_fidelity(work)]
    frame_fid = Float64[product_frame_fidelity(work)]
    amplitude_infid = Float64[amplitude_infidelity_from_squared(fid[1])]
    frame_amplitude_infid =
        Float64[amplitude_infidelity_from_squared(frame_fid[1])]
    cumulative = 0
    certified_at = nothing

    for layer in 1:config.max_depth
        remaining = config.max_gates - cumulative
        remaining <= 0 && break
        step = aspire_step(work; config=config, pairs=pairs, layer=layer,
            gate_limit=remaining, layout=layout)
        append!(diagnostics, step.diagnostics)
        push!(modes, step.batch_mode)
        isempty(step.gates) && break
        append!(gate_log, step.gates)
        work = step.psi
        cumulative += length(step.gates)
        push!(gates, cumulative)
        push!(depth, layer)
        push!(E, total_entanglement(work))
        push!(T, total_correlation(work))
        push!(lambda, lambda_certificate(work))
        push!(fid, zero_product_fidelity(work))
        push!(frame_fid, product_frame_fidelity(work))
        push!(amplitude_infid, amplitude_infidelity_from_squared(fid[end]))
        push!(frame_amplitude_infid,
            amplitude_infidelity_from_squared(frame_fid[end]))
        _advance_layout!(layout, routing, step.gates, config)

        # The bound is vacuous (1.0) while Lambda >= 1, so this cannot fire on a state whose
        # infidelity has not been certified.
        if config.target_infidelity !== nothing &&
                infidelity_upper_bound(lambda[end]) <= config.target_infidelity
            certified_at = layer
            break
        end
    end

    return (psi=work, final=work, gates=gates, depth=depth, E=E, T=T, lambda=lambda,
        certified_at=certified_at,
        zero_product_squared_fidelity=fid,
        zero_product_amplitude_fidelity=amplitude_fidelity_from_squared.(fid),
        product_frame_squared_fidelity=frame_fid,
        product_frame_amplitude_fidelity=amplitude_fidelity_from_squared.(frame_fid),
        zero_product_amplitude_infidelity=amplitude_infid,
        product_frame_amplitude_infidelity=frame_amplitude_infid,
        gate_log=gate_log, diagnostics=diagnostics, modes=modes, config=config)
end

end # module
