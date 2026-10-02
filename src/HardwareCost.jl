"""
    HardwareCost

Native two-qubit gate accounting for MPS circuits executed on a device with
restricted connectivity. Logical gates on site pairs `(i, j)` are converted to
native CNOT counts by routing the two qubits together with SWAPs (3 CNOTs each)
on a coupling graph. The default `:running` convention leaves the SWAP-induced
permutation in place, so a gate at layout distance `d` costs `1 + 3(d-1)`
CNOTs and later gates are priced on the permuted layout; `:restore` undoes the
SWAPs (`1 + 6(d-1)`, pristine distances) and `:no_restore` charges `1 + 3(d-1)`
on pristine distances (a lower bound). Single-qubit gates are free. The module
also finds MPS site orderings that minimise the routing cost of a staircase
layer on a given coupling graph (`routing_ordering`).
"""
module HardwareCost

using ITensors
using Random

export ROUTING_CONVENTIONS, SWAP_GATE_COST, DEFAULT_ROUTING_CONVENTION
export graph_distances, graph_routing_tables, routing_swaps, gate_cost
export RoutingLayout, layout_distance, route_gate!, routed_gate_series
export routed_layer_cost!
export gate_support, circuit_supports
export physical_gate_count, physical_gate_series
export staircase_supports, staircase_cost
export hamiltonian_path, reorder_pairs, routing_ordering

"Native two-qubit gates in one SWAP."
const SWAP_GATE_COST = 3

"Supported routing conventions and the native-gate cost per unit of routing distance."
const ROUTING_CONVENTIONS = Dict(
    :running => SWAP_GATE_COST,
    :no_restore => SWAP_GATE_COST,
    :restore => 2 * SWAP_GATE_COST,
)

"""
Routing convention used by default. `:running` pays 3 native gates per unit of
distance once and prices every later gate on the layout the SWAPs leave behind.
"""
const DEFAULT_ROUTING_CONVENTION = :running

_convention_cost(convention::Symbol) = get(ROUTING_CONVENTIONS, convention) do
    throw(ArgumentError("unknown routing convention `$convention`; expected " *
        "one of $(sort(collect(keys(ROUTING_CONVENTIONS)))))"))
end

_is_running(convention::Symbol) = convention === :running

"""
    graph_distances(N, pairs) -> Matrix{Int}

All-pairs shortest-path distances on the undirected graph with `N` vertices and
edge list `pairs` (1-based index tuples). `D[i, i] == 0`, `D[i, j] == 1` for an
edge, and `D[i, j] == -1` when `j` is unreachable from `i`.

Breadth-first search from every vertex, `O(N (N + E))`.
"""
function graph_distances(N::Int, pairs)
    N >= 0 || throw(ArgumentError("N must be non-negative, got $N"))
    adjacency = [Int[] for _ in 1:N]
    for (a, b) in pairs
        (1 <= a <= N && 1 <= b <= N) ||
            throw(ArgumentError("edge ($a, $b) is outside 1:$N"))
        a == b && throw(ArgumentError("self-loop at vertex $a is not a bond"))
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    D = fill(-1, N, N)
    queue = Vector{Int}(undef, N)
    for source in 1:N
        D[source, source] = 0
        queue[1] = source
        head, tail = 1, 1
        while head <= tail
            v = queue[head]
            head += 1
            for w in adjacency[v]
                D[source, w] >= 0 && continue
                D[source, w] = D[source, v] + 1
                tail += 1
                queue[tail] = w
            end
        end
    end
    return D
end

"""
    graph_routing_tables(N, pairs) -> (distances, predecessors)

All-pairs shortest-path distances and the predecessor table needed to walk one
shortest path. `predecessors[s, v]` is the vertex immediately before `v` on a
shortest path from `s`, and `0` when `v == s` or `v` is unreachable from `s`.

[`graph_distances`](@ref) returns only the distances; routing needs the path
itself, because the SWAPs along it permute the layout.
"""
function graph_routing_tables(N::Int, pairs)
    N >= 0 || throw(ArgumentError("N must be non-negative, got $N"))
    adjacency = [Int[] for _ in 1:N]
    for (a, b) in pairs
        (1 <= a <= N && 1 <= b <= N) ||
            throw(ArgumentError("edge ($a, $b) is outside 1:$N"))
        a == b && throw(ArgumentError("self-loop at vertex $a is not a bond"))
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    D = fill(-1, N, N)
    P = zeros(Int, N, N)
    queue = Vector{Int}(undef, N)
    for source in 1:N
        D[source, source] = 0
        queue[1] = source
        head, tail = 1, 1
        while head <= tail
            v = queue[head]
            head += 1
            for w in adjacency[v]
                D[source, w] >= 0 && continue
                D[source, w] = D[source, v] + 1
                P[source, w] = v
                tail += 1
                queue[tail] = w
            end
        end
    end
    return (distances=D, predecessors=P)
end

"""
    RoutingLayout(N)

Live assignment of logical qubits to physical vertices, starting from the
identity. `site_of[q]` is the vertex holding logical qubit `q`, and `qubit_at[v]`
is its inverse. The two are kept consistent by `swap_sites!`, the only mutation.
The price of a gate under `:running` depends on the current placement, so the
circuit is walked in order with this layout.
"""
struct RoutingLayout
    site_of::Vector{Int}
    qubit_at::Vector{Int}
end

RoutingLayout(N::Int) = RoutingLayout(collect(1:N), collect(1:N))

Base.copy(layout::RoutingLayout) =
    RoutingLayout(copy(layout.site_of), copy(layout.qubit_at))

Base.length(layout::RoutingLayout) = length(layout.site_of)

"Exchange the logical qubits on physical vertices `u` and `v`."
function swap_sites!(layout::RoutingLayout, u::Int, v::Int)
    qu, qv = layout.qubit_at[u], layout.qubit_at[v]
    layout.qubit_at[u], layout.qubit_at[v] = qv, qu
    layout.site_of[qu], layout.site_of[qv] = v, u
    return layout
end

"""
    layout_distance(layout, D, i, j) -> Int

Graph distance between logical qubits `i` and `j` *as currently placed*, read off
the static physical distance matrix `D` at their present vertices. This is the
distance that prices the next gate under `:running`.
"""
layout_distance(layout::RoutingLayout, D::AbstractMatrix, i::Integer, j::Integer) =
    D[layout.site_of[i], layout.site_of[j]]

"""
    route_gate!(layout, D, P, i, j) -> Int

Route logical qubits `i` and `j` together on the live `layout` and return the
native two-qubit gates spent: `1 + 3(d-1)`, with `d` the layout distance.

`i` is carried along a shortest path towards `j` by `d-1` SWAPs and left
adjacent to it; `j` does not move. The induced permutation is kept in `layout`.
"""
function route_gate!(layout::RoutingLayout, D::AbstractMatrix,
        P::AbstractMatrix, i::Integer, j::Integer)
    u, v = layout.site_of[i], layout.site_of[j]
    d = D[u, v]
    d >= 1 || throw(ArgumentError(
        "logical pair ($i, $j) sits on vertices ($u, $v) at distance $d and " *
        "cannot be routed"))
    # Shortest path u -> v from the predecessor table; `i` is swapped along it
    # up to the vertex adjacent to `v`.
    path = Int[v]
    w = v
    while w != u
        w = P[u, w]
        push!(path, w)
    end
    reverse!(path)
    for k in 1:(length(path) - 2)
        swap_sites!(layout, path[k], path[k + 1])
    end
    return 1 + SWAP_GATE_COST * routing_swaps(d)
end

"""
    routed_layer_cost!(layout, tables, supports; gate_cost=1) -> Int

Charge the gates `supports` (pairs of logical qubits) under `:running`, mutating
`layout`, and return the native two-qubit gates spent. `tables` is a
[`graph_routing_tables`](@ref) result; `gate_cost` is the native price of the
gate itself before routing (`1` for a CZ or CNOT, `3` for a generic SU(4)).
Gates are priced in order on the layout left by their predecessors, as in
[`routed_gate_series`](@ref).
"""
function routed_layer_cost!(layout::RoutingLayout, tables, supports;
        gate_cost::Int=1)
    D, P = tables.distances, tables.predecessors
    spent = 0
    for (i, j) in supports
        d = layout_distance(layout, D, i, j)
        spent += gate_cost + SWAP_GATE_COST * routing_swaps(d)
        route_gate!(layout, D, P, i, j)
    end
    return spent
end

"""
    routed_gate_series(supports, pairs; N, layout=RoutingLayout(N))
        -> (series, layout, distances)

Walk a circuit under the `:running` convention. Returns the cumulative native
gate count after each gate (prefixed with `0`, so entry `k+1` is the cost of the
first `k` gates), the final layout, and the layout distance each gate was
actually charged at.

The returned `layout` is the object passed in, mutated in place, so a circuit
can be threaded through several calls.
"""
function routed_gate_series(supports, pairs;
        N::Int=_implied_size(supports, pairs),
        layout::RoutingLayout=RoutingLayout(N))
    length(layout) == N || throw(ArgumentError(
        "layout holds $(length(layout)) qubits but the circuit implies $N"))
    tables = graph_routing_tables(N, pairs)
    D, P = tables.distances, tables.predecessors
    series = Vector{Int}(undef, length(supports) + 1)
    series[1] = 0
    distances = Vector{Int}(undef, length(supports))
    for (k, (i, j)) in enumerate(supports)
        distances[k] = layout_distance(layout, D, i, j)
        series[k + 1] = series[k] + route_gate!(layout, D, P, i, j)
    end
    return (series=series, layout=layout, distances=distances)
end

"""
    routing_swaps(distance) -> Int

SWAPs needed to make two qubits at graph `distance` adjacent: `distance - 1`.
Throws on an unreachable pair (`distance < 0`), which cannot be routed at all.
"""
function routing_swaps(distance::Integer)
    distance >= 1 || throw(ArgumentError(
        "pair is unreachable or ill-formed (distance = $distance)"))
    return Int(distance) - 1
end

"""
    gate_cost(distance; convention=:no_restore) -> Int

Native two-qubit gates spent executing one logical two-qubit gate between qubits
at graph `distance` under a stateless convention: `1 + 3(d-1)` for
`:no_restore`, `1 + 6(d-1)` for `:restore`. A device edge costs 1. Throws for
`:running`, whose price depends on the layout.
"""
function gate_cost(distance::Integer; convention::Symbol=:no_restore)
    _is_running(convention) && throw(ArgumentError(
        "`:running` has no distance-only price; use `layout_distance` with " *
        "`RoutingLayout`, or `physical_gate_count` / `physical_gate_series` / " *
        "`routed_gate_series`."))
    return 1 + _convention_cost(convention) * routing_swaps(distance)
end

"""
    gate_support(tensor, sites) -> Union{Nothing,Tuple{Int,Int}}

Site positions a two-qubit gate `tensor` acts on, as positions into the site
index vector `sites`. Returned in ascending order, or `nothing` for a
single-qubit gate, which has no two-qubit cost.

Gate tensors carry the site index and its primed partner for each qubit they
touch; the unprimed indices are matched against `sites`. Throws on three or more
sites.
"""
function gate_support(tensor::ITensor, sites)
    position = Dict(s => k for (k, s) in enumerate(sites))
    found = Int[]
    for index in inds(tensor)
        plev(index) == 0 || continue
        site = get(position, index, nothing)
        site === nothing && continue
        site in found || push!(found, site)
    end
    length(found) <= 2 || throw(ArgumentError(
        "expected a one- or two-qubit gate, found support $(sort(found))"))
    # Single-qubit gates carry no two-qubit cost.
    length(found) == 2 || return nothing
    return minmax(found[1], found[2])
end

"""
    circuit_supports(circuit, sites) -> Vector{Tuple{Int,Int}}

Site-pair support of every *two-qubit* gate in `circuit`, in circuit order.
Accepts gates that are bare `ITensor`s or any struct with a `tensor` field (e.g.
`MPSCircuits.UnitaryGate`). Single-qubit gates are skipped, since they are free.
"""
function circuit_supports(circuit, sites)
    supports = Tuple{Int,Int}[]
    for gate in circuit
        support = gate_support(_gate_tensor(gate), sites)
        support === nothing && continue
        push!(supports, support)
    end
    return supports
end

_gate_tensor(gate::ITensor) = gate
_gate_tensor(gate) = gate.tensor

"""
    physical_gate_count(supports, pairs; convention=:running, N) -> Int

Total native two-qubit gates for a circuit whose gates act on the site pairs
`supports`, executed on the connectivity graph `pairs`. `N` defaults to the
largest index appearing in either argument.

Under the default `:running` convention the circuit is walked in order and each
gate is priced against the layout its predecessors left behind; under the
stateless conventions every gate is priced on the pristine graph.
"""
function physical_gate_count(supports, pairs;
        convention::Symbol=DEFAULT_ROUTING_CONVENTION,
        N::Int=_implied_size(supports, pairs))
    _is_running(convention) &&
        return routed_gate_series(supports, pairs; N=N).series[end]
    D = graph_distances(N, pairs)
    return sum(gate_cost(D[i, j]; convention=convention) for (i, j) in supports;
        init=0)
end

"""
    physical_gate_series(supports, pairs; convention=:running, N) -> Vector{Int}

Cumulative native two-qubit gate count after each gate of the circuit, prefixed
with `0`: entry `k+1` is the hardware cost of the first `k` gates. Use
[`routed_gate_series`](@ref) to obtain the final layout as well.
"""
function physical_gate_series(supports, pairs;
        convention::Symbol=DEFAULT_ROUTING_CONVENTION,
        N::Int=_implied_size(supports, pairs))
    _is_running(convention) &&
        return routed_gate_series(supports, pairs; N=N).series
    D = graph_distances(N, pairs)
    series = Vector{Int}(undef, length(supports) + 1)
    series[1] = 0
    for (k, (i, j)) in enumerate(supports)
        series[k + 1] = series[k] + gate_cost(D[i, j]; convention=convention)
    end
    return series
end

"""
    staircase_supports(N) -> Vector{Tuple{Int,Int}}

The MPS nearest-neighbour staircase `(1,2), (2,3), ..., (N-1,N)`, the support of
one nearest-neighbour sweep layer.
"""
staircase_supports(N::Int) = [(i, i + 1) for i in 1:(N - 1)]

"""
    staircase_cost(N, pairs; convention=:running) -> Int

Native two-qubit gates in one MPS staircase layer on the connectivity graph
`pairs`. This is the objective for choosing a site ordering: an ordering that
follows a path in the device graph costs `N-1`, the minimum.
"""
staircase_cost(N::Int, pairs; convention::Symbol=DEFAULT_ROUTING_CONVENTION) =
    physical_gate_count(staircase_supports(N), pairs;
        convention=convention, N=N)

"""
    hamiltonian_path(N, pairs) -> Union{Nothing,Vector{Int}}

A Hamiltonian path of the graph, as a vertex sequence, or `nothing` if none
exists. Exact depth-first search, trying continuations in order of increasing
remaining degree.

An ordering along such a path is routing-optimal, since [`staircase_cost`](@ref)
is at least `N-1` and attains it exactly when every consecutive pair is a
device edge.
"""
function hamiltonian_path(N::Int, pairs)
    N >= 1 || return nothing
    adjacency = [Int[] for _ in 1:N]
    for (a, b) in pairs
        push!(adjacency[a], b)
        push!(adjacency[b], a)
    end
    visited = falses(N)
    path = Int[]
    function extend(v)
        push!(path, v)
        visited[v] = true
        length(path) == N && return true
        continuations = sort(adjacency[v];
            by=w -> count(!visited[u] for u in adjacency[w]))
        for w in continuations
            visited[w] && continue
            extend(w) && return true
        end
        pop!(path)
        visited[v] = false
        return false
    end
    for start in 1:N
        empty!(path)
        fill!(visited, false)
        extend(start) && return copy(path)
    end
    return nothing
end

"""
    reorder_pairs(pairs, order) -> Vector{Tuple{Int,Int}}

Relabel an edge list so that MPS position `k` holds the vertex `order[k]`. The
graph is unchanged; only its vertex labels move.
"""
function reorder_pairs(pairs, order)
    position = Dict(v => k for (k, v) in enumerate(order))
    return [minmax(position[a], position[b]) for (a, b) in pairs]
end

"""
    routing_ordering(N, pairs; convention=:running, iterations=200_000,
                     rng=Random.default_rng()) -> (order, pairs, cost, optimal)

Site ordering that minimizes the routing cost of one MPS staircase layer.

Returns the vertex ordering, the edge list relabelled into it
([`reorder_pairs`](@ref)), the resulting [`staircase_cost`](@ref), and whether
the result is provably optimal (`cost == N-1`).

A Hamiltonian path is returned immediately when one exists. Otherwise simulated
annealing over permutations (transpositions and segment reversals) is run from
the identity ordering and the best ordering seen is kept; `optimal` reports
whether the cost reaches `N-1`.

The annealing scores candidates on the pristine distance matrix, which is exact
for the stateless conventions and a proxy for `:running`. The returned `cost` is
the exact [`staircase_cost`](@ref) of the ordering under `convention`.
"""
function routing_ordering(N::Int, pairs;
        convention::Symbol=DEFAULT_ROUTING_CONVENTION,
        iterations::Int=200_000, rng::AbstractRNG=Random.default_rng())
    path = hamiltonian_path(N, pairs)
    if path !== nothing
        relabelled = reorder_pairs(pairs, path)
        return (order=path, pairs=relabelled,
            cost=staircase_cost(N, relabelled; convention=convention),
            optimal=true)
    end
    D = graph_distances(N, pairs)
    unit = _convention_cost(convention)
    # Staircase cost of an ordering on the pristine distance matrix.
    cost_of = order -> sum(1 + unit * (D[order[k], order[k + 1]] - 1)
        for k in 1:(N - 1); init=0)
    order = collect(1:N)
    cost = cost_of(order)
    best_order, best_cost = copy(order), cost
    floor_cost = N - 1
    for iteration in 1:iterations
        best_cost == floor_cost && break
        temperature = 2.0 * (1 - iteration / iterations) + 1e-9
        candidate = copy(order)
        if rand(rng, Bool)
            i, j = rand(rng, 1:N), rand(rng, 1:N)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else
            i, j = minmax(rand(rng, 1:N), rand(rng, 1:N))
            reverse!(candidate, i, j)
        end
        candidate_cost = cost_of(candidate)
        if candidate_cost <= cost ||
                rand(rng) < exp((cost - candidate_cost) / temperature)
            order, cost = candidate, candidate_cost
            if cost < best_cost
                best_order, best_cost = copy(order), cost
            end
        end
    end
    relabelled = reorder_pairs(pairs, best_order)
    exact_cost = staircase_cost(N, relabelled; convention=convention)
    return (order=best_order, pairs=relabelled, cost=exact_cost,
        optimal=exact_cost == floor_cost)
end

# Smallest vertex count containing every index in `supports` and `pairs`.
function _implied_size(supports, pairs)
    N = 0
    for (a, b) in supports
        N = max(N, a, b)
    end
    for (a, b) in pairs
        N = max(N, a, b)
    end
    return N
end

end # module
