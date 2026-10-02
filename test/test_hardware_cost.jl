@testset "HardwareCost" begin
    # A 4-vertex path graph: distances are |i-j|.
    path_pairs = [(1, 2), (2, 3), (3, 4)]

    @testset "graph distances" begin
        D = graph_distances(4, path_pairs)
        @test D == [0 1 2 3; 1 0 1 2; 2 1 0 1; 3 2 1 0]
        @test D == permutedims(D)

        # A disconnected vertex is unreachable, not distance zero.
        D2 = graph_distances(3, [(1, 2)])
        @test D2[1, 3] == -1 && D2[3, 3] == 0

        # A 6-cycle: distance saturates at the halfway point.
        ring = [(i, mod1(i + 1, 6)) for i in 1:6]
        Dring = graph_distances(6, ring)
        @test Dring[1, 4] == 3
        @test all(Dring[1, j] == min(j - 1, 7 - j) for j in 1:6)

        @test_throws ArgumentError graph_distances(2, [(1, 3)])
        @test_throws ArgumentError graph_distances(2, [(1, 1)])
    end

    @testset "gate cost conventions" begin
        # A device edge is free of routing under every stateless convention.
        for convention in (:no_restore, :restore)
            @test gate_cost(1; convention=convention) == 1
        end
        @test routing_swaps(1) == 0 && routing_swaps(4) == 3
        # `gate_cost` prices one gate from a distance, so its default is the
        # stateless `:no_restore` (1 + 3(d-1)); `:restore` is 1 + 6(d-1).
        @test gate_cost(2) == 4
        @test gate_cost(4) == 10
        @test gate_cost(2; convention=:no_restore) == 4
        @test gate_cost(4; convention=:no_restore) == 10
        @test gate_cost(2; convention=:restore) == 7
        @test gate_cost(4; convention=:restore) == 19
        # Restoring the layout is strictly dearer wherever routing happens.
        @test all(gate_cost(d; convention=:restore) > gate_cost(d) for d in 2:6)
        # `:running` has no distance-only price: the distance depends on the
        # layout the circuit has reached.
        @test DEFAULT_ROUTING_CONVENTION === :running
        @test haskey(ROUTING_CONVENTIONS, :running)
        @test_throws ArgumentError gate_cost(2; convention=:running)
        # An unreachable pair cannot be routed and must not silently cost 1.
        @test_throws ArgumentError gate_cost(-1)
        @test_throws ArgumentError gate_cost(0)
        @test_throws ArgumentError gate_cost(2; convention=:teleport)
    end

    @testset "the running layout" begin
        # Distances are read off the live placement, and the SWAPs that route a
        # gate are kept rather than undone.
        layout = RoutingLayout(4)
        D = graph_distances(4, path_pairs)
        tables = graph_routing_tables(4, path_pairs)
        @test tables.distances == D
        @test layout_distance(layout, D, 1, 4) == 3

        cost = route_gate!(layout, tables.distances, tables.predecessors, 1, 4)
        @test cost == 1 + 3 * 2                 # two SWAPs, paid once
        @test layout_distance(layout, D, 1, 4) == 1   # they are adjacent
        @test sort(layout.site_of) == collect(1:4)    # still a permutation
        @test layout.qubit_at[layout.site_of[2]] == 2 # and still consistent

        # Repeating a routed gate is a device-edge gate under `:running`, and a
        # full-price gate under `:no_restore`.
        @test physical_gate_count([(1, 4), (1, 4)], path_pairs; N=4) == 7 + 1
        @test physical_gate_count([(1, 4), (1, 4)], path_pairs; N=4,
            convention=:no_restore) == 7 + 7

        # `routed_layer_cost!` performs the same accounting on a given layout.
        fresh = RoutingLayout(4)
        @test routed_layer_cost!(fresh, tables, [(1, 4)]) == 7
        @test routed_layer_cost!(RoutingLayout(4), tables, [(1, 4)];
            gate_cost=3) == 3 + 3 * 2
    end

    @testset "circuit costing" begin
        # Pristine-graph distances are 1, 2, 3. Under the default `:running`
        # convention only the first is charged at its pristine distance: routing
        # (1,3) carries qubit 1 to vertex 2, so (1,4) is then two hops, not three.
        supports = [(1, 2), (1, 3), (1, 4)]
        @test physical_gate_count(supports, path_pairs) == 1 + 4 + 4
        @test physical_gate_count(supports, path_pairs;
            convention=:no_restore) == 1 + 4 + 7
        @test physical_gate_count(supports, path_pairs;
            convention=:restore) == 1 + 7 + 13
        # Cumulative series is prefixed with zero and ends at the total.
        series = physical_gate_series(supports, path_pairs)
        @test series == [0, 1, 5, 9]
        @test series[end] == physical_gate_count(supports, path_pairs)
        @test physical_gate_count(Tuple{Int,Int}[], path_pairs; N=4) == 0
        @test physical_gate_series(Tuple{Int,Int}[], path_pairs; N=4) == [0]

        # The walk exposes the layout as well as the series, and the distances
        # it reports are the ones actually charged.
        walk = routed_gate_series(supports, path_pairs; N=4)
        @test walk.distances == [1, 2, 2]
        @test walk.series == series
        @test walk.layout.site_of == [3, 1, 2, 4]

        # An all-to-all graph makes every gate local, so cost == gate count.
        dense = [(i, j) for i in 1:4 for j in (i + 1):4]
        @test physical_gate_count(supports, dense) == length(supports)
    end

    @testset "staircase on the heavy-hex snake" begin
        @test staircase_supports(4) == [(1, 2), (2, 3), (3, 4)]
        # A staircase that follows the graph is exactly one gate per bond.
        @test staircase_cost(4, path_pairs) == 3

        # The snake ordering leaves most MPS-neighbour pairs off the device
        # graph, so a staircase layer costs several times its logical gate count.
        # Under `:restore` the per-gate costs at distances 1, 2, 4, 6 are 1, 7,
        # 19, 31 on the pristine graph; under `:running` the layout is permuted
        # as the staircase proceeds, so the totals are pinned directly.
        for (cells, N, running, restoring) in
                ((1, 12, 56, 6 + 2 * 7 + 2 * 19 + 1 * 31),
                 (2, 21, 128, 6 + 6 * 7 + 6 * 19 + 2 * 31),
                 (3, 30, 200, 6 + 10 * 7 + 10 * 19 + 3 * 31))
            patch = heavy_hex_patch(cells; ordering=:snake)
            @test length(patch.coordinates) == N
            @test length(staircase_supports(N)) == N - 1
            @test staircase_cost(N, patch.pairs) == running
            @test running > 1.5 * (N - 1)
            @test staircase_cost(N, patch.pairs; convention=:restore) == restoring
            @test running < restoring
        end

        # Every heavy-hex edge is a device edge and costs 1.
        patch = heavy_hex_patch(3; ordering=:snake)
        @test physical_gate_count(patch.pairs, patch.pairs) == length(patch.pairs)
    end

    @testset "site ordering" begin
        # A path graph is already ordered along its own path.
        @test hamiltonian_path(4, path_pairs) !== nothing
        @test staircase_cost(4, path_pairs) == 3

        # A star has no Hamiltonian path for N > 3: the centre must be revisited.
        @test hamiltonian_path(4, [(1, 2), (1, 3), (1, 4)]) === nothing

        @test reorder_pairs([(1, 3), (3, 2)], [3, 1, 2]) == [(1, 2), (1, 3)]
        # Relabelling preserves the graph, so distances are unchanged up to
        # the permutation.
        patch = heavy_hex_patch(2; ordering=:snake)
        N2 = length(patch.coordinates)
        order = [N2:-1:1;]
        @test sort(reorder_pairs(patch.pairs, order)) ==
            sort([minmax(N2 + 1 - a, N2 + 1 - b) for (a, b) in patch.pairs])

        # cells=1 and cells=2 admit Hamiltonian paths, so the optimal ordering
        # pays no routing at all; the snake ordering is several times worse.
        for cells in 1:2
            patch = heavy_hex_patch(cells; ordering=:snake)
            N = length(patch.coordinates)
            result = routing_ordering(N, patch.pairs)
            @test result.optimal
            @test result.cost == N - 1
            @test sort(result.order) == collect(1:N)
            @test staircase_cost(N, result.pairs) == N - 1
            @test result.cost < staircase_cost(N, patch.pairs)
        end

        # cells=3 has no Hamiltonian path, so some routing is unavoidable; the
        # annealed ordering must still beat the snake by a wide margin.
        patch3 = heavy_hex_patch(3; ordering=:snake)
        N3 = length(patch3.coordinates)
        @test hamiltonian_path(N3, patch3.pairs) === nothing
        annealed = routing_ordering(N3, patch3.pairs;
            iterations=50_000, rng=MersenneTwister(7))
        @test !annealed.optimal
        @test annealed.cost >= N3 - 1
        @test annealed.cost < staircase_cost(N3, patch3.pairs)
        # The reported cost is the exact staircase cost of the ordering under
        # the requested convention, not the search proxy the annealer ranks on.
        @test annealed.cost == staircase_cost(N3, annealed.pairs)
    end

    @testset "gate support extraction" begin
        sites = siteinds("S=1/2", 5)
        U = reshape(Matrix{ComplexF64}(I, 4, 4), 2, 2, 2, 2)
        gate = two_qubit_gate(reshape(U, 4, 4), sites[2], sites[4])
        @test gate_support(gate, sites) == (2, 4)
        # Wrapped gates (any struct with a `tensor` field) are accepted.
        wrapped = (tensor=gate,)
        @test circuit_supports([wrapped, wrapped], sites) == [(2, 4), (2, 4)]
        # A one-site operator has no two-qubit support and no two-qubit cost.
        single = ITensor(ComplexF64, prime(sites[1]), sites[1])
        single[prime(sites[1]) => 1, sites[1] => 1] = 1
        @test gate_support(single, sites) === nothing
        @test circuit_supports([single, gate, single], sites) == [(2, 4)]
        @test physical_gate_count(circuit_supports([single], sites),
            path_pairs; N=5) == 0
        # Three or more sites is still a mis-shaped gate.
        triple = ITensor(ComplexF64, prime(sites[1]), sites[1],
            prime(sites[2]), sites[2], prime(sites[3]), sites[3])
        @test_throws ArgumentError gate_support(triple, sites)
    end
end
