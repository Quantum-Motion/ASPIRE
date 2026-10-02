using Test
using LinearAlgebra
using Random
using ITensors
using ITensorMPS
using MPSCircuits
using ASPIRE

@testset "selection and cost accounting" begin

@testset "step (III) is a maximum-weight matching" begin
    # An odd cycle with one heavy edge: greedy selection by descending weight takes the heavy
    # edge and cannot complete, while the optimum takes the two lighter ones.
    candidates = Dict((1, 2) => 1.0, (2, 3) => 0.9, (3, 4) => 0.9, (4, 1) => 0.1)
    selection = select_layer(candidates)
    @test selection == [(1, 2), (3, 4)]
    @test matching_weight(selection, candidates) ≈ 1.9

    greedy_trap = Dict((1, 2) => 1.0, (2, 3) => 0.6, (3, 4) => 0.6, (4, 5) => 0.6,
        (5, 6) => 1.0)
    exact = select_layer(greedy_trap)
    @test matching_weight(exact, greedy_trap) ≈
        matching_weight(brute_force_matching(
            [(p[1], p[2], w) for (p, w) in greedy_trap]), greedy_trap)

    # Non-positive scores never enter a layer.
    @test isempty(select_layer(Dict((1, 2) => 0.0, (2, 3) => -1.0)))

    # The selected supports are pairwise disjoint, so the per-gate contributions to the
    # decrease of T are additive.
    sites = vcat([[p[1], p[2]] for p in selection]...)
    @test length(unique(sites)) == length(sites)
end

@testset "blossom mode selects a disjoint layer" begin
    psi = normalize!(random_circuit_state(siteinds("S=1/2", 8);
        gates=24, adjacent=false, rng=MersenneTwister(3)))
    config = ASPIREConfig(; batch_mode=:blossom, shortlist=28, threshold=0.0,
        relative_threshold=0.0)
    step = aspire_step(psi; config=config)
    @test step.batch_mode == :blossom
    touched = vcat([[p[1], p[2]] for p in step.proposed]...)
    @test length(unique(touched)) == length(touched)
end

@testset "c_ij includes routing SWAPs" begin
    chain = [(i, i + 1) for i in 1:5]
    # `:restore` pays d-1 SWAPs in and d-1 back out: 6(d-1) added to the gate's own cost.
    restoring = ASPIREConfig(; hardware_graph=chain, pair_gate_cost=3,
        routing_convention=:restore)
    @test pair_native_cost((1, 2), restoring; N=6) == 3
    @test pair_native_cost((1, 4), restoring; N=6) == 3 + 6 * 2
    @test pair_native_cost((1, 6), restoring; N=6) == 3 + 6 * 4
    # The default `:running` convention leaves the permutation in place and pays 3(d-1) once:
    # c_ij = 3(1 + s_ij) CNOTs for s_ij SWAPs.
    config = ASPIREConfig(; hardware_graph=chain, pair_gate_cost=3)
    @test config.routing_convention === :running
    @test pair_native_cost((1, 2), config; N=6) == 3
    @test pair_native_cost((1, 4), config; N=6) == 3 + 3 * 2
    @test pair_native_cost((1, 6), config; N=6) == 3 + 3 * 4
    # Without a hardware graph every pair costs the same.
    flat = ASPIREConfig(; hardware_graph=nothing, pair_gate_cost=3)
    @test pair_native_cost((1, 6), flat; N=6) == 3

    transform = cost_normalised_transform(config, 6)
    @test transform((1, 2), 1.0) ≈ 1 / 3
    @test transform((1, 6), 1.0) ≈ 1 / 15
    @test cost_normalised_transform(ASPIREConfig(; cost_normalised=false), 6) === nothing
end

@testset "the running layout prices a repeated long-range pair once" begin
    # A pair that has been routed together is adjacent afterwards, so a second gate on it is a
    # device-edge gate. `:no_restore` charges the same price per gate and keeps the pristine
    # layout, so it charges the second gate in full again.
    N = 6
    chain = [(i, i + 1) for i in 1:(N - 1)]
    once = physical_gate_count([(1, 6)], chain; N=N)
    twice = physical_gate_count([(1, 6), (1, 6)], chain; N=N)
    @test once == 1 + 3 * 4
    @test twice == once + 1
    @test physical_gate_count([(1, 6), (1, 6)], chain; N=N, convention=:no_restore) == 2 * once
    @test physical_gate_count([(1, 6)], chain; N=N, convention=:restore) == 1 + 6 * 4

    walk = routed_gate_series([(1, 6)], chain; N=N)
    @test walk.distances == [5]
    @test sort(walk.layout.site_of) == collect(1:N)
    @test walk.layout.site_of[1] == 5           # qubit 1 carried next to qubit 6
    @test walk.layout.site_of[6] == 6
    @test layout_distance(walk.layout, graph_distances(N, chain), 1, 6) == 1

    # `:running` is stateful and has no price as a function of distance alone.
    @test_throws ArgumentError gate_cost(3; convention=:running)
end

@testset "the selector prices moves against the live layout" begin
    chain = [(i, i + 1) for i in 1:5]
    config = ASPIREConfig(; hardware_graph=chain, pair_gate_cost=3)
    layout = RoutingLayout(6)
    tables = graph_routing_tables(6, chain)

    @test pair_native_cost((1, 6), config; N=6, layout=layout) == 3 + 3 * 4
    route_gate!(layout, tables.distances, tables.predecessors, 1, 6)
    @test pair_native_cost((1, 6), config; N=6, layout=layout) == 3
    # Without a layout the identity placement is assumed.
    @test pair_native_cost((1, 6), config; N=6) == 3 + 3 * 4

    # The cost-normalised score carries the layout, so a routed pair outranks an unrouted one
    # of equal removable correlation.
    stale = cost_normalised_transform(config, 6)
    live = cost_normalised_transform(config, 6; layout=layout)
    @test live((1, 6), 1.0) > stale((1, 6), 1.0)
    @test live((1, 6), 1.0) ≈ 1 / 3

    # A stateless convention ignores the layout.
    restoring = ASPIREConfig(; hardware_graph=chain, pair_gate_cost=3,
        routing_convention=:restore)
    @test pair_native_cost((1, 6), restoring; N=6, layout=layout) ==
        pair_native_cost((1, 6), restoring; N=6)
end

@testset "truncation is tracked beside T and is off by default" begin
    rng = MersenneTwister(41)
    psi = normalize!(random_mps(rng, ComplexF64, siteinds("S=1/2", 10); linkdims=16))
    U = haar_unitary(4, rng)

    loose = apply_two_qubit_unitary_tracked(psi, 3, 4, U; maxdim=64, cutoff=1e-16)
    tight = apply_two_qubit_unitary_tracked(psi, 3, 4, U; maxdim=2, cutoff=1e-16)
    @test loose.discarded < 1e-12
    @test tight.discarded > 1e-3
    @test loose.discarded <= tight.discarded
    @test 0 <= tight.discarded <= 1
    # The returned state is normalised whatever was discarded.
    @test norm(tight.psi) ≈ 1 atol = 1e-12
    @test isnan(apply_two_qubit_unitary_tracked(psi, 3, 4, U; track=false).discarded)

    target = normalize!(random_mps(rng, ComplexF64, siteinds("S=1/2", 12); linkdims=16))
    previous = global_maxdim()
    try
        set_max_bond_dim!(8)
        base = (max_depth=3, cost_normalised=false)
        off = aspire(copy(target); config=ASPIREConfig(; base..., track_truncation=false))
        on = aspire(copy(target); config=ASPIREConfig(; base..., track_truncation=true))
        @test all(d -> isnan(d.discarded), off.diagnostics)
        @test !isempty(on.diagnostics)
        @test all(d -> isfinite(d.discarded) && d.discarded >= 0, on.diagnostics)
        # Tracking is a measurement: the selected gates are the same.
        @test off.gates[end] == on.gates[end]
    finally
        set_max_bond_dim!(previous)
    end
end

@testset "the Lambda certificate can terminate a run" begin
    psi = normalize!(random_circuit_state(siteinds("S=1/2", 8);
        gates=14, adjacent=false, rng=MersenneTwister(17)))

    base = (max_depth=12, cost_normalised=false)
    unbounded = aspire(psi; config=ASPIREConfig(; base...))
    @test unbounded.certified_at === nothing

    # A loose target is certified no later than the end of the unbounded run, and the run
    # stops there.
    loose = aspire(psi; config=ASPIREConfig(; base..., target_infidelity=0.9))
    if loose.certified_at !== nothing
        @test loose.certified_at <= length(unbounded.depth) - 1
        @test length(loose.depth) - 1 == loose.certified_at
        @test infidelity_upper_bound(loose.lambda[end]) <= 0.9 + 1e-12
        @test loose.gates[end] <= unbounded.gates[end]
    end

    # The bound is vacuous while Lambda >= 1, so an unreachable target never stops a run early.
    tight = aspire(psi; config=ASPIREConfig(; base..., target_infidelity=1e-14))
    @test tight.certified_at === nothing
    for L in tight.lambda
        L >= 1 && @test infidelity_upper_bound(L) == 1.0
    end

    @test_throws ArgumentError aspire(psi; config=ASPIREConfig(; target_infidelity=0.0))
end

@testset "noise-aware admission requires the certificate to beat the injected error" begin
    psi = normalize!(random_circuit_state(siteinds("S=1/2", 8);
        gates=14, adjacent=false, rng=MersenneTwister(17)))
    free = aspire(psi; config=ASPIREConfig(; max_depth=6, cost_normalised=false,
        noise_rate=0.0))
    costly = aspire(psi; config=ASPIREConfig(; max_depth=6, cost_normalised=false,
        noise_rate=10.0))
    @test free.gates[end] > 0
    @test costly.gates[end] == 0            # no gate removes more than 30 units of Lambda
    @test_throws ArgumentError aspire(psi; config=ASPIREConfig(; noise_rate=-1.0))
end

@testset "predicted R_ij differs from the decrease of T after Procrustes refinement" begin
    # A greedy layer lowers T by exactly sum(R_ij) because its gates are the eigenbasis
    # optimisers. After a Procrustes sweep the gates are fitted to the fidelity instead, so the
    # decrease achieved by the refined circuit is a separate quantity.
    sites = siteinds("S=1/2", 6)
    target = normalize!(random_circuit_state(sites; gates=18, adjacent=false,
        rng=MersenneTwister(17)))

    config = ASPIREConfig(; threshold=0.0, relative_threshold=0.0)
    step = aspire_step(target; config=config, layer=1)
    @test !isempty(step.gates)
    sum_R_selected = sum(d.removable_qmi for d in step.diagnostics if d.kept)

    T_before = total_correlation(target)
    @test T_before - total_correlation(step.psi) ≈ sum_R_selected atol = 1e-8

    circuit = prep_circuit_from_gate_log(step.gates, sites)
    oall_sweep!(ITensorMPS.dense(copy(target)), circuit; sweeps=3)
    residual = disentangle_with_prep_circuit(target, circuit)
    @test abs((T_before - total_correlation(residual)) - sum_R_selected) > 1e-6
end

end # testset
