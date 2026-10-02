using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Test

@testset "ASPIRE v2 (opt-in)" begin
    @testset "removable mutual information" begin
        mixed = Matrix{ComplexF64}(I, 4, 4) / 4
        fm = pair_spectrum_features(mixed)
        @test isapprox(fm.qmi, 0.0; atol=1e-12)
        @test isapprox(fm.removable_qmi, 0.0; atol=1e-12)
        @test isapprox(fm.actionability, 0.0; atol=1e-12)
        @test isapprox(fm.residual_qmi, 0.0; atol=1e-12)
        @test isapprox(fm.purity, 0.25; atol=1e-12)

        bell = ComplexF64[1, 0, 0, 1] / sqrt(2)
        rho_bell = bell * bell'
        fb = pair_spectrum_features(rho_bell)
        @test isapprox(fb.qmi, 2.0; atol=1e-10)
        @test isapprox(fb.removable_qmi, 2.0; atol=1e-10)
        @test isapprox(fb.actionability, 1.0; atol=1e-10)
        @test isapprox(fb.residual_qmi, 0.0; atol=1e-10)
        @test isapprox(fb.purity, 1.0; atol=1e-10)

        rank_two = zeros(ComplexF64, 4, 4)
        rank_two[1, 1] = 0.5
        rank_two[4, 4] = 0.5
        fr = pair_spectrum_features(rank_two)
        @test isapprox(fr.qmi, 1.0; atol=1e-10)
        @test isapprox(fr.removable_qmi, 1.0; atol=1e-10)
        @test isapprox(fr.actionability, 1.0; atol=1e-10)
        @test isapprox(fr.purity, 0.5; atol=1e-10)

        psi = random_circuit_state(siteinds("S=1/2", 6); gates=10,
            rng=MersenneTwister(19))
        s = siteinds(psi)
        rho = dense_two_site_matrix(two_site_rdm(psi, 2, 3), s, 2, 3)
        predicted = removable_mutual_information(rho)
        actual = total_correlation(psi) - total_correlation(
            disentangle_pair(psi, 2, 3; disentangler=mi_disentangler))
        @test isapprox(predicted, actual; atol=1e-8)
    end

    @testset "canonical gate inside degenerate eigenspaces" begin
        # The score depends only on the spectrum, so it is stable under any rotation inside a
        # degenerate eigenvalue block, but the eigenvectors returned by LAPACK are not. Symmetric
        # targets (the rung singlets of a Heisenberg ladder) have exactly degenerate pair marginals,
        # so the gate must be fixed on the eigenspace, not on the eigenvectors: perturbing the input
        # by 1e-15 must not change the gate.
        function degenerate_pair(p, seed)
            Random.seed!(seed)
            Q, _ = qr(randn(ComplexF64, 4, 4))
            U = Matrix(Q)
            return U * diagm(collect(p) ./ sum(p)) * U'
        end

        for p in ([0.4, 0.4, 0.1, 0.1],          # two two-fold blocks
                  [0.25, 0.25, 0.25, 0.25],      # one four-fold block
                  [0.5, 0.25, 0.25, 0.0],        # a block between simple ones
                  [0.7, 0.1, 0.1, 0.1])          # a three-fold block
            rho = degenerate_pair(p, 3)
            U0 = eigenbasis_gate(rho)
            R0 = removable_mutual_information(rho)
            @test opnorm(U0 * U0' - I) < 1e-10          # still a unitary

            worst_gate = 0.0
            worst_score = 0.0
            Random.seed!(11)
            for _ in 1:8
                h = randn(ComplexF64, 4, 4)
                h = (h + h') / 2
                h = h / opnorm(h)
                pert = rho + 1e-15 * h
                pert = (pert + pert') / 2
                pert = pert / real(tr(pert))
                worst_gate = max(worst_gate,
                                 opnorm(eigenbasis_gate(pert) - U0))
                worst_score = max(worst_score,
                                  abs(removable_mutual_information(pert) - R0))
            end
            # The gate moves by less than 1e-10 in operator norm and the score by less than 1e-12.
            @test worst_gate < 1e-10
            @test worst_score < 1e-12
        end

        # Idempotent, and independent of which equally valid input eigenbasis of
        # the same operator it is handed.
        rho = degenerate_pair([0.4, 0.4, 0.1, 0.1], 5)
        F = eigen(Hermitian((rho + rho') / 2))
        order = sortperm(F.values; rev=true)
        spec = clamp.(real.(F.values[order]), 0.0, 1.0)
        spec = spec ./ sum(spec)
        A = ASPIRE.Selection._canonical_eigenbasis(F.vectors[:, order], spec)
        @test maximum(abs.(ASPIRE.Selection._canonical_eigenbasis(A, spec) - A)) < 1e-10
        B = copy(F.vectors[:, order])
        rot = ComplexF64[0.6 0.8; -0.8 0.6]
        B[:, 1:2] = B[:, 1:2] * rot
        B[:, 3:4] = B[:, 3:4] * rot
        @test maximum(abs.(ASPIRE.Selection._canonical_eigenbasis(B, spec) - A)) < 1e-9

        # The global phase of an eigenvector is anchored on its largest component; a component
        # near an absolute threshold must not change the anchor under a tiny perturbation.
        v = ComplexF64[1e-12, 0.6, 0.8, 0.0]
        v = v / norm(v)
        w = copy(v)
        w[1] = -1e-13
        a = ASPIRE.Selection._fix_phase(v)
        b = ASPIRE.Selection._fix_phase(w)
        @test maximum(abs.(a - b)) < 1e-9
    end

    @testset "product-frame fidelity" begin
        sites = siteinds("S=1/2", 4)
        product = MPS(ComplexF64, sites, fill("Up", 4))
        H = ComplexF64[1 1; 1 -1] / sqrt(2)
        product = apply_two_qubit_unitary(product, 1, 2, kron(H, H))

        # The state is a product state in a rotated local basis: far from |0...0>, but
        # exactly reproduced by its bond-dimension-one truncation.
        @test zero_product_fidelity(product) < 0.3
        @test isapprox(product_frame_fidelity(product), 1.0; atol=1e-10)

        complex_product = MPS(ComplexF64, sites, fill("Up", 4))
        phase_gate = Diagonal(ComplexF64[1, im])
        local_state = ComplexF64[1 1; im -im] / sqrt(2)
        complex_product = apply_two_qubit_unitary(complex_product, 1, 2,
            kron(local_state, phase_gate))
        @test isapprox(product_frame_fidelity(complex_product), 1.0; atol=1e-10)

        # The frame infidelity is an upper bound on the optimal-frame infidelity, so it cannot be
        # smaller than the infidelity to the bond-dimension-one truncation by construction.
        ghz = ghz_block_state(4, 1, 4)
        @test product_frame_amplitude_infidelity(ghz) > 0.2
    end

    @testset "exact removable-QMI matching" begin
        pairs = [(1, 2), (1, 3), (2, 4)]
        weights = Dict((1, 2) => 10.0, (1, 3) => 6.0, (2, 4) => 6.0)

        # Greedy takes (1,2) with weight 10; the exact disjoint layer has
        # (1,3)+(2,4) with weight 12.
        @test maximum_weight_pair_matching(pairs, weights) ==
            [(1, 3), (2, 4)]
        @test maximum_weight_pair_matching(pairs, weights; max_pairs=1) ==
            [(1, 2)]
        @test isempty(maximum_weight_pair_matching(
            [(1, 2)], Dict((1, 2) => 0.0)))
        @test maximum_weight_pair_matching(
            [(2, 1), (1, 2)], Dict((2, 1) => 3.0)) == [(1, 2)]
        @test_throws ArgumentError maximum_weight_pair_matching(
            pairs, weights; max_edges=2)
        @test_throws ArgumentError maximum_weight_pair_matching(
            [(1, 1)], Dict((1, 1) => 1.0))
        @test_throws DomainError maximum_weight_pair_matching(
            [(1, 2)], Dict((1, 2) => Inf))

        # Randomized tie-break: samples among exactly-tied optima without ever
        # giving up primary weight. Four disjoint pairs all at weight 1.0, so
        # every maximal matching of size two is optimal (weight 2.0).
        tied_pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]
        tied_weights = Dict(pair => 1.0 for pair in tied_pairs)
        deterministic = maximum_weight_pair_matching(tied_pairs, tied_weights;
            max_pairs=2)
        # Without an rng the lowest labels win.
        @test deterministic == [(1, 2), (3, 4)]
        @test maximum_weight_pair_matching(tied_pairs, tied_weights;
            max_pairs=2, rng=nothing) == deterministic

        seen = Set{Vector{Tuple{Int,Int}}}()
        for seed in 1:40
            chosen = maximum_weight_pair_matching(tied_pairs, tied_weights;
                max_pairs=2, rng=MersenneTwister(seed))
            @test length(chosen) == 2
            @test allunique(Iterators.flatten(chosen))
            # Every draw is an exact optimum.
            @test sum(tied_weights[pair] for pair in chosen) ≈ 2.0
            @test issorted(chosen)
            push!(seen, chosen)
        end
        @test length(seen) > 1          # ties really are being sampled
        @test deterministic in seen

        # A strict optimum must be found regardless of the tie-break draw.
        strict_weights = Dict((1, 2) => 10.0, (1, 3) => 6.0, (2, 4) => 6.0)
        for seed in 1:20
            @test maximum_weight_pair_matching(pairs, strict_weights;
                rng=MersenneTwister(seed)) == [(1, 3), (2, 4)]
        end

        # Same guarantees for the cross-QMI-constrained variant.
        for seed in 1:10
            chosen = maximum_weight_cross_qmi_matching(tied_pairs, tied_weights;
                couple=(a, b) -> 0.0, couple_threshold=1.0, max_pairs=2,
                rng=MersenneTwister(seed))
            @test length(chosen) == 2
            @test sum(tied_weights[pair] for pair in chosen) ≈ 2.0
        end

        # `tie_break_seed` is off by default.
        @test ASPIREConfig().tie_break_seed === nothing
        @test ASPIREConfig(; tie_break_seed=7).tie_break_seed == 7

        # Score jitter is off by default and must be reproducible when on.
        @test ASPIREConfig().score_jitter == 0.0
        @test ASPIREConfig().matching_weight_tolerance == 1e-12
        # The configuration invariants are checked on the way into the driver.
        product4 = MPS(siteinds("S=1/2", 4), fill("Up", 4))
        @test_throws ArgumentError aspire_step(product4;
            config=ASPIREConfig(; score_jitter=-1e-8, tie_break_seed=1))
        # Jitter without a seed would draw from global RNG state.
        @test_throws ArgumentError aspire_step(product4;
            config=ASPIREConfig(; score_jitter=1e-6))

        # Jitter perturbs selection scores without corrupting them: the ranking
        # of pairs separated by much more than the jitter must be preserved.
        base = Dict((1, 2) => 1.0, (3, 4) => 0.5, (5, 6) => 0.1)
        for seed in 1:15
            cfg = ASPIREConfig(; score_jitter=1e-6, tie_break_seed=seed)
            jittered = ASPIRE.Selection._jittered_ranks(base, cfg, 1)
            @test keys(jittered) == keys(base)
            for (pair, value) in base
                @test isapprox(jittered[pair], value; rtol=2e-6)
            end
            @test jittered[(1, 2)] > jittered[(3, 4)] > jittered[(5, 6)]
        end
        # Same seed and layer -> same draw; different layer -> different draw.
        cfg = ASPIREConfig(; score_jitter=1e-6, tie_break_seed=3)
        @test ASPIRE.Selection._jittered_ranks(base, cfg, 1) ==
            ASPIRE.Selection._jittered_ranks(base, cfg, 1)
        @test ASPIRE.Selection._jittered_ranks(base, cfg, 1) !=
            ASPIRE.Selection._jittered_ranks(base, cfg, 2)
        # Zero jitter is an exact pass-through, not an approximate one.
        @test ASPIRE.Selection._jittered_ranks(base,
            ASPIREConfig(; score_jitter=0.0), 1) === base

        # Jitter larger than a near-degenerate splitting must be able to flip
        # which of two effectively-tied pairs ranks first, while a splitting far
        # larger than the jitter must never flip.
        degenerate = Dict((1, 2) => 0.16207194468, (3, 4) => 0.16207194100)
        flipped = 0
        for seed in 1:40
            j = ASPIRE.Selection._jittered_ranks(degenerate,
                ASPIREConfig(; score_jitter=1e-6, tie_break_seed=seed), 1)
            j[(3, 4)] > j[(1, 2)] && (flipped += 1)
        end
        @test flipped > 0
        separated = Dict((1, 2) => 0.20, (3, 4) => 0.10)
        @test all(ASPIRE.Selection._jittered_ranks(separated,
            ASPIREConfig(; score_jitter=1e-6, tie_break_seed=s), 1)[(1, 2)] >
            ASPIRE.Selection._jittered_ranks(separated,
            ASPIREConfig(; score_jitter=1e-6, tie_break_seed=s), 1)[(3, 4)]
            for s in 1:40)

        rng = MersenneTwister(211)
        complete_pairs = [(i, j) for i in 1:5 for j in (i + 1):5]
        for _ in 1:12
            sampled = complete_pairs[randperm(rng, length(complete_pairs))[1:7]]
            sampled_weights = Dict(pair => rand(rng) for pair in sampled)
            exact = maximum_weight_pair_matching(sampled, sampled_weights;
                max_pairs=2)
            exact_weight = sum((sampled_weights[pair] for pair in exact); init=0.0)
            brute_weight = 0.0
            for mask in 0:(2^length(sampled) - 1)
                count_ones(mask) <= 2 || continue
                subset = [sampled[k] for k in eachindex(sampled)
                    if !iszero(mask & (1 << (k - 1)))]
                vertices = collect(Iterators.flatten(subset))
                length(unique(vertices)) == length(vertices) || continue
                brute_weight = max(brute_weight,
                    sum((sampled_weights[pair] for pair in subset); init=0.0))
            end
            @test isapprox(exact_weight, brute_weight; atol=1e-12)
        end

        psi = random_circuit_state(siteinds("S=1/2", 4); gates=8,
            adjacent=false, rng=MersenneTwister(223))
        pair_a, pair_b = (1, 2), (3, 4)
        unitary_a = mi_disentangler(psi, pair_a...)
        unitary_b = mi_disentangler(psi, pair_b...)
        sites = siteinds(psi)
        predicted = sum(pair_spectrum_features(dense_two_site_matrix(
                two_site_rdm(psi, pair...), sites, pair...)).removable_qmi
            for pair in (pair_a, pair_b))
        forward, reverse = with_sim_config(maxdim=256, cutoff=0.0) do
            a_then_b = apply_two_qubit_unitary(
                apply_two_qubit_unitary(psi, pair_a..., unitary_a),
                pair_b..., unitary_b)
            b_then_a = apply_two_qubit_unitary(
                apply_two_qubit_unitary(psi, pair_b..., unitary_b),
                pair_a..., unitary_a)
            (a_then_b, b_then_a)
        end
        @test isapprox(squared_fidelity(forward, reverse), 1.0; atol=1e-10)
        @test isapprox(total_correlation(psi) - total_correlation(forward),
            predicted; atol=1e-8)
    end

    @testset "exact cross-QMI-constrained matching" begin
        pairs = [(1, 2), (3, 4), (5, 6)]
        weights = Dict((1, 2) => 10.0, (3, 4) => 6.0, (5, 6) => 6.0)
        blocked = Set(minmax(a, b) for a in (1, 2) for b in (3, 4, 5, 6))
        couple(a, b) = minmax(a, b) in blocked ? 0.2 : 0.01

        # Greedy cross-QMI batching would take the weight-10 edge first and
        # block both remaining edges. Exact constrained matching finds the two
        # mutually compatible weight-6 edges.
        @test maximum_weight_cross_qmi_matching(pairs, weights;
            couple=couple, couple_threshold=0.1) == [(3, 4), (5, 6)]
        @test maximum_weight_cross_qmi_matching(pairs, weights;
            couple=couple, couple_threshold=0.25) == pairs
        @test_throws ArgumentError maximum_weight_cross_qmi_matching(
            pairs, weights; couple=couple, couple_threshold=-0.1)

        psi = random_circuit_state(siteinds("S=1/2", 6); gates=10,
            adjacent=false, rng=MersenneTwister(227))
        common = (max_depth=1, max_gates=3, shortlist=8,
            matching_max_edges=12, threshold=1e-8, relative_threshold=0.25,
            couple_threshold=0.1)
        mincut_step = aspire_step(psi;
            config=ASPIREConfig(; common..., batch_mode=:mincut))
        constrained_step = aspire_step(psi;
            config=ASPIREConfig(; common..., batch_mode=:constrained_matching))

        @test constrained_step.batch_mode == :constrained_matching
        @test length(unique(Iterators.flatten(constrained_step.proposed))) ==
            2 * length(constrained_step.proposed)

        sites = siteinds(psi)
        feature(pair) = pair_spectrum_features(dense_two_site_matrix(
            two_site_rdm(psi, pair...), sites, pair...))
        constrained_weight = sum(
            (feature(pair).removable_qmi for pair in constrained_step.proposed);
            init=0.0)
        mincut_weight = sum(
            (feature(pair).removable_qmi for pair in mincut_step.proposed);
            init=0.0)
        @test constrained_weight + 1e-10 >= mincut_weight

        for left_index in eachindex(constrained_step.proposed)
            for right_index in 1:(left_index - 1)
                left = constrained_step.proposed[left_index]
                right = constrained_step.proposed[right_index]
                cross_qmi = maximum(feature(minmax(a, b)).qmi
                    for a in left for b in right)
                @test cross_qmi <= common.couple_threshold + 1e-10
            end
        end
    end

    @testset "driver is bounded and reports consistent series" begin
        psi = random_circuit_state(siteinds("S=1/2", 6); gates=8,
            rng=MersenneTwister(23))
        config = ASPIREConfig(max_depth=4, max_gates=2, shortlist=3,
            batch_mode=:sequential)
        result = aspire(psi; config=config)

        @test result.gates[end] <= 2
        @test length(result.gate_log) == result.gates[end]
        @test result.T[end] <= result.T[1] + 1e-8
        @test length(result.lambda) == length(result.T) == length(result.gates)
        @test all(result.lambda .<= result.T .+ 1e-9)
        fid = result.zero_product_squared_fidelity
        frame_fid = result.product_frame_squared_fidelity
        @test frame_fid[end] >= 0
        @test all(isapprox.(result.zero_product_amplitude_fidelity,
            amplitude_fidelity_from_squared.(fid); atol=1e-14))
        @test all(isapprox.(result.product_frame_amplitude_fidelity,
            amplitude_fidelity_from_squared.(frame_fid); atol=1e-14))
        @test all(isapprox.(result.zero_product_amplitude_infidelity,
            amplitude_infidelity_from_squared.(fid); atol=1e-14))
        @test all(isapprox.(result.product_frame_amplitude_infidelity,
            amplitude_infidelity_from_squared.(frame_fid); atol=1e-14))
        @test all(d -> haskey(d, :removable_qmi) && haskey(d, :actionability) &&
            haskey(d, :residual_qmi) && haskey(d, :actual_dT) && haskey(d, :objective),
            result.diagnostics)
        # The decrease of T predicted from the two-qubit density matrix is exact for a gate
        # applied without truncation.
        @test all(d -> !d.kept || isapprox(d.dT, d.actual_dT; atol=1e-8), result.diagnostics)

        @test_throws ArgumentError aspire(psi; config=ASPIREConfig(max_gates=-1))
        @test_throws ArgumentError aspire(psi; config=ASPIREConfig(max_rank=0))
        @test_throws ArgumentError aspire(psi; config=ASPIREConfig(matching_max_edges=0))
        @test_throws ArgumentError aspire(psi; config=ASPIREConfig(batch_mode=:not_a_mode))
        @test_throws ArgumentError aspire(psi; config=ASPIREConfig(score_mode=:not_a_mode))

        matching = ASPIREConfig(max_depth=1, max_gates=2, shortlist=4, batch_mode=:matching)
        matching_step = aspire_step(psi; config=matching)
        @test matching_step.batch_mode == :matching
        @test length(unique(Iterators.flatten(matching_step.proposed))) ==
            2 * length(matching_step.proposed)
    end

    @testset "paper defaults" begin
        config = ASPIREConfig()
        @test config.shortlist == 32
        @test config.matching_max_edges == 48
        @test config.batch_mode === :matching
        @test config.couple_threshold == 0.1
        @test config.pair_gate_cost == 3
        @test config.routing_convention === :running
        @test config.entanglement_weight == 0.25
    end

    @testset "score modes" begin
        psi = normalize!(random_mps(MersenneTwister(4), siteinds("S=1/2", 8); linkdims=4))
        pairs = candidate_pairs(8; adjacent=false)
        sites = siteinds(psi)

        base = (max_depth=1, batch_mode=:blossom, threshold=0.0, relative_threshold=0.0,
            cost_normalised=false, accept_only=false, min_objective_gain=-1e6)
        qmi = aspire_step(psi; config=ASPIREConfig(; score_mode=:qmi, base...),
            pairs=pairs, layer=1)
        lam = aspire_step(psi; config=ASPIREConfig(; score_mode=:lambda, base...),
            pairs=pairs, layer=1)
        for pair in pairs
            f = pair_spectrum_features(dense_two_site_matrix(
                two_site_rdm(psi, pair...), sites, pair...))
            @test isapprox(qmi.scores[pair], f.removable_qmi; atol=1e-12)
            @test isapprox(lam.scores[pair], f.removable_lambda; atol=1e-12)
            @test f.removable_lambda >= 0
        end

        # Both modes emit a disjoint layer: the score changes the weights, not the matching.
        for step in (qmi, lam)
            supports = collect(Iterators.flatten((g.i, g.j) for g in step.gates))
            @test length(unique(supports)) == length(supports)
        end
    end
end
