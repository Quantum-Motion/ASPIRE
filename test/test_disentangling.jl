using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Test

@testset "Disentangling" begin
    @testset "candidate_pairs" begin
        @test candidate_pairs(4) == [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        @test length(candidate_pairs(5)) == 10
        @test candidate_pairs(4; adjacent=true) == [(1, 2), (2, 3), (3, 4)]
        @test candidate_pairs(5; max_distance=2) ==
            [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
        @test candidate_pairs(5; max_distance=1) == candidate_pairs(5; adjacent=true)
        @test candidate_pairs(5; max_distance=99) == candidate_pairs(5)
        @test isempty(candidate_pairs(1))
        @test_throws ArgumentError candidate_pairs(5; max_distance=0)
        @test_throws ArgumentError candidate_pairs(5; adjacent=true, max_distance=2)
    end

    @testset "haar_unitary is unitary" begin
        U = haar_unitary(4, MersenneTwister(0))
        @test isapprox(U' * U, I(4); atol=1e-12)
    end

    @testset "entanglement measures of a product state vanish" begin
        psi = MPS(ComplexF64, siteinds("S=1/2", 5), fill("Up", 5))
        @test isapprox(total_entanglement(psi), 0.0; atol=1e-10)
        @test isapprox(total_correlation(psi), 0.0; atol=1e-10)
    end

    @testset "applying the identity leaves the state unchanged" begin
        psi = random_circuit_state(siteinds("S=1/2", 6); gates=10, rng=MersenneTwister(1))
        phi = apply_two_qubit_unitary(psi, 2, 3, Matrix{ComplexF64}(I, 4, 4))
        @test isapprox(abs(inner(psi, phi)), 1.0; atol=1e-10)
    end

    @testset "gate application honors global and per-call truncation settings" begin
        sites = siteinds("S=1/2", 4)
        product = MPS(ComplexF64, sites, fill("Up", 4))
        H = ComplexF64[1 1; 1 -1] / sqrt(2)
        CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        bell_gate = CNOT * kron(H, ComplexF64[1 0; 0 1])

        truncated = with_sim_config(maxdim=1, cutoff=0.0) do
            apply_two_qubit_unitary(product, 1, 4, bell_gate)
        end
        resolved = with_sim_config(maxdim=1, cutoff=0.0) do
            apply_two_qubit_unitary(product, 1, 4, bell_gate; maxdim=8, cutoff=0.0)
        end

        @test maxlinkdim(truncated) == 1
        @test maxlinkdim(resolved) > 1
        @test total_entanglement(resolved) > total_entanglement(truncated) + 1e-8
        @test_throws ArgumentError apply_two_qubit_unitary(product, 1, 4, bell_gate; maxdim=0)
        @test_throws ArgumentError apply_two_qubit_unitary(product, 1, 4, bell_gate; cutoff=-1e-8)
    end

    @testset "tracked application reports the discarded weight" begin
        psi = random_circuit_state(siteinds("S=1/2", 6); gates=14, adjacent=false,
            rng=MersenneTwister(3))
        U = haar_unitary(4, MersenneTwister(4))
        exact = apply_two_qubit_unitary_tracked(psi, 1, 5, U; maxdim=64, cutoff=0.0)
        @test exact.discarded < 1e-12
        lossy = apply_two_qubit_unitary_tracked(psi, 1, 5, U; maxdim=1, cutoff=0.0)
        @test 0 < lossy.discarded < 1
        @test isnan(apply_two_qubit_unitary_tracked(psi, 1, 5, U; track=false).discarded)
    end

    @testset "eig_disentangler removes a Bell pair's entanglement" begin
        sites = siteinds("S=1/2", 4)
        H = (1 / sqrt(2)) * ComplexF64[1 1; 1 -1]
        CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        bell = MPS(ComplexF64, sites, fill("Up", 4))
        bell = apply_two_qubit_unitary(bell, 1, 2, kron(H, ComplexF64[1 0; 0 1]))
        bell = apply_two_qubit_unitary(bell, 1, 2, CNOT)
        @test isapprox(bond_entropy(bell, 1), 1.0; atol=1e-10)

        disentangled = disentangle_pair(bell, 1, 2)
        @test isapprox(bond_entropy(disentangled, 1), 0.0; atol=1e-8)
        @test mi_disentangler === eig_disentangler
    end

    @testset "total correlation is the sum of single-site entropies" begin
        bell = ghz_block_state(2, 1, 2)
        @test isapprox(total_correlation(bell), 2.0; atol=1e-8)
        psi = random_circuit_state(siteinds("S=1/2", 8); gates=24, rng=MersenneTwister(42))
        @test total_correlation(psi) > 0
        @test isapprox(total_correlation(psi), sum(site_entropies(psi)); atol=1e-9)
    end

    @testset "min-cut batch and rank threshold" begin
        psi = random_circuit_state(siteinds("S=1/2", 8); gates=24, rng=MersenneTwister(42))
        s = siteinds(psi)
        pairs = candidate_pairs(8; adjacent=false)
        mi_of = Dict(p => real(quantum_mutual_information(
            dense_two_site_matrix(two_site_rdm(psi, p[1], p[2]), s, p[1], p[2]))) for p in pairs)
        b = mincut_batch(pairs, mi_of; rank_threshold=1e-3, couple_threshold=0.1)
        qubits = collect(Iterators.flatten(b))
        @test length(qubits) == length(unique(qubits))   # pairs are disjoint
        @test all(mi_of[p] >= 1e-3 for p in b)            # only above-threshold pairs

        @test aspire_rank_threshold([1.0, 0.1, 0.01]; threshold=1e-3) == 1e-3
        @test aspire_rank_threshold([1.0, 0.1, 0.01]; threshold=1e-3,
            relative_threshold=0.05) == 0.05
        @test aspire_rank_threshold([1e-4, 1e-5]; threshold=1e-3,
            relative_threshold=0.05) == 1e-3
        @test aspire_rank_threshold([1.0, 0.1]; threshold=ASPIRE_DEFAULT_THRESHOLD,
            relative_threshold=ASPIRE_DEFAULT_RELATIVE_THRESHOLD) == 0.1
        @test_throws ArgumentError aspire_rank_threshold([1.0]; relative_threshold=-0.1)

        capped = mincut_batch(pairs, mi_of; rank_threshold=0.0, couple_threshold=Inf, max_rank=1)
        @test length(capped) <= 1
        @test isempty(mincut_batch(pairs, mi_of; rank_threshold=0.0,
            couple_threshold=Inf, max_rank=0))
        @test_throws ArgumentError mincut_batch(pairs, mi_of; rank_threshold=0.0,
            couple_threshold=Inf, max_rank=-1)
    end

    @testset "layered min-cut disentangling" begin
        psi = random_circuit_state(siteinds("S=1/2", 8); gates=24, rng=MersenneTwister(42))
        r = parallel_disentangling_with_gate_log(psi; couple_threshold=0.1, max_depth=40,
            threshold=1e-7, relative_threshold=0.0)
        @test r.gates >= r.depth                          # a layer holds at least one gate
        @test r.gates == length(r.gate_log)
        @test r.total_correlation[end] <= r.total_correlation[1] + 1e-9
        @test r.gate_counts[end] == r.gates

        base = parallel_disentangling_with_gate_log(psi;
            couple_threshold=0.1, max_depth=1, threshold=1e-7, relative_threshold=0.0)
        adaptive = parallel_disentangling_with_gate_log(psi;
            couple_threshold=0.1, max_depth=1, threshold=1e-3, relative_threshold=0.05)
        top1 = parallel_disentangling_with_gate_log(psi;
            couple_threshold=0.1, max_depth=1, threshold=1e-7, max_rank=1)
        @test adaptive.gates <= base.gates
        @test top1.gates <= 1
    end

    @testset "nearest-neighbour candidate pairs use on-demand cross-pair coupling" begin
        # The coupling test queries qubit pairs that are not candidates when selection is
        # restricted to neighbours, so it must not assume `couple_of` covers every pair.
        psi = random_circuit_state(siteinds("S=1/2", 8); gates=24, rng=MersenneTwister(42))
        r = parallel_disentangling_with_gate_log(psi; pairs=candidate_pairs(8; adjacent=true),
            couple_threshold=0.1, max_depth=40, threshold=1e-7, relative_threshold=0.0)
        @test r.total_correlation[end] <= r.total_correlation[1] + 1e-9
        @test r.gates >= r.depth
    end

    @testset "compression helpers" begin
        psi = random_circuit_state(siteinds("S=1/2", 6); gates=20, adjacent=false,
            rng=MersenneTwister(9))
        summary = bond_dimension_summary(psi)
        @test summary.maxdim == maxlinkdim(psi)
        @test summary.sumdim == sum(summary.dims)
        c = compressed_mps(psi; cutoff=1e-2, maxdim=2)
        @test maxlinkdim(c) <= 2
        @test isapprox(norm(c), 1.0; atol=1e-12)
    end
end
