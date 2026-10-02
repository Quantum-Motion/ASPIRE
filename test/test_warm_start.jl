using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using MPSCircuits
using Random
using Test

# Amplitudes <0..a..b..0|w> of a dense state vector with all sites except i, j projected onto up.
function _slice(w::MPS, i::Int, j::Int)
    s = siteinds(w)
    N = length(w)
    full = Array(reduce(*, w), reverse(s)...)         # indexed [s_N, ..., s_1]
    v = zeros(ComplexF64, 2, 2)
    for a in 1:2, b in 1:2
        idx = ntuple(k -> (N + 1 - k == i ? a : N + 1 - k == j ? b : 1), N)
        v[a, b] = full[idx...]
    end
    return v
end

@testset "Procrustes score P(i,j)" begin
    psi = tfim_snake_state(2, 3; J=1.0, h=1.0, nsweeps=8, maxdim=[8, 16, 32])
    ps = procrustes_pair_scores(psi)
    N = length(psi)

    @testset "score is the norm of the projected amplitudes" begin
        for (i, j) in candidate_pairs(N)
            @test isapprox(ps.score[(i, j)], norm(_slice(psi, i, j)); atol=1e-10)
        end
        @test isapprox(ps.f0, abs(inner(MPS(siteinds(psi), fill("Up", N)), psi)); atol=1e-10)
        @test all(ps.gain[p] >= -1e-12 for p in keys(ps.gain))
    end

    @testset "the optimal gate attains the score" begin
        for p in ((1, 2), (2, 5), (3, 6))
            v = ps.slices[p]
            U = frame_optimal_gate(v)
            @test opnorm(U * U' - I) < 1e-12
            rotated = U * ComplexF64[v[1, 1], v[1, 2], v[2, 1], v[2, 2]]
            @test isapprox(abs(rotated[1]), norm(v); atol=1e-12)
            @test rotated[1] ≈ norm(v) atol = 1e-12            # phase fixed to be non-negative real
            @test isapprox(abs.(rotated[2:4]), zeros(3); atol=1e-12)
        end
        @test frame_optimal_gate(zeros(ComplexF64, 2, 2)) == Matrix{ComplexF64}(I, 4, 4)
    end

    @testset "psel_layer builds a disjoint layer" begin
        for (sel, gt) in ((:proc, :proc), (:qmi, :mi), (:proc, :mi))
            w, gates, info = psel_layer(psi, 10; select=sel, gate=gt, max_batch=3,
                cutoff=1e-12, max_bond_dim=64)
            @test 1 <= length(gates) <= 3
            @test length(info.chosen) == length(gates) == length(info.gains)
            sites = collect(Iterators.flatten(info.chosen))
            @test length(sites) == length(unique(sites))
        end
        # The Procrustes selector raises the overlap with |0...0>; the remaining budget caps the layer.
        w, gates, info = psel_layer(psi, 10; select=:proc, gate=:proc, max_batch=1)
        @test length(gates) == 1
        @test abs(inner(MPS(siteinds(psi), fill("Up", N)), w)) >= ps.f0 - 1e-10
        @test isempty(psel_layer(psi, 0)[2])
        @test_throws ArgumentError psel_layer(psi, 3; select=:nope)
        @test_throws ArgumentError psel_layer(psi, 3; gate=:nope)

        # The cost-aware ordering divides by (2|i-j| + 1)^gamma and so prefers shorter gates.
        _, _, near = psel_layer(psi, 10; select=:qmi, gate=:mi, max_batch=1, span_penalty=4.0)
        _, _, any_span = psel_layer(psi, 10; select=:qmi, gate=:mi, max_batch=1)
        span(p) = p[2] - p[1]
        @test span(near.chosen[1]) <= span(any_span.chosen[1])
    end

    @testset "switch schedule" begin
        sched = switch_schedule(2)
        @test sched(1, String[]) == [(:qmi, :mi)]
        @test sched(2, String[]) == [(:qmi, :mi)]
        @test sched(3, String[]) == [(:proc, :proc)]
        @test switch_schedule(0; warm=(:proc, :mi))(1, String[]) == [(:proc, :mi)]
    end

    @testset "protocol with arbitration" begin
        budget = 2 * (N - 1)
        r = protocol_arb_iter_oall(psi; budget=budget, sweeps=1, look_sweeps=1, max_batch=3,
            cutoff=1e-12, max_bond_dim=64)
        @test r.gates[end] >= budget
        @test issorted(r.gates)
        @test length(r.won) == length(r.gates) == length(r.infid)
        @test all(w -> w in ("qmi/mi", "proc/proc"), r.won)
        @test r.infid[end] < 1
        @test issorted(r.cum_chi3_ops) && issorted(r.look_chi3_ops)

        greedy = protocol_arb_iter_oall(psi; budget=budget, sweeps=0, optimise=false,
            proposals=[(:qmi, :mi)], max_batch=3, span_penalty=0.25,
            cutoff=1e-12, max_bond_dim=64)
        @test all(==("qmi/mi"), greedy.won)
        @test all(iszero, greedy.cum_chi3_ops) && all(iszero, greedy.look_chi3_ops)
        @test greedy.infid[end] < greedy.infid[1]
    end
end
