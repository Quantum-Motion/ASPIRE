using Test
using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

@testset "quantics states" begin
    @testset "index map" begin
        @test quantics_index_map(2, 3; ordering=:serial) == [1 2 3; 4 5 6]
        @test quantics_index_map(2, 3; ordering=:interleaved) == [1 3 5; 2 4 6]
        for d in 1:4, n in 1:4, ord in QUANTICS_ORDERINGS
            @test sort(vec(quantics_index_map(d, n; ordering=ord))) == collect(1:(d * n))
        end
        @test_throws ArgumentError quantics_index_map(2, 3; ordering=:spiral)
        p = quantics_permutation(3, 2; from=:serial, to=:interleaved)
        @test sort(p) == collect(1:6)
        @test p == [1, 4, 2, 5, 3, 6]
    end

    @testset "grid" begin
        @test quantics_grid_point(0, 3, (-1.0, 1.0)) ≈ -1.0
        @test quantics_grid_point(7, 3, (-1.0, 1.0)) ≈ 1.0
        @test quantics_grid_point(0, 3, (0.0, 1.0); endpoint=:exclusive) ≈ 0.0
        @test quantics_grid_point(4, 3, (0.0, 1.0); endpoint=:exclusive) ≈ 0.5
        @test length(quantics_domain((0.0, 1.0), 3)) == 3
        @test quantics_domain([(0, 1), (2, 3)], 2) == [(0.0, 1.0), (2.0, 3.0)]
        @test_throws ArgumentError quantics_domain([(0, 1)], 2)
    end

    @testset "amplitudes reproduce the sampled function" begin
        d, n = 2, 3
        f = x -> exp(-(x[1]^2 + 0.5 * x[1] * x[2] + x[2]^2))
        dom = (-2.0, 2.0)
        for ord in QUANTICS_ORDERINGS
            A = quantics_amplitudes(f, d, n; ordering=ord, domain=dom)
            site = quantics_index_map(d, n; ordering=ord)
            values = similar(A)
            for m1 in 0:(2^n - 1), m2 in 0:(2^n - 1)
                bits = zeros(Int, d * n)
                for (v, m) in ((1, m1), (2, m2)), b in 1:n
                    bits[site[v, b]] = (m >> (n - b)) & 1
                end
                x = [quantics_grid_point(m1, n, dom), quantics_grid_point(m2, n, dom)]
                values[CartesianIndex((bits .+ 1)...)] = f(x)
            end
            @test A ≈ values ./ norm(values)
        end
    end

    @testset "orderings are one state relabelled" begin
        d, n = 2, 4
        f, dom, _ = quantics_function("gaussian", d; rng=MersenneTwister(7), rho=0.6)
        As = quantics_amplitudes(f, d, n; ordering=:serial, domain=dom)
        Ai = quantics_amplitudes(f, d, n; ordering=:interleaved, domain=dom)
        p = quantics_permutation(d, n; from=:serial, to=:interleaved)
        @test permutedims(As, invperm(p)) ≈ Ai
    end

    @testset "product functions have bond dimension one across variables" begin
        d, n = 3, 3
        f = x -> exp(-x[1]^2) * exp(-2 * x[2]^2) * exp(-(x[3] - 0.5)^2)
        psi = quantics_function_state(f, d, n; ordering=:serial, cutoff=1e-14)
        for v in 1:(d - 1)
            @test linkdim(psi, v * n) == 1
        end
        psi_i = quantics_function_state(f, d, n; ordering=:interleaved, cutoff=1e-14)
        @test maxlinkdim(psi_i) > 1
    end

    @testset "state matches its amplitude tensor" begin
        d, n = 2, 3
        f, dom, _ = quantics_function("gaussian", d; rng=MersenneTwister(11), rho=0.6)
        A = quantics_amplitudes(f, d, n; domain=dom)
        psi = quantics_function_state(f, d, n; domain=dom, cutoff=1e-16, maxdim=64)
        recon = Array(prod(psi), siteinds(psi)...)
        @test abs(dot(vec(recon), vec(A))) ≈ 1.0 atol = 1e-10
        @test norm(psi) ≈ 1.0
    end

    @testset "Weierstrass series" begin
        @test weierstrass_terms(16) == 25
        @test weierstrass_terms(10) < weierstrass_terms(16)
        @test weierstrass_series(0.0) == 0.0
        @test weierstrass_series(0.37; terms=1) ≈ sin(pi * 0.37) / pi

        s10 = siteinds("S=1/2", 10)
        det1 = quantics_function_state(1, 10; family="weierstrass", sites=s10,
            rng=MersenneTwister(1))
        det2 = quantics_function_state(1, 10; family="weierstrass", sites=s10,
            rng=MersenneTwister(2))
        @test abs(inner(det1, det2)) ≈ 1.0 atol = 1e-10
        rnd1 = quantics_function_state(1, 10; family="weierstrass", sites=s10,
            randomize=true, rng=MersenneTwister(1))
        rnd2 = quantics_function_state(1, 10; family="weierstrass", sites=s10,
            randomize=true, rng=MersenneTwister(2))
        @test abs(inner(rnd1, rnd2)) < 0.999

        # terms = 0 selects the resolution-matched number of terms.
        spec = Dict{String,Any}("d" => 1, "n" => 10, "family" => "weierstrass",
            "terms" => 0)
        matched = quantics_state_from_spec(spec, 1)
        direct = quantics_function_state(1, 10; family="weierstrass",
            terms=weierstrass_terms(10), sites=siteinds(matched))
        @test abs(inner(matched, direct)) ≈ 1.0 atol = 1e-10
    end

    @testset "families and encodings" begin
        rng = MersenneTwister(5)
        for family in QUANTICS_FAMILIES
            d = family == "weierstrass" ? 1 : 2
            f, dom, label = quantics_function(family, d; rng=rng, rho=0.4)
            @test !isempty(label)
            box = quantics_domain(dom, d)
            @test isfinite(f([0.5 * (lo + hi) for (lo, hi) in box]))
            psi = quantics_function_state(d, 3; family=family,
                rng=MersenneTwister(5), rho=0.4)
            @test norm(psi) ≈ 1.0
            @test length(psi) == 3 * d
        end
        @test_throws ArgumentError quantics_function("weierstrass", 2)
        @test_throws ArgumentError quantics_function("nope", 2)
        @test_throws ArgumentError quantics_function("gaussian", 2; rho=1.0)

        # Different seeds give different members; a repeated seed gives the same.
        s8 = siteinds("S=1/2", 8)
        a = quantics_function_state(2, 4; family="gaussian", rho=0.6, sites=s8,
            rng=MersenneTwister(1))
        b = quantics_function_state(2, 4; family="gaussian", rho=0.6, sites=s8,
            rng=MersenneTwister(2))
        @test abs(inner(a, b)) < 0.999
        c = quantics_function_state(2, 4; family="gaussian", rho=0.6, sites=s8,
            rng=MersenneTwister(1))
        @test abs(inner(a, c)) ≈ 1.0 atol = 1e-12

        psi = quantics_function_state(2, 3; family="gaussian",
            encoding=:sqrt_density, rng=MersenneTwister(2))
        @test norm(psi) ≈ 1.0
        @test_throws ArgumentError quantics_function_state(1, 6; family="weierstrass",
            encoding=:sqrt_density, rng=MersenneTwister(2))
        @test_throws ArgumentError quantics_amplitudes(x -> 1.0, 4, 8; max_qubits=26)
    end

    @testset "spec adapter" begin
        spec = Dict("d" => 2, "n" => 3, "family" => "gaussian", "rho" => 0.6,
            "ordering" => "interleaved", "domain" => [-2.0, 2.0])
        a = quantics_state_from_spec(spec, 1)
        @test norm(a) ≈ 1.0
        @test length(a) == 6

        spec = Dict("d" => 2, "n" => 3, "family" => "gaussian",
            "domain" => [[-2.0, 2.0], [-3.0, 3.0]])
        @test norm(quantics_state_from_spec(spec, 1)) ≈ 1.0
    end
end
