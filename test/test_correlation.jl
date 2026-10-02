# Tests of src/Correlation.jl: site entropies, total correlation, the Lambda
# certified bound, reduced density matrices, and the removable
# correlation R_ij.
using Test
using LinearAlgebra
using Random
using ITensors
using ITensorMPS
using ASPIRE

const CORR = ASPIRE.Correlation

@testset "correlation" begin

@testset "GHZ_8 total correlation and removable correlation" begin
    ghz = ghz_block_state(8, 1, 8)
    @test register_total_correlation(ghz) ≈ 8.0 atol = 1e-10
    pairs = [(i, j) for i in 1:8 for j in (i + 1):8]
    for (i, j) in pairs
        @test total_correlation_support(ghz, [i, j]) ≈ 1.0 atol = 1e-10
    end
    rho = dense_reduced_density_matrix(ghz, [1, 8])
    @test real(quantum_mutual_information(rho)) ≈ 1.0 atol = 1e-10
    # Rank-two marginals saturate R_ij = I(i:j) at every distance.
    for (i, j) in pairs
        R, _ = removable_correlation_pair(dense_reduced_density_matrix(ghz, [i, j]))
        @test R ≈ 1.0 atol = 1e-10
    end
end

@testset "1D cluster state, N = 8" begin
    psi = cluster_state(8)
    @test register_total_correlation(psi) ≈ 8.0 atol = 1e-10
    nonzero = Tuple{Int,Int}[]
    for i in 1:8, j in (i + 1):8
        T_ij = total_correlation_support(psi, [i, j])
        T_ij > 1e-8 ? push!(nonzero, (i, j)) : (@test T_ij ≈ 0.0 atol = 1e-10)
    end
    @test nonzero == [(1, 2), (7, 8)]
    @test total_correlation_support(psi, [1, 2, 3]) ≈ 2.0 atol = 1e-10
    @test total_correlation_support(psi, [6, 7, 8]) ≈ 2.0 atol = 1e-10
    for start in 2:5
        @test total_correlation_support(psi, [start, start + 1, start + 2]) ≈ 1.0 atol = 1e-10
    end
end

@testset "Bell pair removes two bits" begin
    bell = ghz_block_state(2, 1, 2)
    R, U = removable_correlation_pair(dense_reduced_density_matrix(bell, [1, 2]))
    @test R ≈ 2.0 atol = 1e-10
    @test U * U' ≈ Matrix{ComplexF64}(I, 4, 4) atol = 1e-10
end

@testset "entropies are in bits" begin
    @test CORR.spectrum_entropy([0.5, 0.5]) ≈ 1.0 atol = 1e-12
    @test von_neumann_entropy(ComplexF64[0.5 0; 0 0.5]) ≈ 1.0 atol = 1e-12
    @test site_entropies(ghz_block_state(2, 1, 2)) ≈ [1.0, 1.0] atol = 1e-10
end

@testset "T, Lambda and the infidelity bounds on random states" begin
    rng = MersenneTwister(24)
    states = MPS[]
    for _ in 1:6
        push!(states, normalize!(random_circuit_state(siteinds("S=1/2", 6);
            gates=18, adjacent=false, rng=rng)))
    end
    for _ in 1:6
        push!(states, normalize!(random_mps(rng, siteinds("S=1/2", 6); linkdims=4)))
    end
    for psi in states
        T = register_total_correlation(psi)
        Lambda = lambda_certificate(psi)
        @test T >= -1e-9
        @test 2 * Lambda <= T + 1e-9                      # Lambda <= T/2
        @test infidelity_upper_bound(psi) == infidelity_upper_bound(Lambda)
        @test certified_infidelity_bound(psi) == certified_infidelity_bound(Lambda)
        if Lambda <= 1
            @test epsilon_star(psi) <= certified_infidelity_bound(Lambda) + 1e-9
            @test epsilon_star(psi) <= infidelity_upper_bound(Lambda) + 1e-9
        end
        bonds = [bond_entropy(psi, n) for n in 2:length(psi)]
        @test maximum(bonds) <= T + 1e-9

        # 0 <= R_S' <= R_S <= T_S <= |S| for S' a subset of S.
        rho_S = dense_reduced_density_matrix(psi, [2, 3, 4])
        T_S = CORR.spectrum_total_correlation(rho_S, 3)
        R_S, _ = removable_correlation(rho_S, 3; restarts=3, rng=rng)
        R_sub, _ = removable_correlation_pair(marginal(rho_S, 3, [1, 2]))
        @test -1e-9 <= R_sub <= R_S + 1e-7
        @test R_S <= T_S + 1e-7
        @test T_S <= 3 + 1e-9
    end
end

@testset "certified_infidelity_bound" begin
    for L in (0.0, 0.01, 0.2, 0.5, 0.9, 1.0)
        tilde = L / (2 - L)
        @test certified_infidelity_bound(L) ≈ 1 - sqrt(1 - tilde) atol = 1e-14
        @test certified_infidelity_bound(L) <= infidelity_upper_bound(L) + 1e-15
    end
    @test certified_infidelity_bound(0.01) ≈ 0.01 / 4 rtol = 0.02
    @test certified_infidelity_bound(1.5) == 1.0
    @test infidelity_upper_bound(1.5) == 1.0
    # sqrt(1-d)|00> + sqrt(d)|11> has Lambda = 2d and eps_* = 1 - sqrt(1-d).
    d = 0.05
    sites = siteinds("S=1/2", 2)
    v = zeros(2, 2); v[1, 1] = sqrt(1 - d); v[2, 2] = sqrt(d)
    psi = MPS(ITensor(v, sites[1], sites[2]), sites)
    @test lambda_certificate(psi) ≈ 2d atol = 1e-12
end

@testset "disjoint gates change T additively" begin
    rng = MersenneTwister(11)
    psi = normalize!(random_circuit_state(siteinds("S=1/2", 8);
        gates=24, adjacent=false, rng=rng))
    supports = [(1, 2), (3, 4), (5, 6)]
    gates = [haar_unitary(4, rng) for _ in supports]
    T0 = register_total_correlation(psi)
    individual = Float64[]
    for (support, U) in zip(supports, gates)
        moved = apply_two_qubit_unitary(psi, support[1], support[2], U)
        push!(individual, register_total_correlation(moved) - T0)
    end
    layered = psi
    for (support, U) in zip(supports, gates)
        layered = apply_two_qubit_unitary(layered, support[1], support[2], U)
    end
    @test register_total_correlation(layered) - T0 ≈ sum(individual) atol = 1e-10
end

@testset "two-qubit saturation criterion" begin
    # A marginal with spectrum {ab, a(1-b), (1-a)b, (1-a)(1-b)} has p1 p4 = p2 p3.
    a, b = 0.7, 0.6
    spectrum = [a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)]
    rng = MersenneTwister(4)
    V = haar_unitary(4, rng)
    rho = V * Diagonal(ComplexF64.(spectrum)) * V'
    qmi = real(quantum_mutual_information(rho))
    R_fast, _ = removable_correlation_pair(rho)
    @test R_fast ≈ qmi atol = 1e-10

    psi = normalize!(random_circuit_state(siteinds("S=1/2", 6);
        gates=20, adjacent=false, rng=rng))
    generic = dense_reduced_density_matrix(psi, [2, 5])
    p = sort(real.(eigvals(Hermitian((generic + generic') / 2))); rev=true)
    @test abs(p[1] * p[4] - p[2] * p[3]) > 1e-8
    R_generic, _ = removable_correlation_pair(generic; restarts=4)
    @test R_generic < real(quantum_mutual_information(generic)) - 1e-6
end

@testset "weight-preserving removable correlation" begin
    # Basis order |00>, |01>, |10>, |11>.
    bell_sym = ComplexF64[0, 1, 1, 0] / sqrt(2)
    rho_sym = bell_sym * bell_sym'
    R_sym, U_sym, exact = removable_correlation_symmetric(rho_sym)
    @test exact
    @test isapprox(R_sym, 2.0; atol=1e-9)
    @test isapprox(first(removable_correlation_pair(rho_sym)), 2.0; atol=1e-9)
    @test opnorm(U_sym * U_sym' - I) < 1e-10

    for basis in 1:4
        v = zeros(ComplexF64, 4); v[basis] = 1
        @test isapprox(first(removable_correlation_symmetric(v * v')), 0.0; atol=1e-12)
        @test isapprox(first(removable_correlation_pair(v * v')), 0.0; atol=1e-12)
    end

    # (|00> + |11>)/sqrt(2) mixes weight sectors: a generic gate removes two bits,
    # a weight-preserving gate none.
    bell_ghz = ComplexF64[1, 0, 0, 1] / sqrt(2)
    R_ghz, _, exact_ghz = removable_correlation_symmetric(bell_ghz * bell_ghz')
    @test !exact_ghz
    @test isapprox(R_ghz, 0.0; atol=1e-8)
    @test isapprox(first(removable_correlation_pair(bell_ghz * bell_ghz')), 2.0; atol=1e-9)

    # R_sym <= R_ij because the weight-preserving gates are a subgroup of U(4).
    rng = MersenneTwister(19)
    psi = normalize!(random_mps(rng, siteinds("S=1/2", 5); linkdims=3))
    for i in 1:4, j in (i + 1):5
        rho = dense_reduced_density_matrix(psi, [i, j])
        R = first(removable_correlation_pair(rho))
        Rs = first(removable_correlation_symmetric(rho))
        @test -1e-8 <= Rs <= R + 1e-8
    end
end

end # testset correlation
