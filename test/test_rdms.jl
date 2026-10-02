using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Test

function exact_state_mps(amps::AbstractArray, sites)
    "Construct an exact MPS from a dense state tensor."
    return MPS(amps, sites; cutoff=1e-14, maxdim=64)
end

function rdm_matrix(rho, sites, i::Int, j::Int)
    "Convert a two-site RDM tensor to a dense 4x4 matrix."
    if rho isa AbstractMatrix
        return rho
    end

    mat = zeros(ComplexF64, 4, 4)
    basis = ((1, 1), (1, 2), (2, 1), (2, 2))

    for row in 1:4
        a, b = basis[row]
        for col in 1:4
            c, d = basis[col]
            mat[row, col] = rho[
                sites[i] => a,
                sites[j] => b,
                prime(sites[i]) => c,
                prime(sites[j]) => d
            ]
        end
    end
    return mat
end

function product_state_mps(N::Int)
    "Construct a product-state MPS of all Up spins."
    sites = siteinds("S=1/2", N)
    psi = MPS(sites, fill("Up", N))
    return psi, sites
end

function bell_state_mps(N::Int)
    "Construct an N-site state with a Bell pair on sites 1 and 2."
    sites = siteinds("S=1/2", N)
    amps = zeros(ComplexF64, ntuple(_ -> 2, N)...)
    amps[1, 1, ntuple(_ -> 1, N - 2)...] = 1 / sqrt(2)
    amps[2, 2, ntuple(_ -> 1, N - 2)...] = 1 / sqrt(2)
    psi = exact_state_mps(amps, sites)
    return psi, sites
end

function ghz_state_mps(N::Int)
    "Construct an N-site GHZ state."
    sites = siteinds("S=1/2", N)
    amps = zeros(ComplexF64, ntuple(_ -> 2, N)...)
    amps[ntuple(_ -> 1, N)...] = 1 / sqrt(2)
    amps[ntuple(_ -> 2, N)...] = 1 / sqrt(2)
    psi = exact_state_mps(amps, sites)
    return psi, sites
end

@testset "Two-site RDM sanity checks" begin
    @testset "Product state" begin
        psi, sites = product_state_mps(4)
        rho = two_site_rdm(psi, 1, 2)
        rho_mat = dense_two_site_matrix(rho, sites, 1, 2)

        @test isapprox(tr(rho_mat), 1.0; atol=1e-12)
        @test isapprox(quantum_mutual_information(rho_mat), 0.0; atol=1e-12)

        expected = zeros(ComplexF64, 4, 4)
        expected[1, 1] = 1.0
        @test isapprox(rho_mat, expected; atol=1e-12)
    end

    @testset "Bell pair embedded in a larger product state" begin
        psi, sites = bell_state_mps(4)
        rho = two_site_rdm(psi, 1, 2)
        rho_mat = dense_two_site_matrix(rho, sites, 1, 2)

        @test isapprox(tr(rho_mat), 1.0; atol=1e-12)
        @test quantum_mutual_information(rho_mat) > 0

        bell_expected = zeros(ComplexF64, 4, 4)
        bell_expected[1, 1] = 0.5
        bell_expected[1, 4] = 0.5
        bell_expected[4, 1] = 0.5
        bell_expected[4, 4] = 0.5
        @test isapprox(rho_mat, bell_expected; atol=1e-12)
    end

    @testset "GHZ-style state" begin
        psi, sites = ghz_state_mps(4)
        rho = two_site_rdm(psi, 1, 2)
        rho_mat = dense_two_site_matrix(rho, sites, 1, 2)

        @test isapprox(tr(rho_mat), 1.0; atol=1e-12)
        @test quantum_mutual_information(rho_mat) > 0

        ghz_expected = zeros(ComplexF64, 4, 4)
        ghz_expected[1, 1] = 0.5
        ghz_expected[4, 4] = 0.5
        @test isapprox(rho_mat, ghz_expected; atol=1e-12)
    end
end

@testset "Non-adjacent product-state RDM" begin
    psi, sites = product_state_mps(4)
    rho = two_site_rdm(psi, 1, 4)
    rho_mat = dense_two_site_matrix(rho, sites, 1, 4)

    @test isapprox(tr(rho_mat), 1.0; atol=1e-12)
    @test isapprox(quantum_mutual_information(rho_mat), 0.0; atol=1e-12)
end

@testset "Non-adjacent GHZ RDM" begin
    psi, sites = ghz_state_mps(4)
    rho = two_site_rdm(psi, 1, 4)
    rho_mat = dense_two_site_matrix(rho, sites, 1, 4)

    @test isapprox(tr(rho_mat), 1.0; atol=1e-12)
    @test quantum_mutual_information(rho_mat) > 0
end

# The batched sweep is the all-pairs scan every ASPIRE selector runs once per layer, so
# it must agree with the one-pair-at-a-time routine to the last bit, not just to a
# tolerance: a drift here would silently change which gates get selected.
@testset "Batched two-site RDMs match the per-pair routine" begin
    sites = siteinds("S=1/2", 9)
    psi = normalize!(random_mps(MersenneTwister(4), sites; linkdims=8))

    @testset "all pairs" begin
        pairs = candidate_pairs(9)
        batched = all_pairs_two_site_rdms(psi, pairs)
        @test length(batched) == length(pairs)
        for p in pairs
            @test dense_two_site_matrix(batched[p], sites, p...) ==
                  dense_two_site_matrix(two_site_rdm(psi, p...), sites, p...)
        end
    end

    @testset "one sweep, unsorted and non-contiguous js" begin
        js = [9, 4, 7]
        out = two_site_rdms(psi, 2, js)
        @test sort!(collect(keys(out))) == sort(js)
        for j in js
            @test dense_two_site_matrix(out[j], sites, 2, j) ==
                  dense_two_site_matrix(two_site_rdm(psi, 2, j), sites, 2, j)
        end
        @test two_site_rdms(psi, 2, Int[]) |> isempty
        @test_throws ArgumentError two_site_rdms(psi, 5, [3])
        @test_throws ArgumentError two_site_rdms(psi, 5, [10])
    end

    # Both global measures are summed off one sweep of the orthogonality centre; they agree
    # with the per-site and per-bond sums to round-off.
    @testset "global measures match the per-site and per-bond sums" begin
        @test total_correlation(psi) ≈ sum(site_entropies(psi))
        @test total_entanglement(psi) ≈
              sum(ASPIRE.EntanglementBasics.bond_entropy(psi, b) for b in 1:(length(psi) - 1))
        @test total_correlation(product_state_mps(6)[1]) ≈ 0.0 atol = 1e-12
        @test total_entanglement(product_state_mps(6)[1]) ≈ 0.0 atol = 1e-12
    end
end

"""
Test whether a matrix has the basic properties of a density matrix.
"""
function test_density_matrix_properties(rho; atol=1e-12)
    @test isapprox(tr(rho), 1.0; atol=atol)
    @test isapprox(rho, rho'; atol=atol)

    values = eigvals(Hermitian((rho + rho') / 2))
    @test minimum(real.(values)) >= -atol
end