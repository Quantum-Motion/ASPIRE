using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using Test

@testset "Pair entropies" begin
    product = zeros(ComplexF64, 4, 4)
    product[1, 1] = 1.0
    bell = zeros(ComplexF64, 4, 4)
    bell[[1, 4], [1, 4]] .= 0.5
    ghz_marginal = zeros(ComplexF64, 4, 4)
    ghz_marginal[1, 1] = 0.5
    ghz_marginal[4, 4] = 0.5

    @testset "quantum mutual information" begin
        @test isapprox(quantum_mutual_information(product), 0.0; atol=1e-12)
        # A Bell pair carries 2 bits: S(A) + S(B) - S(AB) = 1 + 1 - 0.
        @test isapprox(quantum_mutual_information(bell), 2.0; atol=1e-12)
        # The classically correlated marginal of a GHZ state carries 1 bit.
        @test isapprox(quantum_mutual_information(ghz_marginal), 1.0; atol=1e-12)
    end

    @testset "von Neumann entropy" begin
        @test isapprox(von_neumann_entropy(bell), 0.0; atol=1e-12)
        @test isapprox(von_neumann_entropy(ghz_marginal), 1.0; atol=1e-12)
        partial = [0.8 0.0; 0.0 0.2]
        @test isapprox(von_neumann_entropy(partial), -0.8 * log2(0.8) - 0.2 * log2(0.2); atol=1e-14)
    end
end

@testset "Bond entropy" begin
    bell = ghz_block_state(4, 1, 2)
    @test isapprox(bond_entropy(bell, 1), 1.0; atol=1e-10)
    @test isapprox(bond_entropy(bell, 2), 0.0; atol=1e-10)
    @test isapprox(total_entanglement(bell), 1.0; atol=1e-10)
end
