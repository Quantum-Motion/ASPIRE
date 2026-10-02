using ASPIRE
using ITensorMPS
using Test

@testset "Canonical fidelity metrics" begin
    @testset "scalar conversion uses 1-f, not 1-F" begin
        @test amplitude_fidelity_from_squared(1.0) == 1.0
        @test amplitude_infidelity_from_squared(1.0) == 0.0
        @test amplitude_fidelity_from_squared(0.25) == 0.5
        @test amplitude_infidelity_from_squared(0.25) == 0.5
        @test amplitude_infidelity_from_squared(0.0) == 1.0

        @test clamp_squared_fidelity(1 + 1e-12) == 1.0
        @test clamp_squared_fidelity(-1e-12) == 0.0
        @test_throws DomainError clamp_squared_fidelity(1.1)
        @test_throws DomainError clamp_squared_fidelity(-0.1)
        @test_throws DomainError clamp_squared_fidelity(Inf)
        @test_throws ArgumentError clamp_squared_fidelity(0.5; atol=-1.0)

        series = fidelity_series_from_squared([1.0, 0.25, 0.0])
        @test propertynames(series) ==
            (:squared_fidelity, :amplitude_fidelity, :amplitude_infidelity)
        @test series.squared_fidelity == [1.0, 0.25, 0.0]
        @test series.amplitude_fidelity == [1.0, 0.5, 0.0]
        @test series.amplitude_infidelity == [0.0, 0.5, 1.0]
        @test_throws DomainError fidelity_series_from_squared([0.5, 1.2])
    end

    @testset "MPS preparation and reverse-overlap conventions agree" begin
        sites = siteinds("S=1/2", 3)
        up = MPS(ComplexF64, sites, fill("Up", 3))
        down = MPS(ComplexF64, sites, fill("Dn", 3))

        @test squared_fidelity(up, up) == 1.0
        @test amplitude_fidelity(up, up) == 1.0
        @test amplitude_infidelity(up, up) == 0.0
        @test squared_fidelity(up, down) == 0.0
        @test amplitude_infidelity(up, down) == 1.0
        @test_throws DimensionMismatch squared_fidelity(
            up, MPS(ComplexF64, siteinds("S=1/2", 2), fill("Up", 2)))
    end

    @testset "product-frame helpers expose canonical 1-f" begin
        sites = siteinds("S=1/2", 4)
        product = MPS(ComplexF64, sites, fill("Up", 4))

        @test zero_product_amplitude_infidelity(product) == 0.0
        @test product_frame_amplitude_infidelity(product) == 0.0
        @test isapprox(
            product_frame_amplitude_infidelity(product),
            amplitude_infidelity_from_squared(product_frame_fidelity(product));
            atol=1e-14)
    end
end
