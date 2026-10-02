using ASPIRE
using ITensorMPS
using Random
using Test

@testset "SimConfig global bond-dim / cutoff" begin
    # Save and restore so this testset doesn't leak global state into the others.
    saved_maxdim, saved_cutoff = global_maxdim(), global_cutoff()

    try
        @testset "getters / setters" begin
            set_max_bond_dim!(123)
            set_cutoff!(1e-9)
            @test global_maxdim() == 123
            @test global_cutoff() == 1e-9

            # setters return the new value and coerce types
            @test set_max_bond_dim!(64) == 64
            @test set_cutoff!(1e-12) === 1.0e-12

            # validation
            @test_throws ArgumentError set_max_bond_dim!(0)
            @test_throws ArgumentError set_max_bond_dim!(-5)
            @test_throws ArgumentError set_cutoff!(-1e-3)
        end

        @testset "bond_ramp clamps to ceiling" begin
            @test bond_ramp(100) == [10, 20, 40, 80, 100]
            @test bond_ramp(64) == [10, 20, 40, 64]
            @test bond_ramp(8) == [8]
            @test last(bond_ramp(200)) == 200
        end

        @testset "with_sim_config overrides then restores" begin
            set_max_bond_dim!(64)
            set_cutoff!(1e-12)

            inner = with_sim_config(maxdim=16, cutoff=1e-6) do
                (global_maxdim(), global_cutoff())
            end
            @test inner == (16, 1e-6)
            # restored afterwards
            @test global_maxdim() == 64
            @test global_cutoff() == 1e-12

            # restores even when the block throws
            @test_throws ErrorException with_sim_config(maxdim=4) do
                error("boom")
            end
            @test global_maxdim() == 64
        end

        @testset "global χ is honored end-to-end" begin
            # A tiny Heisenberg ground state should never exceed the global ceiling.
            rng = MersenneTwister(7)
            psi = with_sim_config(maxdim=4, cutoff=1e-12) do
                heisenberg_snake_state(2, 3; nsweeps=4, rng=rng)
            end
            @test maxlinkdim(psi) <= 4
        end
    finally
        set_max_bond_dim!(saved_maxdim)
        set_cutoff!(saved_cutoff)
    end
end
