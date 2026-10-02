using ASPIRE
using ITensors
using ITensorMPS
using Random
using Test

@testset "Target states" begin
    @testset "snake ordering" begin
        # 3x3 grid, boustrophedon order: consecutive chain sites are lattice neighbours.
        idx = snake_index(3, 3)
        @test idx[(1, 1)] == 1 && idx[(3, 1)] == 3
        @test idx[(3, 2)] == 4 && idx[(1, 2)] == 6
        @test length(idx) == 9
        pairs = snake_pairs(3, 3)
        @test length(pairs) == 12
        @test all(i < j for (i, j) in pairs)
        @test maximum(j - i for (i, j) in pairs) == 2 * 3 - 1
    end

    @testset "GHZ block, Bell pairs and cluster state" begin
        ghz = ghz_block_state(6, 2, 5)
        @test isapprox(norm(ghz), 1.0; atol=1e-12)
        @test isapprox(bond_entropy(ghz, 3), 1.0; atol=1e-10)
        @test isapprox(bond_entropy(ghz, 1), 0.0; atol=1e-10)

        decoupled = coupled_bell_pairs(0.0)
        @test isapprox(bond_entropy(decoupled, 2), 0.0; atol=1e-10)
        @test isapprox(bond_entropy(decoupled, 1), 1.0; atol=1e-10)
        coupled = coupled_bell_pairs(0.4)
        @test bond_entropy(coupled, 2) > 0.1

        cluster = cluster_state(6)
        @test maxlinkdim(cluster) == 2
        @test isapprox(total_correlation(cluster), 6.0; atol=1e-8)   # T = N bits
    end

    @testset "TFIM and Heisenberg ground states" begin
        s, H = tfim_setup(6; h=0.5)
        exact, E0 = ed_ground_state(H)
        psi, E = tfim_dmrg(H, s; nsweeps=8, maxdim=[8, 16], cutoff=1e-12, rng=MersenneTwister(1))
        @test isapprox(E, E0; atol=1e-8)
        @test isapprox(energy(psi, H), E0; atol=1e-8)
        @test ed_ground_energy(H) ≈ E0
        @test ed_ground_gap(H) > 0

        heis = heisenberg_snake_state(2, 3; nsweeps=6, maxdim=[8, 16, 32], rng=MersenneTwister(1))
        @test isapprox(norm(heis), 1.0; atol=1e-12)
        # The singlet ground state has zero total magnetisation.
        sz = sum(expect(heis, "Sz"))
        @test abs(sz) < 1e-6
        tfim = tfim_snake_state(2, 3; h=2.0, nsweeps=6, maxdim=[8, 16], rng=MersenneTwister(1))
        @test isapprox(norm(tfim), 1.0; atol=1e-12)
        @test bond_dimension_summary(tfim).maxdim <= 16
    end

    @testset "heavy-hex patch and kicked Ising state" begin
        patch = heavy_hex_patch(1)
        @test length(patch.coordinates) == 12
        @test length(patch.pairs) == 12                 # a 12-cycle
        @test all(i < j for (i, j) in patch.pairs)
        three = heavy_hex_patch(3)
        @test length(three.coordinates) == 30
        @test length(three.pairs) == 32
        # The routing ordering places many chain-adjacent sites on couplers.
        snake = heavy_hex_patch(3; ordering=:snake)
        adjacent_bonds(p) = count(((i, j),) -> j == i + 1, p.pairs)
        @test adjacent_bonds(three) > adjacent_bonds(snake)
        @test_throws ArgumentError heavy_hex_patch(1; ordering=:bogus)

        psi = heavy_hex_kicked_ising_state(1; theta_h=0.4, steps=2)
        @test length(psi) == 12
        @test isapprox(norm(psi), 1.0; atol=1e-10)
        @test total_correlation(psi) > 0.1
        # Without kicks the state is the product state |0...0>.
        zero_steps = graph_kicked_ising_state(4, [(1, 2), (2, 3), (3, 4)]; steps=0)
        @test isapprox(zero_state_fidelity(zero_steps), 1.0; atol=1e-12)
    end
end
