using ASPIRE
using Test

@testset "Global depolarising noise" begin
    @test global_depolarizing_strength(0.015) ≈ 0.016
    @test global_depolarizing_squared_fidelity(0.9, 0, 1e-3, 12) == 0.9      # no gates, no noise
    p, M, F0, N = 1e-3, 40, 0.9, 12
    q = (1 - 16p / 15)^M
    @test global_depolarizing_squared_fidelity(F0, M, p, N) ≈ q * F0 + (1 - q) / 2^N
    # Many gates drive the register to the maximally mixed state.
    @test global_depolarizing_squared_fidelity(0.9, 10^6, 1e-3, 12) ≈ 2.0^-12 atol = 1e-12
    @test global_depolarizing_infidelity(F0, M, p, N) ≈ 1 - sqrt(q * F0 + (1 - q) / 2^N)
    # More noise or more gates never raises the fidelity of a noiseless-fidelity-one circuit.
    @test global_depolarizing_squared_fidelity(1.0, 20, 2e-3, 12) <
          global_depolarizing_squared_fidelity(1.0, 20, 1e-3, 12)
    @test global_depolarizing_squared_fidelity(1.0, 40, 1e-3, 12) <
          global_depolarizing_squared_fidelity(1.0, 20, 1e-3, 12)

    @testset "routing charge" begin
        @test MPD_ROUTING_OVERHEAD ≈ 1.4
        @test native_gate_count(10) == 10
        @test native_gate_count(10; swaps_per_gate=MPD_SWAPS_PER_GATE) == 14
        @test native_gate_count(29; swaps_per_gate=MPD_SWAPS_PER_GATE) == round(Int, 29 * 1.4)
        # A SWAP priced at three emitted gates.
        @test native_gate_count(10; swaps_per_gate=2.0, swap_gate_cost=3) == 70
        @test native_gate_count(0; swaps_per_gate=MPD_SWAPS_PER_GATE) == 0
    end
end
