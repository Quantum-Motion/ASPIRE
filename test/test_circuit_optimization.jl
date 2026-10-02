using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using MPSCircuits
using Random
using Test

@testset "Procrustes optimisation" begin
    @testset "measurement schedule" begin
        @test measurement_schedule(0) == [0]
        sched = measurement_schedule(40; growth=1.15)
        @test sched[1] == 0 && sched[end] == 40
        @test issorted(sched) && allunique(sched)
        @test measurement_schedule(5; growth=2.0) == [0, 1, 2, 4, 5]
        @test_throws ArgumentError measurement_schedule(10; growth=1.0)
    end

    @testset "convergence rule" begin
        ys = [0.5, 0.2, 0.1, 0.0999, 0.09989, 0.0998899]
        at = [0, 1, 2, 3, 4, 5]
        # The last three intervals gained 1e-3, 1e-4 and 1e-6 of their starting value.
        @test patience_converged(ys, at, 2e-3, 1)
        @test patience_converged(ys, at, 2e-3, 3)
        @test !patience_converged(ys, at, 5e-4, 3)
        @test patience_converged(ys, at, 5e-4, 2)
        @test !patience_converged(ys, at, 1e-5, 2)
        @test !patience_converged(ys, at, 1e-9, 1)
        @test !patience_converged(ys, at, nothing, 3)
        # Too short a trace cannot satisfy a patience window.
        @test !patience_converged(ys[1:3], at[1:3], 2e-3, 3)
        # The test is per sweep: one interval of four sweeps may gain four times as much.
        ys2 = [0.1, 0.1 - 4e-8 * 0.1]
        @test patience_converged(ys2, [0, 4], 1e-7, 1)
        @test !patience_converged(ys2, [0, 1], 1e-8, 1)
    end

    @testset "sweeps do not lower the fidelity of a fitted circuit" begin
        psi = tfim_snake_state(2, 3; J=1.0, h=1.0, nsweeps=6, maxdim=[8, 16])
        circuit = mpd_circuit(psi, 2)
        before = rudolph_infidelity(circuit, psi)
        target = ITensorMPS.dense(copy(psi))
        cost = oall_sweep!(target, circuit; sweeps=3, cost=true)
        after = rudolph_infidelity(circuit, psi)
        @test after <= before + 1e-10
        @test cost > 0
        @test oall_sweep!(target, circuit; sweeps=0) == 0.0
        @test oall_sweep!(target, MPSCircuits.UnitaryGate[]; sweeps=2) == 0.0
    end

    @testset "optimize_to_floor" begin
        psi = tfim_snake_state(2, 3; J=1.0, h=1.0, nsweeps=6, maxdim=[8, 16])
        circuit = mpd_circuit(psi, 2)
        r = optimize_to_floor(psi, circuit; rtol=nothing, cap=4, max_bond_dim=64, cutoff=1e-12)
        @test r.sweeps == 4 && r.capped
        @test r.trace_at[1] == 0 && r.trace_at[end] == 4
        @test length(r.trace) == length(r.trace_at)
        @test r.floor <= r.infid0 + 1e-10
        @test r.gates == 2 * (length(psi) - 1)

        converged = optimize_to_floor(psi, circuit; rtol=1e-1, cap=200, patience=1,
            max_bond_dim=64, cutoff=1e-12)
        @test !converged.capped
        @test converged.sweeps < 200
        # The input circuit is left unchanged.
        @test rudolph_infidelity(circuit, psi) ≈ r.infid0 atol = 1e-10
    end

    @testset "product-frame fidelity credits a local basis change" begin
        psi = tfim_snake_state(2, 3; J=1.0, h=4.0, nsweeps=6, maxdim=[8, 16])
        circuit = mpd_circuit(psi, 1)
        fixed = (1 - rudolph_infidelity(circuit, psi))^2
        framed = product_frame_squared_fidelity(psi, circuit)
        @test framed >= fixed - 1e-10
        full = framed_circuit(psi, circuit)
        @test length(full) == length(circuit) + length(psi)
        @test isapprox(abs2(1 - rudolph_infidelity(full, psi)), framed; atol=1e-8)
    end
end
