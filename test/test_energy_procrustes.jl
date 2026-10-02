using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using MPSCircuits
using Random
using Test

@testset "Energy refinement" begin
    function _gate(U::AbstractMatrix, sites, i::Int, j::Int)
        MPSCircuits.UnitaryGate(two_qubit_gate(ComplexF64.(U), sites[i], sites[j]))
    end

    function _matrix(gate)
        reshape(
            array(MPSCircuits.tensor(gate), prime.(gate.site_indices)..., gate.site_indices...),
            4,
            4,
        )
    end

    sites = siteinds("S=1/2", 4)
    H = tfim_hamiltonian(sites; h=0.7)
    rng = MersenneTwister(91)
    circuit = MPSCircuits.UnitaryGate[
        _gate(haar_unitary(4, rng), sites, 1, 2),
        _gate(haar_unitary(4, rng), sites, 3, 4),
        _gate(haar_unitary(4, rng), sites, 2, 3),
    ]
    zero = MPS(sites, fill("0", length(sites)))

    @testset "exact finite-system state matches exact energy" begin
        exact_state, exact_energy = ed_ground_state(H)
        @test isapprox(exact_energy, ed_ground_energy(H); atol=1e-12)
        @test isapprox(energy(exact_state, H), exact_energy; atol=1e-11)
        @test isapprox(real(inner(exact_state, exact_state)), 1.0; atol=1e-12)
    end

    @testset "coordinate sweeps do not raise the energy and keep gates unitary" begin
        original = deepcopy(circuit)
        result = energy_mps_coordinate_refine(
            H,
            circuit;
            n_sweeps=2,
            max_backtracks=5,
            cutoff=1e-12,
            maxdim=32,
        )
        prepared = MPSCircuits.apply_circuit(result.circuit, zero; cutoff=1e-12, maxdim=32)
        @test isapprox(energy(prepared, H), result.energies[end]; atol=1e-9)
        @test all(diff(result.energies) .<= 1e-9)
        @test length(result.sweeps) == 2
        @test all(0 <= sweep.accepted <= length(circuit) for sweep in result.sweeps)
        @test all(
            isapprox(_matrix(gate)' * _matrix(gate), Matrix{ComplexF64}(I, 4, 4); atol=1e-9)
            for gate in result.circuit
        )
        # The input circuit is left untouched.
        @test all(
            isapprox(array(MPSCircuits.tensor(circuit[k])), array(MPSCircuits.tensor(original[k])))
            for k in eachindex(circuit)
        )
    end

    @testset "zero sweeps return the input circuit" begin
        result = energy_mps_coordinate_refine(H, circuit; n_sweeps=0, cutoff=1e-12, maxdim=32)
        @test isempty(result.sweeps)
        @test length(result.energies) == 1
        @test !result.stopped_early
        @test all(
            isapprox(_matrix(result.circuit[k]), _matrix(circuit[k])) for k in eachindex(circuit)
        )
    end

    @testset "single gate update is accepted only if the energy does not rise" begin
        updated = deepcopy(circuit)
        update = optimize_energy_gate_mps!(H, updated, 2; cutoff=1e-12, maxdim=32)
        @test update.after <= update.before + 1e-12
        prepared = MPSCircuits.apply_circuit(updated, zero; cutoff=1e-12, maxdim=32)
        @test isapprox(energy(prepared, H), update.after; atol=1e-9)
    end
end
