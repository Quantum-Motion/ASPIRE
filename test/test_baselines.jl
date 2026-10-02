using ASPIRE
using ITensors
using ITensorMPS
using LinearAlgebra
using MPSCircuits
using Random
using Test

# Squared fidelity of the chi = 2 truncation of the residual after k - 1 MPD layers.
function _mpd_frame_infidelities(psi::MPS, layers::Int)
    work = ITensorMPS.dense(copy(psi))
    out = Float64[]
    for k in 1:layers
        tr2 = ITensorMPS.truncate(deepcopy(work); maxdim=2, cutoff=1e-8)
        f2 = abs2(ITensorMPS.inner(tr2, work) / (norm(tr2) * norm(work)))
        push!(out, 1 - sqrt(clamp(f2, 0.0, 1.0)))
        work, layer, flag = redirect_stdout(devnull) do
            MPSCircuits.generate_layer!(work; tolerance=1e-8, layer_number=k,
                max_bond_dim=global_maxdim(), working_cutoff=global_cutoff())
        end
        (flag || layer === nothing) && break
    end
    return out
end

@testset "MPD baseline" begin
    @testset "framing does not mutate its input" begin
        psi = tfim_snake_state(3, 3; J=1.0, h=4.0)
        work = ITensorMPS.dense(copy(psi))
        work, layer, _ = redirect_stdout(devnull) do
            MPSCircuits.generate_layer!(work; tolerance=1e-8, layer_number=1,
                max_bond_dim=global_maxdim(), working_cutoff=global_cutoff())
        end
        circuit = collect(layer)
        before = copy(circuit)
        framed, fid = MPSCircuits.with_single_qubit_rotations(circuit, work)
        @test circuit == before
        @test length(framed) == length(circuit) + length(psi)
        @test framed[(length(psi)+1):end] == circuit
        @test 0.0 <= fid <= 1.0 + 1e-12
    end

    @testset "unoptimised curve matches the chi = 2 truncation identity" begin
        # The fidelity of the framed circuit after L layers is the squared overlap of the
        # residual after L - 1 layers with its bond-dimension-two truncation. A stale
        # single-qubit frame inside the circuit breaks the identity from layer 2 on.
        psi = tfim_snake_state(3, 3; J=1.0, h=4.0)
        layers = 8
        r = ran_iter_oall_curve(psi; Lmax=layers, sweeps=0, tolerance=1e-8)
        got = [1 - sqrt(clamp(F, 0.0, 1.0)) for F in r.fid][2:end]
        want = _mpd_frame_infidelities(psi, layers)
        @test length(got) == length(want)
        for k in eachindex(want)
            @test isapprox(got[k], want[k]; rtol=1e-6)
        end
    end

    @testset "unoptimised curve is monotone in the gate count" begin
        for (Lx, Ly, h) in ((3, 3, 4.0), (4, 4, 4.0))
            psi = tfim_snake_state(Lx, Ly; J=1.0, h=h)
            r = ran_iter_oall_curve(psi; Lmax=6, sweeps=0, tolerance=1e-8)
            infid = [1 - sqrt(clamp(F, 0.0, 1.0)) for F in r.fid][2:end]
            @test length(infid) >= 2
            for k in 2:length(infid)
                @test infid[k] <= infid[k-1] * (1 + 1e-9)
            end
            @test infid[end] < infid[1]
        end
    end

    @testset "gate counts are two-qubit only" begin
        psi = tfim_snake_state(3, 3; J=1.0, h=4.0)
        N = length(psi)
        r = ran_iter_oall_curve(psi; Lmax=4, sweeps=0, tolerance=1e-8)
        # One staircase layer is N - 1 two-qubit gates; the frame does not inflate the count.
        @test r.gates[2] == N - 1
        @test all(diff(r.gates[2:end]) .== N - 1)
        @test r.tcount[1] == 0.0
        @test issorted(r.tcount)
        @test length(r.circuit) == length(r.two_qubit_circuit) + N
    end

    @testset "optimisation sweeps lower the infidelity at fixed gate count" begin
        psi = tfim_snake_state(3, 3; J=1.0, h=1.0, nsweeps=6)
        plain = ran_iter_oall_curve(psi; Lmax=3, sweeps=0)
        refined = ran_iter_oall_curve(psi; Lmax=3, sweeps=3)
        @test refined.gates == plain.gates
        @test refined.fid[end] >= plain.fid[end] - 1e-10
    end

    @testset "mpd_circuit" begin
        psi = tfim_snake_state(3, 3; J=1.0, h=2.0, nsweeps=6)
        infid = [rudolph_infidelity(mpd_circuit(psi, L), psi) for L in 1:3]
        @test issorted(infid; rev=true)
        @test count(g -> length(g.site_numbers) == 2, mpd_circuit(psi, 2)) ==
            2 * (length(psi) - 1)
    end

    @testset "checkpoint circuits report their own fidelity" begin
        psi = tfim_snake_state(3, 3; J=1.0, h=2.0, nsweeps=6)
        records = mpd_checkpoints(psi; layers=3)
        @test length(records) == 4
        @test records[1].two_qubit_gates == 0
        @test issorted([r.two_qubit_gates for r in records])
        for r in records
            @test isapprox(r.squared_fidelity,
                product_frame_squared_fidelity(ITensorMPS.dense(copy(psi)),
                    filter(g -> length(g.site_numbers) == 2, r.circuit)); atol=1e-8)
        end

        config = ASPIREConfig(; max_depth=3, threshold=0.0, relative_threshold=0.0)
        edges = candidate_pairs(length(psi); max_distance=3)
        ckpts = aspire_checkpoints(psi, config; pairs=edges)
        @test ckpts[1].two_qubit_gates == 0
        @test length(ckpts) >= 2
        @test ckpts[end].squared_fidelity >= ckpts[1].squared_fidelity
    end
end

@testset "circuit conventions" begin
    @testset "prep_from_disentanglers inverts the order" begin
        psi = random_circuit_state(siteinds("S=1/2", 5); gates=8, adjacent=false,
            rng=MersenneTwister(2))
        s = siteinds(psi)
        res = parallel_disentangling_with_gate_log(psi; max_depth=3, threshold=1e-8,
            relative_threshold=0.0)
        circuit = prep_circuit_from_gate_log(res.gate_log, s)
        @test length(circuit) == length(res.gate_log)
        # The inverse circuit applied to psi reproduces the disentangled state.
        residual = disentangle_with_prep_circuit(psi, circuit)
        @test isapprox(abs(inner(residual, res.psi)), 1.0; atol=1e-8)
    end

    @testset "zero_state_fidelity and rudolph_infidelity" begin
        up = MPS(siteinds("S=1/2", 4), fill("Up", 4))
        @test zero_state_fidelity(up) ≈ 1.0
        @test rudolph_infidelity(MPSCircuits.UnitaryGate[], up) == 1.0
    end

    @testset "random_gaussian_mps" begin
        psi = random_gaussian_mps(8; chimax=4, rng=MersenneTwister(1))
        @test length(psi) == 8
        @test maxlinkdim(psi) == 4
        @test norm(psi) ≈ 1.0
        @test maximum(linkdims(psi)) == 4
        again = random_gaussian_mps(8; chimax=4, rng=MersenneTwister(1))
        @test linkdims(again) == linkdims(psi)
        @test array(psi[3]) ≈ array(again[3])      # same seed, same cores
    end

    @testset "aspire_layer_logged is the first layer of the layered driver" begin
        psi = random_circuit_state(siteinds("S=1/2", 6); gates=12, adjacent=false,
            rng=MersenneTwister(5))
        work, gates = aspire_layer_logged(psi)
        ref = parallel_disentangling_with_gate_log(psi; max_depth=1)
        @test length(gates) == length(ref.gate_log)
        @test isapprox(abs(inner(work, ref.psi)), 1.0; atol=1e-8)
        @test total_correlation(work) <= total_correlation(psi) + 1e-9
    end
end
