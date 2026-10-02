import MPSCircuits
using Random: MersenneTwister

@testset "TCount" begin

    # A gate factory: wrap a 4x4 (or 2x2) matrix as an MPSCircuits gate on the
    # given sites, using the index ordering the KAK decomposition reads.
    function gate_on(sites_vec, positions::Vector{Int}, M::AbstractMatrix)
        idx = [sites_vec[p] for p in positions]
        if length(positions) == 1
            t = ITensors.itensor(ComplexF64.(M), prime(idx[1]), idx[1])
        else
            t = ITensors.itensor(ComplexF64.(reshape(M, 2, 2, 2, 2)),
                prime(idx[2]), prime(idx[1]), idx[2], idx[1])
        end
        return MPSCircuits.UnitaryGate(t, copy(positions), idx)
    end

    "A Haar-random SU(4)."
    function random_su4(rng)
        Q, R = qr(randn(rng, ComplexF64, 4, 4))
        M = Matrix(Q) * Diagonal(sign.(diag(R)))
        return M / (det(M))^(1 / 4)
    end

    function random_su2(rng)
        Q, R = qr(randn(rng, ComplexF64, 2, 2))
        M = Matrix(Q) * Diagonal(sign.(diag(R)))
        return M / sqrt(det(M))
    end

    _rz_test(theta) = ComplexF64[exp(-im * theta / 2) 0; 0 exp(im * theta / 2)]

    rng = MersenneTwister(1234)

    @testset "angle classes" begin
        cls = ASPIRE.TCount._angle_class
        @test cls(0.0) === :clifford
        @test cls(pi / 2) === :clifford
        @test cls(-3pi / 2) === :clifford
        @test cls(pi / 4) === :exact_t
        @test cls(-5pi / 4) === :exact_t
        @test cls(0.3) === :approx
        # A nearly Clifford angle is not Clifford.
        @test cls(pi / 2 + 1e-6) === :approx
    end

    @testset "synthesis models" begin
        synth = ASPIRE.TCount._synthesis_t_gates
        @test synth(GRIDSYNTH, 1e-3) ≈ 3.054 * log2(1e3) + 1.83
        # Mean T counts measured with pygridsynth, within the per-angle scatter.
        for (eps, measured) in ((1e-3, 32.07), (1e-6, 62.66), (1e-10, 103.22))
            @test isapprox(synth(GRIDSYNTH, eps), measured; rtol=0.05)
        end
        @test_throws ArgumentError synth(GRIDSYNTH, 1.5)

        # A general block: three calls for gridsynth, one for trasyn.
        bs = ASPIRE.TCount._block_synthesis
        general = [0.3, 0.7, 1.1]
        @test bs(general, GRIDSYNTH) == (calls=3, exact_t=0)
        @test bs(general, TRASYN) == (calls=1, exact_t=0)
        # Angles that are multiples of pi/4 are exactly synthesizable.
        exact = [pi / 4, -pi / 4]
        @test bs(exact, GRIDSYNTH) == (calls=0, exact_t=2)
        @test bs(exact, TRASYN) == (calls=0, exact_t=2)
    end

    @testset "one SU(4) on |00>" begin
        sites = siteinds("Qubit", 2)
        circuit = [gate_on(sites, [1, 2], random_su4(rng))]
        # The raw KAK export has 15 rotations; the two incoming one-qubit
        # unitaries act on |0> and lose their leading Rz, leaving 13.
        @test rotation_count(circuit) == 13
        @test all(b -> length(b) <= 3, ASPIRE.TCount._rotation_blocks(circuit))
    end

    @testset "chain of SU(4)s reaches 9 rotations per gate" begin
        # On a staircase every gate after the first on a wire has its incoming
        # one-qubit unitaries fused into its predecessor's outgoing ones, which
        # is the 15 -> 9 reduction of arXiv:2603.11156.
        N = 6
        sites = siteinds("Qubit", N)
        layers = 4
        circuit = MPSCircuits.UnitaryGate[]
        for _ in 1:layers, i in 1:(N - 1)
            push!(circuit, gate_on(sites, [i, i + 1], random_su4(rng)))
        end
        gates = layers * (N - 1)
        # 9 per gate, plus the 3 outgoing rotations on each of the N wires,
        # minus the leading Rz saved on each wire at |0...0>.
        @test rotation_count(circuit) == 9 * gates + 3 * N - N
        @test rotation_count(circuit) < 15 * gates
        @test isapprox(rotation_count(circuit) / (15 * gates), 0.6; atol=0.05)
    end

    @testset "trasyn vs gridsynth" begin
        N = 5
        sites = siteinds("Qubit", N)
        circuit = [gate_on(sites, [i, i + 1], random_su4(rng)) for i in 1:(N - 1)]

        grid = t_count(circuit; epsilon=1e-3, model=GRIDSYNTH)
        tra = t_count(circuit; epsilon=1e-3, model=TRASYN)

        # Gridsynth pays for every rotation: it is the per-call cost times the
        # rotation count (random angles are never exactly synthesizable).
        per_call = 3.054 * log2(1e3) + 1.83
        @test grid ≈ rotation_count(circuit) * per_call
        # Block synthesis merges rotations, but not the three core rotations
        # that sit alone between CNOTs, so the saving is below threefold.
        @test tra < grid
        @test 1.0 < grid / tra < 3.0
    end

    @testset "Clifford and product gates are free" begin
        sites = siteinds("Qubit", 3)
        CNOT = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

        # A circuit of Cliffords costs nothing.
        clifford = [gate_on(sites, [1, 2], CNOT), gate_on(sites, [2, 3], CNOT)]
        @test rotation_count(clifford) == 0
        @test t_count(clifford; epsilon=1e-3) == 0.0

        # A Clifford entangler is free but still a barrier: the rotations on
        # either side of it cannot fuse.
        split = [gate_on(sites, [1], _rz_test(0.4)), gate_on(sites, [1, 2], CNOT),
                 gate_on(sites, [1], _rz_test(-0.4))]
        @test rotation_count(split; initial_state=:arbitrary) == 2

        # A two-qubit gate that is a product raises no barrier. Each wire holds
        # one SU(2) acting on |0>, with two rotations apiece.
        A, B = random_su2(rng), random_su2(rng)
        product = [gate_on(sites, [1, 2], kron(A, B))]
        @test rotation_count(product) == 4

        # The same two SU(2)s applied as one-qubit gates cost the same.
        singles = [gate_on(sites, [1], A), gate_on(sites, [2], B)]
        @test rotation_count(singles) == rotation_count(product)
    end

    @testset "the |0...0> boundary" begin
        sites = siteinds("Qubit", 2)
        circuit = [gate_on(sites, [1, 2], random_su4(rng))]
        # Dropping the |0> assumption costs exactly the two leading Rz.
        @test rotation_count(circuit; initial_state=:arbitrary) ==
            rotation_count(circuit) + 2
        @test_throws ArgumentError rotation_count(circuit; initial_state=:bogus)

        # A wire whose first block leaves it on a pole is free: the trailing Rz
        # is then a phase on |1>, not a relative phase.
        flip = ComplexF64[0 1; 1 0]
        phased_flip = [gate_on(sites, [1],
            ComplexF64[exp(-im * 0.35) 0; 0 exp(im * 0.35)] * flip)]
        @test rotation_count(phased_flip) == 0
        @test rotation_count(phased_flip; initial_state=:arbitrary) == 1
        # X itself is Clifford, so it is free either way.
        @test rotation_count([gate_on(sites, [1], flip)];
            initial_state=:arbitrary) == 0
    end

    @testset "routing invariance" begin
        # SWAPs are Clifford, so a long-range gate costs the same as a local one.
        sites = siteinds("Qubit", 6)
        U, V = random_su4(rng), random_su4(rng)
        local_circuit = [gate_on(sites, [1, 2], U), gate_on(sites, [3, 4], V)]
        distant = [gate_on(sites, [1, 6], U), gate_on(sites, [3, 4], V)]
        @test rotation_count(local_circuit) == rotation_count(distant)
        @test t_count(local_circuit; epsilon=1e-3) ≈ t_count(distant; epsilon=1e-3)
    end

    @testset "error budget" begin
        N = 5
        sites = siteinds("Qubit", N)
        circuit = [gate_on(sites, [i, i + 1], random_su4(rng)) for i in 1:(N - 1)]

        # A total-error budget is split evenly over the synthesis calls.
        calls = rotation_count(circuit)
        budget = 1e-4
        @test t_count(circuit; total_error=budget, model=GRIDSYNTH) ≈
            t_count(circuit; epsilon=budget / calls, model=GRIDSYNTH)

        @test_throws ArgumentError t_count(circuit)
        @test_throws ArgumentError t_count(circuit; epsilon=1e-3, total_error=1e-3)
    end

    @testset "a compiled preparation circuit" begin
        N = 6
        sites = siteinds("Qubit", N)
        state = random_mps(ComplexF64, sites; linkdims=4)
        normalize!(state)
        circuit = MPSCircuits.compile_mps_circuit(
            state, MPSCircuits.DecomposeAllAnalytical();
            n_layers_max=3, backend=:cpu, precision=:fp64)

        rotations = rotation_count(circuit)
        gates = count(g -> length(g.site_numbers) == 2, circuit)
        @test rotations > 0
        @test t_count(circuit; total_error=1e-3) > 0
        # Fusion never exceeds the 15 rotations per gate of the unfused export.
        @test rotations <= 15 * gates + 3 * N
    end
end
