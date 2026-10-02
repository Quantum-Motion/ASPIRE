"""
    ModelStates

Target states: ground states of the transverse-field Ising model (TFIM) and the Heisenberg model
on chains and snake-mapped square lattices, computed by DMRG; the kicked transverse-field Ising
state on a graph; and the heavy-hex lattice fragment on which the latter is defined.

The `maxdim` and `cutoff` defaults read the global simulation configuration
([`global_maxdim`](@ref), [`global_cutoff`](@ref)); DMRG uses the warm-up schedule
[`bond_ramp`](@ref) capped at the global bond dimension. All of them can be overridden per call.
"""
module ModelStates

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

using ..SimConfig: global_maxdim, global_cutoff
using ..Disentangling: apply_two_qubit_unitary
using ..HardwareCost: routing_ordering

export tfim_hamiltonian, tfim_setup, energy, tfim_dmrg,
    ed_ground_energy, ed_ground_state, ed_ground_gap,
    tfim_snake_state, heisenberg_snake_state,
    snake_index, snake_pairs, bond_ramp,
    heavy_hex_patch, apply_one_qubit_unitary,
    graph_kicked_ising_state, heavy_hex_kicked_ising_state

"""
    tfim_hamiltonian(s; h=2.0) -> MPO

Open-chain transverse-field Ising Hamiltonian `H = -sum_j Sz_j Sz_{j+1} - h sum_j Sx_j` on the
sites `s`. With spin operators `S = sigma/2` the usual Pauli field ratio is `g = 2h`, so the
chain is critical at `h = 0.5` and the square lattice at `h ~ 1.52`; `h = 2` is paramagnetic on
both.
"""
function tfim_hamiltonian(s; h::Float64=2.0)
    N = length(s)
    os = OpSum()
    for j in 1:(N - 1); os += -1.0, "Sz", j, "Sz", j + 1; end
    for j in 1:N;       os += -h,   "Sx", j;               end
    return MPO(os, s)
end

"`(s, H)`: spin-1/2 sites and the open-chain TFIM Hamiltonian on `N` sites at field `h`."
function tfim_setup(N::Int; h::Float64=2.0)
    s = siteinds("S=1/2", N)
    return (s=s, H=tfim_hamiltonian(s; h=h))
end

"Real part of the energy `<psi|H|psi>` of the MPS `psi` under the MPO `H`."
energy(psi::MPS, H::MPO) = real(inner(psi', H, psi))

"""
    bond_ramp(ceiling)

Warm-up bond-dimension schedule `[10, 20, 40, 80, ceiling]` clamped at `ceiling`, with duplicates
removed, for the `maxdim` argument of DMRG.
"""
bond_ramp(ceiling::Integer) = unique(min.([10, 20, 40, 80, Int(ceiling)], Int(ceiling)))

"""
    tfim_dmrg(H, s; nsweeps=10, maxdim=bond_ramp(global_maxdim()), cutoff=global_cutoff(),
              rng=Random.default_rng()) -> (psi, E)

DMRG ground state of `H` from a random bond-dimension-4 start drawn from `rng`. The sweep count
and the `maxdim` schedule set the accuracy, from crude (one sweep, `maxdim=[2]`) to converged.
Returns the normalised MPS and the DMRG energy.
"""
function tfim_dmrg(H::MPO, s; nsweeps::Int=10, maxdim=bond_ramp(global_maxdim()),
        cutoff::Float64=global_cutoff(), rng::AbstractRNG=Random.default_rng())
    psi0 = random_mps(rng, s; linkdims=4)
    E, psi = dmrg(H, psi0;
        nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
    return normalize!(psi), E
end

# Eigendecomposition of a small MPO (N <= 12) contracted to a dense matrix.
function _ed_ground_eigensystem(H::MPO)
    length(H) > 12 && throw(ArgumentError("exact diagonalisation is for N <= 12; got N=$(length(H))."))
    ITensors.disable_warn_order()
    Hfull = prod(H)
    ITensors.reset_warn_order()
    s = filter(i -> ITensors.plev(i) == 0, inds(Hfull))
    M = Array(Hfull, prime.(s)..., s...)
    d = prod(dim, s)
    factorization = eigen(Hermitian(reshape(M, d, d)))
    return (values=real.(factorization.values), vectors=factorization.vectors, sites=s)
end

"Exact ground-state energy of the MPO `H` by dense diagonalisation (`N <= 12`)."
ed_ground_energy(H::MPO) = first(_ed_ground_eigensystem(H).values)

"""
    ed_ground_gap(H) -> Float64

Exact gap `E_1 - E_0` of the MPO `H` by dense diagonalisation (`N <= 12`). An exponentially small
gap means that the returned ground state is an arbitrary member of a near-degenerate multiplet
(for example the cat state of the ordered TFIM), so overlaps with it are not a meaningful accuracy
measure even when the energy is converged.
"""
function ed_ground_gap(H::MPO)
    values = _ed_ground_eigensystem(H).values
    length(values) < 2 && return Inf
    return values[2] - values[1]
end

"""
    ed_ground_state(H) -> (psi, energy)

Exact ground state of the MPO `H` (`N <= 12`) as an MPS on the sites of `H`, with its energy.
"""
function ed_ground_state(H::MPO)
    exact = _ed_ground_eigensystem(H)
    dimensions = dim.(exact.sites)
    tensor = ITensor(
        reshape(exact.vectors[:, 1], dimensions...),
        exact.sites...,
    )
    psi = MPS(
        tensor,
        exact.sites;
        cutoff=0.0,
        maxdim=prod(dimensions),
    )
    return normalize!(psi), first(exact.values)
end

"""
    snake_index(Lx, Ly) -> Dict{Tuple{Int,Int},Int}

Chain position of each site `(x, y)` of an `Lx x Ly` grid (`x` the column, `y` the row) under the
snake ordering: rows are laid out in sequence with every other row reversed, so consecutive chain
sites are always lattice neighbours.
"""
function snake_index(Lx::Int, Ly::Int)
    idx = Dict{Tuple{Int,Int},Int}()
    n = 0
    for y in 1:Ly, x in (isodd(y) ? (1:Lx) : (Lx:-1:1))
        idx[(x, y)] = (n += 1)
    end
    return idx
end

"""
    snake_pairs(Lx, Ly; periodic_y=false) -> Vector{Tuple{Int,Int}}

Nearest-neighbour bonds of the `Lx x Ly` grid as ordered chain pairs under [`snake_index`](@ref).
Horizontal bonds become nearest-neighbour chain pairs and vertical bonds pairs at distance up to
`2 Lx - 1`. With `periodic_y=true` the rows wrap around.
"""
function snake_pairs(Lx::Int, Ly::Int; periodic_y::Bool=false)
    idx = snake_index(Lx, Ly)
    pairs = Tuple{Int,Int}[]
    for y in 1:Ly, x in 1:Lx
        x < Lx && push!(pairs, minmax(idx[(x, y)], idx[(x + 1, y)]))
        y < Ly && push!(pairs, minmax(idx[(x, y)], idx[(x, y + 1)]))
        periodic_y && Ly > 2 && y == Ly && push!(pairs, minmax(idx[(x, y)], idx[(x, 1)]))
    end
    return pairs
end

"""
    tfim_snake_state(Lx, Ly; J=1.0, h=2.0, periodic_y=false, nsweeps=8,
                     maxdim=bond_ramp(global_maxdim()), cutoff=global_cutoff(), rng) -> MPS

DMRG ground state of `H = -J sum_<ab> Sz_a Sz_b - h sum_a Sx_a` on an `Lx x Ly` square lattice
mapped to a chain by [`snake_index`](@ref).
"""
function tfim_snake_state(Lx::Int, Ly::Int; J::Float64=1.0, h::Float64=2.0,
        periodic_y::Bool=false, nsweeps::Int=8, maxdim=bond_ramp(global_maxdim()),
        cutoff::Real=global_cutoff(), rng::AbstractRNG=Random.default_rng())
    sites = siteinds("S=1/2", Lx * Ly)
    os = OpSum()
    for (a, b) in snake_pairs(Lx, Ly; periodic_y=periodic_y)
        os += -J, "Sz", a, "Sz", b
    end
    for j in 1:(Lx * Ly)
        os += -h, "Sx", j
    end
    _, psi = dmrg(MPO(os, sites), random_mps(rng, sites; linkdims=4);
        nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
    return normalize!(psi)
end

"""
    heisenberg_snake_state(Lx, Ly; J=1.0, Delta=1.0, periodic_y=false, nsweeps=8,
                           maxdim=bond_ramp(global_maxdim()), cutoff=global_cutoff(), rng) -> MPS

DMRG ground state of the spin-1/2 XXZ model
`H = J sum_<ab> (Sx_a Sx_b + Sy_a Sy_b + Delta Sz_a Sz_b)` on an `Lx x Ly` square lattice in the
snake ordering of [`snake_index`](@ref). `Delta = 1` is the isotropic antiferromagnetic
Heisenberg model. Vertical lattice bonds become chain pairs at distance up to `2 Lx - 1`, which
gives the state its long-range entanglement.
"""
function heisenberg_snake_state(Lx::Int, Ly::Int;
        J::Real=1.0, Delta::Real=1.0, periodic_y::Bool=false,
        nsweeps::Int=8, maxdim=bond_ramp(global_maxdim()), cutoff::Real=global_cutoff(),
        rng::AbstractRNG=Random.default_rng())
    sites = siteinds("S=1/2", Lx * Ly)
    os = OpSum()
    for (a, b) in snake_pairs(Lx, Ly; periodic_y=periodic_y)
        os += J / 2, "S+", a, "S-", b
        os += J / 2, "S-", a, "S+", b
        os += J * Delta, "Sz", a, "Sz", b
    end
    H = MPO(os, sites)
    psi0 = random_mps(rng, sites; linkdims=4)
    _, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
    return normalize!(psi)
end

"""
    heavy_hex_patch(cells=1; ordering=:routing, routing_iterations=200_000, routing_seed)
        -> (pairs, coordinates)

Open-boundary heavy-hex fragment: a row of `cells` edge-sharing hexagons with every edge
decorated by a midpoint qubit. One cell is the 12-qubit motif (6 corner and 6 link qubits);
the sizes are `N = 12, 21, 30, 39, ...`. Returns the device couplers `pairs` as 1-based site-index
tuples and the lattice `coordinates` of each MPS site, both in the chosen ordering.

`ordering` lays the lattice sites onto the MPS chain:
- `:routing` minimises the native cost of one MPS staircase layer ([`routing_ordering`](@ref)):
  a Hamiltonian path where one exists, otherwise simulated annealing started from the snake
  layout. The annealing is seeded by `routing_seed`, so the ordering is reproducible.
- `:snake` traverses the rows in alternating directions.
- `:rowmajor` sorts by `(y, x)`.

Heavy-hex is bipartite, with every coupler joining a corner to a link qubit, so the snake layout
places most consecutive sites in the same class, where adjacency is impossible. At three cells
only 6 of 29 snake bonds are couplers.
"""
function heavy_hex_patch(cells::Int=1; ordering::Symbol=:routing,
        routing_iterations::Int=200_000, routing_seed::Int=Int(0x13527c6))
    cells >= 1 || throw(ArgumentError("cells must be positive."))
    ordering in (:routing, :snake, :rowmajor) || throw(ArgumentError(
        "unknown ordering `$ordering`; expected :routing, :snake or :rowmajor"))
    raw_vertices = NTuple{2,Float64}[]
    raw_edges = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    key(x, y) = (round(x; digits=8), round(y; digits=8))
    angles = (π / 6, π / 2, 5π / 6, 7π / 6, 3π / 2, 11π / 6)
    for cell in 0:(cells - 1)
        cx = cell * sqrt(3)
        vertices = [key(cx + cos(a), sin(a)) for a in angles]
        append!(raw_vertices, vertices)
        append!(raw_edges, [(vertices[k], vertices[mod1(k + 1, 6)]) for k in 1:6])
    end
    base_vertices = unique(raw_vertices)
    base_edges = unique(map(raw_edges) do (a, b)
        isless(a, b) ? (a, b) : (b, a)
    end)
    decorated_edges = Tuple{NTuple{2,Float64},NTuple{2,Float64}}[]
    all_coordinates = copy(base_vertices)
    for (a, b) in base_edges
        midpoint = key((a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
        push!(all_coordinates, midpoint)
        push!(decorated_edges, (a, midpoint), (midpoint, b))
    end
    coordinates = sort!(unique(all_coordinates); by=xy -> (xy[2], xy[1]))
    if ordering !== :rowmajor
        # Boustrophedon: reverse the x-order of every other y-row. This is the starting
        # point of the annealing under `:routing`.
        rows = Dict{Float64,Vector{NTuple{2,Float64}}}()
        for xy in coordinates
            push!(get!(rows, xy[2], NTuple{2,Float64}[]), xy)
        end
        snaked = NTuple{2,Float64}[]
        for (r, ykey) in enumerate(sort!(collect(keys(rows))))
            row = sort!(rows[ykey]; by=xy -> xy[1])
            iseven(r) && reverse!(row)
            append!(snaked, row)
        end
        coordinates = snaked
    end
    index = Dict(xy => k for (k, xy) in enumerate(coordinates))
    pairs = unique(sort!([minmax(index[a], index[b]) for (a, b) in decorated_edges]))
    if ordering === :routing
        best = routing_ordering(length(coordinates), pairs; convention=:running,
            iterations=routing_iterations, rng=MersenneTwister(routing_seed))
        coordinates = coordinates[best.order]
        pairs = sort!(unique(best.pairs))
    end
    return (pairs=pairs, coordinates=coordinates)
end

"Apply the 2x2 unitary `U` to one site of an MPS and renormalise."
function apply_one_qubit_unitary(psi::MPS, site::Int, U::AbstractMatrix)
    size(U) == (2, 2) || throw(ArgumentError("Expected a 2x2 unitary."))
    s = siteinds(psi)[site]
    G = ITensor(ComplexF64, prime(s), s)
    for a in 1:2, b in 1:2
        G[prime(s) => a, s => b] = U[a, b]
    end
    return normalize!(apply(G, psi))
end

"""
    graph_kicked_ising_state(N, pairs; theta_h=0.6, steps=2) -> MPS

Kicked-Ising state on the connectivity graph `pairs`, following the gate convention of the IBM
Eagle utility experiment (Kim et al., Nature 618, 500 (2023)):

    U(theta_h) = prod_edges exp(i pi/4 Z_i Z_j) prod_sites exp(-i theta_h/2 X_i),

applied for `steps` Trotter steps to the all-up product state.
"""
function graph_kicked_ising_state(N::Int, pairs; theta_h::Real=0.6, steps::Int=2)
    steps >= 0 || throw(ArgumentError("steps must be non-negative."))
    sites = siteinds("S=1/2", N)
    psi = MPS(sites, fill("Up", N))
    X = ComplexF64[0 1; 1 0]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Ux = cos(theta_h / 2) * I2 - im * sin(theta_h / 2) * X
    phase_same, phase_opposite = cis(π / 4), cis(-π / 4)
    Uzz = Diagonal(ComplexF64[
        phase_same, phase_opposite, phase_opposite, phase_same
    ])
    for _ in 1:steps
        for site in 1:N
            psi = apply_one_qubit_unitary(psi, site, Ux)
        end
        for (i, j) in pairs
            psi = apply_two_qubit_unitary(psi, i, j, Uzz)
        end
    end
    return normalize!(psi)
end

"""
    heavy_hex_kicked_ising_state(cells=1; theta_h=0.6, steps=2, ordering=:routing) -> MPS

Kicked-Ising state ([`graph_kicked_ising_state`](@ref)) on the open heavy-hex patch of `cells`
hexagons in the site ordering of [`heavy_hex_patch`](@ref).
"""
function heavy_hex_kicked_ising_state(cells::Int=1; theta_h::Real=0.6,
        steps::Int=2, ordering::Symbol=:routing)
    patch = heavy_hex_patch(cells; ordering=ordering)
    return graph_kicked_ising_state(length(patch.coordinates), patch.pairs;
        theta_h=theta_h, steps=steps)
end

end # module
