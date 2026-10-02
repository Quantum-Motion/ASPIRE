"""
    OptimizerCost

Classical cost of one Procrustes sweep (`MPSCircuits.replace_gates!` with a telescoping environment)
in elementary MPS operations, so that protocols are compared at equal classical compute
rather than at an equal number of sweeps. A sweep applies each gate twice, once to build the cached
bra chain and once to roll the ket forward, and `ITensorMPS.apply` realises a gate on sites `d`
apart by moving them together and back, which is `2d - 1` two-site operations of cost `O(chi^3)`.
Hence `site_ops = 2 sum_g (2 span_g - 1)` and `chi3_ops = 2 sum_g (2 span_g - 1) chi_g^3`, or the
bond-resolved count that charges each operation at the bond dimensions it is performed at. This is
a classical simulation cost; the device cost of a circuit, including routing, is `HardwareCost`.
"""
module OptimizerCost

using ITensors
using ITensorMPS
import MPSCircuits

using ..SimConfig: global_cutoff, global_maxdim

export gate_spans, gate_pairs, sweep_site_ops,
    bra_chain_bond_dims, bra_chain_link_dims, gate_bond_ops, bond_op_weight,
    bond_resolved_chi3_ops, sweep_cost_model

"Site spans `|i-j|` of the two-qubit gates of `circuit`, in circuit order."
gate_spans(circuit) =
    [abs(g.site_numbers[2] - g.site_numbers[1]) for g in circuit if length(g.site_numbers) == 2]

"Ordered site pairs `(i, j)`, `i < j`, of the two-qubit gates of `circuit`, in circuit order."
gate_pairs(circuit) =
    [minmax(g.site_numbers[1], g.site_numbers[2]) for g in circuit if length(g.site_numbers) == 2]

# The maximum-bond-dimension model charges every elementary operation of a span-d gate the same
# weight `maxlinkdim(state)^3`, which is exact only for a flat bond profile. A long-range gate
# leaves a peaked profile, and charging the peak to every operation overstates the cost of the
# protocol with the most peaked profile, the long-range one. The bond-resolved count below
# charges each operation at the bond dimensions it is performed at.

"""
    gate_bond_ops(i, j) -> Vector{Pair{Int,Int}}

Which bonds a two-qubit gate on sites `i < j` touches, and how many elementary two-site
operations land on each.

`ITensorMPS.apply` realises the gate by walking site `j` down to `i+1` (`d-1` swaps at bonds
`j-1 … i+1`), applying the gate at bond `i`, then walking it back (`d-1` swaps at the same
bonds). So bond `i` carries one operation and every interior bond carries two, summing to the
familiar `2d-1`.
"""
function gate_bond_ops(i::Int, j::Int)
    a, b = minmax(i, j)
    ops = Pair{Int,Int}[a => 1]
    for bond in (a + 1):(b - 1)
        push!(ops, bond => 2)
    end
    return ops
end

"""
    bond_op_weight(link_dims, b) -> Float64

Arithmetic weight of one elementary two-site operation at bond `b`, given the chain's link
dimensions.

The operation contracts sites `b` and `b+1` into a `(chi_{b-1} d) x (d chi_{b+1})` matrix and
factorises it, which costs `O(m n min(m, n))`; dropping the physical dimension (a constant
`d^3 = 8` shared by every operation and every protocol, so it cannot shift a comparison) leaves
`chi_L * chi_R * min(chi_L, chi_R)`. On a flat profile this is exactly `chi^3`, so the
bond-resolved model reduces to the maximum-bond-dimension model there and differs only where the
profile is peaked or near a chain end.
"""
function bond_op_weight(link_dims::AbstractVector{<:Integer}, b::Int)
    nb = length(link_dims)                      # bonds 1..nb, sites 1..nb+1
    chi_l = b <= 1 ? 1 : Int(link_dims[b - 1])
    chi_r = b >= nb ? 1 : Int(link_dims[b + 1])
    return float(chi_l) * float(chi_r) * float(min(chi_l, chi_r))
end

"""
    bra_chain_link_dims(target, circuit; cutoff, max_bond_dim) -> Vector{Vector{Int}}

Full bond profile the sweep sees, gate by gate, in circuit order: the bond-resolved
counterpart of [`bra_chain_bond_dims`](@ref), which keeps only each profile's maximum.
"""
function bra_chain_link_dims(target::MPS, circuit;
        cutoff::Real=global_cutoff(), max_bond_dim::Int=global_maxdim())
    isempty(circuit) && return Vector{Int}[]
    bra = ITensorMPS.dense(copy(target))
    profiles = Vector{Int}[collect(Int, linkdims(bra))]
    for g in Iterators.reverse(@view circuit[2:end])
        bra = MPSCircuits.apply_gate(MPSCircuits.dagger(g), bra; mixed_device=:coerce,
            cutoff=cutoff, maxdim=max_bond_dim)
        push!(profiles, collect(Int, linkdims(bra)))
    end
    return reverse(profiles)
end

"""
    bond_resolved_chi3_ops(circuit, profiles) -> Float64

`chi3_ops` counted at the bond dimensions each operation is performed at:
`2 sum_g sum_{b in path(g)} mult_b chi_L chi_R min(chi_L, chi_R)`. The leading 2 is the
bra-chain plus ket-roll-forward factor of the maximum-bond-dimension model.
"""
function bond_resolved_chi3_ops(circuit, profiles::AbstractVector{<:AbstractVector{<:Integer}})
    pairs = gate_pairs(circuit)
    length(pairs) == length(profiles) ||
        throw(ArgumentError("got $(length(pairs)) gates but $(length(profiles)) bond profiles"))
    total = 0.0
    for (pair, ld) in zip(pairs, profiles)
        for (bond, mult) in gate_bond_ops(pair[1], pair[2])
            total += mult * bond_op_weight(ld, bond)
        end
    end
    return 2 * total
end

"""
    sweep_site_ops(circuit) -> Int

Elementary two-site MPS operations in one `replace_gates!` sweep over `circuit`, ignoring
bond dimension: `2 Σ_g (2·span_g − 1)`.

For a nearest-neighbour circuit this is just `2G`, so the model reduces to the gate count;
every unit of span above one is overhead a gate count cannot see.
"""
sweep_site_ops(circuit) = 2 * sum(2s - 1 for s in gate_spans(circuit); init=0)

"""
    bra_chain_bond_dims(target, circuit; cutoff, max_bond_dim) -> Vector{Int}

Working bond dimension the sweep sees, gate by gate, in circuit order.

Replays the bra chain of the telescoping environment, the daggered gates applied to `target`
from the end of the circuit backwards, and records `maxlinkdim` after each. Long-range gates
are the ones that grow the bond dimension, so this is where the `chi^3` factor separates
protocols at sizes where the bond dimension is not yet saturated.
"""
function bra_chain_bond_dims(target::MPS, circuit;
        cutoff::Real=global_cutoff(), max_bond_dim::Int=global_maxdim())
    isempty(circuit) && return Int[]
    bra = ITensorMPS.dense(copy(target))
    chis = Int[maxlinkdim(bra)]
    for g in Iterators.reverse(@view circuit[2:end])
        bra = MPSCircuits.apply_gate(MPSCircuits.dagger(g), bra; mixed_device=:coerce,
            cutoff=cutoff, maxdim=max_bond_dim)
        push!(chis, maxlinkdim(bra))
    end
    return reverse(chis)
end

"""
    sweep_cost_model(target, circuit; cutoff, max_bond_dim, chis=nothing, profiles=nothing,
                     bond_resolved=false) -> NamedTuple

Cost of one Procrustes sweep over `circuit` against `target`, as
`(ngates, site_ops, chi3_ops, chi3_ops_maxchi, chi3_ops_bond, mean_span, max_chi, spans, chis,
bond_resolved)`.

`site_ops` needs no state. The `chi3` counts need one replay of the bra chain, about half a
sweep; pass a precomputed `chis` or `profiles` to skip it when the circuit has not
changed. Both counts are always returned, as `chi3_ops_maxchi` (maximum bond dimension of each
bra) and `chi3_ops_bond` (bond-resolved), and `bond_resolved` selects which one is reported as
`chi3_ops`. The paper's budgets use the bond-resolved count (`bond_resolved=true`); the default
is the maximum-bond-dimension model.
"""
function sweep_cost_model(target::MPS, circuit;
        cutoff::Real=global_cutoff(), max_bond_dim::Int=global_maxdim(), chis=nothing,
        profiles=nothing, bond_resolved::Bool=false)
    spans = gate_spans(circuit)
    isempty(spans) && return (ngates=0, site_ops=0, chi3_ops=0.0, chi3_ops_maxchi=0.0,
                              chi3_ops_bond=0.0, mean_span=0.0, max_chi=0,
                              spans=spans, chis=Int[], bond_resolved=bond_resolved)
    # One bra-chain replay serves both counts: keep the full profile and take its maximum for
    # the maximum-bond-dimension count.
    if profiles === nothing && (bond_resolved || chis === nothing)
        profiles = bra_chain_link_dims(target, circuit; cutoff=cutoff, max_bond_dim=max_bond_dim)
    end
    chis = chis !== nothing ? chis :
           profiles !== nothing ? [isempty(ld) ? 1 : maximum(ld) for ld in profiles] :
           bra_chain_bond_dims(target, circuit; cutoff=cutoff, max_bond_dim=max_bond_dim)
    moves = [2s - 1 for s in spans]
    max_chi_ops = 2 * sum(m * float(c)^3 for (m, c) in zip(moves, chis))
    bond_ops = profiles === nothing ? NaN : bond_resolved_chi3_ops(circuit, profiles)
    return (ngates=length(spans),
            site_ops=2 * sum(moves),
            # Both counts are reported; `bond_resolved` selects the one called `chi3_ops`.
            chi3_ops=bond_resolved ? bond_ops : max_chi_ops,
            chi3_ops_maxchi=max_chi_ops,
            chi3_ops_bond=bond_ops,
            mean_span=sum(spans) / length(spans),
            max_chi=maximum(chis),
            spans=spans, chis=chis, bond_resolved=bond_resolved)
end

end # module
