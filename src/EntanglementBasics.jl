"""
    EntanglementBasics

Entanglement entropy across a bond of an MPS.
"""
module EntanglementBasics

using ITensors
using ITensorMPS

export bond_entropy

"""
    bond_entropy(psi, bond) -> Float64

Von Neumann entanglement entropy in bits across the bond between sites `bond` and `bond + 1`:
`-sum_n p_n log2 p_n` with `p_n` the squared Schmidt coefficients of the cut.
"""
function bond_entropy(psi::MPS, bond::Int)
    psi = orthogonalize(psi, bond)
    left_inds = (linkinds(psi, bond - 1)..., siteinds(psi, bond)...)
    _, s, _ = svd(psi[bond], left_inds)
    entropy = 0.0
    for n in 1:dim(s, 1)
        p = abs2(s[n, n])
        if p > 0
            entropy -= p * log2(p)
        end
    end
    return entropy
end

end # module
