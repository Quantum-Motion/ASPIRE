"""
    SimConfig

Process-global truncation parameters for every tensor-network simulation in the package: the
maximum MPS bond dimension `chi` and the SVD cutoff. Set them once with
[`set_max_bond_dim!`](@ref) and [`set_cutoff!`](@ref); every function that truncates reads
[`global_maxdim`](@ref) and [`global_cutoff`](@ref) as the default of its `maxdim` and `cutoff`
keywords, so per-call overrides still work. [`with_sim_config`](@ref) scopes a temporary override.
"""
module SimConfig

export set_max_bond_dim!, set_cutoff!, global_maxdim, global_cutoff, with_sim_config

const _MAXDIM = Ref{Int}(64)
const _CUTOFF = Ref{Float64}(1e-12)

"Current global maximum MPS bond dimension."
global_maxdim() = _MAXDIM[]

"Current global SVD truncation cutoff."
global_cutoff() = _CUTOFF[]

"""
    set_max_bond_dim!(chi::Integer) -> Int

Set the global maximum bond dimension (initially 64) and return the new value.
"""
function set_max_bond_dim!(chi::Integer)
    chi > 0 || throw(ArgumentError("max bond dimension must be positive, got $chi"))
    _MAXDIM[] = Int(chi)
end

"""
    set_cutoff!(eps::Real) -> Float64

Set the global SVD cutoff (initially 1e-12) and return the new value.
"""
function set_cutoff!(eps::Real)
    eps >= 0 || throw(ArgumentError("cutoff must be non-negative, got $eps"))
    _CUTOFF[] = Float64(eps)
end

"""
    with_sim_config(f; maxdim=global_maxdim(), cutoff=global_cutoff())

Run `f()` with the global truncation parameters temporarily overridden and restore the previous
values afterwards, even if `f` throws. Returns the value of `f()`.

```julia
with_sim_config(maxdim=16) do
    compressed_mps(psi)        # truncates at chi = 16
end                            # global chi restored here
```
"""
function with_sim_config(f; maxdim::Integer=global_maxdim(), cutoff::Real=global_cutoff())
    old_maxdim, old_cutoff = _MAXDIM[], _CUTOFF[]
    set_max_bond_dim!(maxdim)
    set_cutoff!(cutoff)
    try
        return f()
    finally
        _MAXDIM[] = old_maxdim
        _CUTOFF[] = old_cutoff
    end
end

end # module
