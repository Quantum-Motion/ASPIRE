"""
    FidelityMetrics

Fidelity conventions of the package.

The compilation error is the amplitude infidelity

```text
1 - f,  where f = |<target|prepared>| = sqrt(F)
```

with `F` the squared fidelity. The conversion functions carry `_from_squared` in their names so
that `1 - F` and `1 - f` cannot be confused.
"""
module FidelityMetrics

using ITensors
using ITensorMPS

export clamp_squared_fidelity,
    amplitude_fidelity_from_squared, amplitude_infidelity_from_squared,
    fidelity_series_from_squared,
    squared_fidelity, amplitude_fidelity, amplitude_infidelity

const _FIDELITY_ATOL = 1e-10

"""
    clamp_squared_fidelity(F; atol=1e-10)

Validate and clamp a numerically evaluated squared fidelity to `[0, 1]`.
Values outside that interval by more than `atol` raise `DomainError`; small
floating-point excursions are clamped.
"""
function clamp_squared_fidelity(F::Real; atol::Real=_FIDELITY_ATOL)
    atol >= 0 || throw(ArgumentError("atol must be non-negative, got $atol"))
    value = Float64(F)
    isfinite(value) || throw(DomainError(F, "squared fidelity must be finite"))
    (-Float64(atol) <= value <= 1 + Float64(atol)) ||
        throw(DomainError(F, "squared fidelity must lie in [0, 1] up to atol=$atol"))
    return clamp(value, 0.0, 1.0)
end

"""
    amplitude_fidelity_from_squared(F; atol=1e-10)

Return `f = sqrt(F)` from a squared fidelity.
"""
amplitude_fidelity_from_squared(F::Real; atol::Real=_FIDELITY_ATOL) =
    sqrt(clamp_squared_fidelity(F; atol=atol))

"""
    amplitude_infidelity_from_squared(F; atol=1e-10)

Return the canonical compilation metric `1 - f = 1 - sqrt(F)`.
"""
amplitude_infidelity_from_squared(F::Real; atol::Real=_FIDELITY_ATOL) =
    1.0 - amplitude_fidelity_from_squared(F; atol=atol)

"""
    fidelity_series_from_squared(values; atol=1e-10)

Fidelity curve with unambiguous field names: a named tuple with `squared_fidelity`,
`amplitude_fidelity` and `amplitude_infidelity = 1 - f`.
"""
function fidelity_series_from_squared(values; atol::Real=_FIDELITY_ATOL)
    squared = [clamp_squared_fidelity(value; atol=atol) for value in values]
    amplitude = sqrt.(squared)
    return (
        squared_fidelity=squared,
        amplitude_fidelity=amplitude,
        amplitude_infidelity=1 .- amplitude,
    )
end

"""
    squared_fidelity(psi, target)

Squared overlap `F = |<target|psi>|^2` between normalized MPSs.
"""
function squared_fidelity(psi::MPS, target::MPS)
    length(psi) == length(target) ||
        throw(DimensionMismatch("MPS lengths differ: $(length(psi)) and $(length(target))"))
    value = abs2(inner(normalize!(copy(target)), normalize!(copy(psi))))
    return clamp_squared_fidelity(value)
end

"""
    amplitude_fidelity(psi, target)

Overlap amplitude `f = |<target|psi>|` between normalized MPSs.
"""
amplitude_fidelity(psi::MPS, target::MPS) =
    amplitude_fidelity_from_squared(squared_fidelity(psi, target))

"""
    amplitude_infidelity(psi, target)

Canonical compilation error `1-f` between normalized MPSs.
"""
amplitude_infidelity(psi::MPS, target::MPS) =
    amplitude_infidelity_from_squared(squared_fidelity(psi, target))

end # module
