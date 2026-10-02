"""
    Saturation

Stopping rule for gate-versus-infidelity curves. An arm is declared saturated
when doubling its gate budget would reduce the infidelity by less than a factor
`gain_per_doubling` (default 1.10), and this remains true after the budget is
extended a further factor `confirm` (default 1.5). The rule is scale-free,
needs no extrapolation, and can be evaluated while the curve is being built.
The `confirm` guard is needed because staircase-shaped curves have long flat
treads followed by abrupt drops, so a single quiet doubling can lie on a tread.
"""
module Saturation

export SaturationRule, saturation_rule, curve_infidelity_at, doubling_gain,
    saturation_state, saturation_gate_cap

"""
    SaturationRule(; gain_per_doubling, confirm, floor, min_gates_per_site,
                     max_gates_per_site, min_samples)

- `gain_per_doubling`: stop once doubling the budget would buy less than this
  factor (`1.10` = 10%). Equivalently a local log-log slope `log2(1.10) = 0.14`.
- `confirm`: the flatness must survive extending the budget a further `confirm`x.
- `floor`: an arm reaching this infidelity is saturated regardless of its slope.
- `min_gates_per_site` / `max_gates_per_site`: the budget bracket in two-qubit
  gates per site, independent of `N`. `max` is the censoring point: an arm
  still paying there is reported as censored, not saturated.
- `min_samples`: checkpoints required inside the trailing doubling before the
  rule is evaluated; coarser sampling is reported as not saturated.
"""
Base.@kwdef struct SaturationRule
    gain_per_doubling::Float64 = 1.10
    confirm::Float64 = 1.5
    floor::Float64 = 1e-10
    min_gates_per_site::Float64 = 4.0
    max_gates_per_site::Float64 = 96.0
    min_samples::Int = 3
end

"""
    saturation_rule(spec) -> Union{Nothing,SaturationRule}

Build a [`SaturationRule`](@ref) from a dictionary with string keys `"mode"`,
`"gain_per_doubling"`, `"confirm"`, `"floor"`, `"min_gates_per_site"`,
`"max_gates_per_site"` and `"min_samples"`. Omitted keys take the
`SaturationRule` defaults. Returns `nothing` when `spec` is `nothing`. The only
supported `"mode"` is `"saturation"`; any other value throws an error.
"""
function saturation_rule(spec)
    spec === nothing && return nothing
    mode = String(get(spec, "mode", "saturation"))
    mode == "saturation" ||
        error("unsupported stop mode: $mode (only \"saturation\")")
    d = SaturationRule()
    return SaturationRule(
        gain_per_doubling=Float64(get(spec, "gain_per_doubling", d.gain_per_doubling)),
        confirm=Float64(get(spec, "confirm", d.confirm)),
        floor=Float64(get(spec, "floor", d.floor)),
        min_gates_per_site=Float64(get(spec, "min_gates_per_site", d.min_gates_per_site)),
        max_gates_per_site=Float64(get(spec, "max_gates_per_site", d.max_gates_per_site)),
        min_samples=Int(get(spec, "min_samples", d.min_samples)))
end

"""
    saturation_gate_cap(rule, N) -> Int

Gate cap `ceil(max_gates_per_site * N)` (at least 1) for a system of `N` sites.
"""
saturation_gate_cap(rule::SaturationRule, N::Integer) =
    max(1, ceil(Int, rule.max_gates_per_site * N))

"""Infidelity at gate count `g`, log-linearly interpolated, clamped at both ends.

`gates`/`infid` may include the zero-gate sample; it is dropped, since it prices
the bare product state rather than a circuit."""
function curve_infidelity_at(gates, infid, g::Real)
    idx = [i for i in eachindex(gates) if gates[i] > 0 && infid[i] > 0]
    isempty(idx) && return NaN
    gs = Float64[gates[i] for i in idx]
    es = Float64[infid[i] for i in idx]
    g <= gs[1] && return es[1]
    g >= gs[end] && return es[end]
    for i in 2:length(gs)
        if gs[i] >= g
            w = (g - gs[i-1]) / (gs[i] - gs[i-1])
            return exp(log(es[i-1]) + w * (log(es[i]) - log(es[i-1])))
        end
    end
    return es[end]
end

"""`eps(g/window) / eps(g)`: what the last doubling of the budget bought.

`Inf` when the window reaches back past the first sample, where the curve gives
no information."""
function doubling_gain(gates, infid, g::Real=NaN; window::Real=2.0)
    idx = [i for i in eachindex(gates) if gates[i] > 0 && infid[i] > 0]
    length(idx) < 2 && return Inf
    gs = Float64[gates[i] for i in idx]
    g = isnan(g) ? gs[end] : Float64(g)
    g / window < gs[1] && return Inf
    return curve_infidelity_at(gates, infid, g / window) /
           curve_infidelity_at(gates, infid, g)
end

"""
    saturation_state(gates, infid, rule; N)

`(saturated::Bool, reason::String, g_first::Union{Float64,Nothing})`.

`g_first` is the earliest measured gate count from which the quiet-doubling
condition holds continuously up to the endpoint, i.e. where a shorter run could
have stopped. Saturation is declared only if the endpoint is at least
`confirm * g_first`. `N`, if given, enforces `min_gates_per_site * N` as a
minimum endpoint.
"""
function saturation_state(gates, infid, rule::SaturationRule; N::Union{Nothing,Integer}=nothing)
    idx = [i for i in eachindex(gates) if gates[i] > 0 && infid[i] > 0]
    length(idx) < 3 && return (false, "too few checkpoints", nothing)
    gs = Float64[gates[i] for i in idx]
    es = Float64[infid[i] for i in idx]
    es[end] <= rule.floor && return (true, "below floor", gs[end])

    g_end = gs[end]
    N !== nothing && g_end < rule.min_gates_per_site * N &&
        return (false, "under minimum budget", nothing)
    count(g -> g >= g_end / 2, gs) < rule.min_samples &&
        return (false, "too coarsely sampled", nothing)

    doubling_gain(gs, es, g_end) > rule.gain_per_doubling &&
        return (false, "still paying at endpoint", nothing)

    g_first = nothing
    for g in gs
        if doubling_gain(gs, es, g) > rule.gain_per_doubling
            g_first = nothing
        elseif g_first === nothing
            g_first = g
        end
    end
    g_first === nothing && return (false, "no sustained quiet window", nothing)
    g_end < g_first * rule.confirm && return (false, "quiet but unconfirmed", g_first)
    return (true, "saturated", g_first)
end

end # module
