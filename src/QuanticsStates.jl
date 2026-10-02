"""
    QuanticsStates

Quantics-encoded multivariate function states: MPS targets whose amplitudes are a
function `f(x_1, ..., x_d)` sampled on a `2^n`-point grid per variable, stored bit
by bit on `N = d*n` qubits. Two bit orderings are provided: `:serial` (the bits of
each variable are contiguous) and `:interleaved` (bits of equal significance are
adjacent); they differ by a site permutation. The function families are a
correlated multivariate Gaussian and a truncated Weierstrass series, and the
states are built densely (all `2^N` amplitudes, compressed by successive SVDs).
"""
module QuanticsStates

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

using ..SimConfig: global_maxdim, global_cutoff

export quantics_index_map, quantics_permutation, quantics_grid_point,
    quantics_domain, quantics_function, quantics_amplitudes,
    quantics_function_state, quantics_state_from_spec,
    weierstrass_series, weierstrass_terms,
    QUANTICS_ORDERINGS, QUANTICS_FAMILIES

const QUANTICS_ORDERINGS = (:serial, :interleaved)
const QUANTICS_FAMILIES = ("gaussian", "weierstrass")

"""
    quantics_index_map(d, n; ordering=:serial) -> Matrix{Int}

`site[v, b]` is the chain site holding bit `b` (1 = most significant) of variable
`v`. A variable's integer grid index is `m = sum_b sigma_b 2^(n-b)` with
`sigma_b in {0,1}`. The serial ordering is `site = (v-1)*n + b` and the
interleaved ordering is `site = (b-1)*d + v`; both are bijections onto `1:d*n`.
"""
function quantics_index_map(d::Int, n::Int; ordering::Symbol=:serial)
    d >= 1 || throw(ArgumentError("need at least one variable, got d=$d"))
    n >= 1 || throw(ArgumentError("need at least one bit per variable, got n=$n"))
    ordering in QUANTICS_ORDERINGS ||
        throw(ArgumentError("unknown ordering $ordering; expected one of $QUANTICS_ORDERINGS"))
    site = Matrix{Int}(undef, d, n)
    for v in 1:d, b in 1:n
        site[v, b] = ordering === :serial ? (v - 1) * n + b : (b - 1) * d + v
    end
    @assert sort(vec(site)) == collect(1:(d * n)) "quantics index map is not a bijection"
    return site
end

"""
    quantics_permutation(d, n; from=:serial, to=:interleaved) -> Vector{Int}

`p[s]` is the site that site `s` of the `from` ordering becomes under the `to`
ordering. The two orderings of a given function are the same state up to this
permutation of the qubits.
"""
function quantics_permutation(d::Int, n::Int; from::Symbol=:serial,
        to::Symbol=:interleaved)
    src = quantics_index_map(d, n; ordering=from)
    dst = quantics_index_map(d, n; ordering=to)
    p = Vector{Int}(undef, d * n)
    for v in 1:d, b in 1:n
        p[src[v, b]] = dst[v, b]
    end
    return p
end

"""
    quantics_grid_point(m, n, (a, b); endpoint=:inclusive) -> Float64

Grid point `m` in `0:2^n-1` of a variable spanning `[a, b]`. `:inclusive` places
the first and last grid points exactly on `a` and `b`, `a + (b-a) m / (2^n - 1)`;
`:exclusive` uses the half-open convention `a + (b-a) m / 2^n`.
"""
function quantics_grid_point(m::Integer, n::Integer, dom::Tuple{<:Real,<:Real};
        endpoint::Symbol=:inclusive)
    a, b = float(dom[1]), float(dom[2])
    if endpoint === :inclusive
        return a + (b - a) * m / (2^n - 1)
    elseif endpoint === :exclusive
        return a + (b - a) * m / 2^n
    end
    throw(ArgumentError("unknown endpoint convention $endpoint"))
end

"""
    quantics_domain(domain, d) -> Vector{Tuple{Float64,Float64}}

Normalise a domain to one interval per variable. A single `(a, b)` is broadcast
across all `d` variables; a vector must hold exactly `d` intervals.
"""
quantics_domain(dom::Tuple{<:Real,<:Real}, d::Int) =
    fill((float(dom[1]), float(dom[2])), d)
function quantics_domain(dom::AbstractVector, d::Int)
    length(dom) == d ||
        throw(ArgumentError("domain has $(length(dom)) intervals for d=$d variables"))
    return [(float(t[1]), float(t[2])) for t in dom]
end

# Numbers in family labels are rounded rather than printed at full Float64 width.
_fmt(x::Real) = string(round(float(x); digits=4))

"""
    weierstrass_series(x; terms=25, exponent=3.0, amps=nothing, phases=nothing) -> Float64

The truncated Weierstrass-like series
`sum_{k=1}^{terms} a_k sin(pi k^e x + p_k) / (pi k^e)` with `e = exponent`. The
amplitudes `a_k` default to 1 and the phases `p_k` to 0; `amps` and `phases`
replace them without changing the spectral content.

The `k = terms` mode has frequency `terms^e`, so `n` bits resolve the series only
while `2^n >> terms^e`; at smaller `n` the top modes alias. Use
[`weierstrass_terms`](@ref) to hold the resolution ratio fixed as `n` varies.
"""
function weierstrass_series(x::Real; terms::Integer=25, exponent::Real=3.0,
        amps=nothing, phases=nothing)
    terms >= 1 || throw(ArgumentError("weierstrass needs terms >= 1, got $terms"))
    acc = 0.0
    for k in 1:terms
        w = float(k)^exponent
        a = amps === nothing ? 1.0 : amps[k]
        p = phases === nothing ? 0.0 : phases[k]
        acc += a * sin(pi * w * x + p) / (pi * w)
    end
    return acc
end

"""
    weierstrass_terms(n; exponent=3.0, samples_per_period=4) -> Int

Largest `K` whose top mode `K^e` is still resolved by `2^n` grid points at
`samples_per_period` samples per period, i.e. `K^e <= 2^n / samples_per_period`.
For `exponent = 3` and `n = 16` this gives `K = 25`.
"""
function weierstrass_terms(n::Integer; exponent::Real=3.0,
        samples_per_period::Real=4.0)
    K = floor(Int, (2.0^n / samples_per_period)^(1 / exponent))
    return max(1, K)
end

"""
    quantics_function(family, d; rng, rho, terms, exponent, randomize) -> (f, domain, label)

Callable `f(x::Vector{Float64})` for one draw of a function family, together with
the default domain of that family and a short label. `family` is one of
`QUANTICS_FAMILIES`.

  * `"gaussian"`: the unnormalised correlated Gaussian
    `exp(-(x - mu)' P (x - mu) / 2)` on `(-3, 3)^d`, with `P` the inverse of
    `Sigma = diag(sigma) C diag(sigma)`. Here `C_ij = rho s_i s_j` for `i != j`
    and `C_ii = 1`, `s_i = +-1` are random signs, and the widths `sigma_i` are
    uniform in `[0.8, 1.2]`. For `0 <= rho < 1`, `C` is a sign congruence of the
    positive-definite equicorrelation matrix, so every draw is a valid
    covariance whose correlations have magnitude `rho`. The means `mu_i` are
    uniform in `[-0.4, 0.4]`.
  * `"weierstrass"`: [`weierstrass_series`](@ref) with `terms` and `exponent` on
    `(0, 1)`; requires `d = 1`. It is deterministic unless `randomize = true`, in
    which case signed random amplitudes and random phases are drawn from `rng`,
    leaving the mode frequencies unchanged.

All random draws come from `rng`, so a seed selects one member of the family.
"""
function quantics_function(family::AbstractString, d::Int;
        rng::AbstractRNG=Random.default_rng(), rho::Real=0.5,
        terms::Int=25, exponent::Real=3.0, randomize::Bool=false)
    if family == "gaussian"
        0 <= rho < 1 || throw(ArgumentError("rho must be in [0, 1), got $rho"))
        s = rand(rng, (-1.0, 1.0), d)
        sigma = 0.8 .+ 0.4 .* rand(rng, d)
        C = [(i == j ? 1.0 : rho * s[i] * s[j]) for i in 1:d, j in 1:d]
        Sigma = Diagonal(sigma) * C * Diagonal(sigma)
        P = inv(Sigma)
        mu = 0.4 .* (2 .* rand(rng, d) .- 1)
        f = function (x)
            z = x .- mu
            return exp(-0.5 * dot(z, P, z))
        end
        return (f, (-3.0, 3.0), "gaussian(rho=$(round(rho; digits=2)))")
    elseif family == "weierstrass"
        d == 1 || throw(ArgumentError("weierstrass is a single-variable family, got d=$d"))
        amps = randomize ? (0.5 .+ rand(rng, terms)) .* rand(rng, (-1.0, 1.0), terms) : nothing
        phases = randomize ? 2pi .* rand(rng, terms) : nothing
        f = x -> weierstrass_series(x[1]; terms=terms, exponent=exponent,
            amps=amps, phases=phases)
        return (f, (0.0, 1.0), "weierstrass(a=$(_fmt(exponent)),K=$terms)")
    end
    throw(ArgumentError("unknown quantics family $family; expected one of $QUANTICS_FAMILIES"))
end

"""
    quantics_amplitudes(f, d, n; ordering, domain, encoding, endpoint, max_qubits) -> Array

The `2^(d*n)` amplitude tensor of `f` sampled on the quantics grid, normalised in
`L2`. Dimension `s` of the result is the qubit at chain site `s`.

`encoding = :amplitude` stores `f`; `:sqrt_density` stores `sqrt(f)` and requires
`f >= 0`. The construction evaluates `f` at all `2^(d*n)` grid points and throws
if `d*n > max_qubits`.
"""
function quantics_amplitudes(f, d::Int, n::Int; ordering::Symbol=:serial,
        domain=(-3.0, 3.0), encoding::Symbol=:amplitude,
        endpoint::Symbol=:inclusive, max_qubits::Int=26)
    N = d * n
    N <= max_qubits || throw(ArgumentError(
        "dense quantics construction materialises 2^$N amplitudes; raise " *
        "max_qubits deliberately"))
    doms = quantics_domain(domain, d)
    site = quantics_index_map(d, n; ordering=ordering)
    grids = [[quantics_grid_point(m, n, doms[v]; endpoint=endpoint)
              for m in 0:(2^n - 1)] for v in 1:d]

    A = Array{ComplexF64}(undef, ntuple(_ -> 2, N))
    x = Vector{Float64}(undef, d)
    for idx in CartesianIndices(A)
        for v in 1:d
            m = 0
            for b in 1:n
                m |= (idx[site[v, b]] - 1) << (n - b)
            end
            x[v] = grids[v][m + 1]
        end
        value = f(x)
        if encoding === :sqrt_density
            real(value) >= 0 || throw(ArgumentError(
                "encoding=:sqrt_density needs a non-negative function; got $value"))
            value = sqrt(real(value))
        elseif encoding !== :amplitude
            throw(ArgumentError("unknown encoding $encoding"))
        end
        A[idx] = value
    end
    nrm = norm(A)
    nrm > 0 || throw(ArgumentError("function is identically zero on the grid"))
    return A ./ nrm
end

"""
    quantics_function_state(f, d, n; ...) -> MPS
    quantics_function_state(d, n; family="gaussian", ...) -> MPS

Normalised MPS of a quantics-encoded function on `N = d*n` qubits, obtained by
compressing the dense amplitude tensor of [`quantics_amplitudes`](@ref) with
successive SVDs.

The first form takes a callable `f(x::Vector{Float64})`. The second draws one
member of a named family (see [`quantics_function`](@ref)) from `rng` and uses
the family's default domain unless `domain` is given.

Keyword arguments:

  * `ordering`: `:serial` or `:interleaved`.
  * `domain`: one `(a, b)` for all variables, or a vector of `d` intervals.
  * `encoding`: `:amplitude` or `:sqrt_density`.
  * `endpoint`: `:inclusive` or `:exclusive` grid convention.
  * `cutoff`, `maxdim`: SVD truncation; the defaults are the package-wide
    `global_cutoff()` and `global_maxdim()`.
  * `sites`: optional vector of `N` site indices to build the MPS on.
  * `max_qubits`: guard on the dense construction size.

The family form also accepts `rho`, `terms`, `exponent`, `randomize` and `rng`.
"""
function quantics_function_state(f, d::Int, n::Int; ordering::Symbol=:serial,
        domain=(-3.0, 3.0), encoding::Symbol=:amplitude,
        endpoint::Symbol=:inclusive, cutoff::Real=global_cutoff(),
        maxdim::Int=global_maxdim(), sites=nothing, max_qubits::Int=26)
    N = d * n
    A = quantics_amplitudes(f, d, n; ordering=ordering, domain=domain,
        encoding=encoding, endpoint=endpoint, max_qubits=max_qubits)
    s = sites === nothing ? siteinds("S=1/2", N) : sites
    length(s) == N || throw(ArgumentError("expected $N site indices, got $(length(s))"))
    psi = MPS(ITensor(A, s...), s; cutoff=cutoff, maxdim=maxdim)
    return normalize!(psi)
end

function quantics_function_state(d::Int, n::Int; family::AbstractString="gaussian",
        rng::AbstractRNG=Random.default_rng(), rho::Real=0.5,
        terms::Int=25, exponent::Real=3.0, randomize::Bool=false,
        domain=nothing, kwargs...)
    f, default_domain, _ = quantics_function(family, d; rng=rng, rho=rho,
        terms=terms, exponent=exponent, randomize=randomize)
    return quantics_function_state(f, d, n;
        domain=(domain === nothing ? default_domain : domain), kwargs...)
end

"""
    quantics_state_from_spec(spec, seed) -> MPS

Target MPS from a dictionary of string keys. Recognised keys:

    d, n          variables and bits per variable; N = d*n
    family        one of `QUANTICS_FAMILIES` (default "gaussian")
    ordering      "serial" (default) or "interleaved"
    rho           Gaussian correlation magnitude (default 0.5)
    terms,
    exponent      Weierstrass truncation K and exponent a (default 25, 3);
                  `terms = 0` means the resolution-matched `weierstrass_terms(n)`
    randomize     draw random Weierstrass amplitudes and phases from `seed`
    encoding      "amplitude" (default) or "sqrt_density"
    endpoint      "inclusive" (default) or "exclusive" grid convention
    domain        optional [a, b] override of the family's default domain, or a
                  list of `d` intervals
    cutoff,
    max_bond_dim  SVD truncation (defaults: `global_cutoff()`, `global_maxdim()`)
    max_qubits    guard on the dense construction size (default 26)

`seed` seeds the random number generator that draws the member of the family.
"""
function quantics_state_from_spec(spec::AbstractDict, seed::Integer)
    rng = Random.MersenneTwister(seed)
    dom = get(spec, "domain", nothing)
    domain = if dom === nothing
        nothing
    elseif dom[1] isa AbstractVector || dom[1] isa Tuple
        [(Float64(t[1]), Float64(t[2])) for t in dom]
    else
        (Float64(dom[1]), Float64(dom[2]))
    end
    exponent = Float64(get(spec, "exponent", 3.0))
    terms = Int(get(spec, "terms", 25))
    terms == 0 && (terms = weierstrass_terms(Int(spec["n"]); exponent=exponent))
    return quantics_function_state(Int(spec["d"]), Int(spec["n"]);
        family=String(get(spec, "family", "gaussian")),
        ordering=Symbol(get(spec, "ordering", "serial")),
        encoding=Symbol(get(spec, "encoding", "amplitude")),
        endpoint=Symbol(get(spec, "endpoint", "inclusive")),
        rho=Float64(get(spec, "rho", 0.5)),
        terms=terms,
        exponent=exponent,
        randomize=Bool(get(spec, "randomize", false)),
        domain=domain,
        cutoff=Float64(get(spec, "cutoff", global_cutoff())),
        maxdim=Int(get(spec, "max_bond_dim", global_maxdim())),
        max_qubits=Int(get(spec, "max_qubits", 26)),
        rng=rng)
end

end # module
