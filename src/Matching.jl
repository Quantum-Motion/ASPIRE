"""
    Matching

Exact maximum-weight matching on a general graph. A set of pairwise-disjoint
gate supports contributes to the objective additively, so choosing the best set
of two-qubit moves is a maximum-weight matching on the connectivity graph.

`maximum_weight_matching` implements Edmonds' blossom algorithm (Edmonds 1965)
in the `O(V^3)` primal-dual formulation of Galil (1986). Edge weights are
quantised to `Int64` on a fixed scale (`WEIGHT_SCALE`, 1e-9 units) so that every
slack comparison in the dual updates is exact and the search terminates.
Non-positive edges are dropped, since they never improve a matching. The result
agrees with `networkx.max_weight_matching` with `maxcardinality=false`.
`brute_force_matching` is the exponential reference used in the tests.
"""
module Matching

export maximum_weight_matching, matching_weight, brute_force_matching

"Quantisation of edge weights onto the integer lattice used by the blossom search."
const WEIGHT_SCALE = 1e9

# 0-based view over a `Vector`. The endpoint indexing `p ^ 1` used by the blossom
# algorithm pairs the two ends of edge `p div 2`, which is natural only with
# 0-based indices.
struct Zero{T}
    data::Vector{T}
end
Base.getindex(z::Zero, i::Integer) = z.data[i + 1]
Base.setindex!(z::Zero, v, i::Integer) = (z.data[i + 1] = v)
Base.length(z::Zero) = length(z.data)
zeros_of(value, n::Int) = Zero(fill(value, n))

"""
    maximum_weight_matching(edges; maxcardinality=false) -> Vector{Tuple{Int,Int}}

Maximum-weight set of pairwise vertex-disjoint edges. `edges` is any iterable of
`(i, j, w)` with `Int` vertex labels (arbitrary, need not be contiguous) and a
real weight. Edges with `w <= 0` are discarded. The result is sorted, with each
pair in ascending vertex order.

Exact for arbitrary graphs, including odd cycles, which the blossom
contraction handles. Runs in `O(V^3)`.
"""
function maximum_weight_matching(edges; maxcardinality::Bool=false)
    triples = Tuple{Int,Int,Float64}[]
    for e in edges
        i, j, w = Int(e[1]), Int(e[2]), Float64(e[3])
        i == j && throw(ArgumentError("self-loop at vertex $i is not a matchable edge"))
        isfinite(w) || throw(DomainError(w, "matching weights must be finite"))
        w > 0 || continue
        push!(triples, (min(i, j), max(i, j), w))
    end
    isempty(triples) && return Tuple{Int,Int}[]

    labels = sort!(collect(Set(v for t in triples for v in (t[1], t[2]))))
    index_of = Dict(v => k - 1 for (k, v) in enumerate(labels))   # 0-based
    quantised = [(index_of[i], index_of[j], _quantise(w)) for (i, j, w) in triples]

    mate = _blossom(quantised, length(labels), maxcardinality)

    result = Tuple{Int,Int}[]
    for v in 0:(length(labels) - 1)
        w = mate[v]
        w >= 0 && v < w && push!(result, (labels[v + 1], labels[w + 1]))
    end
    return sort!(result)
end

"""
    maximum_weight_matching(weight_of::AbstractDict; maxcardinality=false)

Convenience form taking `(i, j) => weight`.
"""
maximum_weight_matching(weight_of::AbstractDict; maxcardinality::Bool=false) =
    maximum_weight_matching(((p[1], p[2], w) for (p, w) in weight_of);
        maxcardinality=maxcardinality)

function _quantise(w::Float64)
    scaled = round(w * WEIGHT_SCALE)
    abs(scaled) < 4.0e18 ||
        throw(DomainError(w, "weight overflows the integer matching lattice"))
    return Int64(scaled)
end

"Total weight of `pairs` under `weight_of`, for reporting and for the tests."
function matching_weight(pairs, weight_of::AbstractDict)
    total = 0.0
    for p in pairs
        key = (min(p[1], p[2]), max(p[1], p[2]))
        total += Float64(haskey(weight_of, key) ? weight_of[key] : weight_of[p])
    end
    return total
end

"""
    brute_force_matching(edges) -> Vector{Tuple{Int,Int}}

Exhaustive maximum-weight matching by enumeration. Exponential; exists so the
blossom implementation can be checked against something with no cleverness in
it. Refuses graphs with more than `max_edges` positive edges.
"""
function brute_force_matching(edges; max_edges::Int=20)
    list = [(min(Int(e[1]), Int(e[2])), max(Int(e[1]), Int(e[2])), Float64(e[3]))
            for e in edges if Float64(e[3]) > 0]
    length(list) <= max_edges || throw(ArgumentError(
        "brute force refuses $(length(list)) edges; it is a reference for small graphs"))
    best = Tuple{Int,Int}[]
    best_weight = 0.0
    n = length(list)
    for mask in 0:(2^n - 1)
        used = Set{Int}()
        weight = 0.0
        chosen = Tuple{Int,Int}[]
        ok = true
        for k in 1:n
            (mask >> (k - 1)) & 1 == 1 || continue
            i, j, w = list[k]
            (i in used || j in used) && (ok = false; break)
            push!(used, i); push!(used, j)
            weight += w
            push!(chosen, (i, j))
        end
        ok || continue
        if weight > best_weight + 1e-12
            best_weight = weight
            best = sort!(chosen)
        end
    end
    return best
end

# Primal-dual blossom search on quantised weights; returns the 0-based mate array.
function _blossom(edges::Vector{Tuple{Int,Int,Int64}}, nvertex::Int,
        maxcardinality::Bool)
    nedge = length(edges)
    maxweight = max(Int64(0), maximum(e[3] for e in edges))

    endpoint = Zero([edges[div(p, 2) + 1][mod(p, 2) + 1] for p in 0:(2 * nedge - 1)])
    neighbend = Zero([Int[] for _ in 1:nvertex])
    for (k, (i, j, _)) in enumerate(edges)
        push!(neighbend[i], 2 * (k - 1) + 1)
        push!(neighbend[j], 2 * (k - 1))
    end

    mate = zeros_of(-1, nvertex)
    label = zeros_of(0, 2 * nvertex)
    labelend = zeros_of(-1, 2 * nvertex)
    inblossom = Zero(collect(0:(nvertex - 1)))
    blossomparent = zeros_of(-1, 2 * nvertex)
    blossomchilds = Zero(Vector{Union{Nothing,Vector{Int}}}(nothing, 2 * nvertex))
    blossombase = Zero(vcat(collect(0:(nvertex - 1)), fill(-1, nvertex)))
    blossomendps = Zero(Vector{Union{Nothing,Vector{Int}}}(nothing, 2 * nvertex))
    bestedge = zeros_of(-1, 2 * nvertex)
    blossombestedges = Zero(Vector{Union{Nothing,Vector{Int}}}(nothing, 2 * nvertex))
    unusedblossoms = collect(nvertex:(2 * nvertex - 1))
    dualvar = Zero(vcat(fill(maxweight, nvertex), fill(Int64(0), nvertex)))
    allowedge = zeros_of(false, nedge)
    queue = Int[]

    slack(k) = dualvar[edges[k + 1][1]] + dualvar[edges[k + 1][2]] -
               2 * edges[k + 1][3]

    function blossom_leaves(b, out=Int[])
        if b < nvertex
            push!(out, b)
        else
            for t in blossomchilds[b]
                t < nvertex ? push!(out, t) : blossom_leaves(t, out)
            end
        end
        return out
    end

    function assign_label(w, t, p)
        b = inblossom[w]
        label[w] = t; label[b] = t
        labelend[w] = p; labelend[b] = p
        bestedge[w] = -1; bestedge[b] = -1
        if t == 1
            append!(queue, blossom_leaves(b))
        elseif t == 2
            base = blossombase[b]
            assign_label(endpoint[mate[base]], 1, mate[base] ⊻ 1)
        end
        return nothing
    end

    function scan_blossom(v, w)
        path = Int[]
        base = -1
        while v != -1 || w != -1
            b = inblossom[v]
            if label[b] & 4 != 0
                base = blossombase[b]
                break
            end
            push!(path, b)
            label[b] = 5
            if labelend[b] == -1
                v = -1
            else
                v = endpoint[labelend[b]]
                b = inblossom[v]
                v = endpoint[labelend[b]]
            end
            if w != -1
                v, w = w, v
            end
        end
        for b in path
            label[b] = 1
        end
        return base
    end

    function add_blossom(base, k)
        v, w, _ = edges[k + 1]
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = pop!(unusedblossoms)
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        path = Int[]
        endps = Int[]
        blossomchilds[b] = path
        blossomendps[b] = endps
        while bv != bb
            blossomparent[bv] = b
            push!(path, bv)
            push!(endps, labelend[bv])
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        end
        push!(path, bb)
        reverse!(path)
        reverse!(endps)
        push!(endps, 2 * k)
        while bw != bb
            blossomparent[bw] = b
            push!(path, bw)
            push!(endps, labelend[bw] ⊻ 1)
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        end
        label[b] = 1
        labelend[b] = labelend[bb]
        dualvar[b] = 0
        for u in blossom_leaves(b)
            label[inblossom[u]] == 2 && push!(queue, u)
            inblossom[u] = b
        end
        bestedgeto = zeros_of(-1, 2 * nvertex)
        for bv2 in path
            nblists = if blossombestedges[bv2] === nothing
                [[div(p, 2) for p in neighbend[u]] for u in blossom_leaves(bv2)]
            else
                [blossombestedges[bv2]]
            end
            for nblist in nblists, kk in nblist
                i, j, _ = edges[kk + 1]
                if inblossom[j] == b
                    i, j = j, i
                end
                bj = inblossom[j]
                if bj != b && label[bj] == 1 &&
                        (bestedgeto[bj] == -1 || slack(kk) < slack(bestedgeto[bj]))
                    bestedgeto[bj] = kk
                end
            end
            blossombestedges[bv2] = nothing
            bestedge[bv2] = -1
        end
        best = Int[]
        for x in 0:(2 * nvertex - 1)
            bestedgeto[x] != -1 && push!(best, bestedgeto[x])
        end
        blossombestedges[b] = best
        bestedge[b] = -1
        for kk in best
            if bestedge[b] == -1 || slack(kk) < slack(bestedge[b])
                bestedge[b] = kk
            end
        end
        return nothing
    end

    function expand_blossom(b, endstage)
        for s in blossomchilds[b]
            blossomparent[s] = -1
            if s < nvertex
                inblossom[s] = s
            elseif endstage && dualvar[s] == 0
                expand_blossom(s, endstage)
            else
                for u in blossom_leaves(s)
                    inblossom[u] = s
                end
            end
        end
        if !endstage && label[b] == 2
            entrychild = inblossom[endpoint[labelend[b] ⊻ 1]]
            children = blossomchilds[b]
            j = findfirst(==(entrychild), children) - 1     # 0-based position
            jstep = 0
            endptrick = 0
            if j & 1 != 0
                j -= length(children)
                jstep = 1
                endptrick = 0
            else
                jstep = -1
                endptrick = 1
            end
            p = labelend[b]
            while j != 0
                label[endpoint[p ⊻ 1]] = 0
                label[endpoint[_at(blossomendps[b], j - endptrick) ⊻ endptrick ⊻ 1]] = 0
                assign_label(endpoint[p ⊻ 1], 2, p)
                allowedge[div(_at(blossomendps[b], j - endptrick), 2)] = true
                j += jstep
                p = _at(blossomendps[b], j - endptrick) ⊻ endptrick
                allowedge[div(p, 2)] = true
                j += jstep
            end
            bv = _at(children, j)
            label[endpoint[p ⊻ 1]] = 2; label[bv] = 2
            labelend[endpoint[p ⊻ 1]] = p; labelend[bv] = p
            bestedge[bv] = -1
            j += jstep
            while _at(children, j) != entrychild
                bv = _at(children, j)
                if label[bv] == 1
                    j += jstep
                    continue
                end
                v = -1
                for u in blossom_leaves(bv)
                    v = u
                    label[u] != 0 && break
                end
                if v != -1 && label[v] != 0
                    label[v] = 0
                    label[endpoint[mate[blossombase[bv]]]] = 0
                    assign_label(v, 2, labelend[v])
                end
                j += jstep
            end
        end
        label[b] = 0
        labelend[b] = -1
        blossomchilds[b] = nothing
        blossomendps[b] = nothing
        blossombase[b] = -1
        blossombestedges[b] = nothing
        bestedge[b] = -1
        push!(unusedblossoms, b)
        return nothing
    end

    function augment_blossom(b, v)
        t = v
        while blossomparent[t] != b
            t = blossomparent[t]
        end
        t >= nvertex && augment_blossom(t, v)
        children = blossomchilds[b]
        i = findfirst(==(t), children) - 1
        j = i
        jstep = 0
        endptrick = 0
        if i & 1 != 0
            j -= length(children)
            jstep = 1
            endptrick = 0
        else
            jstep = -1
            endptrick = 1
        end
        while j != 0
            j += jstep
            t = _at(children, j)
            p = _at(blossomendps[b], j - endptrick) ⊻ endptrick
            t >= nvertex && augment_blossom(t, endpoint[p])
            j += jstep
            t = _at(children, j)
            t >= nvertex && augment_blossom(t, endpoint[p ⊻ 1])
            mate[endpoint[p]] = p ⊻ 1
            mate[endpoint[p ⊻ 1]] = p
        end
        blossomchilds[b] = vcat(children[(i + 1):end], children[1:i])
        endps = blossomendps[b]
        blossomendps[b] = vcat(endps[(i + 1):end], endps[1:i])
        blossombase[b] = blossombase[blossomchilds[b][1]]
        return nothing
    end

    function augment_matching(k)
        v, w, _ = edges[k + 1]
        for (s0, p0) in ((v, 2 * k + 1), (w, 2 * k))
            s = s0
            p = p0
            while true
                bs = inblossom[s]
                bs >= nvertex && augment_blossom(bs, s)
                mate[s] = p
                labelend[bs] == -1 && break
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ⊻ 1]
                bt >= nvertex && augment_blossom(bt, j)
                mate[j] = labelend[bt]
                p = labelend[bt] ⊻ 1
            end
        end
        return nothing
    end

    for _stage in 1:nvertex
        for x in 0:(2 * nvertex - 1)
            label[x] = 0
            bestedge[x] = -1
        end
        for x in nvertex:(2 * nvertex - 1)
            blossombestedges[x] = nothing
        end
        for x in 0:(nedge - 1)
            allowedge[x] = false
        end
        empty!(queue)
        for v in 0:(nvertex - 1)
            if mate[v] == -1 && label[inblossom[v]] == 0
                assign_label(v, 1, -1)
            end
        end

        augmented = false
        while true
            while !isempty(queue) && !augmented
                v = pop!(queue)
                for p in neighbend[v]
                    k = div(p, 2)
                    w = endpoint[p]
                    inblossom[v] == inblossom[w] && continue
                    kslack = Int64(0)
                    if !allowedge[k]
                        kslack = slack(k)
                        kslack <= 0 && (allowedge[k] = true)
                    end
                    if allowedge[k]
                        if label[inblossom[w]] == 0
                            assign_label(w, 2, p ⊻ 1)
                        elseif label[inblossom[w]] == 1
                            base = scan_blossom(v, w)
                            if base >= 0
                                add_blossom(base, k)
                            else
                                augment_matching(k)
                                augmented = true
                                break
                            end
                        elseif label[w] == 0
                            label[w] = 2
                            labelend[w] = p ⊻ 1
                        end
                    elseif label[inblossom[w]] == 1
                        b = inblossom[v]
                        if bestedge[b] == -1 || kslack < slack(bestedge[b])
                            bestedge[b] = k
                        end
                    elseif label[w] == 0
                        if bestedge[w] == -1 || kslack < slack(bestedge[w])
                            bestedge[w] = k
                        end
                    end
                end
            end
            augmented && break

            deltatype = -1
            delta = Int64(0)
            deltaedge = -1
            deltablossom = -1

            if !maxcardinality
                deltatype = 1
                delta = minimum(dualvar[v] for v in 0:(nvertex - 1))
            end
            for v in 0:(nvertex - 1)
                if label[inblossom[v]] == 0 && bestedge[v] != -1
                    d = slack(bestedge[v])
                    if deltatype == -1 || d < delta
                        delta = d; deltatype = 2; deltaedge = bestedge[v]
                    end
                end
            end
            for b in 0:(2 * nvertex - 1)
                if blossomparent[b] == -1 && label[b] == 1 && bestedge[b] != -1
                    kslack = slack(bestedge[b])
                    # Both endpoints sit in S-blossoms, whose duals move together,
                    # so this slack is even and the halving is exact.
                    d = div(kslack, 2)
                    if deltatype == -1 || d < delta
                        delta = d; deltatype = 3; deltaedge = bestedge[b]
                    end
                end
            end
            for b in nvertex:(2 * nvertex - 1)
                if blossombase[b] >= 0 && blossomparent[b] == -1 && label[b] == 2 &&
                        (deltatype == -1 || dualvar[b] < delta)
                    delta = dualvar[b]; deltatype = 4; deltablossom = b
                end
            end
            if deltatype == -1
                deltatype = 1
                delta = max(Int64(0), minimum(dualvar[v] for v in 0:(nvertex - 1)))
            end

            for v in 0:(nvertex - 1)
                lab = label[inblossom[v]]
                if lab == 1
                    dualvar[v] -= delta
                elseif lab == 2
                    dualvar[v] += delta
                end
            end
            for b in nvertex:(2 * nvertex - 1)
                if blossombase[b] >= 0 && blossomparent[b] == -1
                    if label[b] == 1
                        dualvar[b] += 2 * delta
                    elseif label[b] == 2
                        dualvar[b] -= 2 * delta
                    end
                end
            end

            if deltatype == 1
                break
            elseif deltatype == 2
                allowedge[deltaedge] = true
                i, j, _ = edges[deltaedge + 1]
                label[inblossom[i]] == 0 && ((i, j) = (j, i))
                push!(queue, i)
            elseif deltatype == 3
                allowedge[deltaedge] = true
                i, _, _ = edges[deltaedge + 1]
                push!(queue, i)
            elseif deltatype == 4
                expand_blossom(deltablossom, false)
            end
        end

        augmented || break

        for b in nvertex:(2 * nvertex - 1)
            if blossomparent[b] == -1 && blossombase[b] >= 0 && label[b] == 1 &&
                    dualvar[b] == 0
                expand_blossom(b, true)
            end
        end
    end

    for v in 0:(nvertex - 1)
        mate[v] >= 0 && (mate[v] = endpoint[mate[v]])
    end
    return mate
end

# Index a 1-based vector with a 0-based position that may be negative, counting
# from the end (the child and endpoint walks in `_blossom` step in both directions).
_at(v::AbstractVector, j::Integer) = j >= 0 ? v[j + 1] : v[length(v) + j + 1]

end # module
