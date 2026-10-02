# Where the advantage of ASPIRE comes from, in native CNOTs on a linear chain.
#
#   julia --project=experiments experiments/advantage_decomposition.jl
#
# Arms, each recorded gate by gate on the same states (N = 12; Heisenberg 4x3, TFIM 3x4, random MPS):
#   mpd                    the MPD staircase, undone gate by gate;
#   mpd_pattern            ASPIRE eigenbasis gates on the supports of the MPD, in its order;
#   aspire_r1              ASPIRE restricted to nearest-neighbour pairs;
#   aspire_inf_routed      ASPIRE on all pairs, ranked on R_ij, charged for its SWAPs on the chain;
#   aspire_inf_free        the same trajectory charged as if all pairs were coupled;
#   aspire_inf_costaware   ASPIRE ranked on R_ij / c_ij with the chain costs, charged for its SWAPs.
# A generic gate costs 3 CNOTs and a SWAP 3 CNOTs; SWAPs are counted on the running layout, so the
# routing permutation left by one gate is kept for the next (c_ij = 3 (1 + s_ij)). Infidelity is
# against the fixed |0...0>. For the selector arms the admission threshold rho and the layer width
# (all disjoint pairs or one pair) are scanned; each arm is drawn with the setting of lowest median
# infidelity within the CNOT budget.
include(joinpath(@__DIR__, "settings.jl"))
import MPSCircuits
using Plots

S = DECOMPOSITION
out = outdir("advantage_decomposition")
write_json(joinpath(out, "settings.json"), S)

chain(N) = [(i, i + 1) for i in 1:(N - 1)]

# Routing accountant: SWAPs needed by each gate on the chain, given the permutation left by the
# gates before it.
struct Charger
    tables::Any
    layout::RoutingLayout
end
Charger(N::Int) = Charger(graph_routing_tables(N, chain(N)), RoutingLayout(N))
function charge!(c::Charger, i::Int, j::Int)
    d = layout_distance(c.layout, c.tables.distances, i, j)
    route_gate!(c.layout, c.tables.distances, c.tables.predecessors, i, j)
    return routing_swaps(d)
end

# A record is (gate index, cumulative SWAPs, infidelity).
function mpd_records(psi, S)
    sim = S.simulation
    circuit = mpd_circuit(psi, S.mpd.layers; tolerance=S.mpd.tolerance,
        max_bond_dim=sim.max_bond_dim, cutoff=sim.cutoff)
    sites = siteinds(psi)
    work = ITensorMPS.dense(copy(psi))
    charger = Charger(length(psi))
    recs = Tuple{Int,Int,Float64}[]; supports = Tuple{Int,Int}[]
    swaps = 0
    for g in reverse(circuit)
        work = MPSCircuits.apply_gate(MPSCircuits.dagger(g), work; mixed_device=:coerce,
            cutoff=sim.cutoff, maxdim=sim.max_bond_dim)
        support = gate_support(g.tensor, sites)
        support === nothing && continue          # single-qubit frame gates are free
        i, j = minmax(support...)
        push!(supports, (i, j))
        swaps += charge!(charger, i, j)
        push!(recs, (length(supports), swaps, zero_product_amplitude_infidelity(normalize!(copy(work)))))
    end
    return recs, supports
end

function pattern_records(psi, supports, config)
    work = copy(psi); charger = Charger(length(psi))
    recs = Tuple{Int,Int,Float64}[]; swaps = 0
    for (k, (i, j)) in enumerate(supports)
        step = aspire_support_step(work, [(i, j)]; config=config, layer=k)
        isempty(step.gates) && break
        work = step.psi
        swaps += charge!(charger, i, j)
        push!(recs, (k, swaps, zero_product_amplitude_infidelity(work)))
    end
    return recs
end

function selector_records(psi, pairs, config, S; track_layout=false)
    N = length(psi); work = copy(psi); charger = Charger(N)
    recs = Tuple{Int,Int,Float64}[]; gate = 0; swaps = 0
    for layer in 1:config.max_depth
        gate >= S.logical_cap && break
        step = aspire_step(work; config=config, pairs=pairs, layer=layer,
            gate_limit=S.logical_cap - gate, layout=track_layout ? charger.layout : nothing)
        isempty(step.gates) && break
        for g in step.gates
            i, j = minmax(g.i, g.j)
            work = apply_two_qubit_unitary(work, g.i, g.j, g.U)
            gate += 1
            # The tracked layout is advanced by the charge; otherwise charging uses its own copy.
            swaps += charge!(charger, i, j)
            push!(recs, (gate, swaps, zero_product_amplitude_infidelity(work)))
        end
    end
    return recs
end

rows = Any[]   # (target, arm, rho, batch, seed, gate, cnots_routed, cnots_free, infidelity)
add!(recs, t, arm, rho, batch, seed) = for (g, sw, e) in recs
    push!(rows, (t, arm, rho, batch, seed, g, S.pair_gate_cost * g + S.swap_cost * sw,
        S.pair_gate_cost * g, floor_eps(e)))
end

sel = S.aspire
for t in S.targets, seed in S.seeds
    @info "$(t.name) seed $seed"
    with_truncation(S.simulation) do
        psi = build_target(t, seed; dmrg_sweeps=S.dmrg_sweeps,
            dmrg_maxdim=get(t, :dmrg_maxdim, nothing), dmrg_cutoff=S.dmrg_cutoff)
        N = length(psi)
        mrecs, supports = mpd_records(psi, S)
        add!(mrecs, t.name, "mpd", NaN, "-", seed)
        for rho in S.rhos, batch in S.batches
            width = batch === :all ? typemax(Int) : batch
            kw = (; relative_threshold=rho, max_batch_size=width, score_jitter=sel.score_jitter,
                max_depth=sel.max_depth)
            free = aspire_config(sel; tie_break_seed=seed, kw...)
            aware = aspire_config(sel; tie_break_seed=seed, hardware_graph=chain(N),
                cost_normalised=true, kw...)
            add!(pattern_records(psi, supports, free), t.name, "mpd_pattern", rho, batch, seed)
            add!(selector_records(psi, radius_graph(N, 1), free, S), t.name, "aspire_r1", rho, batch, seed)
            inf = selector_records(psi, radius_graph(N, N - 1), free, S)
            add!(inf, t.name, "aspire_inf_routed", rho, batch, seed)
            add!(inf, t.name, "aspire_inf_free", rho, batch, seed)
            add!(selector_records(psi, radius_graph(N, N - 1), aware, S;
                track_layout=S.cost_aware_tracks_layout), t.name, "aspire_inf_costaware", rho, batch, seed)
        end
    end
end
write_csv(joinpath(out, "gate_records.csv"),
    ["target", "arm", "rho", "batch", "seed", "gate", "cnots_routed", "cnots_free", "infidelity"], rows)

# Arm -> which cost column it is charged on.
arms = [("mpd", :routed), ("mpd_pattern", :routed), ("aspire_r1", :routed),
    ("aspire_inf_routed", :routed), ("aspire_inf_free", :free), ("aspire_inf_costaware", :routed)]
colors = Dict("mpd" => :firebrick, "mpd_pattern" => :seagreen, "aspire_r1" => :gray,
    "aspire_inf_routed" => :royalblue, "aspire_inf_free" => :black, "aspire_inf_costaware" => :darkorange)
cost_of(r, mode) = mode === :routed ? r[7] : r[8]

# Median over seeds at each gate index of the cost (x) and the infidelity (y), within the budget.
function curve(sel_rows, mode)
    gates = sort(unique(r[6] for r in sel_rows))
    xs = Float64[]; ys = Float64[]
    for g in gates
        at = [r for r in sel_rows if r[6] == g]
        x = median([cost_of(r, mode) for r in at])
        x > S.native_budget && break
        push!(xs, x); push!(ys, median([r[9] for r in at]))
    end
    return xs, ys
end

panels = Any[]
summary = Any[]
for t in S.targets
    p = plot(; xlabel="native CNOTs", ylabel="infidelity", yscale=:log10, title=t.name,
        legend=t === last(S.targets) ? :outerright : false, legendfontsize=7,
        xlims=(0, S.native_budget))
    for (arm, mode) in arms
        pool = [r for r in rows if r[1] == t.name && r[2] == arm]
        settings = unique((r[3], r[4]) for r in pool)
        best = nothing
        for (rho, batch) in settings
            sub = [r for r in pool if isequal(r[3], rho) && r[4] == batch]
            xs, ys = curve(sub, mode)
            isempty(ys) && continue
            if best === nothing || ys[end] < best.y
                best = (xs=xs, ys=ys, y=ys[end], rho=rho, batch=batch)
            end
        end
        best === nothing && continue
        plot!(p, best.xs, best.ys; label=arm, color=colors[arm])
        push!(summary, (t.name, arm, best.rho, best.batch, best.y))
    end
    push!(panels, p)
end
write_csv(joinpath(out, "best_settings.csv"),
    ["target", "arm", "rho", "batch", "median_infidelity_at_budget"], summary)
plt = plot(panels...; layout=(1, length(panels)), size=(450 * length(panels) + 150, 400), margin=6Plots.mm)
savefig(plt, joinpath(out, "advantage_decomposition.png"))
savefig(plt, joinpath(out, "advantage_decomposition.pdf"))
@info "wrote $out"
