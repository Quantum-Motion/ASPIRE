# Converged Procrustes floors of ASPIRE and the MPD at fixed layer count.
#
#   julia --project=experiments experiments/converged_floors.jl
#
# For each target and layer count L the MPD circuit of L layers and the ASPIRE circuit with the same
# number of gates, L (N - 1), are each refit by Procrustes sweeps until the infidelity stops improving
# (`optimize_to_floor`). ASPIRE places its gates by greedy disjoint packing of R_ij / (2|i-j|+1)^gamma;
# the reported ASPIRE floor is the best over the gamma and layer widths of the settings,. Infidelity is measured against the fixed |0...0>. A floor that stopped at the sweep cap is a
# bound, marked by a hollow point.
include(joinpath(@__DIR__, "settings.jl"))
using Plots

S = FLOORS
out = outdir("converged_floors")
write_json(joinpath(out, "settings.json"), S)

sim, opt = S.simulation, S.optimizer
floor_of(psi, circuit) = optimize_to_floor(psi, circuit; rtol=opt.rtol, cap=opt.cap,
    patience=opt.patience, max_bond_dim=sim.max_bond_dim, cutoff=sim.cutoff,
    trace_growth=opt.trace_growth, bond_resolved=opt.bond_resolved)

rows = Any[]
with_truncation(sim) do
    for t in S.targets, seed in t.seeds
        psi = build_target(t, seed; dmrg_sweeps=S.dmrg_sweeps, dmrg_maxdim=S.dmrg_maxdim,
            dmrg_cutoff=S.dmrg_cutoff)
        truncate!(psi; maxdim=S.target_chi); normalize!(psi)
        N = length(psi)
        for L in S.layers
            @info "$(t.name) seed $seed, $L layers"
            mpd = mpd_circuit(psi, L; tolerance=S.mpd.tolerance, max_bond_dim=sim.max_bond_dim,
                cutoff=sim.cutoff)
            r = floor_of(psi, mpd)
            push!(rows, (t.name, seed, L, "MPD", 0.0, 0, r.gates, r.infid0, r.floor, r.sweeps, r.capped))
            a = S.aspire
            settings = vcat([(g, a.max_batch[1]) for g in a.gamma_max_batch4],
                [(g, 1) for g in a.gamma_max_batch1])
            for (gamma, width) in settings
                res = protocol_arb_iter_oall(psi; budget=L * (N - 1), sweeps=0, optimise=false,
                    proposals=[(a.select, a.gate)], span_penalty=gamma, max_batch=width,
                    min_gain=a.min_gain, cutoff=sim.cutoff, max_bond_dim=sim.max_bond_dim)
                r = floor_of(psi, res.circuits[end])
                push!(rows, (t.name, seed, L, "ASPIRE", gamma, width, r.gates, r.infid0, r.floor,
                    r.sweeps, r.capped))
            end
        end
    end
end
write_csv(joinpath(out, "floors.csv"),
    ["target", "seed", "layers", "arm", "gamma", "max_batch", "gates", "eps_before", "eps_floor",
        "sweeps", "capped"], rows)

# Best ASPIRE setting per (target, seed, layers); geometric mean over seeds.
p = plot(; xlabel="MPD layers", ylabel="floor infidelity", yscale=:log10, legend=:outerright,
    size=(780, 380), xticks=S.layers, margin=6Plots.mm)
colors = Dict(t.name => c for (t, c) in zip(S.targets, (:royalblue, :darkorange, :seagreen)))
for t in S.targets
    for (arm, ls, mk) in (("MPD", :dash, :square), ("ASPIRE", :solid, :circle))
        xs = Int[]; ys = Float64[]; capped = Bool[]
        for L in S.layers
            sel = [r for r in rows if r[1] == t.name && r[3] == L && r[4] == arm]
            isempty(sel) && continue
            vals = [minimum(r[9] for r in sel if r[2] == s) for s in t.seeds]
            push!(xs, L); push!(ys, geomean(floor_eps.(vals)))
            push!(capped, any(r[11] for r in sel))
        end
        plot!(p, xs, ys; label="$(t.name) $arm", color=colors[t.name], linestyle=ls, marker=mk,
            markerstrokecolor=colors[t.name], markercolor=[c ? :white : colors[t.name] for c in capped])
    end
end
savefig(p, joinpath(out, "converged_floors.png"))
savefig(p, joinpath(out, "converged_floors.pdf"))
@info "wrote $out"
