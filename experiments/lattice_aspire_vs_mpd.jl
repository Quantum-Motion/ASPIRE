# ASPIRE against the MPD on TFIM ground states of square lattices, at equal gate count.
#
#   julia --project=experiments experiments/lattice_aspire_vs_mpd.jl
#
# For every lattice and seed the MPD is run for its layer budget (unoptimised), ASPIRE (exact matching,
# no thresholds, connectivity radius 2 Lx - 1) is run to the same number of gates, and both are
# scored by the product-frame infidelity. The T count of the framed circuits is recorded along the
# curves. Both final circuits are then refit by Procrustes sweeps (rolling environment) up to the same
# classical cost. Left panel: infidelity against gates for the largest lattice. Right panel: the
# advantage over the lattice size, as the infidelity ratio at equal gates and the gate saving at
# equal infidelity.
include(joinpath(@__DIR__, "settings.jl"))
import MPSCircuits
using Plots

S = LATTICE
out = outdir("lattice_aspire_vs_mpd")
write_json(joinpath(out, "settings.json"), S)

# Procrustes sweeps of the two-qubit gates of a framed circuit, frame held fixed, until the cost
# budget (bond-resolved chi^3 operations of the two-qubit circuit `bare`) or the sweep cap is reached.
function optimise_to_budget(psi, framed, bare, opt, sim)
    nframe = length(framed) - count(g -> length(g.site_numbers) == 2, framed)
    per_sweep = sweep_cost_model(psi, bare; cutoff=sim.cutoff, max_bond_dim=sim.max_bond_dim,
        bond_resolved=opt.bond_resolved).chi3_ops
    nsweeps = min(opt.max_sweeps, floor(Int, opt.chi3_budget / per_sweep))
    circuit = deepcopy(framed)
    target = ITensorMPS.dense(copy(psi))
    for _ in 1:nsweeps
        MPSCircuits.replace_gates!(target, circuit, MPSCircuits.RollingEnvironment(),
            (nframe + 1):length(circuit); max_bond_dim=sim.max_bond_dim, working_cutoff=sim.cutoff)
    end
    return nsweeps, rudolph_infidelity(circuit, psi; cutoff=sim.cutoff, max_bond_dim=sim.max_bond_dim)
end

curves = Any[]; summary = Any[]
for lat in S.lattices, seed in lat.seeds
    N = lat.L^2
    @info "lattice $(lat.L)x$(lat.L), seed $seed"
    with_truncation(S.simulation) do
        t = merge(S.target, (Lx=lat.L, Ly=lat.L))
        psi = build_target(t, seed; dmrg_sweeps=lat.dmrg_sweeps, dmrg_maxdim=lat.dmrg_maxdim,
            dmrg_cutoff=S.target.dmrg_cutoff)
        sites = siteinds(psi)

        mpd = ran_iter_oall_curve(psi; Lmax=lat.layers, sweeps=S.mpd.sweeps,
            tolerance=S.mpd.tolerance, t_error=S.tcount.total_error)
        budget = mpd.gates[end]
        eps_mpd = amplitude_infidelity_from_squared.(mpd.fid)
        eps_mpd[1] = product_frame_amplitude_infidelity(psi)     # zero-gate point carries the frame

        cfg = aspire_config(S.aspire; max_gates=budget)
        asp = aspire(psi; config=cfg, pairs=radius_graph(N, 2 * lat.L - 1))
        eps_asp = asp.product_frame_amplitude_infidelity

        # T count of the framed circuits at about a dozen layer boundaries, frame charged to both arms.
        stride = max(1, cld(length(asp.gates) - 1, 12))
        tcount_asp = fill(NaN, length(asp.gates)); tcount_asp[1] = 0.0
        for k in 2:stride:length(asp.gates)
            circ = prep_circuit_from_gate_log(asp.gate_log[1:asp.gates[k]], sites)
            tcount_asp[k] = t_count(framed_circuit(psi, circ); total_error=S.tcount.total_error)
        end
        for k in eachindex(mpd.gates)
            push!(curves, (N, seed, "MPD", mpd.depth[k], mpd.gates[k], eps_mpd[k], mpd.tcount[k]))
        end
        for k in eachindex(asp.gates)
            push!(curves, (N, seed, "ASPIRE", asp.depth[k], asp.gates[k], eps_asp[k], tcount_asp[k]))
        end

        # Optimised arms at the final gate count.
        n_mpd, opt_mpd = optimise_to_budget(psi, mpd.circuit, mpd.two_qubit_circuit, S.optimizer,
            S.simulation)
        bare_asp = prep_circuit_from_gate_log(asp.gate_log, sites)
        n_asp, opt_asp = optimise_to_budget(psi, framed_circuit(psi, bare_asp), bare_asp,
            S.optimizer, S.simulation)

        g_save = gates_to_reach(asp.gates, eps_asp, eps_mpd[end])
        push!(summary, (N, seed, budget, asp.gates[end], eps_mpd[end], eps_asp[end],
            eps_mpd[end] / eps_asp[end], budget / g_save, n_mpd, opt_mpd, n_asp, opt_asp))
    end
end

write_csv(joinpath(out, "curves.csv"),
    ["N", "seed", "arm", "layer", "gates", "infidelity", "t_count"], curves)
write_csv(joinpath(out, "summary.csv"),
    ["N", "seed", "mpd_gates", "aspire_gates", "eps_mpd", "eps_aspire", "eps_ratio",
        "gate_saving_at_equal_eps", "sweeps_mpd", "eps_mpd_optimised", "sweeps_aspire",
        "eps_aspire_optimised"], summary)

Nmax = maximum(r[1] for r in summary)
pa = plot(; xlabel="two-qubit gates", ylabel="infidelity", yscale=:log10, legend=:bottomleft,
    title="N = $Nmax")
for (arm, c) in (("MPD", :firebrick), ("ASPIRE", :royalblue))
    sel = [r for r in curves if r[1] == Nmax && r[2] == first(S.lattices[end].seeds) && r[3] == arm]
    plot!(pa, [r[5] for r in sel], floor_eps.([r[6] for r in sel]); label=arm, color=c, marker=:circle)
end
last = [r for r in summary if r[1] == Nmax][1]
scatter!(pa, [last[3]], [last[10]]; color=:firebrick, marker=:diamond, label="MPD, optimised")
scatter!(pa, [last[4]], [last[12]]; color=:royalblue, marker=:diamond, label="ASPIRE, optimised")

Ns = sort(unique(r[1] for r in summary))
ratio = [geomean([r[7] for r in summary if r[1] == N]) for N in Ns]
saving = [geomean([r[8] for r in summary if r[1] == N && isfinite(r[8])]) for N in Ns]
pb = plot(Ns, ratio; marker=:circle, label="infidelity ratio at equal gates", xlabel="N",
    ylabel="MPD / ASPIRE", yscale=:log10, color=:black, legend=:topright, ylims=(0.9, 12))
plot!(pb, Ns, saving; marker=:square, label="gate ratio at equal infidelity", color=:gray)
hline!(pb, [1.0]; color=:gray, linestyle=:dash, label="")
plt = plot(pa, pb; layout=(1, 2), size=(900, 380), margin=4Plots.mm)
savefig(plt, joinpath(out, "lattice_aspire_vs_mpd.png"))
savefig(plt, joinpath(out, "lattice_aspire_vs_mpd.pdf"))
@info "wrote $out"
