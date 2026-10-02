# Infidelity of ASPIRE at a fixed gate budget as a function of the connectivity radius.
#
#   julia --project=experiments experiments/connectivity_radius.jl
#
# The MPD runs 6 layers (66 gates on N = 12) without optimisation. ASPIRE with the greedy min-cut
# selector on I(i:j) (thresholds tau, rho, cross-pair bound kappa) is restricted to pairs with
# |i - j| <= r and run to the same gate count, for every radius r. ASPIRE is scored against the
# fixed |0...0>, the MPD circuit carries its analytic frame. Per target the figure shows the
# seed-geometric-mean infidelity against r, the MPD level, and the first radius at which ASPIRE
# is below the MPD (star). The table also gives the onset by a median rule:
# the first radius at which the median over seeds of log10(eps_MPD / eps_ASPIRE) is positive.
include(joinpath(@__DIR__, "settings.jl"))
using Plots

S = CONNECTIVITY
out = outdir("connectivity_radius")
write_json(joinpath(out, "settings.json"), S)

sel = S.aspire
rows = Any[]        # (target, seed, radius, gates, eps_aspire, eps_mpd)
for t in S.targets, seed in S.seeds
    @info "$(t.name) seed $seed"
    with_truncation(S.simulation) do
        psi = build_target(t, seed; dmrg_sweeps=S.dmrg_sweeps, dmrg_maxdim=t.dmrg_maxdim,
            dmrg_cutoff=S.dmrg_cutoff)
        N = length(psi)
        mpd = ran_iter_oall_curve(psi; Lmax=S.mpd.layers, sweeps=S.mpd.sweeps,
            tolerance=S.mpd.tolerance)
        eps_mpd = amplitude_infidelity_from_squared(mpd.fid[end])
        for r in S.radii
            gates, eps, _ = aspire_to_budget(psi; pairs=radius_graph(N, r), budget=mpd.gates[end],
                tau=sel.tau, rho=sel.rho, kappa=sel.kappa)
            push!(rows, (t.name, seed, r, gates[end], floor_eps(eps[end]), floor_eps(eps_mpd)))
        end
    end
end
write_csv(joinpath(out, "radius_scan.csv"),
    ["target", "seed", "radius", "gates", "eps_aspire", "eps_mpd"], rows)

summary = Any[]
panels = Any[]
for t in S.targets
    rs = collect(S.radii)
    gm_a = [geomean([x[5] for x in rows if x[1] == t.name && x[3] == r]) for r in rs]
    gm_m = [geomean([x[6] for x in rows if x[1] == t.name && x[3] == r]) for r in rs]
    med = [median([log10(x[6] / x[5]) for x in rows if x[1] == t.name && x[3] == r]) for r in rs]
    star = findfirst(gm_a .<= gm_m)
    onset = findfirst(med .> 0)
    push!(summary, (t.name, star === nothing ? -1 : rs[star], onset === nothing ? -1 : rs[onset]))
    p = plot(rs, gm_a; marker=:circle, color=:royalblue, label="ASPIRE", yscale=:log10,
        xlabel="radius r", ylabel="infidelity", title=t.name, legend=:topright)
    hline!(p, [gm_m[1]]; color=:firebrick, linestyle=:dash, label="MPD")
    star === nothing || scatter!(p, [rs[star]], [gm_a[star]]; marker=:star5, markersize=10,
        color=:gold, label="first r below MPD")
    push!(panels, p)
end
write_csv(joinpath(out, "onset.csv"), ["target", "first_radius_below_mpd", "first_radius_median_positive"],
    summary)
plt = plot(panels...; layout=(1, length(panels)), size=(450 * length(panels), 380), margin=4Plots.mm)
savefig(plt, joinpath(out, "connectivity_radius.png"))
savefig(plt, joinpath(out, "connectivity_radius.pdf"))
@info "wrote $out" summary
