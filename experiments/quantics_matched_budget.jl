# Correlated Gaussian quantics states, ASPIRE against the MPD at a matched gate budget.
#
#   julia --project=experiments experiments/quantics_matched_budget.jl
#
# Target: the Gaussian density of `d` variables with correlation rho = 0.6 on a grid of 2^n points per
# variable, serial bit ordering, N = d n qubits. The MPD runs 8 layers (8 (N - 1) gates) without
# optimisation. ASPIRE (exact matching, no thresholds, all-to-all) is run to the same number of gates.
# Each arm is read at the best point of its own curve within the budget, in the product-frame
# infidelity. Left panel: infidelity against n for each d. Right panel: the ratio MPD / ASPIRE against d.
include(joinpath(@__DIR__, "settings.jl"))
using Plots

S = QUANTICS
out = outdir("quantics_matched_budget")
write_json(joinpath(out, "settings.json"), S)

rows = Any[]          # (d, n, seed, eps_mpd, eps_aspire)
for (d, ns) in S.grid, n in ns, seed in S.seeds
    N = d * n
    @info "d = $d, n = $n, seed $seed"
    with_truncation(S.simulation) do
        psi = quantics_target(S, d, n, seed)
        budget = S.mpd.layers * (N - 1)
        mpd = ran_iter_oall_curve(psi; Lmax=S.mpd.layers, sweeps=S.mpd.sweeps,
            tolerance=S.mpd.tolerance)
        eps_mpd = amplitude_infidelity_from_squared.(mpd.fid)
        eps_mpd[1] = product_frame_amplitude_infidelity(psi)
        asp = aspire(psi; config=aspire_config(S.aspire; max_gates=budget))
        push!(rows, (d, n, seed, floor_eps(minimum(eps_mpd[mpd.gates .<= budget])),
            floor_eps(minimum(asp.product_frame_amplitude_infidelity))))
    end
end
write_csv(joinpath(out, "matched_budget.csv"), ["d", "n", "seed", "eps_mpd", "eps_aspire"], rows)

ds = sort(unique(r[1] for r in rows))
pa = plot(; xlabel="bits per variable n", ylabel="infidelity at 8 (N-1) gates", yscale=:log10,
    legend=:outerright)
pb = plot(; xlabel="variables d", ylabel="MPD / ASPIRE", yscale=:log10, legend=:outerright)
for (c, d) in zip(palette(:tab10), ds)
    ns = sort(unique(r[2] for r in rows if r[1] == d))
    agg(col) = [geomean([r[col] for r in rows if r[1] == d && r[2] == n]) for n in ns]
    plot!(pa, ns, agg(4); color=c, linestyle=:dash, marker=:square, label="MPD, d = $d")
    plot!(pa, ns, agg(5); color=c, marker=:circle, label="ASPIRE, d = $d")
end
for n in sort(unique(r[2] for r in rows))
    dn = sort(unique(r[1] for r in rows if r[2] == n))
    length(dn) > 1 || continue
    ratio = [geomean([r[4] / r[5] for r in rows if r[1] == d && r[2] == n]) for d in dn]
    plot!(pb, dn, ratio; marker=:circle, label="n = $n")
end
hline!(pb, [1.0]; color=:gray, linestyle=:dash, label="")
plt = plot(pa, pb; layout=(1, 2), size=(1000, 380), margin=4Plots.mm)
savefig(plt, joinpath(out, "quantics_matched_budget.png"))
savefig(plt, joinpath(out, "quantics_matched_budget.pdf"))
@info "wrote $out"
