# Kicked-Ising state on a heavy-hex fragment under global depolarising noise.
#
#   julia --project=experiments experiments/heavy_hex_noise.jl
#
# Target: kicked Ising state (theta_h, two kicks) on `cells` edge-sharing decorated hexagons, ordered
# along the MPS chain to minimise the cost of an MPD staircase (N = 12 at one cell). The MPD runs
# its staircase layers with Procrustes sweeps after each layer. ASPIRE is restricted to the device
# couplers, so each emitted gate is one native gate; three admission rules are run (no thresholds,
# rho = 0.9, tau = 1e-3). Every circuit checkpoint carries its noiseless squared fidelity F0 with the
# product-frame layer; the noisy infidelity follows from F = q F0 + (1 - q) 2^-N with
# q = (1 - 16p/15)^M for M native gates. The MPD is charged MPD_SWAPS_PER_GATE SWAPs
# per emitted gate. Left panel: infidelity against native gates for the noise levels p. Right panel:
# the best infidelity of each arm against p (minimum over checkpoints for the MPD, last checkpoint
# for the ASPIRE arms, which stop by their admission rule or at the gate budget).
include(joinpath(@__DIR__, "settings.jl"))
using Plots

S = HEAVY_HEX
out = outdir("heavy_hex_noise")
write_json(joinpath(out, "settings.json"), S)

native_mpd(g) = native_gate_count(g; swaps_per_gate=S.noise.mpd_swaps_per_gate,
    swap_gate_cost=S.noise.swap_gate_cost)

records = Dict{String,Any}()      # arm => (native gate counts, F0 values)
N = 0
with_truncation(S.simulation) do
    global N
    patch = heavy_hex_patch(S.cells; ordering=S.ordering)
    N = length(patch.coordinates)
    @info "heavy-hex fragment: N = $N, $(length(patch.pairs)) couplers"
    psi = graph_kicked_ising_state(N, patch.pairs; theta_h=S.theta_h, steps=S.steps)
    @info "target link dimension $(maxlinkdim(psi))"

    mpd = mpd_checkpoints(psi; layers=S.mpd.layers, sweeps=S.mpd.sweeps, tolerance=S.mpd.tolerance)
    records["MPD"] = ([native_mpd(r.two_qubit_gates) for r in mpd], [r.squared_fidelity for r in mpd],
        [r.two_qubit_gates for r in mpd])

    a = S.aspire
    for arm in S.aspire_arms
        cfg = aspire_config(a; max_depth=a.max_depth, max_gates=a.gate_budget,
            threshold=arm.tau, relative_threshold=arm.rho, entanglement_weight=a.entanglement_weight)
        ck = aspire_checkpoints(psi, cfg; pairs=patch.pairs, sweeps=a.sweeps,
            gate_budget=a.gate_budget, checkpoint_interval=a.checkpoint_interval)
        gates = [r.two_qubit_gates for r in ck]
        records["ASPIRE " * arm.name] = (gates, [r.squared_fidelity for r in ck], gates)
    end
end

noisy(F0, M, p) = global_depolarizing_infidelity(F0, M, p, N)
curves = Any[]; best = Any[]
for (arm, (native, F0, logical)) in records, p in unique(vcat(S.noise.p, S.noise.p_scan))
    eps = [noisy(F0[k], native[k], p) for k in eachindex(F0)]
    p in S.noise.p && for k in eachindex(F0)
        push!(curves, (arm, p, k - 1, logical[k], native[k], F0[k], eps[k]))
    end
    push!(best, (arm, p, minimum(eps), arm == "MPD" ? minimum(eps) : eps[end]))
end
write_csv(joinpath(out, "curves.csv"),
    ["arm", "p", "checkpoint", "logical_gates", "native_gates", "F0", "infidelity"], curves)
write_csv(joinpath(out, "best.csv"), ["arm", "p", "min_over_checkpoints", "reading"], best)

pa = plot(; xlabel="native two-qubit gates", ylabel="infidelity", yscale=:log10, legend=:outertop,
    xlims=(0, S.noise.x_max))
palette = [:royalblue, :darkorange, :seagreen, :purple]
for (c, p) in zip(palette, S.noise.p)
    for arm in ("MPD", "ASPIRE fill")
        sel = [r for r in curves if r[1] == arm && r[2] == p]
        plot!(pa, [r[5] for r in sel], [r[7] for r in sel]; color=c,
            linestyle=arm == "MPD" ? :dash : :solid, marker=:circle, markersize=3,
            label="$arm, p = $p")
    end
end
pb = plot(; xlabel="p", ylabel="best infidelity", xscale=:log10, yscale=:log10, legend=:topleft)
for (c, arm) in zip(palette, sort(collect(keys(records))))
    sel = sort([r for r in best if r[1] == arm]; by=r -> r[2])
    plot!(pb, [r[2] for r in sel], [r[4] for r in sel]; color=c, marker=:circle, label=arm)
end
plt = plot(pa, pb; layout=(1, 2), size=(1100, 400), margin=6Plots.mm)
savefig(plt, joinpath(out, "heavy_hex_noise.png"))
savefig(plt, joinpath(out, "heavy_hex_noise.pdf"))
@info "wrote $out"
