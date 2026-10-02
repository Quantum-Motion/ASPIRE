@testset "Saturation" begin
    rule = SaturationRule()

    @testset "rule parsing" begin
        @test saturation_rule(nothing) === nothing
        r = saturation_rule(Dict("mode" => "saturation", "gain_per_doubling" => 1.2))
        @test r.gain_per_doubling == 1.2
        @test r.confirm == SaturationRule().confirm      # untouched keys keep defaults
        @test_throws ErrorException saturation_rule(Dict("mode" => "fixed"))
        # The cap is quoted per site and converted to a total gate count.
        @test saturation_gate_cap(SaturationRule(max_gates_per_site=10.0), 16) == 160
    end

    @testset "curve arithmetic" begin
        g = [0, 10, 20, 40]
        e = [1.0, 1e-1, 1e-2, 1e-3]
        # The zero-gate sample prices the product state, not a circuit: dropped.
        @test curve_infidelity_at(g, e, 10) ≈ 1e-1
        @test curve_infidelity_at(g, e, 5) ≈ 1e-1          # clamped below
        @test curve_infidelity_at(g, e, 400) ≈ 1e-3        # clamped above
        @test curve_infidelity_at(g, e, 30) ≈ sqrt(1e-2 * 1e-3)   # log-linear
        @test doubling_gain(g, e) ≈ 1e-2 / 1e-3
        # A window reaching back past the first sample cannot be read as flat.
        @test doubling_gain([0, 10, 15], [1.0, 1e-1, 1e-2]) == Inf
    end

    @testset "saturation state" begin
        gates = collect(0:10:800)
        # A curve that has genuinely flattened: geometric descent onto a floor,
        # run well past the point where it went quiet so `confirm` is paid.
        flat = [k == 0 ? 1.0 : 1e-3 + 1.0 * exp(-k / 1.5) for k in 0:80]
        sat, reason, g_first = saturation_state(gates, flat, rule)
        @test sat
        @test reason == "saturated"
        @test g_first < gates[end]        # reports where it went quiet, not the endpoint

        # Still descending as a power law: never saturates, whatever the budget.
        power = [k == 0 ? 1.0 : 1.0 / k for k in 0:80]
        @test !first(saturation_state(gates, power, rule))
        @test last(saturation_state(gates, power, rule)) === nothing

        # Staircase guard: the curve sits on a flat tread and then drops. On the
        # tread the last doubling is quiet, so only `confirm` prevents a stop.
        tread_gates = [0, 10, 20, 30, 40, 50]
        tread = [1.0, 5e-1, 2.0e-1, 1.99e-1, 1.98e-1, 1.97e-1]
        sat_t, reason_t, quiet = saturation_state(tread_gates, tread, rule)
        @test !sat_t
        @test reason_t == "quiet but unconfirmed"
        @test quiet !== nothing            # the quiet window is still reported

        # Extending the budget past the confirmation stretch exposes the drop,
        # and the tread is not reported as saturation.
        cliff_gates = vcat(tread_gates, [60, 70, 80])
        cliff = vcat(tread, [4.0e-2, 3.9e-2, 3.85e-2])
        @test !first(saturation_state(cliff_gates, cliff, rule))

        # Solved target: below the floor is done regardless of the taper.
        solved = [1.0, 1e-4, 1e-8, 1e-12, 1e-14]
        @test first(saturation_state([0, 10, 20, 30, 40], solved, rule))

        # Too coarse to evaluate: only one sample in the trailing doubling.
        @test !first(saturation_state([0, 10, 100], [1.0, 1e-2, 9.9e-3], rule))
        @test last(saturation_state([0, 10, 100], [1.0, 1e-2, 9.9e-3], rule)) === nothing

        # The minimum-budget bracket is per site, so it scales with N.
        @test !first(saturation_state(gates, flat, rule; N=400))
    end

    @testset "threshold monotonicity" begin
        # A laxer threshold can only declare saturation earlier, never later.
        gates = collect(0:10:400)
        curve = [k == 0 ? 1.0 : 1e-3 + exp(-k / 6) for k in 0:40]
        strict = saturation_state(gates, curve, SaturationRule(gain_per_doubling=1.02))
        lax = saturation_state(gates, curve, SaturationRule(gain_per_doubling=1.30))
        @test lax[3] === nothing || strict[3] === nothing || lax[3] <= strict[3]
    end
end
