using Test
using LinearAlgebra
using ITensors
using ITensorMPS
using ASPIRE

include("test_sim_config.jl")
include("test_fidelity_metrics.jl")
include("test_rdms.jl")
include("test_entanglement_metrics.jl")
include("test_correlation.jl")
include("test_disentangling.jl")
include("test_aspire.jl")
include("test_selection.jl")
include("test_hardware_cost.jl")
include("test_toy_states.jl")
include("test_quantics_states.jl")
include("test_saturation.jl")
include("test_t_count.jl")
include("test_circuit_optimization.jl")
include("test_baselines.jl")
include("test_warm_start.jl")
include("test_noise.jl")
include("test_energy_procrustes.jl")
