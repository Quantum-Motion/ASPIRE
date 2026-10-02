"""
    ASPIRE

State preparation for matrix product states by entanglement-informed gate placement (ASPIRE),
with the matrix product disentangler (MPD) as baseline. The submodules implement the quantities
of the accompanying paper: removable mutual information and the Lambda certificate (`Correlation`, `Selection`),
the ASPIRE selector (`Selection`), native-gate accounting under routing (`HardwareCost`), Procrustes
optimisation (`CircuitOptimization`, `WarmStartSelection`, `EnergyProcrustes`), the baseline
(`Baselines`) and the target states (`ModelStates`, `ToyStates`, `QuanticsStates`). All their
public names are exported here.
"""
module ASPIRE

include("SimConfig.jl")
include("FidelityMetrics.jl")
include("Matching.jl")
include("EntanglementBasics.jl")
include("RDM.jl")
include("HardwareCost.jl")
include("Correlation.jl")
include("Disentangling.jl")
include("Selection.jl")
include("ModelStates.jl")
include("ToyStates.jl")
include("QuanticsStates.jl")
include("Saturation.jl")
include("TCount.jl")
include("OptimizerCost.jl")
include("EnergyProcrustes.jl")
include("CircuitOptimization.jl")
include("Baselines.jl")
include("WarmStartSelection.jl")
include("Noise.jl")

using .SimConfig, .FidelityMetrics, .Matching, .EntanglementBasics, .RDM, .HardwareCost
using .Correlation, .Disentangling, .Selection, .ModelStates, .ToyStates, .QuanticsStates
using .Saturation, .TCount, .OptimizerCost, .EnergyProcrustes, .CircuitOptimization
using .Baselines, .WarmStartSelection, .Noise

# Truncation parameters
export global_cutoff, global_maxdim, set_cutoff!, set_max_bond_dim!, with_sim_config

# Fidelity conventions
export amplitude_fidelity, amplitude_fidelity_from_squared, amplitude_infidelity,
    amplitude_infidelity_from_squared, clamp_squared_fidelity, fidelity_series_from_squared,
    squared_fidelity

# Maximum-weight matching
export brute_force_matching, matching_weight, maximum_weight_matching

# Bond entropy
export bond_entropy

# Two-site reduced density matrices
export all_pairs_two_site_rdms, dense_two_site_matrix, quantum_mutual_information,
    two_site_rdm, two_site_rdms, von_neumann_entropy

# Native two-qubit gate accounting with routing
export DEFAULT_ROUTING_CONVENTION, ROUTING_CONVENTIONS, RoutingLayout, SWAP_GATE_COST,
    circuit_supports, gate_cost, gate_support, graph_distances, graph_routing_tables,
    hamiltonian_path, layout_distance, physical_gate_count, physical_gate_series,
    reorder_pairs, route_gate!, routed_gate_series, routed_layer_cost!, routing_ordering,
    routing_swaps, staircase_cost, staircase_supports

# Correlation quantities and the Lambda certificate
export certified_infidelity_bound, dense_reduced_density_matrix, infidelity_upper_bound,
    lambda_certificate, marginal, reduced_density_matrix, register_total_correlation,
    removable_correlation, removable_correlation_pair, removable_correlation_symmetric,
    site_entropies, total_correlation_support

# Two-qubit gates and disentanglers
export DisentanglingGate, ASPIRE_DEFAULT_RELATIVE_THRESHOLD, ASPIRE_DEFAULT_THRESHOLD,
    apply_two_qubit_unitary, apply_two_qubit_unitary_tracked, bond_dimension_summary,
    candidate_pairs, compressed_mps, disentangle_pair, eig_disentangler, haar_unitary,
    aspire_rank_threshold, mi_disentangler, mincut_batch, parallel_disentangling_with_gate_log,
    random_circuit_state, total_correlation, total_entanglement, two_qubit_gate

# ASPIRE selector
export ASPIREPairFeatures, ASPIREConfig, cost_normalised_transform, eigenbasis_gate, aspire,
    aspire_step, aspire_support_step, maximum_weight_cross_qmi_matching,
    maximum_weight_pair_matching, pair_native_cost, pair_spectrum_features,
    product_frame_amplitude_infidelity, product_frame_fidelity, removable_lambda,
    removable_mutual_information, select_layer, zero_product_amplitude_infidelity,
    zero_product_fidelity

# Model ground states and kicked-Ising states
export apply_one_qubit_unitary, bond_ramp, ed_ground_energy, ed_ground_gap, ed_ground_state,
    energy, graph_kicked_ising_state, heavy_hex_kicked_ising_state, heavy_hex_patch,
    heisenberg_snake_state, snake_index, snake_pairs, tfim_dmrg, tfim_hamiltonian, tfim_setup,
    tfim_snake_state

# Toy states
export cluster_state, coupled_bell_pairs, ghz_block_state

# Quantics function states
export QUANTICS_FAMILIES, QUANTICS_ORDERINGS, quantics_amplitudes, quantics_domain,
    quantics_function, quantics_function_state, quantics_grid_point, quantics_index_map,
    quantics_permutation, quantics_state_from_spec, weierstrass_series, weierstrass_terms

# Saturation stopping rule
export SaturationRule, curve_infidelity_at, doubling_gain, saturation_gate_cap,
    saturation_rule, saturation_state

# T-gate count
export GRIDSYNTH, TRASYN, rotation_count, t_count

# Cost of the Procrustes sweep
export bond_op_weight, bond_resolved_chi3_ops, bra_chain_bond_dims, bra_chain_link_dims,
    gate_bond_ops, gate_pairs, gate_spans, sweep_cost_model, sweep_site_ops

# Energy refinement
export energy_mps_coordinate_refine, optimize_energy_gate_mps!

# Circuits and Procrustes optimisation
export OPTIMIZER_CUTOFF, OPTIMIZER_MAXDIM, circuit_checkpoint, disentangle_with_prep_circuit,
    framed_circuit, measurement_schedule, oall_sweep!, optimize_to_floor, patience_converged,
    prep_circuit_from_gate_log, prep_from_disentanglers, product_frame_squared_fidelity,
    rudolph_infidelity

# Matrix product disentangler baseline
export aspire_checkpoints, aspire_layer_logged, mpd_checkpoints, mpd_circuit,
    ran_iter_oall_curve, random_gaussian_mps, zero_state_fidelity

# Warm-start Procrustes selection
export ARB_PROPOSALS, frame_optimal_gate, procrustes_pair_scores, protocol_arb_iter_oall,
    psel_layer, switch_schedule

# Global depolarising noise
export MPD_ROUTING_OVERHEAD, MPD_SWAPS_PER_GATE, global_depolarizing_infidelity,
    global_depolarizing_squared_fidelity, global_depolarizing_strength, native_gate_count

end
