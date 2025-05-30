#!/usr/bin/env python
"""
Extinction probability analysis for Lotka-Volterra model.

This script compares extinction probabilities across three threshold strategies:
1. Large constant threshold (1000) - pure continuous
2. Medium constant threshold (30) - hybrid  
3. Dynamic jacobian-based threshold - adaptive

Each configuration is tested with 500 Monte Carlo runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
from jsf.analysis_utils import run_monte_carlo_extinction, compare_extinction_results
from jsf.dynamic_scaling import jacobian_lotka_volterra
import json
import time


def setup_lotka_volterra_model():
    """Set up Lotka-Volterra model parameters."""
    # Model parameters
    x0 = [50, 10]  # Initial: 50 prey, 10 predator
    alpha = 2.0    # Prey birth rate
    beta = 0.05    # Predation rate  
    gamma = 1.5    # Predator death rate
    
    # Rate function
    rates = lambda x, t: [alpha * x[0], gamma * x[1], beta * x[0] * x[1]]
    
    # Stoichiometric matrix
    reactant_matrix = [[1, 0], [0, 1], [1, 1]]
    product_matrix = [[2, 0], [0, 0], [0, 2]]
    
    stoich = {
        "nu": [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(product_matrix, reactant_matrix)],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }
    
    # Jacobian function for dynamic threshold strategy
    def jacobian_func(state):
        return jacobian_lotka_volterra(state, alpha, beta, gamma)
    
    return {
        "x0": x0,
        "rates": rates,
        "stoich": stoich,
        "jacobian_func": jacobian_func,
        "parameters": {"alpha": alpha, "beta": beta, "gamma": gamma}
    }


def create_test_configurations(jacobian_func):
    """Create the three test configurations."""
    
    # Base configuration shared by all cases
    base_config = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "method": "operator-splitting"
    }
    
    # Case A: Large constant threshold (pure continuous)
    config_large = base_config.copy()
    config_large.update({
        "SwitchingThreshold": [1000, 1000],
        "EnableDynamicThreshold": False
    })
    
    # Case B: Medium constant threshold (hybrid)
    config_medium = base_config.copy()
    config_medium.update({
        "SwitchingThreshold": [30, 30],
        "EnableDynamicThreshold": False
    })
    
    # Case C: Dynamic jacobian-based threshold
    config_dynamic = base_config.copy()
    config_dynamic.update({
        "SwitchingThreshold": [30, 30],  # Initial threshold
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": jacobian_func,
            "omega_min": 5.0,
            "omega_max": 100.0,
            "k": 2.5,
            "score_threshold": 0.1,
            "debug": False
        }
    })
    
    return {
        "large_threshold": config_large,
        "medium_threshold": config_medium, 
        "dynamic_jacobian": config_dynamic
    }


def run_extinction_analysis(n_runs=500, t_max=10.0, base_seed=42):
    """Run the complete extinction probability analysis."""
    
    print("="*60)
    print("EXTINCTION PROBABILITY ANALYSIS - LOTKA-VOLTERRA MODEL")
    print("="*60)
    print(f"Configuration:")
    print(f"  Number of runs per configuration: {n_runs}")
    print(f"  Simulation time: {t_max}")
    print(f"  Extinction threshold: 0.0 (exact zero)")
    print(f"  Base random seed: {base_seed}")
    print()
    
    # Set up model
    model = setup_lotka_volterra_model()
    print(f"Model parameters:")
    print(f"  Initial populations: {model['x0']}")
    print(f"  Alpha (prey birth): {model['parameters']['alpha']}")
    print(f"  Beta (predation): {model['parameters']['beta']}")
    print(f"  Gamma (predator death): {model['parameters']['gamma']}")
    print()
    
    # Create configurations
    configs = create_test_configurations(model["jacobian_func"])
    
    results = {}
    total_start_time = time.time()
    
    # Run each configuration
    for config_name, config in configs.items():
        print(f"Running configuration: {config_name.upper()}")
        print(f"  Threshold settings: {config.get('SwitchingThreshold', 'N/A')}")
        print(f"  Dynamic threshold: {config.get('EnableDynamicThreshold', False)}")
        
        start_time = time.time()
        
        # Run Monte Carlo simulation
        result = run_monte_carlo_extinction(
            base_config=config,
            model_params=model,
            n_runs=n_runs,
            base_seed=base_seed,
            extinction_threshold=0.0,
            t_max=t_max
        )
        
        end_time = time.time()
        results[config_name] = result
        
        # Display preliminary results
        prob = result["extinction_probability"]
        ci = result["confidence_interval"]
        print(f"  Extinction probability: {prob:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        print(f"  Extinctions: {result['extinctions']}/{n_runs}")
        print(f"  Runtime: {end_time - start_time:.1f} seconds")
        print()
    
    total_end_time = time.time()
    print(f"Total analysis time: {total_end_time - total_start_time:.1f} seconds")
    print()
    
    return results


def generate_summary_report(results, comparison):
    """Generate a summary report of the extinction analysis."""
    
    print("="*60)
    print("EXTINCTION PROBABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Results table
    print("Configuration Results:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Ext. Prob.':<12} {'95% CI':<20} {'Mean Ext. Time':<15}")
    print("-" * 80)
    
    for config_name, result in results.items():
        prob = result["extinction_probability"]
        ci = result["confidence_interval"]
        mean_time = result["mean_extinction_time"]
        
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        time_str = f"{mean_time:.2f}" if mean_time is not None else "N/A"
        
        print(f"{config_name:<20} {prob:<12.4f} {ci_str:<20} {time_str:<15}")
    
    print("-" * 80)
    print()
    
    # Statistical comparisons
    print("Pairwise Comparisons:")
    print("-" * 50)
    
    for comparison_name, stats in comparison["statistical_tests"].items():
        overlap = stats["confidence_intervals_overlap"]
        diff = stats["prob_diff"]
        
        status = "Overlapping CIs" if overlap else "Non-overlapping CIs"
        print(f"{comparison_name}: {status} (|Δp| = {diff:.4f})")
    
    print()
    
    # Threshold statistics for dynamic case
    if "dynamic_jacobian" in results and results["dynamic_jacobian"]["threshold_stats"]:
        thresh_stats = results["dynamic_jacobian"]["threshold_stats"]
        print("Dynamic Threshold Statistics:")
        print("-" * 40)
        print(f"  Mean threshold: {thresh_stats['mean']:.2f}")
        print(f"  Std deviation: {thresh_stats['std']:.2f}")
        print(f"  Range: [{thresh_stats['min']:.0f}, {thresh_stats['max']:.0f}]")
        print()
    
    # Key findings
    print("Key Findings:")
    print("-" * 20)
    
    # Find configuration with lowest extinction probability
    min_config = min(results.keys(), key=lambda k: results[k]["extinction_probability"])
    max_config = max(results.keys(), key=lambda k: results[k]["extinction_probability"])
    
    min_prob = results[min_config]["extinction_probability"]
    max_prob = results[max_config]["extinction_probability"]
    
    print(f"• Lowest extinction risk: {min_config} ({min_prob:.4f})")
    print(f"• Highest extinction risk: {max_config} ({max_prob:.4f})")
    print(f"• Risk ratio: {max_prob/min_prob:.2f}x" if min_prob > 0 else "• Risk ratio: ∞")
    
    # Interpretation
    print("\nInterpretation:")
    if min_config == "large_threshold":
        print("• Pure continuous simulation (large threshold) shows lowest extinction risk")
    elif min_config == "dynamic_jacobian":
        print("• Dynamic jacobian-based thresholds provide optimal extinction prevention")
    else:
        print("• Medium threshold hybrid approach shows lowest extinction risk")


def save_results(results, comparison, filename="extinction_analysis_results.json"):
    """Save detailed results to JSON file."""
    
    # Prepare data for JSON serialization
    json_data = {
        "results": {},
        "comparison": comparison,
        "metadata": {
            "analysis_type": "extinction_probability",
            "model": "lotka_volterra",
            "configurations": list(results.keys())
        }
    }
    
    # Convert results to JSON-serializable format
    for config_name, result in results.items():
        json_result = result.copy()
        
        # Convert numpy arrays to lists if present
        if isinstance(json_result.get("mean_final_populations"), list):
            json_result["mean_final_populations"] = json_result["mean_final_populations"]
        if isinstance(json_result.get("std_final_populations"), list):
            json_result["std_final_populations"] = json_result["std_final_populations"]
            
        # Remove threshold histories (too large for JSON)
        json_result.pop("threshold_histories", None)
        json_result.pop("final_populations", None)
        
        json_data["results"][config_name] = json_result
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Detailed results saved to: {filename}")


def main():
    """Main execution function."""
    
    # Run analysis
    results = run_extinction_analysis(n_runs=500, t_max=10.0, base_seed=42)
    
    # Compare results
    comparison = compare_extinction_results(results)
    
    # Generate reports
    generate_summary_report(results, comparison)
    #save_results(results, comparison)
    
    # Create visualizations (optional - requires matplotlib)
    try:
        from jsf.visualization_utils import create_analysis_dashboard
        model = setup_lotka_volterra_model()
        print("\nGenerating visualization dashboard...")
        create_analysis_dashboard(results, model)
    except ImportError:
        print("\nNote: matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"\nWarning: Visualization generation failed: {e}")
    
    print("\nExtinction probability analysis completed! ✅")


if __name__ == "__main__":
    main()