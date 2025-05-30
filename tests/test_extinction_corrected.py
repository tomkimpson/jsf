#!/usr/bin/env python
"""
Corrected extinction probability analysis for Lotka-Volterra model.

This fixes the issues found in the original analysis:
1. Uses stable Lotka-Volterra parameters
2. Correct threshold configuration for desired regimes  
3. Appropriate initial conditions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
from jsf.analysis_utils import run_monte_carlo_extinction, compare_extinction_results
from jsf.dynamic_scaling import jacobian_lotka_volterra
import json
import time


def setup_stable_lotka_volterra():
    """Set up Lotka-Volterra with more stable parameters."""
    
    # More stable parameters (closer to classic LV oscillations)
    x0 = [100, 50]  # Larger initial populations
    alpha = 1.0      # Reduced prey birth rate  
    beta = 0.01      # Reduced predation rate
    gamma = 0.5      # Reduced predator death rate
    
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
    
    # Jacobian function
    def jacobian_func(state):
        return jacobian_lotka_volterra(state, alpha, beta, gamma)
    
    return {
        "x0": x0,
        "rates": rates,
        "stoich": stoich,
        "jacobian_func": jacobian_func,
        "parameters": {"alpha": alpha, "beta": beta, "gamma": gamma}
    }


def create_corrected_configurations(jacobian_func):
    """Create corrected test configurations."""
    
    base_config = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "method": "operator-splitting"
    }
    
    # Case A: Pure discrete (threshold infinitely large → always stochastic)
    config_discrete = base_config.copy()
    config_discrete.update({
        "SwitchingThreshold": [1000000, 1000000],  # Infinitely large → pure discrete
        "EnableDynamicThreshold": False
    })
    
    # Case B: Pure continuous (threshold below initial populations)
    config_continuous = base_config.copy()
    config_continuous.update({
        "SwitchingThreshold": [10, 10],  # Below initial [100, 50] → continuous
        "EnableDynamicThreshold": False
    })
    
    # Case C: Hybrid regime (threshold between typical population ranges)
    config_hybrid = base_config.copy()
    config_hybrid.update({
        "SwitchingThreshold": [50, 25],  # Mixed continuous/discrete
        "EnableDynamicThreshold": False
    })
    
    # Case D: Dynamic jacobian-based threshold
    config_dynamic = base_config.copy()
    config_dynamic.update({
        "SwitchingThreshold": [50, 25],  # Initial threshold
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": jacobian_func,
            "omega_min": 10.0,    # Minimum threshold
            "omega_max": 200.0,   # Maximum threshold  
            "k": 2.0,             # Sigmoid steepness
            "score_threshold": 0.05,  # Score at midpoint
            "debug": False
        }
    })
    
    return {
        "pure_discrete": config_discrete,
        "pure_continuous": config_continuous,
        "hybrid_regime": config_hybrid,
        "dynamic_jacobian": config_dynamic
    }


def verify_regime_behavior():
    """Verify that configurations produce expected regimes."""
    
    print("VERIFYING SIMULATION REGIMES")
    print("="*50)
    
    model = setup_stable_lotka_volterra()
    configs = create_corrected_configurations(model["jacobian_func"])
    
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Threshold: {config.get('SwitchingThreshold', 'N/A')}")
        print(f"  Initial populations: {model['x0']}")
        
        # Determine expected regime
        threshold = config.get('SwitchingThreshold', [0, 0])
        initial_regime = []
        for i, (pop, thresh) in enumerate(zip(model['x0'], threshold)):
            regime = "discrete" if pop <= thresh else "continuous"
            initial_regime.append(regime)
            print(f"  Compartment {i}: {pop} vs {thresh} → {regime}")
        
        # Run quick test
        test_config = config.copy()
        test_config["seed"] = 42
        
        start_time = time.time()
        result = jsf.jsf(
            model["x0"], model["rates"], model["stoich"], 
            2.0, config=test_config, method="operator-splitting"
        )
        runtime = time.time() - start_time
        
        times = result[1]
        prey = result[0][0]
        predator = result[0][1]
        
        print(f"  Runtime: {runtime:.4f}s")
        print(f"  Time points: {len(times)}")
        print(f"  Final populations: [{prey[-1]:.1f}, {predator[-1]:.1f}]")
        
        # Count time in each regime
        thresh_prey, thresh_pred = threshold
        discrete_prey = sum(1 for p in prey if p <= thresh_prey)
        discrete_pred = sum(1 for p in predator if p <= thresh_pred)
        
        print(f"  Discrete fraction: Prey={discrete_prey/len(prey):.2f}, Predator={discrete_pred/len(predator):.2f}")


def run_corrected_extinction_analysis(n_runs=100, t_max=20.0):
    """Run corrected extinction analysis with stable parameters."""
    
    print("\n" + "="*60)
    print("CORRECTED EXTINCTION PROBABILITY ANALYSIS")
    print("="*60)
    print(f"Configuration:")
    print(f"  Number of runs per configuration: {n_runs}")
    print(f"  Simulation time: {t_max}")
    print(f"  Extinction threshold: 0.0")
    print()
    
    # Set up stable model
    model = setup_stable_lotka_volterra()
    print(f"Stable model parameters:")
    print(f"  Initial populations: {model['x0']}")
    print(f"  Alpha (prey birth): {model['parameters']['alpha']}")
    print(f"  Beta (predation): {model['parameters']['beta']}")
    print(f"  Gamma (predator death): {model['parameters']['gamma']}")
    print()
    
    # Create corrected configurations
    configs = create_corrected_configurations(model["jacobian_func"])
    
    results = {}
    total_start_time = time.time()
    
    # Run each configuration
    for config_name, config in configs.items():
        print(f"Running configuration: {config_name.upper()}")
        
        start_time = time.time()
        
        # Run Monte Carlo simulation
        result = run_monte_carlo_extinction(
            base_config=config,
            model_params=model,
            n_runs=n_runs,
            base_seed=42,
            extinction_threshold=0.0,
            t_max=t_max
        )
        
        end_time = time.time()
        results[config_name] = result
        
        # Display results
        prob = result["extinction_probability"]
        ci = result["confidence_interval"]
        print(f"  Extinction probability: {prob:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        print(f"  Extinctions: {result['extinctions']}/{n_runs}")
        print(f"  Runtime: {end_time - start_time:.1f} seconds")
        print()
    
    total_time = time.time() - total_start_time
    print(f"Total analysis time: {total_time:.1f} seconds")
    print()
    
    return results, model


def generate_corrected_summary(results):
    """Generate summary with corrected interpretation."""
    
    print("="*60)
    print("CORRECTED ANALYSIS SUMMARY")
    print("="*60)
    
    # Results table
    print("Configuration Results:")
    print("-" * 80)
    print(f"{'Configuration':<20} {'Ext. Prob.':<12} {'95% CI':<20} {'Expected Regime':<15}")
    print("-" * 80)
    
    regime_descriptions = {
        "pure_discrete": "Discrete",
        "pure_continuous": "Continuous",
        "hybrid_regime": "Hybrid", 
        "dynamic_jacobian": "Adaptive"
    }
    
    for config_name, result in results.items():
        prob = result["extinction_probability"]
        ci = result["confidence_interval"]
        regime = regime_descriptions.get(config_name, "Unknown")
        
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        
        print(f"{config_name:<20} {prob:<12.4f} {ci_str:<20} {regime:<15}")
    
    print("-" * 80)
    print()
    
    # Expected ordering check
    discrete_prob = results["pure_discrete"]["extinction_probability"]
    continuous_prob = results["pure_continuous"]["extinction_probability"]
    hybrid_prob = results["hybrid_regime"]["extinction_probability"]
    dynamic_prob = results["dynamic_jacobian"]["extinction_probability"]
    
    print("Expected vs Observed Ordering:")
    print("-" * 50)
    print(f"Pure discrete (true stochastic): {discrete_prob:.4f}")
    print(f"Pure continuous (should be lowest): {continuous_prob:.4f}")
    print(f"Hybrid regime: {hybrid_prob:.4f}")
    print(f"Dynamic jacobian: {dynamic_prob:.4f}")
    print()
    
    # Check if ordering matches expectation
    print("Expected ordering: Continuous < Hybrid ≤ Dynamic ≤ Discrete")
    print("Ordering checks:")
    
    if continuous_prob <= hybrid_prob:
        print("✓ Continuous ≤ Hybrid: Expected")
    else:
        print("✗ Continuous > Hybrid: Unexpected!")
    
    if hybrid_prob <= discrete_prob:
        print("✓ Hybrid ≤ Discrete: Expected")
    else:
        print("✗ Hybrid > Discrete: Unexpected!")
    
    if dynamic_prob <= discrete_prob:
        print("✓ Dynamic ≤ Discrete: Expected")
    else:
        print("✗ Dynamic > Discrete: Unexpected!")
    
    print("\nInterpretation:")
    print("• Pure discrete provides 'true' stochastic extinction probability")
    print("• Pure continuous should show lowest extinction (deterministic)")
    print("• Hybrid and dynamic should fall between continuous and discrete")
    print("• Dynamic jacobian adapts thresholds based on system stability")


def main():
    """Main execution function."""
    
    # First verify regime behavior
    verify_regime_behavior()
    
    # Run corrected analysis
    results, model = run_corrected_extinction_analysis(n_runs=100, t_max=20.0)
    
    # Generate summary
    generate_corrected_summary(results)
    
    print("\nCorrected extinction analysis completed! ✅")


if __name__ == "__main__":
    main()