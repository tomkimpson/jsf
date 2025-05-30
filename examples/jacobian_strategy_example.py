#!/usr/bin/env python
"""
Example demonstrating the jacobian_based dynamic threshold strategy.

This example shows how to use the scale-aware switching threshold
that adapts based on the noise-to-drift ratio and system stability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
from jsf.dynamic_scaling import jacobian_lotka_volterra
import matplotlib.pyplot as plt


def run_lotka_volterra_example():
    """Run Lotka-Volterra with jacobian_based strategy."""
    print("Running Lotka-Volterra with jacobian-based thresholds...")
    
    # Model parameters
    x0 = [50, 10]  # Initial prey and predator populations
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
    
    # Jacobian function for this model
    def jacobian_func(state):
        return jacobian_lotka_volterra(state, alpha, beta, gamma)
    
    # Configuration with jacobian_based strategy
    config = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "SwitchingThreshold": [30, 30],  # Initial threshold
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": jacobian_func,
            "omega_min": 5.0,      # Minimum threshold
            "omega_max": 100.0,    # Maximum threshold
            "k": 2.5,              # Sigmoid steepness
            "score_threshold": 0.1, # Score at sigmoid midpoint
            "debug": False
        },
        "seed": 42
    }
    
    # Run simulation
    print("  Running simulation...")
    result = jsf.jsf(x0, rates, stoich, 5.0, config=config, method="operator-splitting")
    
    # Extract results
    times = result[1]
    prey = result[0][0]
    predator = result[0][1]
    thresholds = [th[0] for th in result[2]]  # Both compartments have same threshold
    
    print(f"  Simulation completed with {len(times)} time points")
    print(f"  Threshold range: {min(thresholds)} - {max(thresholds)}")
    print(f"  Final populations: Prey={prey[-1]:.1f}, Predator={predator[-1]:.1f}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Population dynamics
    ax1.plot(times, prey, 'b-', label='Prey', linewidth=2)
    ax1.plot(times, predator, 'r-', label='Predator', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population')
    ax1.set_title('Lotka-Volterra Dynamics with Jacobian-Based Thresholds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dynamic thresholds
    ax2.plot(times, thresholds, 'g-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Switching Threshold')
    ax2.set_title('Dynamic Threshold Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jacobian_strategy_example.png', dpi=300, bbox_inches='tight')
    print("  Plot saved as 'jacobian_strategy_example.png'")
    
    return result


def compare_strategies():
    """Compare static vs jacobian_based strategies."""
    print("\nComparing static vs jacobian-based strategies...")
    
    # Same model setup as above
    x0 = [50, 10]
    alpha, beta, gamma = 2.0, 0.05, 1.5
    rates = lambda x, t: [alpha * x[0], gamma * x[1], beta * x[0] * x[1]]
    
    reactant_matrix = [[1, 0], [0, 1], [1, 1]]
    product_matrix = [[2, 0], [0, 0], [0, 2]]
    stoich = {
        "nu": [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(product_matrix, reactant_matrix)],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }
    
    def jacobian_func(state):
        return jacobian_lotka_volterra(state, alpha, beta, gamma)
    
    # Static threshold configuration
    static_config = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "SwitchingThreshold": [30, 30],
        "seed": 123
    }
    
    # Dynamic threshold configuration
    dynamic_config = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "SwitchingThreshold": [30, 30],
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": jacobian_func,
            "omega_min": 10.0,
            "omega_max": 80.0,
            "k": 3.0,
            "score_threshold": 0.15,
        },
        "seed": 123
    }
    
    # Run both simulations
    static_result = jsf.jsf(x0, rates, stoich, 3.0, config=static_config, method="operator-splitting")
    dynamic_result = jsf.jsf(x0, rates, stoich, 3.0, config=dynamic_config, method="operator-splitting")
    
    # Compare results
    print(f"  Static simulation: {len(static_result[1])} time points")
    print(f"  Dynamic simulation: {len(dynamic_result[1])} time points")
    
    if len(dynamic_result) > 2:  # Has threshold history
        thresholds = [th[0] for th in dynamic_result[2]]
        print(f"  Dynamic threshold range: {min(thresholds)} - {max(thresholds)}")
    
    print("  Comparison complete!")


if __name__ == "__main__":
    run_lotka_volterra_example()
    compare_strategies()
    print("\nJacobian-based strategy example completed! ✅")