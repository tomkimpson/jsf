#!/usr/bin/env python
"""
Debug script to investigate unexpected extinction behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
from jsf.dynamic_scaling import jacobian_lotka_volterra
import time
import numpy as np


def debug_single_runs():
    """Debug individual simulation runs to understand behavior."""
    
    print("DEBUGGING EXTINCTION BEHAVIOR")
    print("="*50)
    
    # Model setup
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
    
    # Test configurations
    configs = {
        "large_threshold": {
            "EnforceDo": [0, 0],
            "dt": 0.01,
            "SwitchingThreshold": [1000, 1000],
            "EnableDynamicThreshold": False,
            "seed": 42
        },
        "medium_threshold": {
            "EnforceDo": [0, 0], 
            "dt": 0.01,
            "SwitchingThreshold": [30, 30],
            "EnableDynamicThreshold": False,
            "seed": 42
        }
    }
    
    t_max = 10.0
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name}:")
        print(f"  Threshold: {config['SwitchingThreshold']}")
        
        # Time the simulation
        start_time = time.time()
        
        result = jsf.jsf(x0, rates, stoich, t_max, config=config, method="operator-splitting")
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Analyze results
        times = result[1]
        prey = result[0][0]
        predator = result[0][1]
        
        # Check for extinction
        min_prey = min(prey)
        min_predator = min(predator)
        final_prey = prey[-1]
        final_predator = predator[-1]
        
        extinction = min_prey <= 0.0 or min_predator <= 0.0
        
        print(f"  Runtime: {runtime:.4f} seconds")
        print(f"  Time points: {len(times)}")
        print(f"  Final populations: Prey={final_prey:.3f}, Predator={final_predator:.3f}")
        print(f"  Minimum populations: Prey={min_prey:.3f}, Predator={min_predator:.3f}")
        print(f"  Extinction occurred: {extinction}")
        
        # Sample trajectory points
        n_sample = min(10, len(times))
        print(f"  Sample trajectory (first {n_sample} points):")
        for i in range(n_sample):
            print(f"    t={times[i]:.2f}: Prey={prey[i]:.2f}, Predator={predator[i]:.2f}")


def debug_regime_behavior():
    """Check which simulation regime is actually being used."""
    
    print("\n" + "="*50)
    print("DEBUGGING SIMULATION REGIME")
    print("="*50)
    
    # Create a simple test to see how switching thresholds affect behavior
    x0 = [50, 10]  # Start above both thresholds
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
    
    thresholds_to_test = [10, 30, 100, 1000]
    
    for threshold in thresholds_to_test:
        config = {
            "EnforceDo": [0, 0],
            "dt": 0.01,
            "SwitchingThreshold": [threshold, threshold],
            "EnableDynamicThreshold": False,
            "seed": 42
        }
        
        print(f"\nThreshold = {threshold}:")
        
        # Check initial regime
        initial_regime_prey = "discrete" if x0[0] <= threshold else "continuous"
        initial_regime_predator = "discrete" if x0[1] <= threshold else "continuous"
        
        print(f"  Initial regime: Prey={initial_regime_prey}, Predator={initial_regime_predator}")
        
        # Run short simulation
        start_time = time.time()
        result = jsf.jsf(x0, rates, stoich, 2.0, config=config, method="operator-splitting")
        end_time = time.time()
        
        times = result[1]
        prey = result[0][0]
        predator = result[0][1]
        
        print(f"  Runtime: {end_time - start_time:.4f}s")
        print(f"  Time points: {len(times)}")
        print(f"  Final: Prey={prey[-1]:.2f}, Predator={predator[-1]:.2f}")
        
        # Check how much time spent in each regime
        discrete_time_prey = sum(1 for p in prey if p <= threshold)
        discrete_time_predator = sum(1 for p in predator if p <= threshold)
        
        print(f"  Discrete time points: Prey={discrete_time_prey}/{len(prey)}, Predator={discrete_time_predator}/{len(predator)}")


def debug_deterministic_vs_stochastic():
    """Compare deterministic (large threshold) vs stochastic (small threshold) behavior."""
    
    print("\n" + "="*50)
    print("DEBUGGING DETERMINISTIC VS STOCHASTIC")
    print("="*50)
    
    x0 = [5, 2]  # Small populations to see stochastic effects
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
    
    configs = {
        "deterministic": {"SwitchingThreshold": [1000, 1000]},  # Force continuous
        "stochastic": {"SwitchingThreshold": [10, 10]}          # Allow discrete
    }
    
    for config_name, config_specific in configs.items():
        print(f"\n{config_name.upper()} (Threshold = {config_specific['SwitchingThreshold']}):")
        
        extinctions = 0
        n_test = 5
        
        for seed in range(n_test):
            config = {
                "EnforceDo": [0, 0],
                "dt": 0.01,
                "EnableDynamicThreshold": False,
                "seed": seed,
                **config_specific
            }
            
            result = jsf.jsf(x0, rates, stoich, 5.0, config=config, method="operator-splitting")
            
            prey = result[0][0]
            predator = result[0][1]
            
            extinct = min(prey) <= 0.0 or min(predator) <= 0.0
            if extinct:
                extinctions += 1
                
            print(f"  Run {seed}: Final=[{prey[-1]:.2f}, {predator[-1]:.2f}], Extinct={extinct}")
        
        print(f"  Extinction rate: {extinctions}/{n_test} = {extinctions/n_test:.2f}")


if __name__ == "__main__":
    debug_single_runs()
    debug_regime_behavior()
    debug_deterministic_vs_stochastic()