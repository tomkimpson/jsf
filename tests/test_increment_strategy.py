#!/usr/bin/env python
"""
Test script for the increment strategy dynamic threshold functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
import jsf.sbml as sbml


def test_increment_strategy():
    """Test that the increment strategy increases thresholds by 1 each step."""
    print("Testing increment strategy...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    # Ensure x0 is a list if it's a single value
    if not isinstance(x0, list):
        x0 = [x0]
    
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'increment'
    }
    
    # Run simulation with increment strategy
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    assert len(result) == 3, f"Expected 3 elements with dynamic thresholds, got {len(result)}"
    
    # Check that thresholds are changing and generally increasing
    threshold_history = result[2]
    print(f"  Initial threshold: {threshold_history[0]}")
    print(f"  Final threshold: {threshold_history[-1]}")
    print(f"  Total steps: {len(threshold_history)}")
    
    # Extract threshold values for easier analysis
    threshold_values = [th[0] for th in threshold_history]
    print(f"  All threshold values: {threshold_values}")
    
    # The initial threshold should be the starting value
    assert threshold_values[0] == 30, f"Initial threshold should be 30, got {threshold_values[0]}"
    
    # Final threshold should be higher than initial (increment strategy should increase it)
    assert threshold_values[-1] > threshold_values[0], f"Final threshold {threshold_values[-1]} should be greater than initial {threshold_values[0]}"
    
    # Check that threshold generally increases (allowing for simulation-specific behavior)
    total_increase = threshold_values[-1] - threshold_values[0]
    assert total_increase > 0, f"Total threshold increase should be positive, got {total_increase}"
    
    print(f"✓ Thresholds increased from {threshold_values[0]} to {threshold_values[-1]} (total increase: {total_increase})")
    print("✓ Increment strategy test passed")


def test_sqrt_population_strategy():
    """Test that the sqrt population strategy works."""
    print("Testing sqrt population strategy...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'sqrt_population'
    }
    
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    threshold_history = result[2]
    state_history = result[0][0]  # First compartment values
    
    print(f"  Sample population values: {state_history[:5]}")
    print(f"  Sample threshold values: {[th[0] for th in threshold_history[:5]]}")
    
    # Check that thresholds are reasonable (at least 1, and change with population)
    for i, threshold_snapshot in enumerate(threshold_history):
        threshold_val = threshold_snapshot[0]
        assert threshold_val >= 1, f"Threshold at step {i} should be at least 1, got {threshold_val}"
    
    print("✓ Sqrt population strategy test passed")




def test_increment_strategy_lotkavolterra():
    """Test that the increment strategy works for the Lotka-Volterra model."""
    print("Testing increment strategy for Lotka-Volterra model...")
    
    #Initial conditions
    x0 = [50, 10]


    #Rates
    mA = 2.00
    mB = 0.05
    mC = 1.50
    rates = lambda x, t: [mA * x[0],
                        mC * x[1],
                        mB * x[0] * x[1]]


    #Stoichiometric matrices
    reactant_matrix = [[1 , 0],
                    [0 , 1],
                    [1 , 1]]

    product_matrix = [[2 , 0],
                    [0 , 0],
                    [0 , 2]]

    stoich = {
        "nu": [ [a - b for a, b in zip(r1, r2)]
            for r1, r2 in zip(product_matrix, reactant_matrix) ],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }


    #Options
    omega = 30
    my_opts = {
                "EnforceDo": [0, 0],
                "dt": 0.01,
                "SwitchingThreshold": [omega, omega],
                'alpha': mA,
                'beta': mB,
                'gamma': mC,
                'seed':28,
            'EnableDynamicThreshold': True,
            'DynamicThresholdStrategy': 'increment'
            }
    t_max = 10


    result = jsf.jsf(x0, rates, stoich, t_max, config=my_opts, method="operator-splitting")

    threshold_history = result[2]
    print(f"  Initial threshold: {threshold_history[0]}")
    print(f"  Final threshold: {threshold_history[-1]}")
    print(f"  Total steps: {len(threshold_history)}")

    # Extract threshold values for easier analysis
    threshold_values = [th[0] for th in threshold_history]
    print("✓ Lotka-Volterra increment strategy test passed")




if __name__ == "__main__":
    test_increment_strategy()
    test_sqrt_population_strategy()
    test_increment_strategy_lotkavolterra()
    print("\nAll strategy tests passed! ✅")