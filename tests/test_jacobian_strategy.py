#!/usr/bin/env python
"""
Test script for jacobian_based_strategy in dynamic threshold functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
import jsf.sbml as sbml
from jsf.dynamic_scaling import jacobian_lotka_volterra
import numpy as np


def test_jacobian_strategy_lotka_volterra():
    """Test jacobian_based strategy with Lotka-Volterra model."""
    print("Testing jacobian_based strategy with Lotka-Volterra model...")
    
    # Initial conditions
    x0 = [50, 10]
    
    # Rates
    mA = 2.00
    mB = 0.05
    mC = 1.50
    rates = lambda x, t: [mA * x[0], mC * x[1], mB * x[0] * x[1]]
    
    # Stoichiometric matrices
    reactant_matrix = [[1, 0], [0, 1], [1, 1]]
    product_matrix = [[2, 0], [0, 0], [0, 2]]
    
    stoich = {
        "nu": [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(product_matrix, reactant_matrix)],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }
    
    # Create Jacobian function with fixed parameters
    def lotka_volterra_jacobian(state):
        return jacobian_lotka_volterra(state, mA, mB, mC)
    
    # Options with jacobian_based strategy
    my_opts = {
        "EnforceDo": [0, 0],
        "dt": 0.01,
        "SwitchingThreshold": [30, 30],
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": lotka_volterra_jacobian,
            "omega_min": 5.0,
            "omega_max": 100.0,
            "k": 2.5,
            "score_threshold": 0.1,
            "debug": False  # Disable debug output for cleaner tests
        },
        "seed": 42
    }
    
    # Run simulation
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method="operator-splitting")
    
    # Check results
    times = result[1]
    compartments = result[0]
    thresholds = result[2]
    
    print(f"  Simulation completed with {len(times)} time points")
    print(f"  Final populations: {[comp[-1] for comp in compartments]}")
    print(f"  Initial threshold: {thresholds[0]}")
    print(f"  Final threshold: {thresholds[-1]}")
    
    # Check that thresholds are reasonable
    threshold_values = [th[0] for th in thresholds]
    assert all(th >= 1 for th in threshold_values), "All thresholds should be at least 1"
    assert all(th <= 100 for th in threshold_values), "All thresholds should be at most omega_max=100"
    
    print("✓ Jacobian-based strategy test passed")


def test_jacobian_strategy_error_handling():
    """Test error handling when jacobian_func is missing."""
    print("Testing jacobian_based strategy error handling...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Options without jacobian_func - should raise error
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'jacobian_based',
        # Missing JacobianParams with jacobian_func
    }
    
    try:
        result = jsf.jsf(x0, rates, stoich, 0.5, config=my_opts, method='operator-splitting')
        assert False, "Should have raised ValueError for missing jacobian_func"
    except ValueError as e:
        assert "jacobian_func must be provided" in str(e)
        print("✓ Correctly raised error for missing jacobian_func")
    
    print("✓ Error handling test passed")


def test_jacobian_strategy_parameters():
    """Test different parameter configurations for jacobian_based strategy."""
    print("Testing jacobian_based strategy with different parameters...")
    
    x0 = [50, 10]
    mA, mB, mC = 2.0, 0.05, 1.5
    rates = lambda x, t: [mA * x[0], mC * x[1], mB * x[0] * x[1]]
    
    reactant_matrix = [[1, 0], [0, 1], [1, 1]]
    product_matrix = [[2, 0], [0, 0], [0, 2]]
    
    stoich = {
        "nu": [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(product_matrix, reactant_matrix)],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }
    
    def lotka_volterra_jacobian(state):
        return jacobian_lotka_volterra(state, mA, mB, mC)
    
    # Test configuration 1: Low threshold range
    config1 = {
        "EnforceDo": [0, 0],
        "dt": 0.05,
        "SwitchingThreshold": [30, 30],
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": lotka_volterra_jacobian,
            "omega_min": 1.0,
            "omega_max": 10.0,
            "k": 1.0,
            "score_threshold": 0.5,
            "debug": False
        },
        "seed": 123
    }
    
    # Test configuration 2: High threshold range
    config2 = {
        "EnforceDo": [0, 0],
        "dt": 0.05,
        "SwitchingThreshold": [30, 30],
        "EnableDynamicThreshold": True,
        "DynamicThresholdStrategy": "jacobian_based",
        "JacobianParams": {
            "jacobian_func": lotka_volterra_jacobian,
            "omega_min": 50.0,
            "omega_max": 200.0,
            "k": 5.0,
            "score_threshold": 0.01,
            "debug": False
        },
        "seed": 123
    }
    
    # Run both configurations
    result1 = jsf.jsf(x0, rates, stoich, 0.5, config=config1, method="operator-splitting")
    result2 = jsf.jsf(x0, rates, stoich, 0.5, config=config2, method="operator-splitting")
    
    # Extract threshold values
    thresholds1 = [th[0] for th in result1[2]]
    thresholds2 = [th[0] for th in result2[2]]
    
    print(f"  Config 1 threshold range: {min(thresholds1)} - {max(thresholds1)}")
    print(f"  Config 2 threshold range: {min(thresholds2)} - {max(thresholds2)}")
    
    # Check that different configurations produce different threshold ranges
    # Note: thresholds may start at initial values but should trend toward configured ranges
    assert min(thresholds1) >= 1, "Config 1 should have thresholds ≥ omega_min=1"
    assert any(th <= 10 for th in thresholds1), "Config 1 should have some low thresholds"
    assert any(th >= 50 for th in thresholds2), "Config 2 should have some high thresholds"
    
    print("✓ Parameter configuration test passed")


def test_simple_birth_death_jacobian():
    """Test jacobian strategy with simple birth-death model."""
    print("Testing jacobian_based strategy with birth-death model...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Simple Jacobian for birth-death model: dX/dt = (birth_rate - death_rate) * X
    # J = birth_rate - death_rate (assuming rates are [birth_rate * X, death_rate * X])
    def birth_death_jacobian(state):
        # For birth-death, assume rates are proportional to population
        # This is a simplified example - real Jacobian would depend on rate function
        return np.array([[-0.1]])  # Simple negative eigenvalue
    
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'jacobian_based',
        'JacobianParams': {
            'jacobian_func': birth_death_jacobian,
            'omega_min': 2.0,
            'omega_max': 50.0,
            'debug': False
        },
        'seed': 456
    }
    
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    thresholds = [th[0] for th in result[2]]
    print(f"  Birth-death threshold range: {min(thresholds)} - {max(thresholds)}")
    
    # Check basic properties
    assert all(th >= 2 for th in thresholds), "All thresholds should be at least omega_min=2"
    assert all(th <= 50 for th in thresholds), "All thresholds should be at most omega_max=50"
    
    print("✓ Birth-death jacobian test passed")


if __name__ == "__main__":
    test_jacobian_strategy_error_handling()
    test_simple_birth_death_jacobian() 
    test_jacobian_strategy_parameters()
    test_jacobian_strategy_lotka_volterra()
    print("\nAll jacobian-based strategy tests passed! ✅")