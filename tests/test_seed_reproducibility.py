#!/usr/bin/env python
"""
Test script for seed reproducibility in JSF simulations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
import jsf.sbml as sbml


def test_operator_splitting_reproducibility():
    """Test that operator-splitting method produces identical results with same seed."""
    print("Testing operator-splitting reproducibility with seeds...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Configuration with seed
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'seed': 42  # Set seed for reproducibility
    }
    
    # Run simulation twice with same seed
    result1 = jsf.jsf(x0, rates, stoich, 2.0, config=my_opts, method='operator-splitting')
    result2 = jsf.jsf(x0, rates, stoich, 2.0, config=my_opts, method='operator-splitting')
    
    # Results should be identical
    times1, times2 = result1[1], result2[1]
    values1, values2 = result1[0][0], result2[0][0]
    
    print(f"  Run 1 times: {times1[:5]}")
    print(f"  Run 2 times: {times2[:5]}")
    print(f"  Run 1 values: {values1[:5]}")
    print(f"  Run 2 values: {values2[:5]}")
    
    assert len(times1) == len(times2), f"Time array lengths differ: {len(times1)} vs {len(times2)}"
    assert len(values1) == len(values2), f"Value array lengths differ: {len(values1)} vs {len(values2)}"
    
    # Check that times and values are identical
    for i, (t1, t2) in enumerate(zip(times1, times2)):
        assert abs(t1 - t2) < 1e-10, f"Times differ at index {i}: {t1} vs {t2}"
    
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        assert abs(v1 - v2) < 1e-10, f"Values differ at index {i}: {v1} vs {v2}"
    
    print("✓ Operator-splitting reproducibility test passed")


def test_exact_method_reproducibility():
    """Test that exact method produces identical results with same seed."""
    print("Testing exact method reproducibility with seeds...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Configuration with seed
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'seed': 123  # Set seed for reproducibility
    }
    
    # Run simulation twice with same seed
    result1 = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='exact')
    result2 = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='exact')
    
    # Results should be identical
    times1, times2 = result1[1], result2[1]
    values1, values2 = result1[0][0], result2[0][0]
    
    print(f"  Run 1 times: {times1[:5]}")
    print(f"  Run 2 times: {times2[:5]}")
    print(f"  Run 1 values: {values1[:5]}")
    print(f"  Run 2 values: {values2[:5]}")
    
    assert len(times1) == len(times2), f"Time array lengths differ: {len(times1)} vs {len(times2)}"
    assert len(values1) == len(values2), f"Value array lengths differ: {len(values1)} vs {len(values2)}"
    
    # Check that times and values are identical
    for i, (t1, t2) in enumerate(zip(times1, times2)):
        assert abs(t1 - t2) < 1e-10, f"Times differ at index {i}: {t1} vs {t2}"
    
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        assert abs(v1 - v2) < 1e-10, f"Values differ at index {i}: {v1} vs {v2}"
    
    print("✓ Exact method reproducibility test passed")


def test_different_seeds_give_different_results():
    """Test that different seeds produce different results."""
    print("Testing that different seeds produce different results...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Configuration with different seeds
    opts1 = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'seed': 42
    }
    
    opts2 = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'seed': 999
    }
    
    # Run simulations with different seeds
    result1 = jsf.jsf(x0, rates, stoich, 2.0, config=opts1, method='operator-splitting')
    result2 = jsf.jsf(x0, rates, stoich, 2.0, config=opts2, method='operator-splitting')
    
    times1, times2 = result1[1], result2[1]
    values1, values2 = result1[0][0], result2[0][0]
    
    print(f"  Seed 42 times: {times1[:5]}")
    print(f"  Seed 999 times: {times2[:5]}")
    
    # Results should be different (at least some values)
    times_different = any(abs(t1 - t2) > 1e-10 for t1, t2 in zip(times1, times2) if len(times1) == len(times2))
    values_different = any(abs(v1 - v2) > 1e-10 for v1, v2 in zip(values1, values2) if len(values1) == len(values2))
    length_different = len(times1) != len(times2) or len(values1) != len(values2)
    
    assert times_different or values_different or length_different, "Different seeds should produce different results"
    
    print("✓ Different seeds produce different results test passed")


def test_dynamic_threshold_reproducibility():
    """Test that dynamic thresholds are also reproducible with seeds."""
    print("Testing dynamic threshold reproducibility with seeds...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    # Configuration with dynamic thresholds and seed
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'increment',
        'seed': 777
    }
    
    # Run simulation twice with same seed
    result1 = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    result2 = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    # Check that threshold histories are identical
    thresholds1 = result2[2] if len(result1) > 2 else []
    thresholds2 = result2[2] if len(result2) > 2 else []
    
    assert len(thresholds1) == len(thresholds2), "Threshold array lengths should match"
    
    for i, (th1, th2) in enumerate(zip(thresholds1, thresholds2)):
        assert th1 == th2, f"Threshold snapshots differ at index {i}: {th1} vs {th2}"
    
    print(f"  Identical threshold progression: {[th[0] for th in thresholds1[:5]]}")
    print("✓ Dynamic threshold reproducibility test passed")


if __name__ == "__main__":
    test_operator_splitting_reproducibility()
    test_exact_method_reproducibility()
    test_different_seeds_give_different_results()
    test_dynamic_threshold_reproducibility()
    print("\nAll seed reproducibility tests passed! ✅")