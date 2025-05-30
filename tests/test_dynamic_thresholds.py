#!/usr/bin/env python
"""
Test script for dynamic threshold functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
import jsf.sbml as sbml


def test_backward_compatibility():
    """Test that existing code still works unchanged."""
    print("Testing backward compatibility...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    # Ensure x0 is a list if it's a single value
    if not isinstance(x0, list):
        x0 = [x0]
    my_opts = {'EnforceDo': [0], 'dt': 0.1, 'SwitchingThreshold': [30]}
    
    # Test static behavior (should return Trajectory)
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    assert len(result[0]) == 1, f"Expected 1 compartment, got {len(result[0])}"
    assert len(result[1]) > 0, "Expected non-empty time array"
    
    print("✓ Backward compatibility test passed")


def test_dynamic_threshold_tracking():
    """Test that dynamic threshold tracking works."""
    print("Testing dynamic threshold tracking...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    # Ensure x0 is a list if it's a single value
    if not isinstance(x0, list):
        x0 = [x0]
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True
    }
    
    # Test dynamic behavior (should return TrajectoryWithThresholds)
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    assert len(result) == 3, f"Expected 3 elements with dynamic thresholds, got {len(result)}"
    assert len(result[0]) == 1, f"Expected 1 compartment, got {len(result[0])}"
    assert len(result[1]) > 0, "Expected non-empty time array"
    assert len(result[2]) > 0, "Expected non-empty threshold array"
    
    # Debug output
    print(f"  Time array length: {len(result[1])}")
    print(f"  Threshold array length: {len(result[2])}")
    print(f"  Compartment 0 trajectory length: {len(result[0][0])}")
    
    # For now, just check that we have threshold data
    assert len(result[2]) > 0, "Expected non-empty threshold array"
    
    # Check that threshold values are lists of integers
    for i, threshold_snapshot in enumerate(result[2]):
        assert isinstance(threshold_snapshot, list), f"Threshold at index {i} should be a list"
        assert len(threshold_snapshot) == 1, f"Expected 1 threshold value, got {len(threshold_snapshot)}"
        assert isinstance(threshold_snapshot[0], int), f"Threshold value should be int, got {type(threshold_snapshot[0])}"
    
    print("✓ Dynamic threshold tracking test passed")


def test_threshold_consistency():
    """Test that thresholds remain consistent when dynamic scaling is disabled."""
    print("Testing threshold consistency...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    # Ensure x0 is a list if it's a single value
    if not isinstance(x0, list):
        x0 = [x0]
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.1, 
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True  # Enabled but should return unchanged thresholds
    }
    
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method='operator-splitting')
    
    # All threshold snapshots should be the same since adjust_threshold returns unchanged values
    initial_threshold = result[2][0]
    for i, threshold_snapshot in enumerate(result[2]):
        assert threshold_snapshot == initial_threshold, f"Threshold at index {i} changed: {threshold_snapshot} != {initial_threshold}"
    
    print("✓ Threshold consistency test passed")


if __name__ == "__main__":
    test_backward_compatibility()
    test_dynamic_threshold_tracking()
    test_threshold_consistency()
    print("\nAll dynamic threshold tests passed! ✅")