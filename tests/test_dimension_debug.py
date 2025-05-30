#!/usr/bin/env python
"""
Diagnostic test to investigate ThresholdArr vs TauArr dimension mismatch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jsf
import jsf.sbml as sbml


def test_dimension_debug():
    """Debug the dimension mismatch between ThresholdArr and TauArr."""
    print("Investigating ThresholdArr vs TauArr dimensions...")
    
    x0, rates, stoich = sbml.read_sbml('tests/data/birth-death-model.xml')
    if not isinstance(x0, list):
        x0 = [x0]
    
    print(f"Initial state x0: {x0}")
    
    # Test with a shorter simulation for easier debugging
    my_opts = {
        'EnforceDo': [0], 
        'dt': 0.5,  # Larger timestep for fewer iterations
        'SwitchingThreshold': [30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'increment'
    }
    
    result = jsf.jsf(x0, rates, stoich, 2.0, config=my_opts, method='operator-splitting')  # Shorter simulation
    
    times = result[1]
    compartments = result[0][0]  # First compartment
    thresholds = result[2]
    
    print(f"\nResults:")
    print(f"  len(times) = {len(times)}")
    print(f"  len(compartments) = {len(compartments)}")
    print(f"  len(thresholds) = {len(thresholds)}")
    
    print(f"\nTimes: {times}")
    print(f"Compartment values: {compartments}")
    print(f"Threshold values: {[th[0] for th in thresholds]}")
    
    print(f"\nExpected: len(ThresholdArr) should equal len(TauArr)")
    print(f"Actual: len(ThresholdArr)={len(thresholds)}, len(TauArr)={len(times)}")
    print(f"Difference: {len(times) - len(thresholds)}")
    
    # Check if compartment array length matches either
    print(f"Compartment length matches times: {len(compartments) == len(times)}")
    print(f"Compartment length matches thresholds: {len(compartments) == len(thresholds)}")


def test_lotka_volterra_debug():
    """Debug with Lotka-Volterra model to see if issue persists."""
    print("\n" + "="*50)
    print("Testing Lotka-Volterra model dimensions...")
    
    x0 = [50, 10]
    
    mA = 2.00
    mB = 0.05
    mC = 1.50
    rates = lambda x, t: [mA * x[0], mC * x[1], mB * x[0] * x[1]]
    
    reactant_matrix = [[1, 0], [0, 1], [1, 1]]
    product_matrix = [[2, 0], [0, 0], [0, 2]]
    
    stoich = {
        "nu": [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(product_matrix, reactant_matrix)],
        "DoDisc": [1, 1],
        "nuReactant": reactant_matrix,
        "nuProduct": product_matrix,
    }
    
    my_opts = {
        "EnforceDo": [0, 0],
        "dt": 0.1,
        "SwitchingThreshold": [30, 30],
        'EnableDynamicThreshold': True,
        'DynamicThresholdStrategy': 'increment'
    }
    
    result = jsf.jsf(x0, rates, stoich, 1.0, config=my_opts, method="operator-splitting")
    
    times = result[1]
    compartments = result[0]  # Both compartments
    thresholds = result[2]
    
    print(f"\nLotka-Volterra Results:")
    print(f"  len(times) = {len(times)}")
    print(f"  len(compartment[0]) = {len(compartments[0])}")
    print(f"  len(compartment[1]) = {len(compartments[1])}")
    print(f"  len(thresholds) = {len(thresholds)}")
    
    print(f"\nFirst few times: {times[:5]}")
    print(f"First few comp[0]: {compartments[0][:5]}")
    print(f"First few comp[1]: {compartments[1][:5]}")
    print(f"First few thresholds: {thresholds[:5]}")
    
    print(f"\nDimension analysis:")
    print(f"  Times vs Thresholds difference: {len(times) - len(thresholds)}")
    print(f"  Compartment[0] vs Times: {len(compartments[0]) == len(times)}")
    print(f"  Compartment[0] vs Thresholds: {len(compartments[0]) == len(thresholds)}")


if __name__ == "__main__":
    test_dimension_debug()
    test_lotka_volterra_debug()