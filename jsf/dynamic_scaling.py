"""
Dynamic threshold scaling module for JSF simulations.

This module provides functionality to dynamically adjust switching thresholds
during simulation based on current system state.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from .types import SystemState, Time


def adjust_threshold(
    state: SystemState, 
    stoich_matrix: List[List[float]], 
    rates: List[float], 
    current_time: Time, 
    current_threshold: List[int], 
    options: Optional[Dict[str, Any]] = None
) -> List[int]:
    """
    Adjust switching threshold based on current system state.
    
    This function enables dynamic scaling of switching thresholds during
    simulation. By default, it returns the unchanged threshold to maintain
    backward compatibility.
    
    Args:
        state: Current system state (X values)
        stoich_matrix: Stoichiometric matrix (nu)
        rates: Current reaction rates 
        current_time: Current simulation time
        current_threshold: Current threshold values
        options: Options dictionary (for future extensibility)
        
    Returns:
        List[int]: Updated threshold values
    """
    # Check if dynamic scaling is enabled
    if options and options.get("EnableDynamicThreshold", False):
        # Get the strategy (default to "increment")
        strategy = options.get("DynamicThresholdStrategy", "increment")
        
        if strategy == "increment":
            return _increment_strategy(current_threshold)
        elif strategy == "sqrt_population":
            return _sqrt_population_strategy(state)
        elif strategy == "rate_based":
            return _rate_based_strategy(state, rates)
        elif strategy == "jacobian_based":
            return _jacobian_based_strategy(state, stoich_matrix, rates, current_time, options)
        else:
            # Unknown strategy - return unchanged
            return current_threshold
    else:
        # Static threshold - return unchanged (backward compatible)
        return current_threshold


def _increment_strategy(current_threshold: List[int]) -> List[int]:
    """
    Simple strategy that increases each threshold by 1 every step.
    
    This is a "dumb" strategy for testing dynamic threshold functionality.
    
    Args:
        current_threshold: Current threshold values
        
    Returns:
        List[int]: Threshold values incremented by 1
    """
    return [threshold + 1 for threshold in current_threshold]


def _sqrt_population_strategy(state: SystemState) -> List[int]:
    """
    Strategy that sets threshold proportional to sqrt of population size.
    
    Args:
        state: Current system state
        
    Returns:
        List[int]: Threshold values based on sqrt(population)
    """
    import math
    return [max(1, int(math.sqrt(x))) for x in state]


def _rate_based_strategy(state: SystemState, rates: List[float]) -> List[int]:
    """
    Strategy that adjusts threshold based on reaction rates.
    
    Args:
        state: Current system state
        rates: Current reaction rates
        
    Returns:
        List[int]: Threshold values based on reaction rates
    """
    # Simple example: threshold proportional to total reaction rate affecting each compartment
    total_rate = sum(rates) if rates else 1.0
    return [max(1, int(total_rate * 0.1)) for _ in state]


def _jacobian_based_strategy(
    state: SystemState, 
    stoich_matrix: List[List[float]], 
    rates: List[float],
    current_time: Time,
    options: Dict[str, Any]
) -> List[int]:
    """
    Strategy that computes scale-aware switching score S(V) = (sigma / |f|) * lambda_max(J).
    
    This strategy uses the noise-to-drift ratio multiplied by the maximum eigenvalue
    of the Jacobian to determine appropriate switching thresholds.
    
    Args:
        state: Current system state
        stoich_matrix: Stoichiometric matrix (nu)
        rates: Current reaction rates
        current_time: Current simulation time
        options: Options dictionary containing JacobianParams
        
    Returns:
        List[int]: Threshold values based on scale-aware score
        
    Raises:
        ValueError: If required jacobian_func is not provided in JacobianParams
    """
    # Extract jacobian parameters from options
    jacobian_params = options.get('JacobianParams', {})
    
    # Required parameters
    jacobian_func = jacobian_params.get('jacobian_func')
    if jacobian_func is None:
        raise ValueError("jacobian_func must be provided in JacobianParams for jacobian_based strategy")
    
    # Default sigmoid parameters
    omega_min = jacobian_params.get('omega_min', 5.0)
    omega_max = jacobian_params.get('omega_max', 100.0)
    k = jacobian_params.get('k', 2.5)
    score_threshold = jacobian_params.get('score_threshold', 0.1)
    debug = jacobian_params.get('debug', False)
    
    # Convert inputs to numpy arrays for computation
    state_arr = np.array(state)
    stoich_arr = np.array(stoich_matrix)
    rates_arr = np.array(rates)
    
    # Compute drift: f = stoich_matrix.T @ rates (transpose since nu is reactions x species)
    f = stoich_arr.T @ rates_arr
    drift_norm = np.linalg.norm(f)
    
    # Compute local noise variance: sigma^2 = sum_k ||eta_k||^2 * lambda_k
    # Note: stoich_matrix rows are reactions, columns are species, so eta_k is row k
    sigma2 = sum(np.linalg.norm(stoich_arr[k, :])**2 * rates[k] for k in range(len(rates)))
    sigma = np.sqrt(sigma2)
    
    # Compute Jacobian and maximum eigenvalue
    J = jacobian_func(state)
    lambda_max = np.max(np.real(np.linalg.eigvals(J)))
    
    # Compute switching score with numerical stability
    raw_stochasticity_score = (sigma / (drift_norm + 1e-10)) * max(0.0, lambda_max)
    # Apply sigmoid mapping to get dynamic threshold
    sigmoid = 1 / (1 + np.exp(-k * (raw_stochasticity_score - score_threshold)))
    dynamic_threshold = omega_min + (omega_max - omega_min) * sigmoid
    
    # Debug output if requested
    if debug:
        print(f"Jacobian-based strategy debug:")
        print(f"  Stochasticity score: {raw_stochasticity_score:.6f}")
        print(f"  Score threshold: {score_threshold}")
        print(f"  Dynamic threshold: {dynamic_threshold:.2f}")
        print(f"  Sigmoid value: {sigmoid:.6f}")
    
    # Ensure threshold is at least 1 and return for all compartments
    threshold_value = max(1, int(dynamic_threshold))
    return [threshold_value] * len(state)


def _calculate_adaptive_threshold(
    state: SystemState,
    stoich_matrix: List[List[float]], 
    rates: List[float]
) -> List[int]:
    """
    Helper function for calculating adaptive thresholds.
    
    This is a placeholder for future implementation of more sophisticated
    threshold calculation algorithms.
    
    Args:
        state: Current system state
        stoich_matrix: Stoichiometric matrix
        rates: Current reaction rates
        
    Returns:
        List[int]: Calculated threshold values
    """
    # Placeholder implementation
    # Could implement strategies like:
    # - Threshold based on reaction flux through stoichiometric matrix
    # - Time-dependent thresholds
    # - Machine learning-based adaptive thresholds
    
    return [max(1, int(x * 0.1)) for x in state]


# Helper Jacobian functions for common models
def jacobian_lotka_volterra(state: SystemState, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Jacobian of the Lotka-Volterra ODE system.
    
    For the system:
    dV1/dt = alpha * V1 - beta * V1 * V2
    dV2/dt = beta * V1 * V2 - gamma * V2
    
    Args:
        state: Current system state [V1, V2]
        alpha: Prey birth rate
        beta: Predation rate
        gamma: Predator death rate
        
    Returns:
        2x2 Jacobian matrix
    """
    state_arr = np.asarray(state)
    V1, V2 = state_arr
    return np.array([
        [alpha - beta * V2, -beta * V1],
        [beta * V2,         beta * V1 - gamma]
    ])