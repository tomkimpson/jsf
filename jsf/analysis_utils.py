"""
Analysis utilities for JSF simulations.

This module provides tools for extinction analysis, Monte Carlo simulations,
and statistical analysis of JSF results.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from .types import SystemState, Time


def detect_extinction(trajectory: List[List[float]], times: List[Time], extinction_threshold: float = 0.0) -> Tuple[bool, Optional[float]]:
    """
    Detect if extinction occurred during simulation.
    
    Args:
        trajectory: List of compartment trajectories [compartment][timepoint]
        times: List of time points
        extinction_threshold: Population level considered extinct (default: 0.0)
        
    Returns:
        (extinction_occurred, extinction_time): Bool and time of first extinction, or None
    """
    for t_idx, time in enumerate(times):
        for compartment_idx, compartment_trajectory in enumerate(trajectory):
            if t_idx < len(compartment_trajectory):
                population = compartment_trajectory[t_idx]
                if population <= extinction_threshold:
                    return True, time
    
    return False, None


def calculate_binomial_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate binomial proportion confidence interval using normal approximation.
    
    Args:
        successes: Number of successes (extinctions)
        trials: Total number of trials (simulations)
        confidence: Confidence level (default: 0.95)
        
    Returns:
        (proportion, lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
        
    p = successes / trials
    
    # Normal approximation z-score for confidence interval
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    # Standard error for binomial proportion
    se = math.sqrt(p * (1 - p) / trials)
    
    # Confidence interval
    margin_error = z * se
    lower = max(0.0, p - margin_error)
    upper = min(1.0, p + margin_error)
    
    return p, lower, upper


def run_monte_carlo_extinction(
    base_config: Dict[str, Any],
    model_params: Dict[str, Any], 
    n_runs: int = 500,
    base_seed: int = 42,
    extinction_threshold: float = 0.0,
    t_max: float = 10.0
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to estimate extinction probability.
    
    Args:
        base_config: Base JSF configuration
        model_params: Model-specific parameters (x0, rates, stoich)
        n_runs: Number of simulation runs
        base_seed: Base random seed (will add run index)
        extinction_threshold: Population level considered extinct
        t_max: Simulation time
        
    Returns:
        Dictionary with simulation results and statistics
    """
    import jsf
    
    extinctions = 0
    extinction_times = []
    final_populations = []
    threshold_histories = []  # For dynamic threshold cases
    
    print(f"Running {n_runs} Monte Carlo simulations...")
    
    for run in range(n_runs):
        # Set unique seed for this run
        config = base_config.copy()
        config["seed"] = base_seed + run
        
        # Progress tracking
        if (run + 1) % 100 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs...")
        
        # Run simulation
        try:
            result = jsf.jsf(
                model_params["x0"], 
                model_params["rates"], 
                model_params["stoich"], 
                t_max, 
                config=config, 
                method="operator-splitting"
            )
            
            # Extract results
            times = result[1]
            trajectory = result[0]
            
            # Store final populations
            final_pops = [traj[-1] for traj in trajectory]
            final_populations.append(final_pops)
            
            # Store threshold history if available
            if len(result) > 2:  # Has threshold history
                threshold_histories.append([th[0] for th in result[2]])
            
            # Check for extinction
            extinct, ext_time = detect_extinction(trajectory, times, extinction_threshold)
            
            if extinct:
                extinctions += 1
                extinction_times.append(ext_time)
            else:
                extinction_times.append(None)
                
        except Exception as e:
            print(f"  Warning: Run {run} failed with error: {e}")
            # Count failed runs as non-extinctions
            extinction_times.append(None)
            final_populations.append([0.0] * len(model_params["x0"]))
    
    # Calculate statistics
    prob, lower, upper = calculate_binomial_confidence_interval(extinctions, n_runs)
    
    # Mean extinction time (excluding non-extinctions)
    valid_extinction_times = [t for t in extinction_times if t is not None]
    mean_extinction_time = np.mean(valid_extinction_times) if valid_extinction_times else None
    
    # Final population statistics
    final_pops_array = np.array(final_populations)
    mean_final_pops = np.mean(final_pops_array, axis=0)
    std_final_pops = np.std(final_pops_array, axis=0)
    
    # Threshold statistics (if available)
    threshold_stats = None
    if threshold_histories:
        # Calculate mean and std of threshold values across runs
        all_thresholds = [th for hist in threshold_histories for th in hist]
        threshold_stats = {
            "mean": np.mean(all_thresholds),
            "std": np.std(all_thresholds),
            "min": np.min(all_thresholds),
            "max": np.max(all_thresholds)
        }
    
    return {
        "n_runs": n_runs,
        "extinctions": extinctions,
        "extinction_probability": prob,
        "confidence_interval": (lower, upper),
        "extinction_times": extinction_times,
        "mean_extinction_time": mean_extinction_time,
        "final_populations": final_populations,
        "mean_final_populations": mean_final_pops.tolist(),
        "std_final_populations": std_final_pops.tolist(),
        "threshold_histories": threshold_histories,
        "threshold_stats": threshold_stats
    }


def compare_extinction_results(results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare extinction results across different configurations.
    
    Args:
        results_dict: Dictionary mapping config names to results
        
    Returns:
        Comparison statistics and analysis
    """
    comparison = {
        "configurations": list(results_dict.keys()),
        "extinction_probabilities": {},
        "confidence_intervals": {},
        "mean_extinction_times": {},
        "statistical_tests": {}
    }
    
    for config_name, results in results_dict.items():
        comparison["extinction_probabilities"][config_name] = results["extinction_probability"]
        comparison["confidence_intervals"][config_name] = results["confidence_interval"]
        comparison["mean_extinction_times"][config_name] = results["mean_extinction_time"]
    
    # Simple pairwise comparisons (overlap of confidence intervals)
    config_names = list(results_dict.keys())
    overlaps = {}
    
    for i, config1 in enumerate(config_names):
        for j, config2 in enumerate(config_names):
            if i < j:  # Avoid duplicate comparisons
                ci1 = results_dict[config1]["confidence_interval"]
                ci2 = results_dict[config2]["confidence_interval"]
                
                # Check if confidence intervals overlap
                overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
                overlaps[f"{config1}_vs_{config2}"] = {
                    "confidence_intervals_overlap": overlap,
                    "prob_diff": abs(results_dict[config1]["extinction_probability"] - 
                                   results_dict[config2]["extinction_probability"])
                }
    
    comparison["statistical_tests"] = overlaps
    
    return comparison