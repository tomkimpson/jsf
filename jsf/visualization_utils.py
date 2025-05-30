"""
Visualization utilities for JSF extinction analysis.

This module provides plotting functions for extinction probability analysis
and Monte Carlo simulation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional


def plot_extinction_probabilities(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot extinction probabilities with confidence intervals.
    
    Args:
        results: Dictionary mapping config names to results
        save_path: Optional path to save the plot
    """
    config_names = list(results.keys())
    probabilities = [results[name]["extinction_probability"] for name in config_names]
    confidence_intervals = [results[name]["confidence_interval"] for name in config_names]
    
    # Extract confidence interval bounds
    lower_bounds = [ci[0] for ci in confidence_intervals]
    upper_bounds = [ci[1] for ci in confidence_intervals]
    
    # Calculate error bars
    lower_errors = [p - l for p, l in zip(probabilities, lower_bounds)]
    upper_errors = [u - p for p, u in zip(upper_bounds, probabilities)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(config_names))
    bars = ax.bar(x_pos, probabilities, 
                  yerr=[lower_errors, upper_errors],
                  capsize=5, capthick=2, 
                  color=['skyblue', 'lightcoral', 'lightgreen'],
                  edgecolor='black', linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Extinction Probability', fontsize=12)
    ax.set_title('Extinction Probability Comparison\n(Lotka-Volterra Model)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in config_names], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, prob, ci) in enumerate(zip(bars, probabilities, confidence_intervals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + upper_errors[i] + 0.01,
                f'{prob:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Extinction probability plot saved to: {save_path}")
    
    plt.show()


def plot_sample_trajectories(results: Dict[str, Dict[str, Any]], model_params: Dict[str, Any], 
                           n_samples: int = 5, save_path: Optional[str] = None):
    """
    Plot sample trajectories from each configuration.
    
    Args:
        results: Dictionary mapping config names to results
        model_params: Model parameters for generating sample trajectories
        n_samples: Number of sample trajectories per configuration
        save_path: Optional path to save the plot
    """
    import jsf
    from jsf.dynamic_scaling import jacobian_lotka_volterra
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Configuration setups (simplified)
    configs = {
        "large_threshold": {"SwitchingThreshold": [1000, 1000], "EnableDynamicThreshold": False},
        "medium_threshold": {"SwitchingThreshold": [30, 30], "EnableDynamicThreshold": False},
        "dynamic_jacobian": {
            "SwitchingThreshold": [30, 30], 
            "EnableDynamicThreshold": True,
            "DynamicThresholdStrategy": "jacobian_based",
            "JacobianParams": {
                "jacobian_func": model_params["jacobian_func"],
                "omega_min": 5.0, "omega_max": 100.0, "k": 2.5, "score_threshold": 0.1
            }
        }
    }
    
    base_config = {"EnforceDo": [0, 0], "dt": 0.01}
    
    for i, (config_name, config_specific) in enumerate(configs.items()):
        ax = axes[i]
        
        for sample in range(n_samples):
            # Merge configurations
            full_config = {**base_config, **config_specific, "seed": 100 + sample}
            
            # Run simulation
            try:
                result = jsf.jsf(
                    model_params["x0"], 
                    model_params["rates"], 
                    model_params["stoich"], 
                    10.0, 
                    config=full_config, 
                    method="operator-splitting"
                )
                
                times = result[1]
                prey = result[0][0]
                predator = result[0][1]
                
                # Plot trajectories
                alpha = 0.7 if sample == 0 else 0.4  # Highlight first trajectory
                ax.plot(times, prey, 'b-', alpha=alpha, linewidth=1.5, label='Prey' if sample == 0 else "")
                ax.plot(times, predator, 'r-', alpha=alpha, linewidth=1.5, label='Predator' if sample == 0 else "")
                
            except Exception as e:
                print(f"Warning: Sample trajectory {sample} failed for {config_name}: {e}")
        
        # Customize subplot
        ax.set_xlabel('Time')
        ax.set_ylabel('Population')
        ax.set_title(config_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.suptitle('Sample Population Trajectories by Configuration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample trajectories plot saved to: {save_path}")
    
    plt.show()


def plot_threshold_evolution(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """
    Plot threshold evolution for dynamic configuration.
    
    Args:
        results: Dictionary mapping config names to results
        save_path: Optional path to save the plot
    """
    # Find dynamic configuration results
    dynamic_results = None
    for config_name, result in results.items():
        if result.get("threshold_histories") and len(result["threshold_histories"]) > 0:
            dynamic_results = result
            break
    
    if not dynamic_results:
        print("No dynamic threshold data found for plotting.")
        return
    
    # Plot threshold evolution for several runs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    threshold_histories = dynamic_results["threshold_histories"]
    n_plot = min(10, len(threshold_histories))  # Plot up to 10 runs
    
    # Individual runs
    for i in range(n_plot):
        if i < len(threshold_histories):
            hist = threshold_histories[i]
            times = np.linspace(0, 10, len(hist))  # Approximate time axis
            alpha = 0.6 if i == 0 else 0.3
            ax1.plot(times, hist, alpha=alpha, linewidth=1)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Switching Threshold')
    ax1.set_title(f'Threshold Evolution (Sample of {n_plot} runs)')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of all threshold values
    all_thresholds = [th for hist in threshold_histories for th in hist]
    
    ax2.hist(all_thresholds, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Threshold Values (All Runs)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats = dynamic_results["threshold_stats"]
    if stats:
        ax2.axvline(stats["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold evolution plot saved to: {save_path}")
    
    plt.show()


def create_analysis_dashboard(results: Dict[str, Dict[str, Any]], model_params: Dict[str, Any]):
    """
    Create a comprehensive dashboard with all analysis plots.
    
    Args:
        results: Dictionary mapping config names to results
        model_params: Model parameters
    """
    print("Creating analysis dashboard...")
    
    # Create plots
    plot_extinction_probabilities(results, "extinction_probabilities.png")
    plot_sample_trajectories(results, model_params, n_samples=3, save_path="sample_trajectories.png")
    plot_threshold_evolution(results, "threshold_evolution.png")
    
    print("Analysis dashboard created! Check the generated PNG files.")