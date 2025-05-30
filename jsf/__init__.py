from typing import Any, Dict
from jsf.types import SystemState, Time, Trajectory
from jsf.operator_splitting import OperatorSplittingSimulator
from jsf.exact_method import ExactSimulator
from jsf import sbml


def read_sbml(sbml_xml: str):
    """
    Read an SBML file and return the initial state, rates, and
    stoichiometric matrix.

    Args:
        sbml_xml: The SBML file to read.

    Returns:
        x0: Initial state.
        rates: Function that computes reaction rates.
        stoich: Stoichiometry matrix.
    """
    return sbml.read_sbml(sbml_xml)


def jsf(x0: SystemState, rates, stoich, t_max: Time, **kwargs) -> Trajectory:
    """Generates a sample from the JSF process.

    Args:
        x0: The initial state of the system.
        rates: A function that takes the current state and time and
            returns the rates of each reaction.
        stoich: A dictionary containing the stoichiometry of the
            system.
        t_max: The final time of the simulation.
        **kwargs: A dictionary containing the simulation options.

    Returns:
        A list containing the time series of the state of the system.

    Raises:
        RuntimeError: If the requested method is not implemented.
    """
    method = kwargs.get('method')
    config = kwargs['config']
    
    if method is None or method == 'exact':
        simulator = ExactSimulator(x0, rates, stoich, t_max, config)
    elif method == 'operator-splitting':
        simulator = OperatorSplittingSimulator(x0, rates, stoich, t_max, config)
    else:
        raise RuntimeError(f"Requested method is not supported: {method}")

    return simulator.simulate()


# Legacy function for backward compatibility
def JumpSwitchFlowSimulator(x0: SystemState, rates, stoich, t_max: Time, options: Dict[str, Any]) -> Trajectory:
    """Legacy wrapper for OperatorSplittingSimulator."""
    simulator = OperatorSplittingSimulator(x0, rates, stoich, t_max, options)
    return simulator.simulate()