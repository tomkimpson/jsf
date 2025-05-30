from abc import ABC, abstractmethod
from typing import Any, Callable, Dict
from jsf.types import Time, SystemState, Trajectory


class BaseSimulator(ABC):
    """Abstract base class for all JSF simulators."""
    
    def __init__(self, 
                 x0: SystemState,
                 rates: Callable[[SystemState, Time], list[float]],
                 stoich: Dict[str, Any],
                 t_max: Time,
                 options: Dict[str, Any]):
        self.x0 = x0
        self.rates = rates
        self.stoich = stoich
        self.t_max = t_max
        self.options = options
        
        # Extract common parameters
        self.nu = stoich["nu"]
        self.nu_reactant = stoich["nuReactant"]
        self.nu_product = stoich.get("nuProduct", [])
        self.n_rates = len(self.nu)
        self.n_compartments = len(self.nu[0])
        self.switching_threshold = options["SwitchingThreshold"]
        self.dt = options["dt"]
    
    @abstractmethod
    def simulate(self) -> Trajectory:
        """Run the simulation and return the trajectory."""
        pass
    
    def _is_discrete(self, state: SystemState) -> list[bool]:
        """Determine which compartments are discrete based on switching thresholds."""
        return [x <= threshold for x, threshold in zip(state, self.switching_threshold)]