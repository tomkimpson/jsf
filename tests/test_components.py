import unittest
import random
from jsf.base import BaseSimulator
from jsf.operator_splitting import OperatorSplittingSimulator
from jsf.exact_method import ExactSimulator
from jsf.utils import array_plus_ab, array_multiply_ab, num_non_zero
from jsf.types import SystemState, Time


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in jsf.utils."""
    
    def test_array_plus_ab(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = array_plus_ab(a, b)
        self.assertEqual(result, [5, 7, 9])
    
    def test_array_multiply_ab(self):
        a = [2, 3, 4]
        b = [5, 6, 7]
        result = array_multiply_ab(a, b)
        self.assertEqual(result, [10, 18, 28])
    
    def test_num_non_zero(self):
        array = [0, 1, 0, 3, 0, 5]
        result = num_non_zero(array)
        self.assertEqual(result, 3)


class TestBaseSimulator(unittest.TestCase):
    """Test base simulator functionality."""
    
    def setUp(self):
        self.x0 = SystemState([10, 5])
        self.rates = lambda x, t: [1.0, 0.5]
        self.stoich = {
            "nu": [[1, -1], [-1, 1]],
            "nuReactant": [[1, 0], [0, 1]],
            "nuProduct": [[0, 1], [1, 0]]
        }
        self.t_max = Time(1.0)
        self.options = {
            "dt": 0.1,
            "SwitchingThreshold": [15, 15],
            "EnforceDo": [0, 0]
        }
    
    def test_base_simulator_initialization(self):
        """Test that BaseSimulator initializes correctly."""
        # We can't instantiate BaseSimulator directly, so test via subclass
        simulator = OperatorSplittingSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        self.assertEqual(simulator.x0, self.x0)
        self.assertEqual(simulator.t_max, self.t_max)
        self.assertEqual(simulator.n_rates, 2)
        self.assertEqual(simulator.n_compartments, 2)
        self.assertEqual(simulator.switching_threshold, [15, 15])
    
    def test_is_discrete_method(self):
        """Test the _is_discrete method."""
        simulator = OperatorSplittingSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        # Test with values above threshold
        state_above = SystemState([20, 18])
        result = simulator._is_discrete(state_above)
        self.assertEqual(result, [False, False])
        
        # Test with values below threshold
        state_below = SystemState([10, 12])
        result = simulator._is_discrete(state_below)
        self.assertEqual(result, [True, True])
        
        # Test with mixed values
        state_mixed = SystemState([20, 12])
        result = simulator._is_discrete(state_mixed)
        self.assertEqual(result, [False, True])


class TestOperatorSplittingSimulator(unittest.TestCase):
    """Test operator splitting simulator specific functionality."""
    
    def setUp(self):
        random.seed(1234)  # For reproducible tests
        self.x0 = SystemState([5, 10])
        self.rates = lambda x, t: [0.1 * x[0], 0.2 * x[1]]
        self.stoich = {
            "nu": [[1, -1], [-1, 1]],
            "nuReactant": [[1, 0], [0, 1]],
            "nuProduct": [[0, 1], [1, 0]]
        }
        self.t_max = Time(1.0)
        self.options = {
            "dt": 0.1,
            "SwitchingThreshold": [15, 15],
            "EnforceDo": [0, 0]
        }
    
    def test_compute_dx_dt(self):
        """Test derivative computation."""
        simulator = OperatorSplittingSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        props = [1.0, 2.0]
        frozen = [False, False]
        dx_dt = simulator._compute_dx_dt(props, frozen)
        
        # dx_dt[0] = 1.0 * 1 + 2.0 * (-1) = -1.0
        # dx_dt[1] = 1.0 * (-1) + 2.0 * 1 = 1.0
        self.assertEqual(dx_dt, [-1.0, 1.0])
    
    def test_compute_dx_dt_with_frozen_reactions(self):
        """Test derivative computation with frozen reactions."""
        simulator = OperatorSplittingSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        props = [1.0, 2.0]
        frozen = [True, False]  # First reaction frozen
        dx_dt = simulator._compute_dx_dt(props, frozen)
        
        # dx_dt[0] = 0 * 1 + 2.0 * (-1) = -2.0
        # dx_dt[1] = 0 * (-1) + 2.0 * 1 = 2.0
        self.assertEqual(dx_dt, [-2.0, 2.0])


class TestExactSimulator(unittest.TestCase):
    """Test exact simulator specific functionality."""
    
    def setUp(self):
        random.seed(1234)  # For reproducible tests
        self.x0 = SystemState([5, 10])
        self.rates = lambda x, t: [0.1 * x[0], 0.2 * x[1]]
        self.stoich = {
            "nu": [[1, -1], [-1, 1]],
            "nuReactant": [[1, 0], [0, 1]],
            "nuProduct": [[0, 1], [1, 0]]
        }
        self.t_max = Time(1.0)
        self.options = {
            "dt": 0.1,
            "SwitchingThreshold": [15, 15],
            "EnforceDo": [0, 0]
        }
    
    def test_is_jumping_all_continuous(self):
        """Test jumping detection when all compartments are continuous."""
        simulator = ExactSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        # All compartments above threshold - no reactions should be jumping
        state = SystemState([20, 25])
        result = simulator._is_jumping(state)
        self.assertEqual(result, [False, False])
    
    def test_is_jumping_mixed_regimes(self):
        """Test jumping detection with mixed continuous/discrete compartments."""
        simulator = ExactSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        # First compartment discrete, second continuous
        state = SystemState([10, 20])
        result = simulator._is_jumping(state)
        # Both reactions should be jumping since they involve the discrete compartment
        self.assertEqual(result, [True, True])


class TestSimulatorComparison(unittest.TestCase):
    """Test that both simulators produce similar statistical results."""
    
    def setUp(self):
        random.seed(1234)
        self.x0 = SystemState([50, 10])  # Start with larger numbers
        self.rates = lambda x, t: [0.01 * x[0], 0.02 * x[1]]
        self.stoich = {
            "nu": [[1, -1], [-1, 1]],
            "nuReactant": [[1, 0], [0, 1]],
            "nuProduct": [[0, 1], [1, 0]]
        }
        self.t_max = Time(2.0)
        self.options = {
            "dt": 0.01,
            "SwitchingThreshold": [30, 30],
            "EnforceDo": [0, 0]
        }
    
    def test_both_simulators_run_without_error(self):
        """Test that both simulators complete without errors."""
        op_sim = OperatorSplittingSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        exact_sim = ExactSimulator(
            self.x0, self.rates, self.stoich, self.t_max, self.options)
        
        # Should not raise any exceptions
        op_result = op_sim.simulate()
        exact_result = exact_sim.simulate()
        
        # Check basic structure
        self.assertEqual(len(op_result), 2)  # (trajectories, times)
        self.assertEqual(len(exact_result), 2)
        self.assertEqual(len(op_result[0]), 2)  # Two compartments
        self.assertEqual(len(exact_result[0]), 2)


if __name__ == '__main__':
    unittest.main()