import random
import math
from typing import Any, Callable, Dict, List
from jsf.types import Time, SystemState, CompartmentValue, Trajectory
from jsf.base import BaseSimulator
from jsf.utils import array_plus_ab, array_multiply_ab


class OperatorSplittingSimulator(BaseSimulator):
    """Jump-Switch-Flow simulator using operator splitting method."""
    
    def __init__(self, 
                 x0: SystemState,
                 rates: Callable[[SystemState, Time], List[float]],
                 stoich: Dict[str, Any],
                 t_max: Time,
                 options: Dict[str, Any]):
        super().__init__(x0, rates, stoich, t_max, options)
        
        # Initialize operator-splitting specific parameters
        self.enforce_do = [(not (ed == 0)) for ed in options["EnforceDo"]]
        self._setup_compartment_tracking()
        self._initialize_stochastic_variables()
    
    def _setup_compartment_tracking(self):
        """Set up compartment and reaction tracking structures."""
        do_disc = self._is_discrete(self.x0)
        
        # Identify which compartments are in which reactions
        nu_comp = [[value != 0 for value in row] for row in self.nu]
        react_comp = [[value != 0 for value in row] for row in self.nu_reactant]
        self.compart_in_nu = [[value != 0 for value in row] 
                             for row in self._matrix_plus_ab(nu_comp, react_comp)]
        
        # Track which reactions are frozen (discrete)
        self.frozen_reaction = [False] * self.n_rates
        for idx in range(self.n_compartments):
            for reaction_idx in range(self.n_rates):
                if not self.enforce_do[idx]:
                    if do_disc[idx] and self.compart_in_nu[reaction_idx][idx]:
                        self.frozen_reaction[reaction_idx] = True
                else:
                    if do_disc[idx] and self.compart_in_nu[reaction_idx][idx]:
                        self.frozen_reaction[reaction_idx] = True
    
    def _initialize_stochastic_variables(self):
        """Initialize stochastic variables for jump processes."""
        self.integral_of_firing_times = [0.0] * self.n_rates
        self.rand_times = [random.random() for _ in range(self.n_rates)]
        self.tau_array = [Time(0.0)] * self.n_rates
    
    def simulate(self) -> Trajectory:
        """Run the operator splitting simulation."""
        # Initialize solution arrays
        X = [[self.x0[i]] for i in range(self.n_compartments)]
        tau_arr = [Time(0.0)]
        iters = 0
        
        # Track time
        abs_t = Time(0.0)
        cont_t = Time(0.0)
        
        x_prev = self.x0
        x_curr = self.x0
        do_disc = self._is_discrete(self.x0)
        
        newly_disc_comp_index = None
        correct_integer = 0
        
        while cont_t < self.t_max:
            dtau = self.dt
            x_prev = SystemState([x[iters] for x in X])
            props = self.rates(x_prev, cont_t)
            
            # Forward Euler step
            dx_dt = self._compute_dx_dt(props, self.frozen_reaction)
            
            # Check for regime changes
            (dtau, correct_integer, do_disc, self.frozen_reaction, 
             new_do_disc, new_frozen_reaction, newly_disc_comp_index) = \
                self._update_compartment_regime(dtau, x_prev, dx_dt, props, do_disc)
            
            # Apply forward Euler step for continuous compartments
            x_curr = SystemState([
                CompartmentValue(X[i][iters] + (0 if do_disc[i] else dtau * dx_dt[i])) 
                for i in range(self.n_compartments)
            ])
            
            # Update discrete compartments if needed
            original_do_disc = do_disc[:]
            if correct_integer == 1:
                new_do_disc, new_frozen_reaction = self._is_discrete_update(
                    x_prev, do_disc, self.frozen_reaction)
                self.frozen_reaction = new_frozen_reaction[:]
                do_disc = new_do_disc[:]
            
            # Stochastic loop
            abs_t = cont_t
            dtau_cont_step = dtau
            time_passed = Time(0.0)
            
            stay_while = any(do_disc)
            while stay_while:
                if time_passed > 0:
                    props = self.rates(x_curr, abs_t)
                
                integral_step = self._compute_integral_of_firing_times(
                    dtau, props, x_prev, x_curr, abs_t)
                self.integral_of_firing_times = array_plus_ab(
                    self.integral_of_firing_times,
                    array_multiply_ab(integral_step, self.frozen_reaction))
                
                # Update for newly discrete compartments
                if correct_integer == 1:
                    self._handle_newly_discrete_compartment(newly_disc_comp_index)
                
                fired_reactions = [
                    ((0 > (rand - (1 - math.exp(-integral)))) and disc)
                    for rand, integral, disc in zip(
                        self.rand_times, self.integral_of_firing_times, self.frozen_reaction)
                ]
                
                if any(fired_reactions):
                    self.tau_array = self._compute_firing_times(
                        fired_reactions, integral_step, props, dtau)
                    
                    if self._num_non_zero(self.tau_array) > 0:
                        (x_curr, x_prev, self.integral_of_firing_times, integral_step,
                         self.rand_times, time_passed, abs_t, dtau_min) = \
                            self._implement_fired_reaction(
                                x_prev, x_curr, integral_step, time_passed, abs_t, 
                                X, iters, dx_dt, original_do_disc, props)
                        
                        # Random rounding for discrete values
                        for ix, x in enumerate(x_curr):
                            if (x < self.switching_threshold[ix] and 
                                abs(x - round(x)) > pow(10, -10)):
                                x_floor = math.floor(x)
                                if random.uniform(0, 1) < (x - x_floor):
                                    x_curr[ix] = x_floor + 1
                                else:
                                    x_curr[ix] = x_floor
                                stay_while = False
                        
                        iters += 1
                        for i in range(self.n_compartments):
                            X[i].append(x_curr[i])
                        tau_arr.append(abs_t)
                        dtau = dtau - dtau_min
                    else:
                        stay_while = False
                else:
                    stay_while = False
                
                if time_passed >= dtau_cont_step:
                    stay_while = False
            
            iters += 1
            cont_t = Time(cont_t + dtau_cont_step)
            tau_arr.append(cont_t)
            
            for i in range(len(X)):
                X[i].append(CompartmentValue(
                    X[i][iters - 1] + (0 if do_disc[i] else 
                                      (dtau_cont_step - time_passed) * dx_dt[i])))
            
            if correct_integer == 1:
                self._finalize_newly_discrete_compartment(
                    newly_disc_comp_index, X, iters)
        
        return Trajectory((X, tau_arr))
    
    def _compute_dx_dt(self, props: List[float], frozen_reaction: List[bool]) -> List[float]:
        """Compute derivative of state vector for continuous compartments."""
        return [sum(0 if frozen_reaction[i] else props[i] * self.nu[i][j]
                   for i in range(len(props)))
               for j in range(self.n_compartments)]
    
    def _update_compartment_regime(self, dt, x_prev, dx_dt, props, do_disc):
        """Check if any compartments need to switch regimes."""
        new_do_disc, new_frozen_reaction = self._is_discrete_update(
            x_prev, do_disc, self.frozen_reaction)
        
        correct_integer = 0
        newly_disc_comp_index = None
        dtau = dt
        
        x_step = [(0 if is_disc else dtau * dxi) 
                 for dxi, is_disc in zip(dx_dt, new_do_disc)]
        
        # Check for compartments switching to discrete
        switch_candidates = [(x + dxi <= thresh and not is_disc) 
                           for x, is_disc, thresh, dxi in 
                           zip(x_prev, new_do_disc, self.switching_threshold, x_step)]
        
        if any(switch_candidates):
            possible_dtau = [dt]
            for i, (x, is_disc, thresh, dxi) in enumerate(
                zip(x_prev, new_do_disc, self.switching_threshold, x_step)):
                if switch_candidates[i]:
                    x_prev_pos = x_prev[i]
                    rounded_x_prev_pos = math.ceil(x_prev_pos + x_step[i])
                    dx_dt_pos = dx_dt[i]
                    possible_dtau.append(abs((rounded_x_prev_pos - x_prev_pos) / dx_dt_pos))
            
            if len(possible_dtau) > 1:
                dtau = min(possible_dtau)
                pos = possible_dtau.index(dtau)
                if pos > 0:
                    newly_disc_comp_index = pos - 1
                    correct_integer = 1
            else:
                self.frozen_reaction = new_frozen_reaction
                do_disc = new_do_disc
        else:
            self.frozen_reaction = new_frozen_reaction
            do_disc = new_do_disc
        
        return (dtau, correct_integer, do_disc, self.frozen_reaction, 
                new_do_disc, new_frozen_reaction, newly_disc_comp_index)
    
    def _is_discrete_update(self, x, do_disc, frozen_reaction):
        """Update discrete compartment status."""
        do_disc_tmp = [x_val <= threshold 
                      for x_val, threshold in zip(x, self.switching_threshold)]
        
        for idx, enforce in enumerate(self.enforce_do):
            if enforce:
                do_disc_tmp[idx] = do_disc[idx]
        
        are_equal = [do_disc_tmp[idx] == do_disc[idx] 
                    for idx in range(self.n_compartments)]
        
        if all(are_equal):
            frozen_reaction_tmp = frozen_reaction.copy()
        else:
            frozen_reaction_tmp = [False] * self.n_rates
            for idx in range(self.n_compartments):
                for reaction_idx in range(self.n_rates):
                    if not self.enforce_do[idx]:
                        if (do_disc_tmp[idx] and 
                            self.compart_in_nu[reaction_idx][idx]):
                            frozen_reaction_tmp[reaction_idx] = True
                    else:
                        if (do_disc[idx] and 
                            self.compart_in_nu[reaction_idx][idx]):
                            frozen_reaction_tmp[reaction_idx] = True
        
        return do_disc_tmp, frozen_reaction_tmp
    
    def _compute_integral_of_firing_times(self, dtau, props, x_prev, x_curr, abs_t):
        """Compute integral of firing times using trapezoidal rule."""
        return [dtau * 0.5 * (p + r) 
               for p, r in zip(props, self.rates(x_curr, abs_t + dtau))]
    
    def _compute_firing_times(self, fired_reactions, integral_step, props, dt):
        """Compute firing times for reactions that have fired."""
        tau_array = [0.0] * self.n_rates
        for kk in range(self.n_rates):
            if fired_reactions[kk]:
                integral_t0_ti = integral_step[kk] - self.integral_of_firing_times[kk]
                integral = integral_t0_ti - math.log((1 - self.rand_times[kk]))
                tau_array[kk] = integral / props[kk]
        return tau_array
    
    def _implement_fired_reaction(self, x_prev, x_curr, integral_step, time_passed, 
                                abs_t, X, iters, dx_dt, original_do_disc, props):
        """Implement a fired reaction and update the system state."""
        tau_array = [float('inf') if tau == 0.0 else tau for tau in self.tau_array]
        dtau_min = min(tau_array)
        pos = tau_array.index(dtau_min)
        
        time_passed = time_passed + dtau_min
        abs_t = abs_t + dtau_min
        
        x_curr = [X[i][iters] + self.nu[pos][i] + 
                 (0 if original_do_disc[i] else dtau_min * dx_dt[i]) 
                 for i in range(self.n_compartments)]
        x_prev = [X[i][iters] for i in range(self.n_compartments)]
        
        # Update integrals
        self.integral_of_firing_times = [
            integral - step * disc 
            for integral, step, disc in zip(
                self.integral_of_firing_times, integral_step, self.frozen_reaction)]
        
        integral_step = self._compute_integral_of_firing_times(
            dtau_min, props, x_prev, x_curr, abs_t)
        
        self.integral_of_firing_times = [
            integral + step * disc 
            for integral, step, disc in zip(
                self.integral_of_firing_times, integral_step, self.frozen_reaction)]
        
        self.integral_of_firing_times[pos] = 0.0
        self.rand_times[pos] = random.random()
        
        return (x_curr, x_prev, self.integral_of_firing_times, integral_step,
                self.rand_times, time_passed, abs_t, dtau_min)
    
    def _handle_newly_discrete_compartment(self, newly_disc_comp_index):
        """Handle compartments that have just become discrete."""
        if newly_disc_comp_index is not None:
            for ii in range(self.n_compartments):
                if newly_disc_comp_index == ii and not self.enforce_do[ii]:
                    for jj in range(self.n_rates):
                        if self.compart_in_nu[jj][ii]:
                            self.frozen_reaction[jj] = True
                            self.integral_of_firing_times[jj] = 0.0
                            self.rand_times[jj] = random.random()
    
    def _finalize_newly_discrete_compartment(self, newly_disc_comp_index, X, iters):
        """Finalize compartments that have become discrete."""
        if newly_disc_comp_index is not None:
            pos = newly_disc_comp_index
            X[pos][iters] = round(X[pos][iters])
            
            for jj in range(self.n_rates):
                if self.compart_in_nu[jj][pos]:
                    self.frozen_reaction[jj] = True
                    self.integral_of_firing_times[jj] = 0.0
                    self.rand_times[jj] = random.random()
    
    @staticmethod
    def _matrix_plus_ab(matrix_a, matrix_b):
        """Add two matrices element-wise."""
        return [[a + b for a, b in zip(row1, row2)] 
               for row1, row2 in zip(matrix_a, matrix_b)]
    
    @staticmethod
    def _num_non_zero(array):
        """Count non-zero elements in array."""
        return sum(1 for element in array if element != 0)