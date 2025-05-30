import random
import math
import copy
from typing import Any, Callable, Dict, List, NewType, Tuple
from dataclasses import dataclass
from jsf.types import Time, SystemState, CompartmentValue, Trajectory
from jsf.base import BaseSimulator


@dataclass
class JumpClock:
    clock: Time
    start_time: Time

    def set_clock(self, new_clock: float) -> None:
        self.clock = Time(new_clock)


ExtendedState = NewType('ExtendedState',
                        Tuple[SystemState, List[bool], List[JumpClock], Time])


class ExactSimulator(BaseSimulator):
    """Jump-Switch-Flow simulator using exact method (no operator splitting)."""
    
    def simulate(self) -> Trajectory:
        """Run the exact simulation without operator splitting."""
        is_jumping = self._is_jumping(self.x0)
        jump_clocks = [self._new_jump_clock(is_j) for is_j in is_jumping]
        ext_state = ExtendedState((self.x0, is_jumping, jump_clocks, Time(0.0)))

        time = ext_state[3]
        times = [time]
        state_history = [ext_state[0]]

        while time < self.t_max:
            ext_state = self._update(ext_state, self.dt)
            time = ext_state[3]
            times.append(time)
            state_history.append(ext_state[0])

        compartment_histories = list(map(list, zip(*state_history)))
        return Trajectory((compartment_histories, times))

    def _is_jumping(self, x0: SystemState) -> List[bool]:
        """
        Determine which reactions are jumping.

        A reaction is jumping if at least one of the reactants or products is discrete.
        """
        is_discrete = self._is_discrete(x0)
        is_reactant = [[n != 0 for n in r] for r in self.nu_reactant]
        is_product = [[n != 0 for n in r] for r in self.nu_product]
        result = [False] * self.n_rates

        for rix in range(self.n_rates):
            if any([is_discrete[ix] for ix in range(self.n_compartments)
                    if is_reactant[rix][ix] or is_product[rix][ix]]):
                result[rix] = True

        return result

    def _new_jump_clock(self, is_jumping: bool) -> JumpClock:
        """
        Generate a new jump clock.

        If the system is not jumping then the jump clock is set to infinity.
        Otherwise, it takes a random value from the uniform distribution on [0, 1].
        """
        u = Time(random.uniform(0.0, 1.0))
        return JumpClock(
            clock=u if is_jumping else Time(float('inf')),
            start_time=u if is_jumping else Time(float('inf'))
        )

    def _update_state(self, state: SystemState, is_jumping: List[bool],
                     reaction_rates: List[float], delta_time: Time) -> SystemState:
        """Update the state of the system with a single forward-Euler step."""
        dx_dt = [
            sum(0 if is_jumping[rix] else reaction_rates[rix] * self.nu[rix][cix]
                for rix in range(len(reaction_rates)))
            for cix in range(len(state))
        ]
        next_state = [
            state[cix] + dx_dt[cix] * delta_time
            for cix in range(len(state))
        ]
        assert all([v >= 0 for v in next_state])
        return SystemState(next_state)

    def _update_jump_clocks(self, curr_time: Time, delta_time: Time,
                           curr_jump_clocks: List[JumpClock], curr_state: SystemState,
                           curr_reaction_rates: List[float], next_state: SystemState,
                           next_reaction_rates: List[float]) -> List[JumpClock]:
        """
        Update the jump clocks using the trapezoidal rule and the current
        and next system state to approximate the reaction rates.
        """
        next_jump_clocks = copy.deepcopy(curr_jump_clocks)
        for ix, jc in enumerate(next_jump_clocks):
            if math.isfinite(jc.clock):
                integral = 0.5 * delta_time * (next_reaction_rates[ix] + curr_reaction_rates[ix])
                jc.set_clock(jc.clock + (math.exp(-integral) - 1) * (jc.clock + 1 - jc.start_time))
                next_jump_clocks[ix] = jc
        return next_jump_clocks

    def _update(self, x0: ExtendedState, delta_time: Time) -> ExtendedState:
        """Update state of the (extended) system using forward-Euler."""
        # Establish current state
        curr_state = x0[0]
        curr_is_discrete = self._is_discrete(curr_state)
        curr_is_jumping = x0[1]
        curr_jump_clocks = x0[2]
        curr_time = x0[3]
        curr_r_rates = self.rates(curr_state, curr_time)

        # Propose new states
        next_time = Time(curr_time + delta_time)
        next_state = self._update_state(
            curr_state, curr_is_jumping, curr_r_rates, delta_time)
        next_r_rates = self.rates(next_state, next_time)
        next_is_discrete = self._is_discrete(next_state)
        next_is_jumping = self._is_jumping(next_state)
        next_jump_clocks = self._update_jump_clocks(
            curr_time, delta_time, curr_jump_clocks, curr_state,
            curr_r_rates, next_state, next_r_rates)

        # Exit early if no jumps or switches
        no_switches = all([cid == nid
                          for cid, nid in zip(curr_is_discrete, next_is_discrete)])
        no_jumps = all([jc.clock >= 0 for jc in next_jump_clocks])

        if no_jumps and no_switches:
            return ExtendedState(
                (next_state, next_is_jumping, next_jump_clocks, next_time)
            )

        # Handle first event (switch or jump)
        first_switch_ix, first_switch_time = (
            (-1, Time(float('inf'))) if no_switches else
            self._first_switch(curr_state, next_state, curr_time, next_time))
        first_jump_ix, first_jump_time = (
            (-1, Time(float('inf'))) if no_jumps else
            self._first_jump(curr_r_rates, next_r_rates, curr_time, next_time,
                           curr_jump_clocks, next_jump_clocks))

        next_time = Time(min(first_switch_time, first_jump_time))
        new_delta_time = Time(next_time - curr_time)
        next_state = self._update_state(
            curr_state, curr_is_jumping, curr_r_rates, new_delta_time)

        if next_time == first_switch_time:
            next_state[first_switch_ix] = self.switching_threshold[first_switch_ix]
        else:
            assert next_time == first_jump_time
            next_state = self._jump(next_state, first_jump_ix)

        next_is_jumping = self._is_jumping(next_state)
        next_jump_clocks = [self._new_jump_clock(is_j) for is_j in next_is_jumping]
        next_ext_state = ExtendedState(
            (next_state, next_is_jumping, next_jump_clocks, next_time)
        )
        remaining_delta_time = Time(delta_time - new_delta_time)
        return self._update(next_ext_state, remaining_delta_time)

    def _first_switch(self, x0: SystemState, x1: SystemState, t0: Time, t1: Time) -> Tuple[int, Time]:
        """Return the index and time of the first switch that occurred."""
        earliest_switch_time, earliest_switch_ix = float('inf'), int(10**4299)

        for ix, v0 in enumerate(x0):
            v1 = x1[ix]
            if v0 > self.switching_threshold[ix] and v1 <= self.switching_threshold[ix]:
                s_t = (self.switching_threshold[ix] * (t1 - t0) + v1 * t0 - v0 * t1) / (v1 - v0)
                assert t0 < s_t and s_t < t1
                assert s_t != earliest_switch_time
                if s_t < earliest_switch_time:
                    earliest_switch_time = s_t
                    earliest_switch_ix = ix

        assert earliest_switch_time < float('inf')
        return earliest_switch_ix, Time(earliest_switch_time)

    def _first_jump(self, r0s: List[float], r1s: List[float], t0: Time, t1: Time,
                   j0s: List[JumpClock], j1s: List[JumpClock]) -> Tuple[int, Time]:
        """Return the index and time of the first jump that occurred."""
        delta_t = t1 - t0
        min_h = float('inf')
        earliest_jump_time, earliest_jump_ix = float('inf'), int(10**4299)

        for ix, j0 in enumerate(j0s):
            if j0.clock > 0 and j1s[ix].clock <= 0:
                alpha = math.log((j0.clock + 1 - j0.start_time) / (1 - j0.start_time))
                delta_r = r1s[ix] - r0s[ix]
                if delta_r != 0:
                    h = (-delta_t * r0s[ix] + math.sqrt((delta_t * r0s[ix])**2 + 2 * alpha * delta_t * delta_r)) / delta_r
                else:
                    h = alpha / r0s[ix]

                assert 0 < h and h < delta_t
                if h < min_h:
                    min_h = h
                    earliest_jump_time = t0 + min_h
                    earliest_jump_ix = ix

        assert earliest_jump_time < float('inf')
        return earliest_jump_ix, Time(earliest_jump_time)

    def _jump(self, x0: SystemState, reaction_ix: int) -> SystemState:
        """
        Jump the system to a new state.

        If the jump results in a value below the switching threshold but not an integer,
        randomly round to preserve average behavior.
        """
        x1 = [x0[ix] + self.nu[reaction_ix][ix] for ix in range(len(x0))]

        # Randomly round values if necessary
        for ix, x in enumerate(x1):
            if (x < self.switching_threshold[ix]) and ((x - round(x)) > 1e-10):
                x_floor = math.floor(x)
                if random.uniform(0, 1) < (x - x_floor):
                    x1[ix] = x_floor + 1
                else:
                    x1[ix] = x_floor

        return SystemState(x1)


# Legacy function for backward compatibility
def JumpSwitchFlowExact(x0: SystemState, rates: Callable[[SystemState, Time], List[float]],
                       stoich: Dict[str, Any], t_max: Time, options: Dict[str, Any]) -> Trajectory:
    """Legacy wrapper for ExactSimulator."""
    simulator = ExactSimulator(x0, rates, stoich, t_max, options)
    return simulator.simulate()