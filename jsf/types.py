from typing import List, NewType, Tuple, Optional

CompartmentValue = NewType("CompartmentValue", float)
SystemState = NewType("SystemState", List[CompartmentValue])
Time = NewType("Time", float)
Trajectory = NewType("Trajectory", Tuple[List[List[CompartmentValue]], List[Time]])
TrajectoryWithThresholds = NewType("TrajectoryWithThresholds", Tuple[List[List[CompartmentValue]], List[Time], List[List[int]]])
