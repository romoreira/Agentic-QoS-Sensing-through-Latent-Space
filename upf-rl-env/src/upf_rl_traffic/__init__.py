"""upf_rl_traffic — shared traffic dynamics + TRex driver.

Public API:
    - SliceTrafficGenerator, TrafficDynamicsConfig, TrafficSample
    - SLICE_DEFS, SliceDef
    - TrexConductor, ConductorConfig
"""
from upf_rl_traffic.dynamics import (
    SliceTrafficGenerator,
    TrafficDynamicsConfig,
    TrafficSample,
)
from upf_rl_traffic.trex_profile import SLICE_DEFS, SliceDef

__all__ = [
    "SliceTrafficGenerator",
    "TrafficDynamicsConfig",
    "TrafficSample",
    "SLICE_DEFS",
    "SliceDef",
]

# TrexConductor is exported but imported lazily — it depends on trex_stl_lib
# which is not available in plain Python environments.
