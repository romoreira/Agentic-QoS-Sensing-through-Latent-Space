"""Action appliers — translate action index into MBR enforcement.

MockActionApplier logs the would-be pfcpctl command and updates internal
state. RealActionApplier (stub) shows the shape of the real call against
the BESS-UPF and is left for the user to fill in once the testbed is wired.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from upf_rl_env.action.space import action_to_mbr_kbps

logger = logging.getLogger(__name__)


class ActionApplier(ABC):
    @abstractmethod
    def apply(self, action_idx: int) -> np.ndarray:
        """Apply action and return the resulting MBR vector (kbps, shape (3,))."""

    def reset(self) -> np.ndarray:
        """Reset to nominal MBR (action 0). Returns the MBR vector."""
        return self.apply(0)


class MockActionApplier(ActionApplier):
    """Pure in-memory applier. Updates internal MBR; no side effects."""

    def __init__(
        self,
        slice_teids: Sequence[int],
        nominal_mbr_kbps: Tuple[int, int, int],
    ) -> None:
        self.slice_teids = tuple(slice_teids)
        self.nominal_mbr_kbps = tuple(nominal_mbr_kbps)
        self.current_mbr_kbps = np.array(nominal_mbr_kbps, dtype=np.int64)
        self.last_action_idx: int = 0

    def apply(self, action_idx: int) -> np.ndarray:
        self.current_mbr_kbps = action_to_mbr_kbps(action_idx, self.nominal_mbr_kbps)
        self.last_action_idx = action_idx
        logger.debug(
            "MockApplier action=%d -> MBR(kbps) teid=%s mbr=%s",
            action_idx,
            self.slice_teids,
            self.current_mbr_kbps.tolist(),
        )
        return self.current_mbr_kbps.copy()


class RealActionApplier(ActionApplier):
    """Stub — replace `_pfcp_modify` with a real call when wiring the testbed.

    Expected mechanism (typical for aether-upf-bess / OMEC):
        pfcpctl service session modify --teid <TEID> --mbr-uplink <kbps> --mbr-downlink <kbps>

    Or via the pfcpiface gRPC if exposed. Either way the contract is:
    given (teid, mbr_kbps), enforce the new MBR on the session.
    """

    def __init__(
        self,
        slice_teids: Sequence[int],
        nominal_mbr_kbps: Tuple[int, int, int],
        pfcpctl_endpoint: str = "localhost:8080",
    ) -> None:
        self.slice_teids = tuple(slice_teids)
        self.nominal_mbr_kbps = tuple(nominal_mbr_kbps)
        self.endpoint = pfcpctl_endpoint
        self.current_mbr_kbps = np.array(nominal_mbr_kbps, dtype=np.int64)

    def _pfcp_modify(self, teid: int, mbr_kbps: int) -> None:
        # TODO: implement against your specific UPF-BESS deployment.
        # Examples:
        #   - subprocess.run(["pfcpctl", "service", "session", "modify",
        #                      "--teid", str(teid),
        #                      "--mbr-uplink", str(mbr_kbps),
        #                      "--mbr-downlink", str(mbr_kbps)], check=True)
        #   - or gRPC call to pfcpiface
        raise NotImplementedError(
            "Wire RealActionApplier._pfcp_modify to your UPF-BESS / pfcpctl. "
            "Use MockActionApplier until then."
        )

    def apply(self, action_idx: int) -> np.ndarray:
        new_mbr = action_to_mbr_kbps(action_idx, self.nominal_mbr_kbps)
        for teid, mbr in zip(self.slice_teids, new_mbr.tolist()):
            self._pfcp_modify(int(teid), int(mbr))
        self.current_mbr_kbps = new_mbr
        return new_mbr.copy()
