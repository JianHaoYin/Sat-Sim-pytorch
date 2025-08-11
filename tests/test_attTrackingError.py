# test_attTrackingError.py
from typing import Any, Callable
import pytest
import torch

from satsim.architecture import Timer
from satsim.fsw_algorithm.attGuidance.attTrackingError import (
    AttTrackingError,
    AttTrackingErrorStateDict,
)


@pytest.mark.parametrize("sigma_R0N", [torch.tensor([0.35, -0.25, 0.15])])
@pytest.mark.parametrize("omega_RN_N", [torch.tensor([0.018, -0.032, 0.015])])
@pytest.mark.parametrize("domega_RN_N", [torch.tensor([0.048, -0.022, 0.025])])
@pytest.mark.parametrize("sigma_BN", [torch.tensor([0.25, -0.45, 0.75])])
@pytest.mark.parametrize("omega_BN_B", [torch.tensor([-0.015, -0.012, 0.005])])
def test_unit_attTrackingError(sigma_R0N : torch.Tensor,
                               omega_RN_N : torch.Tensor,
                               domega_RN_N : torch.Tensor,
                               sigma_BN : torch.Tensor,
                               omega_BN_B : torch.Tensor) -> None:

        sigma_R0R_test = torch.tensor([0.01, 0.05, -0.55])
        attTrackingError = AttTrackingError(sigma_R0R=sigma_R0R_test)


        # initialization


        # expected values
        expected_sigma_BR = torch.tensor([0.1836841481753408, -0.0974447769418166, -0.09896069560518146])
        expected_omega_BR_B = torch.tensor([-0.01181207648013235, -0.008916032420030655, -0.0344122606253076])
        expected_omega_RN_B = torch.tensor([-0.003187923519867655, -0.003083967579969345, 0.0394122606253076])
        expected_domega_RN_B = torch.tensor([-0.02388623421245188, -0.02835600277714878, 0.04514847640452802])

            

        state_dict, (sigma_BR, omega_BR_B, omega_RN_B, domega_RN_B) = attTrackingError.forward(
            None,
            sigma_R0N,
            omega_RN_N,
            domega_RN_N,
            sigma_BN,
            omega_BN_B
        )

        assert torch.allclose(sigma_BR, expected_sigma_BR, atol=1e-12)
        assert torch.allclose(omega_BR_B, expected_omega_BR_B, atol=1e-12)
        assert torch.allclose(omega_RN_B, expected_omega_RN_B, atol=1e-12)
        assert torch.allclose(domega_RN_B, expected_domega_RN_B, atol=1e-12)



if __name__ == "__main__":
    raise RuntimeError("This test does not support direct run")

    