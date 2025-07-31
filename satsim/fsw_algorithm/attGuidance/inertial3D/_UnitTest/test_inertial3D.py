import math

import numpy as np
from typing import Any, Callable
import pytest
import torch

from satsim.architecture import Timer
from satsim.fsw_algorithm.attGuidance.inertial3D import (
    Inertial3D,
)

import pytest

@pytest.mark.parametrize("celMsgSet", [True])

def test_inertial3D(celMsgSet):
    r"""

    """
    state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N) = Inertial3DTestFunction(celMsgSet)

    expected_sigma_RN = torch.tensor([[[0.1, 0.2, 0.3]]],dtype=torch.float32)
    expected_omega_RN_N = torch.tensor([[[0.0, 0.0, 0.0]]],dtype=torch.float32)
    expected_dot_omega_RN_N = torch.tensor([[[0.0, 0.0, 0.0]]],dtype=torch.float32)

    assert torch.allclose(sigma_RN, expected_sigma_RN, atol=1e-12)
    assert torch.allclose(omega_RN_N, expected_omega_RN_N, atol=1e-12)
    assert torch.allclose(dot_omega_RN_N, expected_dot_omega_RN_N, atol=1e-12)




def Inertial3DTestFunction(celMsgSet):
    """Test method"""


    module = Inertial3D()
    module.sigma_R0N = torch.tensor([[0.1, 0.2, 0.3]],dtype=torch.float32)
    state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N) = module.forward(
        None,
    )

    return state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N)



if __name__ == "__main__":
    test_inertial3D(True)
    raise RuntimeError("This test does not support direct run")
