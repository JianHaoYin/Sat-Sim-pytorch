import math

import numpy as np
from typing import Any, Callable
import pytest
import torch

from satsim.architecture import Timer
from satsim.fsw_algorithm.attGuidance.velocityPoint import (
    VelocityPoint,
)

import pytest

@pytest.mark.parametrize("celMsgSet", [True])

def test_velocityPoint(celMsgSet):
    r"""

    """
    state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N) = VelocityPointTestFunction(celMsgSet)

    REQ_EARTH = 6378.1366
    D2R = np.pi / 180.0
    MU_EARTH = 398600.436
    a = REQ_EARTH * 2.8 * 1000 # m
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    f = 60 * D2R
    mu = MU_EARTH*1e9
    f_dot = np.sqrt(mu/(a*a*a))


    expected_sigma_RN = torch.tensor([[[0., 0., 0.267949192431]]], dtype=torch.float32)
    expected_omega_RN_N = torch.tensor([[[0.0, 0.0, f_dot]]], dtype=torch.float32)
    expected_dot_omega_RN_N = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)

    assert torch.allclose(sigma_RN, expected_sigma_RN, atol=1e-12)
    assert torch.allclose(omega_RN_N, expected_omega_RN_N, atol=1e-12)
    assert torch.allclose(dot_omega_RN_N, expected_dot_omega_RN_N, atol=1e-12)




def VelocityPointTestFunction(celMsgSet):
    """Test method"""


    module = VelocityPoint()
    
    REQ_EARTH = 6378.1366
    D2R = np.pi / 180.0
    MU_EARTH = 398600.436
    a = REQ_EARTH * 2.8 * 1000 # m
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    f = 60 * D2R
    mu = MU_EARTH*1e9
    (r, v) = OE2RV(mu, a, e, i, Omega, omega, f)
    module.mu = torch.tensor([mu],dtype=torch.float32)

    r_BN_N = torch.tensor(r,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    v_BN_N = torch.tensor(v,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    planetPos = torch.tensor([[[0.0, 0.0, 0.0]]],dtype=torch.float32)
    planetVel = torch.tensor([[[0.0, 0.0, 0.0]]],dtype=torch.float32)


    state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N) = module.forward(
        r_BN_N=r_BN_N,
        v_BN_N=v_BN_N,
        r_celestialObjectN_N=planetPos,
        v_celestialObjectN_N=planetVel,
    )

    return state_dict, (sigma_RN, omega_RN_N, dot_omega_RN_N)


def OE2RV(mu, a, e, i, Omega, w, nu):
    """OE to (r,v) conversion"""
    if e != 1:
        p = a*(1-e*e)
    else:
        print('ERROR: parabolic case')
        return
    c = np.cos(nu)
    s = np.sin(nu)
    r_PQW = np.array([p*c / (1 + e*c), p*s / (1 + e*c), 0])
    v_PQW = np.array([-s * np.sqrt(mu/p), (e + c)*np.sqrt(mu/p), 0])
    return PQW2IJK(Omega, i, w, r_PQW, v_PQW)


def PQW2IJK(Omega, i, w, r_PQW, v_PQW):
    sO = np.sin(Omega)
    cO = np.cos(Omega)
    si = np.sin(i)
    ci = np.cos(i)
    sw = np.sin(w)
    cw = np.cos(w)

    C11 = cO*cw - sO*sw*ci
    C12 = -cO*sw - sO*cw*ci
    C13 = sO*si
    C21 = sO*cw + cO*sw*ci
    C22 = -sO*sw + cO*cw*ci
    C23 = -cO*si
    C31 = sw*si
    C32 = cw*si
    C33 = ci
    C = np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])

    r_IJK = C.dot(r_PQW)
    v_IJK = C.dot(v_PQW)
    return (r_IJK, v_IJK)

if __name__ == "__main__":
    test_velocityPoint(True)
    raise RuntimeError("This test does not support direct run")
