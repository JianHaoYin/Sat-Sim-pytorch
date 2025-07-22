__all__ = ["AttTrackingErrorStateDict", "AttTrackingError"]

from typing import TypedDict

import torch
from satsim.architecture import Module
#from satsim.utils import to_rotation_matrix

class AttTrackingErrorStateDict(TypedDict):
    pass


class AttTrackingError(Module[AttTrackingErrorStateDict]):

    def __init__(
            self,
            *args,
            sigma_R0R: torch.Tensor | None = None,
            **kwargs):
        
        super().__init__(*args, **kwargs)
        sigma_R0R = torch.tensor(
            [0.01, 0.05, -0.55], dtype=torch.float32
        ) if sigma_R0R is None else sigma_R0R

        self.register_buffer(
            "sigma_R0R",
            sigma_R0R,
            persistent=False,
        )


    def reset(self) -> AttTrackingErrorStateDict | None:
        pass

    def forward(
        self,
        state_dict: AttTrackingErrorStateDict | None,
        sigma_R0N: torch.Tensor,
        omega_RN_N: torch.Tensor,
        domega_RN_N: torch.Tensor,
        sigma_BN: torch.Tensor,
        omega_BN_B: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[
            AttTrackingErrorStateDict | None,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ]:


        sigma_RR0 = -self.sigma_R0R
        sigma_RN = _add_mrp(sigma_R0N, sigma_RR0)

        sigma_BR = _sub_mrp(sigma_BN, sigma_RN)

        dcm_BN = _mrp_to_dcm(sigma_BN)

        omega_RN_B = torch.matmul(dcm_BN, omega_RN_N.unsqueeze(-1)).squeeze(-1)
        domega_RN_B = torch.matmul(dcm_BN, domega_RN_N.unsqueeze(-1)).squeeze(-1)

        omega_BR_B = omega_BN_B - omega_RN_B

        return state_dict, (sigma_BR, omega_BR_B, omega_RN_B, domega_RN_B)
    

def _add_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1.clone()
    q2 = q2.clone()
    s1 = q1
    d1 = 1 + (s1 @ s1) * (q2 @ q2) - 2 * (s1 @ q2)
    if torch.abs(d1) < 0.1:
        mag = s1 @ s1
        s1 = -s1 / mag
        d1 = 1 + (s1 @ s1) * (q2 @ q2) - 2 * (s1 @ q2)

    cross = torch.cross(s1, q2)
    term1 = (1 - (q2 @ q2)) * s1
    term2 = 2 * cross
    term3 = (1 - (s1 @ s1)) * q2

    result = term1 + term2 + term3
    result = result / d1

    norm2 = result @ result
    result = torch.where(norm2 > 1.0, -result / norm2, result)

    return result


def _sub_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    s1 = q1.clone()
    d1 = 1 + (s1 @ s1) * (q2 @ q2) + 2 * (s1 @ q2)
    if torch.abs(d1) < 0.1:
        mag = s1 @ s1
        s1 = -s1 / mag
        d1 = 1 + (s1 @ s1) * (q2 @ q2) + 2 * (s1 @ q2)

    cross = torch.cross(s1, q2)
    term1 = (1 - (q2 @ q2)) * s1
    term2 = 2 * cross
    term3 = (1 - (s1 @ s1)) * q2

    result = term1 + term2 - term3
    result = result / d1

    norm2 = result @ result
    result = torch.where(norm2 > 1.0, -result / norm2, result)

    return result


def _mrp_to_dcm(sigma: torch.Tensor) -> torch.Tensor:
    q1, q2, q3 = sigma
    q2_sum = sigma @ sigma
    S = 1 - q2_sum
    d = (1 + q2_sum) ** 2

    c00 = 4 * (2 * q1 * q1 - q2_sum) + S * S
    c01 = 8 * q1 * q2 + 4 * q3 * S
    c02 = 8 * q1 * q3 - 4 * q2 * S
    c10 = 8 * q2 * q1 - 4 * q3 * S
    c11 = 4 * (2 * q2 * q2 - q2_sum) + S * S
    c12 = 8 * q2 * q3 + 4 * q1 * S
    c20 = 8 * q3 * q1 + 4 * q2 * S
    c21 = 8 * q3 * q2 - 4 * q1 * S
    c22 = 4 * (2 * q3 * q3 - q2_sum) + S * S

    C = torch.stack([
        torch.stack([c00, c01, c02]),
        torch.stack([c10, c11, c12]),
        torch.stack([c20, c21, c22])
    ])

    return C / d