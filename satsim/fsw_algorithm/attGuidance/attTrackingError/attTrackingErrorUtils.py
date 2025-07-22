import torch


def add_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def sub_mrp(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def mrp_to_dcm(sigma: torch.Tensor) -> torch.Tensor:
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