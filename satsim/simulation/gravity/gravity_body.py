__all__ = ['GravityBody', 'PointMassGravityBody','SphericalHarmonicGravityBody']

from abc import ABC, abstractmethod
import csv
import math
from typing import Never,Self
from satsim.architecture.timer import Timer
import torch
from torch import Tensor

from satsim.architecture import Module, VoidStateDict, constants


class GravityBody(Module[VoidStateDict], ABC):

    def __init__(
        self,
        *args,
        name: str,
        gm: float,
        equatorial_radius: float,
        polar_radius: float | None = None,
        is_central: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._name = name
        if polar_radius is None:
            polar_radius = equatorial_radius

        self._gm = gm
        self._equatorial_radius = equatorial_radius
        self._polar_radius = polar_radius
        self._is_central = is_central

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_central(self) -> bool:
        return self._is_central


    def set_central(self):
        self._is_central = True

    def forward(
        self,
        state_dict: VoidStateDict,
        *args,
        **kwargs,
    ) -> Never:
        raise NotImplementedError

    @abstractmethod
    def compute_gravitational_acceleration(
        self,
        relative_position: Tensor,
    ) -> Tensor:
        """
        Computes the gravitational field for a set of point masses at specified positions.
        
        Args:
            position (Tensor): Position tensor with shape [batch_size, num_positions, 3],
                representing the 3D position vectors of points relative to each planet.
        
        Returns:
            Tensor: Gravitational field tensor with shape [batch_size, num_positions, 3],
                representing the total gravitational field at each position due to all planets.
        """
        pass

    @classmethod
    def create_sun(cls, timer, **kwargs) -> Self:
        return cls(
            timer = timer,
            name='SUN',
            gm=constants.MU_SUN * 1e9,
            equatorial_radius=constants.REQ_SUN,
            **kwargs,
        )

    @classmethod
    def create_earth(cls, timer, **kwargs) -> Self:
        return cls(
            timer = timer,
            name='EARTH',
            gm=constants.MU_EARTH * 1e9,
            equatorial_radius=constants.REQ_EARTH,
            **kwargs,
        )


class PointMassGravityBody(GravityBody):

    def compute_gravitational_acceleration(
        self,
        relative_position: Tensor,
    ) -> Tensor:
        """
        Computes the gravitational field for a set of point masses at specified positions.

        Args:
            relative_position (Tensor): Position tensor with shape [
            num_positions, 3],
            representing the 3D position vectors of points relative to each planet.

        Returns:
            Tensor: Gravitational field tensor with shape [num_positions, 3],
                representing the total gravitational field at each position due to all planets.

        Notes:
            - The gravitational field is computed using the inverse-square law, where the force
            magnitude is proportional to -mu / r^3, and r is the distance to each planet.
            - The field contributions from all planets are summed along the planet dimension.
        """

        r = torch.norm(relative_position, dim=-1, keepdim=True)

        force_magnitude = -self._gm / (r**3)  # [b, num_position, 1]
        grav_field = force_magnitude * relative_position
        if grav_field.dim() > 1:
            grav_field = grav_field.sum(-2)
        return grav_field


class SphericalHarmonicGravityBody(GravityBody):
    
    def __init__(
        self,
        *args,
        gravity_file = '/supportData/LocalGravData/GGM03S-J2-only.txt',
        max_degree = 2,
        **kwargs,

    ) -> None:
        
        super().__init__(*args, **kwargs)

        self.loadGravFromFile(gravity_file, maxDeg=max_degree)
        self.initializeParameters()


    def loadGravFromFile(
            self,
            fileName: str, 
            maxDeg: int = 2
        ):

        [clmList, slmList, mu, radEquator] = loadGravFromFileToList(fileName, maxDeg=2)
        
        self.muBody = mu
        self.radEquator = radEquator

        clmList = clmList[:maxDeg + 1]
        slmList = slmList[:maxDeg + 1]
        #Convert a lower triangular matrix into a full square matrix
        self.cBar = lower_to_full_square(clmList)
        self.sBar = lower_to_full_square(slmList)
        self.maxDeg = maxDeg

        #load check
        self._loaded = True

    def initializeParameters(self):
        # initialize the parameters
        n1 = []
        n2 = []
        aBar = []
        nQuot1 = []
        nQuot2 = []

        #init the basic parameters
        # calculate aBar / n1 / n2
        for i in range(self.maxDeg + 2):
            aRow = [0.0] * (self.maxDeg + 2)
            if i == 0:
                aRow[0] = 1.0
            else:
                aRow[i] = math.sqrt(((2 * i + 1) * getK(i)) / (2 * i * getK(i - 1))) * aBar[i - 1][i - 1]
            
            n1Row = [0.0] * (self.maxDeg + 2)
            n2Row = [0.0] * (self.maxDeg + 2)
            for m in range(i + 1):
                if i >= m + 2:
                    n1Row[m] = math.sqrt(((2 * i + 1) * (2 * i - 1)) / ((i - m) * (i + m)))
                    n2Row[m] = math.sqrt(((i + m - 1) * (2 * i + 1) * (i - m - 1)) /
                                        ((i + m) * (i - m) * (2 * i - 3)))
            aBar.append(aRow)
            n1.append(n1Row)
            n2.append(n2Row)

        # init nQuot1 / nQuot2
        for l in range(self.maxDeg + 1):
            nq1Row = [0.0] * (self.maxDeg + 1)
            nq2Row = [0.0] * (self.maxDeg + 1)
            for m in range(l + 1):
                if m < l:
                    nq1Row[m] = math.sqrt(((l - m) * getK(m) * (l + m + 1)) / getK(m + 1))
                nq2Row[m] = math.sqrt(((l + m + 2) * (l + m + 1) * (2 * l + 1) * getK(m)) /
                                    ((2 * l + 3) * getK(m + 1)))
            nQuot1.append(nq1Row)
            nQuot2.append(nq2Row)

        self.aBar = aBar
        self.n1 = n1
        self.n2 = n2
        self.nQuot1 = nQuot1
        self.nQuot2 = nQuot2

    #@line_profiler.profile
    def compute_gravitational_acceleration(
            self, 
            relative_position: Tensor,
    ) -> Tensor:
        """
        Computes the gravitational acceleration for a celestial body using a spherical
        harmonics gravity model.

        Args:
            relative_position (Tensor): Position tensor with shape [num_positions, 3],
                representing the 3D position vectors of points relative to the body's
                reference frame (usually body-fixed or inertial frame).

        Returns:
            Tensor: Gravitational acceleration tensor with shape [num_positions, 3],
                representing the total gravitational acceleration at each position
                computed from the spherical harmonics expansion.

        Notes:
            - The gravitational potential is represented as a series expansion in
            spherical harmonics, using normalized coefficients C̄ₗₘ and S̄ₗₘ up to
            a specified maximum degree and order.
            - The acceleration is computed by taking the gradient of the potential,
            which includes central term and higher-order perturbation terms capturing
            the body's non-spherical mass distribution.
            - Coefficients are typically loaded from a gravity model file (e.g., EGM2008)
            and expressed in a body-fixed reference frame.
            - Compared to the point-mass model, this method captures effects such as
            oblateness (J₂ term), tesseral, and sectoral harmonics, improving accuracy
            for near-body orbital dynamics.
        """

        #TODO:check the datastructure of cBar/sBar when its void
        if not hasattr(self, "_loaded") or not self._loaded:
            raise RuntimeError("please load by method:loadGravFromFile")
        
        #shape_prefix = relative_position.shape[:-1]
        device = relative_position.device
        dtype = relative_position.dtype

        #compute gravity field
        degree = self.maxDeg
        include_zero_degree = True
        order = degree

        x = relative_position[..., 0].unsqueeze(-1)
        y = relative_position[..., 1].unsqueeze(-1)
        z = relative_position[..., 2].unsqueeze(-1)

        r = torch.norm(relative_position, dim=-1, keepdim=True)
        s = x / r #shape (...,1)
        t = y / r #shape (...,1)
        u = z / r #shape (...,1)

        aBar_t = expand_matrix(self.aBar, relative_position, dtype=dtype, device=device)
        n1_t   = expand_matrix(self.n1, relative_position, dtype=dtype, device=device)
        n2_t   = expand_matrix(self.n2, relative_position, dtype=dtype, device=device)
        
        ### Original implementation of initializing aBar_t ###
        # # Calcute the coefficients aBar_t
        # for l in range(1, degree+2):
        #     aBar_t[..., l, l-1] = (math.sqrt((2*l*getK(l-1))/getK(l)) * aBar_t[..., l, l].unsqueeze(-1) * u).squeeze(-1)  # shape (..., degree+2, degree+2)

        # for m in range(order + 2):
        #     for l in range(m + 2, degree + 2):

        #         aBar_t[..., l, m] = (u * n1_t[..., l, m].unsqueeze(-1) * aBar_t[..., l-1, m].unsqueeze(-1) - n2_t[..., l, m].unsqueeze(-1) * aBar_t[..., l-2, m].unsqueeze(-1)).squeeze(-1)  # shape (..., degree+2, degree+2)


        # # Calcute the coefficients aBar_t
        #  (l, l-1) terms
        l_idx = torch.arange(1, degree + 2, device=aBar_t.device)                # [1,2,...,degree+1]
        # coef(l) = 1 (l==1) else sqrt(2*l)
        coef = torch.where(
            l_idx == 1,
            torch.ones_like(l_idx, dtype=u.dtype),
            torch.sqrt(2.0 * l_idx.to(u.dtype))
        )
        # aBar_t[..., l, l-1] = coef * aBar_t[..., l, l] * u
        aBar_t[..., l_idx, l_idx - 1] = coef * aBar_t[..., l_idx, l_idx] * u


        m_all = torch.arange(order + 2, device=aBar_t.device)

        for l in range(2, degree + 2):                     # need l-2，start from 2
            # m: 0..(l-2)
            limit = min(l - 1, order + 2)                  # l-2 是上界（含），长度为 l-1
            if limit <= 0:
                continue
            vm = m_all[:limit]                             # shape (<=order+2,)

            # a_{l,m} = u * n1_{l,m} * a_{l-1,m} - n2_{l,m} * a_{l-2,m}
            aBar_t[..., l, vm] = (
                u * n1_t[..., l, vm] * aBar_t[..., l - 1, vm]
                -     n2_t[..., l, vm] * aBar_t[..., l - 2, vm]
            )


        ### Original implementation of initializing rE and iM ###

            # if m == 0:
            #     rE[..., 0] = 1.0
            #     iM[..., 0] = 0.0
            # else:

            #     rE[..., m] = (s * rE[..., m - 1].unsqueeze(-1) - t * iM[..., m - 1].unsqueeze(-1)).squeeze(-1)
            #     iM[..., m] = (s * iM[..., m - 1].unsqueeze(-1) + t * rE[..., m - 1].unsqueeze(-1)).squeeze(-1)

        # Generate real / imaginary parts (rE, iM) of (2+j*t)^m
        # shape rE and iM are (..., 1)
        rE = torch.zeros(relative_position.shape[:-1] + (order+2,),
                        device=relative_position.device,
                        dtype=relative_position.dtype)

        iM = torch.zeros_like(rE)

        z_st = s + 1j * t
        m_idx = torch.arange(order+2, device=z_st.device)
        z_pow = z_st ** m_idx                     # (..., order+2)
        rE = z_pow.real                           # (..., order+2)
        iM = z_pow.imag                           # (..., order+2)

        # Compute the radial coefficient rhol
        rho = self.radEquator / r  # shape (..., 1)
        rhol_0 = self.muBody / r   # shape (..., 1)

        ### Original implementation of initializing rhol ###
        # rhol = torch.zeros(relative_position.shape[:-1] + (degree+2,),
        #                 device=relative_position.device,
        #                 dtype=relative_position.dtype)
        
        # rhol[..., 0] = rhol_0.squeeze(-1)
        # rhol[..., 1] = (rhol_0 * rho).squeeze(-1)

        powers = torch.arange(degree+2, device=rho.device, dtype=rho.dtype)
        rhol = rhol_0 * rho.pow(powers)  # shape (..., degree+2)

        #Gravity components
        a1 = torch.zeros_like(r) #shape (...,1)
        a2 = torch.zeros_like(r)
        a3 = torch.zeros_like(r)
        a4 = torch.zeros_like(r)
        if include_zero_degree:
            a4[..., 0] = -rhol[..., 1] / self.radEquator


        sum_a1 = torch.zeros_like(r) #shape (...,1)
        sum_a2 = torch.zeros_like(r)
        sum_a3 = torch.zeros_like(r)
        sum_a4 = torch.zeros_like(r)

        cBar_t = expand_matrix(self.cBar, relative_position, dtype=dtype, device=device)
        sBar_t = expand_matrix(self.sBar, relative_position, dtype=dtype, device=device)
        nQuot1_t = expand_matrix(self.nQuot1, relative_position, dtype=dtype, device=device)
        nQuot2_t = expand_matrix(self.nQuot2, relative_position, dtype=dtype, device=device)



        for l in range(1, degree + 1):
            #rhol[..., l + 1] = (rho * rhol[..., l].unsqueeze(-1)).squeeze(-1)  # shape (..., degree+2)

            # for m in range(l + 1):
                # D = cBar_t[..., l, m] * rE[..., m] + sBar_t[...,l,m] * iM[...,m]
                # #D = D.unsqueeze(-1)  # shape (..., 1)
                # if m == 0:
                #     E = torch.zeros_like(r)  # shape (...,1)
                #     F = torch.zeros_like(r)  # shape (...,1)
                # else:
                #     E = cBar_t[..., l, m] * rE[..., m - 1] + sBar_t[...,l,m] * iM[..., m - 1]
                #     E = E.unsqueeze(-1)  # shape (..., 1)
                #     F = sBar_t[..., l, m] * rE[..., m - 1] - cBar_t[..., l, m] * iM[..., m - 1]
                #     F = F.unsqueeze(-1)  # shape (..., 1)

                # sum_a1 += m * aBar_t[..., l, m].unsqueeze(-1) * E
                # sum_a2 += m * aBar_t[..., l, m].unsqueeze(-1) * F
                # if m < l:
                #     sum_a3 += nQuot1_t[..., l, m].unsqueeze(-1) * aBar_t[..., l, m + 1].unsqueeze(-1) * D.unsqueeze(-1) #shape(...,)
                # sum_a4 += nQuot2_t[..., l, m].unsqueeze(-1) * aBar_t[..., l + 1, m + 1].unsqueeze(-1) * D.unsqueeze(-1)

            # a1 += rhol[..., l + 1].unsqueeze(-1) / self.radEquator * sum_a1
            # a2 += rhol[..., l + 1].unsqueeze(-1) / self.radEquator * sum_a2
            # a3 += rhol[..., l + 1].unsqueeze(-1) / self.radEquator * sum_a3
            # a4 -= rhol[..., l + 1].unsqueeze(-1) / self.radEquator * sum_a4

            M = l + 1

            cL = cBar_t[..., l, :M]        # (..., M)
            sL = sBar_t[..., l, :M]        # (..., M)
            rE_m = rE[..., :M]             # (..., M)
            iM_m = iM[..., :M]             # (..., M)

            D = cL * rE_m + sL * iM_m      # (..., M)

            rE_prev = torch.cat([torch.zeros_like(rE_m[..., :1]), rE_m[..., :-1]], dim=-1)  # (..., M)
            iM_prev = torch.cat([torch.zeros_like(iM_m[..., :1]), iM_m[..., :-1]], dim=-1)  # (..., M)

            E = cL * rE_prev + sL * iM_prev  # (..., M)
            F = sL * rE_prev - cL * iM_prev  # (..., M)

            m_idx = torch.arange(M, device=relative_position.device, dtype=relative_position.dtype)  # (M,)

            aL          = aBar_t[..., l, :M]       # (..., M)
            aL_mplus1   = aBar_t[..., l, 1:M]      # (..., M-1)
            aLp1_mplus1 = aBar_t[..., l+1, 1:M+1]  # (..., M)

            nQ1 = nQuot1_t[..., l, :M-1]           # (..., M-1)
            nQ2 = nQuot2_t[..., l, :M]             # (..., M)

            sum_a1 = (m_idx * aL * E).sum(dim=-1, keepdim=True)                 # (...,1)
            sum_a2 = (m_idx * aL * F).sum(dim=-1, keepdim=True)                 # (...,1)
            sum_a3 = (nQ1 * aL_mplus1 * D[..., :M-1]).sum(dim=-1, keepdim=True) # (...,1)
            sum_a4 = (nQ2 * aLp1_mplus1 * D).sum(dim=-1, keepdim=True)          # (...,1)

            coeff = rhol[..., l + 1].unsqueeze(-1) / self.radEquator
            a1 += coeff * sum_a1
            a2 += coeff * sum_a2
            a3 += coeff * sum_a3
            a4 -= coeff * sum_a4

        
        ax = a1 + s * a4
        ay = a2 + t * a4
        az = a3 + u * a4

        ax = ax.squeeze(-1)  # shape (...,)
        ay = ay.squeeze(-1)  # shape (...,)
        az = az.squeeze(-1)  # shape (...,)

        acceleration = torch.stack([ax, ay, az], dim=-1)

        return acceleration
    
    @classmethod
    def create_sun(cls,
                timer : Timer,
                gravity_file : str = '/supportData/LocalGravData/GGM03S-J2-only.txt',
                max_degree : int = 2, 
                **kwargs) -> Self:
        return cls(
            name='SUN',
            gm=constants.MU_SUN * 1e9,
            equatorial_radius=constants.REQ_SUN,
            gravity_file=gravity_file,
            max_degree=max_degree,
            timer=timer,
            **kwargs,
        )

    @classmethod
    def create_earth(cls,
                timer : Timer,
                gravity_file : str = '/supportData/LocalGravData/GGM03S-J2-only.txt',
                max_degree : int = 2, 
                **kwargs) -> Self:
        return cls(
            name='EARTH',
            gm=constants.MU_EARTH * 1e9,
            equatorial_radius=constants.REQ_EARTH,
            gravity_file=gravity_file,
            max_degree=max_degree,
            timer=timer,
            **kwargs,
        )

def loadGravFromFileToList(fileName: str, maxDeg: int = 2):
    with open(fileName, 'r') as csvfile:
        gravReader = csv.reader(csvfile, delimiter=',')
        firstRow = next(gravReader)
        clmList = []
        slmList = []

        try:
            radEquator = float(firstRow[0])
            mu = float(firstRow[1])
            # firstRow[2] is uncertainty in mu, not needed for Basilisk
            maxDegreeFile = int(firstRow[3])
            maxOrderFile = int(firstRow[4])
            coefficientsNormalized = int(firstRow[5]) == 1
            refLong = float(firstRow[6])
            refLat = float(firstRow[7])
        except Exception as ex:
            raise ValueError("File is not in the expected JPL format for "
                             "spherical Harmonics", ex)

        if maxDegreeFile < maxDeg or maxOrderFile < maxDeg:
            raise ValueError(f"Requested using Spherical Harmonics of degree {maxDeg}"
                             f", but file '{fileName}' has maximum degree/order of"
                             f"{min(maxDegreeFile, maxOrderFile)}")
        
        if not coefficientsNormalized:
            raise ValueError("Coefficients in given file are not normalized. This is "
                            "not currently supported in Basilisk.")

        if refLong != 0 or refLat != 0:
            raise ValueError("Coefficients in given file use a reference longitude"
                             " or latitude that is not zero. This is not currently "
                             "supported in Basilisk.")

        clmRow = []
        slmRow = []
        currDeg = 0
        for gravRow in gravReader:
            while int(gravRow[0]) > currDeg:
                if (len(clmRow) < currDeg + 1):
                    clmRow.extend([0.0] * (currDeg + 1 - len(clmRow)))
                    slmRow.extend([0.0] * (currDeg + 1 - len(slmRow)))
                clmList.append(clmRow)
                slmList.append(slmRow)
                clmRow = []
                slmRow = []
                currDeg += 1
            clmRow.append(float(gravRow[2]))
            slmRow.append(float(gravRow[3]))

        return [clmList, slmList, mu, radEquator]
    
def getK(degree: int) -> float:
    return 1.0 if degree == 0 else 2.0

def expand_matrix(matrix_list, ref_tensor, dtype=None, device=None):
    batch, num = ref_tensor.shape[0], ref_tensor.shape[1]  # get batch,num from ref_tensor
    mat = torch.tensor(matrix_list, dtype=dtype, device=device)  # (n, n)
    # expand to (batch,num,n,n)
    return mat.unsqueeze(0).unsqueeze(0).expand(batch, num, -1, -1).clone()

def lower_to_full_square(lower_list):
    """
    input: lower_list
    output: square_list
    """
    n = len(lower_list)
    full_matrix = []

    for i, row in enumerate(lower_list):
        
        new_row = row + [0.0] * (n - len(row))
        full_matrix.append(new_row)

    return full_matrix