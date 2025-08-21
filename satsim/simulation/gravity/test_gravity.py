
import torch
import gravity_body
import numpy as np
import math
import time
import csv
from satsim.architecture.timer import Timer

def computeGravityTo20(positionVector):
    #This code follows the formulation in Vallado, page 521, second edition and uses data from UTexas CSR for
    #gravitation harmonics parameters
    #Written 201780807 by Scott Carnahan
    #AVS Lab | CU Boulder

    #INPUTS
    #positionVector - [x,y,z] coordinates list of spacecraft in [m] in earth body frame so that lat, long can be calculated

    def legendres(degree, alpha):
        P = np.zeros((degree+1,degree+1))
        P[0,0] = 1
        P[1,0] = alpha
        cosPhi = np.sqrt(1-alpha**2)
        P[1,1] = cosPhi

        for l in range(2,degree+1):
            for m in range(0,l+1):
                if m == 0 and l >= 2:
                    P[l,m] = ((2*l-1)*alpha*P[l-1,0]-(l-1)*P[l-2,0]) / l
                elif m != 0 and m < l:
                    P[l, m] = (P[l-2, m]+(2*l-1)*cosPhi*P[l-1,m-1])
                elif m == l and l != 0:
                    P[l,m] = (2*l-1)*cosPhi*P[l-1,m-1]
                else:
                    print(l,", ", m)
        return P

    maxDegree = 20
    cList = np.zeros(maxDegree+2)
    sList = np.zeros(maxDegree+2)
    muEarth = 0.
    radEarth = 0.
    [cList, sList, muEarth, radEarth]  = loadGravFromFileToList('/home/d632/YJH/Sat-Sim-pytorch/supportData/LocalGravData/GGM03S.txt', maxDegree+2)

    r = np.linalg.norm(positionVector)
    rHat = positionVector / r
    gHat = rHat
    grav0 = -gHat * muEarth / r**2

    rI = positionVector[0]
    rJ = positionVector[1]
    rK = positionVector[2]

    rIJ = np.sqrt(rI**2 + rJ**2)
    if rIJ != 0.:
        phi = math.atan(rK / rIJ) #latitude in radians
    else:
        phi = math.copysign(np.pi/2., rK)
    if rI != 0.:
        lambdaSat = math.atan(rJ / rI) #longitude in radians
    else:
        lambdaSat = math.copysign(np.pi/2., rJ)

    P = legendres(maxDegree+1,np.sin(phi))

    dUdr = 0.
    dUdphi = 0.
    dUdlambda = 0.

    for l in range(0, maxDegree+1):
        for m in range(0,l+1):
            if m == 0:
                k = 1
            else:
                k = 2
            num = math.factorial(l+m)
            den = math.factorial(l-m)*k*(2*l+1)
            PI = np.sqrt(float(num)/float(den))
            cList[l][m] = cList[l][m] / PI
            sList[l][m] = sList[l][m] / PI

    for l in range(2,maxDegree+1): #can only do for max degree minus 1
        for m in range(0,l+1):
            dUdr = dUdr + (((radEarth/r)**l)*(l+1)*P[l,m]) * (cList[l][m]*np.cos(m*lambdaSat)+sList[l][m]*np.sin(m*lambdaSat))
            dUdphi = dUdphi + (((radEarth/r)**l)*P[l,m+1] - m*np.tan(phi)*P[l,m]) * (cList[l][m]*np.cos(m*lambdaSat) + sList[l][m]*np.sin(m*lambdaSat))
            dUdlambda = dUdlambda + (((radEarth/r)**l)*m*P[l,m]) * (sList[l][m]*np.cos(m*lambdaSat) - cList[l][m]*np.sin(m*lambdaSat))

    dUdr = -muEarth * dUdr / r**2
    dUdphi = muEarth * dUdphi / r
    dUdlambda = muEarth * dUdlambda / r


    if rI != 0. and rJ != 0.:
        accelerationI = (dUdr/r - rK*dUdphi/(r**2)/((rI**2+rJ**2)**0.5))*rI - (dUdlambda/(rI**2+rJ**2))*rJ + grav0[0]
        accelerationJ = (dUdr/r - rK*dUdphi/(r**2)/((rI**2+rJ**2)**0.5))*rJ + (dUdlambda/(rI**2+rJ**2))*rI + grav0[1]
    else:
        accelerationI = dUdr/r + grav0[0]
        accelerationJ = dUdr/r + grav0[1]
    accelerationK = (dUdr/r)*rK + (((rI**2+rJ**2)**0.5)*dUdphi/(r**2)) + grav0[2]

    accelerationVector = [accelerationI, accelerationJ, accelerationK]

    return accelerationVector

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
relative_position = torch.tensor([[[15000., 10000.0, 6378.1363*1e3],[15000.0, 10000.0, 6378.1363*1e3]]],device=device, dtype=dtype)  # shape (1, 2, 3)

# 批量复制，生成 shape (16, 100, 3)
batch_size = 16
num_positions = 100

# 先 repeat 原始的 2 个位置到 100
relative_position_batch = relative_position.repeat(1, num_positions // 2, 1)  # (1, 100, 3)
# 再 repeat batch 维度
relative_position_batch = relative_position_batch.repeat(batch_size, 1, 1)  # (16, 100, 3)
relative_position = relative_position_batch

#gravity_file_path = '/home/d632/YJH/Sat-Sim-pytorch/supportData/LocalGravData/GGM03S-J2-only.txt'
gravity_file_path = '/home/d632/YJH/Sat-Sim-pytorch/supportData/LocalGravData/GGM03S.txt'

timer = Timer()
gravbody = gravity_body.SphericalHarmonicGravityBody.create_earth(timer=timer,gravity_file=gravity_file_path, max_degree=20)
gravCheck = computeGravityTo20([15000., 10000., 6378.1363E3])
gravOut = gravbody.compute_gravitational_acceleration(relative_position)


gravOutMag = torch.norm(gravOut, dim=-1)
gravOutMag_0 = gravOutMag[..., 0]
gravCheckMag = np.linalg.norm(gravCheck)

accuracy = 1e-12
relative = (gravCheckMag-gravOutMag_0)/gravCheckMag

print(f'relative:{relative}')


start = time.time()
for i in range(3600):
    gravOut = gravbody.compute_gravitational_acceleration(relative_position)
end = time.time()
print(f'time_spharm:{end - start}')

# start = time.time()
# for i in range(10000):
#     gravOut = gravbody.compute_gravitational_acceleration(relative_position)
# end = time.time()
# print(f'time10000:{end - start}')

gravbody_pointmass = gravity_body.PointMassGravityBody.create_earth(timer=timer)
start = time.time()
for i in range(3600):
    gravOut_pointmass = gravbody_pointmass.compute_gravitational_acceleration(relative_position)
#gravOut_pointmass = gravbody_pointmass.compute_gravitational_acceleration(relative_position)
end = time.time()
print(f'time_pointmass:{end - start}')


# from line_profiler import LineProfiler
# # 初始化分析器并指定函数
# lp = LineProfiler()
# lp_wrapper = lp(gravbody.compute_gravitational_acceleration(relative_position))
# lp_wrapper()  # 运行函数

# # 生成报告（或用命令行：kernprof -l -v my_script.py）
# lp.print_stats()