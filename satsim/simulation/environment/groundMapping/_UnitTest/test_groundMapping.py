import math

import numpy as np
from typing import Any, Callable
import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.environment.groundMapping import (
    GroundMapping,
    GroundMappingStateDict,
    GroundStateDict,
    AccessDict,
)

import pytest

@pytest.mark.parametrize("maxRange", [1e9, -1.0, 2.0])

def test_groundMapping(maxRange):
    r"""
    This test checks two points to determine if they are accessible for mapping or not. One point should be mapped,
    and one point should not be mapped.

    The inertial, planet-fixed planet-centered, and spacecraft body frames are all aligned.
    The spacecraft is in the -y direction of the inertial frame. The first point is along the line from the spacecraft
    to the origin. The second point is along the z-axis. The first point should be accessible because a.) the spacecraft
    is within the point's visibility cone and the point is within the spacecraft's visibility cone. The second point is
    not accessible because the spacecraft is not within the point's visibility cone and the point is not within the
    spacecraft's visibility cone.
    """
    state_dict, (accessDict,currentGroundState) = groundMappingTestFunction(maxRange)

    expected_hasaccess = torch.tensor([True, False])
    accessDict_hasaccess = torch.tensor(
        [accessDict[0]["hasAccess"],
         accessDict[1]["hasAccess"]
        ],dtype=torch.bool)
    assert torch.equal(accessDict_hasaccess, expected_hasaccess)
    return accessDict,currentGroundState



    #assert testResults < 1, testMessage


def groundMappingTestFunction(maxRange):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    # unitTestSim = SimulationBaseClass.SimBaseClass()
    # testProcessRate = macros.sec2nano(0.5)
    # testProc = unitTestSim.CreateNewProcess(unitProcessName)
    # testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Configure blank module input messages

    J20002Pfix = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    PositionVector = torch.tensor([0., 0., 0.])



    r_BN_N = torch.tensor([0., -1., 0.])
    sigma_BN = torch.tensor([0., 0., 0.])
    v_BN_N = torch.tensor([0., 0., 0.])

    # Create the initial imaging target
    groundMap = GroundMapping(maximumRange=torch.tensor(maxRange))

    groundMap.addPointToModel(torch.tensor([0., -0.1, 0.]))
    groundMap.addPointToModel(torch.tensor([0., 0., math.tan(np.radians(22.5))+0.1]))
    groundMap.minimumElevation = torch.tensor(np.radians(45.))

    groundMap.cameraPos_B = torch.tensor([0., 0., 0.])
    groundMap.nHat_B = torch.tensor([0., 1., 0.])
    groundMap.halfFieldOfView = torch.tensor(np.radians(22.5))


    # # Setup the logging for the mapping locations
    # mapLog = []
    # for idx in range(0, 2):
    #     mapLog.append(groundMap.accessOutMsgs[idx].recorder())
    #     unitTestSim.AddModelToTask(unitTaskName, mapLog[idx])

    # # subscribe input messages to module
    # groundMap.planetInMsg.subscribeTo(planetInMsg)
    # groundMap.scStateInMsg.subscribeTo(scStateInMsg)

    # # setup output message recorder objects
    # unitTestSim.InitializeSimulation()
    # unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))
    # unitTestSim.ExecuteSimulation()

    # # pull module data and make sure it is correct
    # map_access = np.zeros(2, dtype=bool)
    # for idx in range(0, 2):
    #     access = mapLog[idx].hasAccess
    #     if sum(access):
    #         map_access[idx] = 1

    # # If the first target is not mapped, failure
    # if not map_access[0] and (maxRange > 1.0 or maxRange < 0.0) :
    #     testFailCount += 1

    # # If the second target is mapped, failure
    # if map_access[1]:
    #     testFailCount += 1

    # if testFailCount == 0:
    #     print("PASSED: " + groundMap.ModelTag)
    # else:
    #     print(testMessages)

    # return [testFailCount, "".join(testMessages)]
    state_dict, (accessDict,currentGroundState) = groundMap.forward(
        None,
        J20002Pfix=J20002Pfix,
        PositionVector=PositionVector,
        r_BN_N=r_BN_N,
        v_BN_N=v_BN_N,
        sigma_BN=sigma_BN
    )

    return state_dict, (accessDict,currentGroundState)


if __name__ == "__main__":
    accessDict,currentGroundState = test_groundMapping(1e9)
    #accessDict,currentGroundState = test_groundMapping(0.001)
    #accessDict,currentGroundState = test_groundMapping(1e-12)
    #accessDict,currentGroundState = test_groundMapping(2.0)
    #accessDict,currentGroundState = test_groundMapping(-1.0)
    #raise RuntimeError("This test does not support direct run")
