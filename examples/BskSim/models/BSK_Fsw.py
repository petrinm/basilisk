#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

import math

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (hillPoint, inertial3D, attTrackingError, mrpFeedback,
                                    rwMotorTorque,
                                    velocityPoint, mrpSteering, rateServoFullNonlinear,
                                    sunSafePoint, cssWlsEst, lambertPlanner, lambertSolver, lambertValidator,
                                    lambertSurfaceRelativeVelocity, lambertSecondDV,
                                    dvGuidance, thrForceMapping, thrFiringRemainder,
                                    attRefCorrection, dvExecuteGuidance)
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import (fswSetupRW, fswSetupThrusters)
from Basilisk.utilities import macros as mc


class BSKFswModels:
    """Defines the bskSim FSW class"""
    def __init__(self, SimBase, fswRate):
        # define empty class variables
        self.vcMsg = None
        self.fswRwConfigMsg = None
        self.cmdTorqueMsg = None
        self.cmdTorqueDirectMsg = None
        self.attRefMsg = None
        self.attGuidMsg = None
        self.cmdRwMotorMsg = None
        self.dvBurnCmdMsg = None
        self.acsOnTimeCmdMsg = None
        self.dvOnTimeCmdMsg = None

        # by default, use ACS thrusters if attitude is controlled with thrusters
        self.useDvThrusters = False

        # Define process name and default time-step for all FSW tasks defined later on
        self.processName = SimBase.FSWProcessName
        self.processTasksTimeStep = mc.sec2nano(fswRate)

        # Create module data and module wraps
        self.inertial3DData = inertial3D.inertial3DConfig()
        self.inertial3DWrap = SimBase.setModelDataWrap(self.inertial3DData)
        self.inertial3DWrap.ModelTag = "inertial3D"

        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = SimBase.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"

        self.sunSafePointData = sunSafePoint.sunSafePointConfig()
        self.sunSafePointWrap = SimBase.setModelDataWrap(self.sunSafePointData)
        self.sunSafePointWrap.ModelTag = "sunSafePoint"

        self.velocityPointData = velocityPoint.velocityPointConfig()
        self.velocityPointWrap = SimBase.setModelDataWrap(self.velocityPointData)
        self.velocityPointWrap.ModelTag = "velocityPoint"

        self.dvPointData = dvGuidance.dvGuidanceConfig()
        self.dvPointWrap = SimBase.setModelDataWrap(self.dvPointData)
        self.dvPointWrap.ModelTag = "dvPoint"

        self.cssWlsEstData = cssWlsEst.CSSWLSConfig()
        self.cssWlsEstWrap = SimBase.setModelDataWrap(self.cssWlsEstData)
        self.cssWlsEstWrap.ModelTag = "cssWlsEst"

        self.attRefCorrectionData = attRefCorrection.attRefCorrectionConfig()
        self.attRefCorrectionWrap = SimBase.setModelDataWrap(self.attRefCorrectionData)
        self.attRefCorrectionWrap.ModelTag = "attRefCorrection"

        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = SimBase.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"

        self.mrpFeedbackControlData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackControlWrap = SimBase.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"

        self.mrpFeedbackRWsData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackRWsWrap = SimBase.setModelDataWrap(self.mrpFeedbackRWsData)
        self.mrpFeedbackRWsWrap.ModelTag = "mrpFeedbackRWs"

        self.mrpFeedbackTHsData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackTHsWrap = SimBase.setModelDataWrap(self.mrpFeedbackTHsData)
        self.mrpFeedbackTHsWrap.ModelTag = "mrpFeedbackTHs"

        self.mrpSteeringData = mrpSteering.mrpSteeringConfig()
        self.mrpSteeringWrap = SimBase.setModelDataWrap(self.mrpSteeringData)
        self.mrpSteeringWrap.ModelTag = "MRP_Steering"

        self.rateServoData = rateServoFullNonlinear.rateServoFullNonlinearConfig()
        self.rateServoWrap = SimBase.setModelDataWrap(self.rateServoData)
        self.rateServoWrap.ModelTag = "rate_servo"

        self.rwMotorTorqueData = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = SimBase.setModelDataWrap(self.rwMotorTorqueData)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"

        self.thrForceMappingData = thrForceMapping.thrForceMappingConfig()
        self.thrForceMappingWrap = SimBase.setModelDataWrap(self.thrForceMappingData)
        self.thrForceMappingWrap.ModelTag = "thrForceMapping"

        self.thrFiringRemainderData = thrFiringRemainder.thrFiringRemainderConfig()
        self.thrFiringRemainderWrap = SimBase.setModelDataWrap(self.thrFiringRemainderData)
        self.thrFiringRemainderWrap.ModelTag = "thrFiringRemainder"

        self.dvManeuverData = dvExecuteGuidance.dvExecuteGuidanceConfig()
        self.dvManeuverWrap = SimBase.setModelDataWrap(self.dvManeuverData)
        self.dvManeuverWrap.ModelTag = "dvManeuver"

        self.lambertPlannerObject = lambertPlanner.LambertPlanner()
        self.lambertPlannerObject.ModelTag = "LambertPlanner"

        self.lambertSolverObject = lambertSolver.LambertSolver()
        self.lambertSolverObject.ModelTag = "LambertSolver"

        self.lambertValidatorObject = lambertValidator.LambertValidator()
        self.lambertValidatorObject.ModelTag = "LambertValidator"

        self.lambertSurfaceRelativeVelocityObject = lambertSurfaceRelativeVelocity.LambertSurfaceRelativeVelocity()
        self.lambertSurfaceRelativeVelocityObject.ModelTag = "LambertSurfaceRelativeVelocity"

        self.lambertSecondDvObject = lambertSecondDV.LambertSecondDV()
        self.lambertSecondDvObject.ModelTag = "LambertSecondDV"

        # create the FSW module gateway messages
        self.setupGatewayMsgs(SimBase)

        # Initialize all modules
        self.InitAllFSWObjects(SimBase)

        # Create tasks
        SimBase.fswProc.addTask(SimBase.CreateNewTask("inertial3DPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("hillPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("sunSafePointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("velocityPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("dvPointTask", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpFeedbackTask", self.processTasksTimeStep), 10)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpSteeringRWsTask", self.processTasksTimeStep), 10)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpFeedbackRWsTask", self.processTasksTimeStep), 10)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("mrpFeedbackTHsTask", self.processTasksTimeStep), 10)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("dvBurnTask", int(self.processTasksTimeStep/2)), 9)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("lambertGuidanceFirstDV", self.processTasksTimeStep), 20)
        SimBase.fswProc.addTask(SimBase.CreateNewTask("lambertGuidanceSecondDV", self.processTasksTimeStep), 20)

        # Assign initialized modules to tasks
        SimBase.AddModelToTask("inertial3DPointTask", self.inertial3DWrap, self.inertial3DData, 10)
        SimBase.AddModelToTask("inertial3DPointTask", self.trackingErrorWrap, self.trackingErrorData, 9)

        SimBase.AddModelToTask("hillPointTask", self.hillPointWrap, self.hillPointData, 10)
        SimBase.AddModelToTask("hillPointTask", self.trackingErrorWrap, self.trackingErrorData, 9)

        SimBase.AddModelToTask("sunSafePointTask", self.cssWlsEstWrap, self.cssWlsEstData, 10)
        SimBase.AddModelToTask("sunSafePointTask", self.sunSafePointWrap, self.sunSafePointData, 9)

        SimBase.AddModelToTask("velocityPointTask", self.velocityPointWrap, self.velocityPointData, 10)
        SimBase.AddModelToTask("velocityPointTask", self.trackingErrorWrap, self.trackingErrorData, 9)

        SimBase.AddModelToTask("dvPointTask", self.dvPointWrap, self.dvPointData, 10)
        SimBase.AddModelToTask("dvPointTask", self.attRefCorrectionWrap, self.attRefCorrectionData, 9)
        SimBase.AddModelToTask("dvPointTask", self.trackingErrorWrap, self.trackingErrorData, 8)

        SimBase.AddModelToTask("mrpFeedbackTask", self.mrpFeedbackControlWrap, self.mrpFeedbackControlData, 10)

        SimBase.AddModelToTask("mrpSteeringRWsTask", self.mrpSteeringWrap, self.mrpSteeringData, 10)
        SimBase.AddModelToTask("mrpSteeringRWsTask", self.rateServoWrap, self.rateServoData, 9)
        SimBase.AddModelToTask("mrpSteeringRWsTask", self.rwMotorTorqueWrap, self.rwMotorTorqueData, 8)

        SimBase.AddModelToTask("mrpFeedbackRWsTask", self.mrpFeedbackRWsWrap, self.mrpFeedbackRWsData, 9)
        SimBase.AddModelToTask("mrpFeedbackRWsTask", self.rwMotorTorqueWrap, self.rwMotorTorqueData, 8)

        SimBase.AddModelToTask("mrpFeedbackTHsTask", self.mrpFeedbackTHsWrap, self.mrpFeedbackTHsData, 9)
        SimBase.AddModelToTask("mrpFeedbackTHsTask", self.thrForceMappingWrap, self.thrForceMappingData, 8)
        SimBase.AddModelToTask("mrpFeedbackTHsTask", self.thrFiringRemainderWrap, self.thrFiringRemainderData, 7)

        SimBase.AddModelToTask("dvBurnTask", self.dvManeuverWrap, self.dvManeuverData, 10)

        SimBase.AddModelToTask("lambertGuidanceFirstDV", self.lambertPlannerObject, None, 10)
        SimBase.AddModelToTask("lambertGuidanceFirstDV", self.lambertSolverObject, None, 9)
        SimBase.AddModelToTask("lambertGuidanceFirstDV", self.lambertValidatorObject, None, 8)

        SimBase.AddModelToTask("lambertGuidanceSecondDV", self.lambertSurfaceRelativeVelocityObject, None, 10)
        SimBase.AddModelToTask("lambertGuidanceSecondDV", self.lambertSecondDvObject, None, 9)

        # Create events to be called for triggering GN&C maneuvers
        SimBase.fswProc.disableAllTasks()

        SimBase.createNewEvent("initiateStandby", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'standby'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.setAllButCurrentEventActivity('initiateStandby', True)"
                                ])

        SimBase.createNewEvent("initiateAttitudeGuidance", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'inertial3D'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('inertial3DPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateAttitudeGuidance', True)"
                                ])

        SimBase.createNewEvent("initiateAttitudeGuidanceDirect", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'directInertial3D'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('inertial3DPointTask')",
                                "self.enableTask('mrpFeedbackTask')",
                                "self.setAllButCurrentEventActivity('initiateAttitudeGuidanceDirect', True)"
                                ])

        SimBase.createNewEvent("initiateHillPoint", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'hillPoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('hillPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateHillPoint', True)"
                                ])

        SimBase.createNewEvent("initiateSunSafePoint", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'sunSafePoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('sunSafePointTask')",
                                "self.enableTask('mrpSteeringRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateSunSafePoint', True)"
                                ])

        SimBase.createNewEvent("initiateVelocityPoint", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'velocityPoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('velocityPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateVelocityPoint', True)"])

        SimBase.createNewEvent("initiateSteeringRW", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'steeringRW'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('hillPointTask')",
                                "self.enableTask('mrpSteeringRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateSteeringRW', True)"])

        SimBase.createNewEvent("initiateLambertGuidanceFirstDV", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'lambertFirstDV'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('hillPointTask')",
                                "self.enableTask('mrpSteeringRWsTask')",
                                "self.enableTask('lambertGuidanceFirstDV')",
                                "self.setAllButCurrentEventActivity('initiateLambertGuidanceFirstDV', True)"])

        SimBase.createNewEvent("initiateLambertGuidanceSecondDV", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'lambertSecondDV'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.FSWModels.zeroGateWayMsgs()",
                                "self.enableTask('hillPointTask')",
                                "self.enableTask('mrpSteeringRWsTask')",
                                "self.enableTask('lambertGuidanceSecondDV')",
                                "self.setAllButCurrentEventActivity('initiateLambertGuidanceSecondDV', True)"])

        SimBase.createNewEvent("initiateDvPoint", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'dvPoint'"],
                               ["self.fswProc.disableAllTasks()",
                                "self.enableTask('dvPointTask')",
                                "self.enableTask('mrpFeedbackRWsTask')",
                                "self.setAllButCurrentEventActivity('initiateDvPoint', True)"
                                ])

        SimBase.createNewEvent("initiateDvBurn", self.processTasksTimeStep, True,
                               ["self.modeRequest == 'dvBurn'"],
                               ["self.fswProc.disableAllTasks()",
                                "from Basilisk.architecture import messaging",
                                "self.FSWModels.cmdRwMotorMsg.write(messaging.ArrayMotorTorqueMsgPayload())",
                                "self.enableTask('dvPointTask')",
                                "self.enableTask('mrpFeedbackTHsTask')",
                                "self.enableTask('dvBurnTask')",
                                "self.setAllButCurrentEventActivity('initiateDvBurn', True)"
                                ])

    # ------------------------------------------------------------------------------------------- #
    # These are module-initialization methods
    def SetInertial3DPointGuidance(self):
        """Define the inertial 3D guidance module"""
        self.inertial3DData.sigma_R0N = [0.2, 0.4, 0.6]
        messaging.AttRefMsg_C_addAuthor(self.inertial3DData.attRefOutMsg, self.attRefMsg)

    def SetHillPointGuidance(self, SimBase):
        """Define the Hill pointing guidance module"""
        self.hillPointData.transNavInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.transOutMsg)
        self.hillPointData.celBodyInMsg.subscribeTo(SimBase.DynModels.EarthEphemObject.ephemOutMsgs[0])  # earth
        messaging.AttRefMsg_C_addAuthor(self.hillPointData.attRefOutMsg, self.attRefMsg)

    def SetSunSafePointGuidance(self, SimBase):
        """Define the sun safe pointing guidance module"""
        self.sunSafePointData.imuInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.attOutMsg)
        self.sunSafePointData.sunDirectionInMsg.subscribeTo(self.cssWlsEstData.navStateOutMsg)
        self.sunSafePointData.sHatBdyCmd = [0.0, 0.0, 1.0]
        messaging.AttGuidMsg_C_addAuthor(self.sunSafePointData.attGuidanceOutMsg, self.attGuidMsg)

    def SetVelocityPointGuidance(self, SimBase):
        """Define the velocity pointing guidance module"""
        self.velocityPointData.transNavInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.transOutMsg)
        self.velocityPointData.celBodyInMsg.subscribeTo(SimBase.DynModels.EarthEphemObject.ephemOutMsgs[0])
        self.velocityPointData.mu = SimBase.DynModels.gravFactory.gravBodies['earth'].mu
        messaging.AttRefMsg_C_addAuthor(self.velocityPointData.attRefOutMsg, self.attRefMsg)

    def SetDvPointGuidance(self):
        """Define the Delta-V pointing guidance module"""
        self.dvPointData.burnDataInMsg.subscribeTo(self.dvBurnCmdMsg)
        messaging.AttRefMsg_C_addAuthor(self.dvPointData.attRefOutMsg, self.attRefMsg)

    def SetAttRefCorrection(self):
        """Define the attitude reference correction module"""
        self.attRefCorrectionData.sigma_BcB = [0., 0., 0.]
        self.attRefCorrectionData.attRefInMsg.subscribeTo(self.attRefMsg)
        messaging.AttRefMsg_C_addAuthor(self.attRefCorrectionData.attRefOutMsg, self.attRefMsg)

    def SetAttitudeTrackingError(self, SimBase):
        """Define the attitude tracking error module"""
        self.trackingErrorData.attNavInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.attOutMsg)
        self.trackingErrorData.attRefInMsg.subscribeTo(self.attRefMsg)
        messaging.AttGuidMsg_C_addAuthor(self.trackingErrorData.attGuidOutMsg, self.attGuidMsg)

    def SetCSSWlsEst(self, SimBase):
        """Set the FSW CSS configuration information """
        cssConfig = messaging.CSSConfigMsgPayload()
        totalCSSList = []
        nHat_B_vec = [
            [0.0, 0.707107, 0.707107],
            [0.707107, 0., 0.707107],
            [0.0, -0.707107, 0.707107],
            [-0.707107, 0., 0.707107],
            [0.0, -0.965926, -0.258819],
            [-0.707107, -0.353553, -0.612372],
            [0., 0.258819, -0.965926],
            [0.707107, -0.353553, -0.612372]
        ]
        for CSSHat in nHat_B_vec:
            CSSConfigElement = messaging.CSSUnitConfigMsgPayload()
            CSSConfigElement.CBias = 1.0
            CSSConfigElement.nHat_B = CSSHat
            totalCSSList.append(CSSConfigElement)
        cssConfig.cssVals = totalCSSList

        cssConfig.nCSS = len(nHat_B_vec)
        self.cssConfigMsg = messaging.CSSConfigMsg().write(cssConfig)

        self.cssWlsEstData.cssDataInMsg.subscribeTo(SimBase.DynModels.CSSConstellationObject.constellationOutMsg)
        self.cssWlsEstData.cssConfigInMsg.subscribeTo(self.cssConfigMsg)

    def SetMRPFeedbackControl(self, SimBase):
        """Set the MRP feedback module configuration"""
        self.mrpFeedbackControlData.guidInMsg.subscribeTo(self.attGuidMsg)
        self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.vcMsg)
        messaging.CmdTorqueBodyMsg_C_addAuthor(self.mrpFeedbackControlData.cmdTorqueOutMsg, self.cmdTorqueDirectMsg)

        self.mrpFeedbackControlData.K = 3.5
        self.mrpFeedbackControlData.Ki = -1.0  # Note: make value negative to turn off integral feedback
        self.mrpFeedbackControlData.P = 30.0
        self.mrpFeedbackControlData.integralLimit = 2. / self.mrpFeedbackControlData.Ki * 0.1

    def SetMRPFeedbackRWA(self, SimBase):
        """Set the MRP feedback information if RWs are considered"""
        self.mrpFeedbackRWsData.K = 3.5
        self.mrpFeedbackRWsData.Ki = -1  # Note: make value negative to turn off integral feedback
        self.mrpFeedbackRWsData.P = 30.0
        self.mrpFeedbackRWsData.integralLimit = 2. / self.mrpFeedbackRWsData.Ki * 0.1

        self.mrpFeedbackRWsData.vehConfigInMsg.subscribeTo(self.vcMsg)
        self.mrpFeedbackRWsData.rwSpeedsInMsg.subscribeTo(SimBase.DynModels.rwStateEffector.rwSpeedOutMsg)
        self.mrpFeedbackRWsData.rwParamsInMsg.subscribeTo(self.fswRwConfigMsg)
        self.mrpFeedbackRWsData.guidInMsg.subscribeTo(self.attGuidMsg)
        messaging.CmdTorqueBodyMsg_C_addAuthor(self.mrpFeedbackRWsData.cmdTorqueOutMsg, self.cmdTorqueMsg)

    def SetMRPFeedbackTH(self, SimBase):
        """Set the MRP feedback information if Thrusters are considered"""
        self.mrpFeedbackTHsData.K = 3.5*10
        self.mrpFeedbackTHsData.Ki = 0.0002  # Note: make value negative to turn off integral feedback
        self.mrpFeedbackTHsData.P = 30.0*10
        self.mrpFeedbackTHsData.integralLimit = 2. / self.mrpFeedbackTHsData.Ki * 0.1

        self.mrpFeedbackTHsData.vehConfigInMsg.subscribeTo(self.vcMsg)
        self.mrpFeedbackTHsData.guidInMsg.subscribeTo(self.attGuidMsg)
        messaging.CmdTorqueBodyMsg_C_addAuthor(self.mrpFeedbackTHsData.cmdTorqueOutMsg, self.cmdTorqueMsg)

    def SetMRPSteering(self):
        """Set the MRP Steering module"""
        self.mrpSteeringData.K1 = 0.05
        self.mrpSteeringData.ignoreOuterLoopFeedforward = False
        self.mrpSteeringData.K3 = 0.75
        self.mrpSteeringData.omega_max = 1.0 * mc.D2R
        self.mrpSteeringData.guidInMsg.subscribeTo(self.attGuidMsg)

    def SetRateServo(self, SimBase):
        """Set the rate servo module"""
        self.rateServoData.guidInMsg.subscribeTo(self.attGuidMsg)
        self.rateServoData.vehConfigInMsg.subscribeTo(self.vcMsg)
        self.rateServoData.rwParamsInMsg.subscribeTo(self.fswRwConfigMsg)
        self.rateServoData.rwSpeedsInMsg.subscribeTo(SimBase.DynModels.rwStateEffector.rwSpeedOutMsg)
        self.rateServoData.rateSteeringInMsg.subscribeTo(self.mrpSteeringData.rateCmdOutMsg)
        messaging.CmdTorqueBodyMsg_C_addAuthor(self.rateServoData.cmdTorqueOutMsg, self.cmdTorqueMsg)

        self.rateServoData.Ki = 5.0
        self.rateServoData.P = 150.0
        self.rateServoData.integralLimit = 2. / self.rateServoData.Ki * 0.1
        self.rateServoData.knownTorquePntB_B = [0., 0., 0.]

    def SetVehicleConfiguration(self):
        """Set the spacecraft configuration information"""
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        # use the same inertia in the FSW algorithm as in the simulation
        vehicleConfigOut.ISCPntB_B = [900.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 600.0]
        self.vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    def SetRWConfigMsg(self):
        """Set the RW device information"""
        # Configure RW pyramid exactly as it is in the Dynamics (i.e. FSW with perfect knowledge)
        rwElAngle = np.array([40.0, 40.0, 40.0, 40.0]) * mc.D2R
        rwAzimuthAngle = np.array([45.0, 135.0, 225.0, 315.0]) * mc.D2R
        wheelJs = 50.0 / (6000.0 * math.pi * 2.0 / 60)

        fswSetupRW.clearSetup()
        for elAngle, azAngle in zip(rwElAngle, rwAzimuthAngle):
            gsHat = (rbk.Mi(-azAngle, 3).dot(rbk.Mi(elAngle, 2))).dot(np.array([1, 0, 0]))
            fswSetupRW.create(gsHat,  # spin axis
                              wheelJs,  # kg*m^2
                              0.2)  # Nm        uMax

        self.fswRwConfigMsg = fswSetupRW.writeConfigMessage()

    def SetRWMotorTorque(self):
        """Set the RW motor torque information"""
        controlAxes_B = [
            1.0, 0.0, 0.0
            , 0.0, 1.0, 0.0
            , 0.0, 0.0, 1.0
        ]
        self.rwMotorTorqueData.controlAxes_B = controlAxes_B
        self.rwMotorTorqueData.vehControlInMsg.subscribeTo(self.cmdTorqueMsg)
        messaging.ArrayMotorTorqueMsg_C_addAuthor(self.rwMotorTorqueData.rwMotorTorqueOutMsg, self.cmdRwMotorMsg)
        self.rwMotorTorqueData.rwParamsInMsg.subscribeTo(self.fswRwConfigMsg)

    def SetThrConfigMsg(self):
        """Set the Thruster information"""

        if self.useDvThrusters:
            # 6 DV thrusters
            thPos = [[0, 0.95, -1.1],
                     [0.8227241335952166, 0.4750000000000003, -1.1],
                     [0.8227241335952168, -0.47499999999999976, -1.1],
                     [0, -0.95, -1.1],
                     [-0.8227241335952165, -0.4750000000000004, -1.1],
                     [-0.822724133595217, 0.4749999999999993, -1.1]]
            thDir = [[0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0]]
            maxThrust = 22
        else:
            # 8 thrusters are modeled that act in pairs to provide the desired torque
            thPos = [[825.5/1000.0, 880.3/1000.0, 1765.3/1000.0],
                     [825.5/1000.0, 880.3/1000.0, 260.4/1000.0],
                     [880.3/1000.0, 825.5/1000.0, 1765.3/1000.0],
                     [880.3/1000.0, 825.5/1000.0, 260.4/1000.0],
                     [-825.5/1000.0, -880.3/1000.0, 1765.3/1000.0],
                     [-825.5/1000.0, -880.3/1000.0, 260.4/1000.0],
                     [-880.3/1000.0, -825.5/1000.0, 1765.3/1000.0],
                     [-880.3/1000.0, -825.5/1000.0, 260.4/1000.0]]
            thDir = [[0.0, -1.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]]
            maxThrust = 1

        fswSetupThrusters.clearSetup()
        for pos_B, dir_B in zip(thPos, thDir):
            fswSetupThrusters.create(pos_B, dir_B, maxThrust)

        self.fswThrConfigMsg = fswSetupThrusters.writeConfigMessage()

    def SetThrForceMapping(self):
        """Set the Thrust Force Mapping information"""
        if self.useDvThrusters:
            controlAxes_B = [1, 0, 0,
                             0, 1, 0]
            thrForceSign = -1
        else:
            controlAxes_B = [1, 0, 0,
                             0, 1, 0,
                             0, 0, 1]
            thrForceSign = +1

        self.thrForceMappingData.thrForceSign = thrForceSign
        self.thrForceMappingData.controlAxes_B = controlAxes_B

        self.thrForceMappingData.cmdTorqueInMsg.subscribeTo(self.cmdTorqueMsg)
        self.thrForceMappingData.thrConfigInMsg.subscribeTo(self.fswThrConfigMsg)
        self.thrForceMappingData.vehConfigInMsg.subscribeTo(self.vcMsg)

    def SetThrFiringRemainder(self):
        """Set the Thrust Firing Remainder information"""
        self.thrFiringRemainderData.thrMinFireTime = 0.002
        if self.useDvThrusters:
            self.thrFiringRemainderData.baseThrustState = 1

        self.thrFiringRemainderData.thrConfInMsg.subscribeTo(self.fswThrConfigMsg)
        self.thrFiringRemainderData.thrForceInMsg.subscribeTo(self.thrForceMappingData.thrForceCmdOutMsg)
        if self.useDvThrusters:
            messaging.THRArrayOnTimeCmdMsg_C_addAuthor(self.thrFiringRemainderData.onTimeOutMsg, self.dvOnTimeCmdMsg)
        else:
            messaging.THRArrayOnTimeCmdMsg_C_addAuthor(self.thrFiringRemainderData.onTimeOutMsg, self.acsOnTimeCmdMsg)

    def SetDvManeuver(self, SimBase):
        """Set the Dv Burn Maneuver information"""
        self.dvManeuverData.navDataInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.transOutMsg)
        self.dvManeuverData.burnDataInMsg.subscribeTo(self.dvBurnCmdMsg)
        messaging.THRArrayOnTimeCmdMsg_C_addAuthor(self.dvManeuverData.thrCmdOutMsg, self.dvOnTimeCmdMsg)

    def SetLambertPlannerObject(self, SimBase):
        """Set the lambert planner object."""
        self.lambertPlannerObject.navTransInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.transOutMsg)


    def SetLambertSolverObject(self):
        """Set the lambert solver object."""
        self.lambertSolverObject.lambertProblemInMsg.subscribeTo(self.lambertPlannerObject.lambertProblemOutMsg)


    def SetLambertValidatorObject(self, SimBase):
        """Set the lambert validator object."""
        self.lambertValidatorObject.navTransInMsg.subscribeTo(SimBase.DynModels.simpleNavObject.transOutMsg)
        self.lambertValidatorObject.lambertProblemInMsg.subscribeTo(self.lambertPlannerObject.lambertProblemOutMsg)
        self.lambertValidatorObject.lambertPerformanceInMsg.subscribeTo(
            self.lambertSolverObject.lambertPerformanceOutMsg)
        self.lambertValidatorObject.lambertSolutionInMsg.subscribeTo(self.lambertSolverObject.lambertSolutionOutMsg)
        self.lambertValidatorObject.dvBurnCmdOutMsg = self.dvBurnCmdMsg


    def SetLambertSurfaceRelativeVelocityObject(self, SimBase):
        """Set the lambert surface relative velocity object."""
        self.lambertSurfaceRelativeVelocityObject.lambertProblemInMsg.subscribeTo(
            self.lambertPlannerObject.lambertProblemOutMsg)
        self.lambertSurfaceRelativeVelocityObject.ephemerisInMsg.subscribeTo(
            SimBase.DynModels.EarthEphemObject.ephemOutMsgs[0])


    def SetLambertSecondDvObject(self):
        """Set the lambert second DV object."""
        self.lambertSecondDvObject.lambertSolutionInMsg.subscribeTo(self.lambertSolverObject.lambertSolutionOutMsg)
        self.lambertSecondDvObject.desiredVelocityInMsg.subscribeTo(
            self.lambertSurfaceRelativeVelocityObject.desiredVelocityOutMsg)
        self.lambertSecondDvObject.dvBurnCmdOutMsg = self.dvBurnCmdMsg

    # Global call to initialize every module
    def InitAllFSWObjects(self, SimBase):
        """Initialize all the FSW objects"""

        # note that the order in which these routines are called is important.
        # To subscribe to a message that message must already exit.
        self.SetVehicleConfiguration()
        self.SetRWConfigMsg()
        self.SetInertial3DPointGuidance()
        self.SetHillPointGuidance(SimBase)
        self.SetCSSWlsEst(SimBase)
        self.SetSunSafePointGuidance(SimBase)
        self.SetVelocityPointGuidance(SimBase)
        self.SetDvPointGuidance()
        self.SetAttRefCorrection()
        self.SetAttitudeTrackingError(SimBase)
        self.SetMRPFeedbackControl(SimBase)
        self.SetMRPFeedbackRWA(SimBase)
        self.SetMRPFeedbackTH(SimBase)
        self.SetMRPSteering()
        self.SetRateServo(SimBase)
        self.SetRWMotorTorque()
        self.SetThrConfigMsg()
        self.SetThrForceMapping()
        self.SetThrFiringRemainder()
        self.SetDvManeuver(SimBase)
        self.SetLambertPlannerObject(SimBase)
        self.SetLambertSolverObject()
        self.SetLambertValidatorObject(SimBase)
        self.SetLambertSurfaceRelativeVelocityObject(SimBase)
        self.SetLambertSecondDvObject()

    def setupGatewayMsgs(self, SimBase):
        """create C-wrapped gateway messages such that different modules can write to this message
        and provide a common input msg for down-stream modules"""
        self.cmdTorqueMsg = messaging.CmdTorqueBodyMsg_C()
        self.cmdTorqueDirectMsg = messaging.CmdTorqueBodyMsg_C()
        self.attRefMsg = messaging.AttRefMsg_C()
        self.attGuidMsg = messaging.AttGuidMsg_C()
        self.cmdRwMotorMsg = messaging.ArrayMotorTorqueMsg_C()
        self.acsOnTimeCmdMsg = messaging.THRArrayOnTimeCmdMsg_C()
        self.dvOnTimeCmdMsg = messaging.THRArrayOnTimeCmdMsg_C()

        # C++ wrapped gateway messages
        self.dvBurnCmdMsg = messaging.DvBurnCmdMsg()

        self.zeroGateWayMsgs()

        # connect gateway FSW effector command msgs with the dynamics
        SimBase.DynModels.extForceTorqueObject.cmdTorqueInMsg.subscribeTo(self.cmdTorqueDirectMsg)
        SimBase.DynModels.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.cmdRwMotorMsg)
        SimBase.DynModels.thrustersDynamicEffectorACS.cmdsInMsg.subscribeTo(self.acsOnTimeCmdMsg)
        SimBase.DynModels.thrustersDynamicEffectorDV.cmdsInMsg.subscribeTo(self.dvOnTimeCmdMsg)

    def zeroGateWayMsgs(self):
        """Zero all the FSW gateway message payloads"""
        self.cmdTorqueMsg.write(messaging.CmdTorqueBodyMsgPayload())
        self.cmdTorqueDirectMsg.write(messaging.CmdTorqueBodyMsgPayload())
        self.attRefMsg.write(messaging.AttRefMsgPayload())
        self.attGuidMsg.write(messaging.AttGuidMsgPayload())
        self.cmdRwMotorMsg.write(messaging.ArrayMotorTorqueMsgPayload())
        self.acsOnTimeCmdMsg.write(messaging.THRArrayOnTimeCmdMsgPayload())
        self.dvOnTimeCmdMsg.write(messaging.THRArrayOnTimeCmdMsgPayload())
        self.dvBurnCmdMsg.write(messaging.DvBurnCmdMsgPayload())

    def SetAttThrusters(self, useDvThrusters):
        """Change thruster type used for attitude control (ACS or DV)"""
        self.useDvThrusters = useDvThrusters
        self.SetThrConfigMsg()
        self.SetThrForceMapping()
        self.SetThrFiringRemainder()
