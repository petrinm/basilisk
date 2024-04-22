/*
 ISC License

 Copyright (c) 2024, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#ifndef LINEAR_TRANSLATION_N_DOF_STATE_EFFECTOR_H
#define LINEAR_TRANSLATION_N_DOF_STATE_EFFECTOR_H

#include <Eigen/Dense>
#include "simulation/dynamics/_GeneralModuleFiles/stateEffector.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/avsEigenMRP.h"
#include "architecture/utilities/bskLogging.h"

#include "architecture/msgPayloadDefC/ArrayMotorForceMsgPayload.h"
#include "architecture/msgPayloadDefC/ArrayEffectorLockMsgPayload.h"
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/msgPayloadDefC/LinearTranslationRigidBodyMsgPayload.h"
#include "architecture/messaging/messaging.h"

struct translatingBody {
public:
    /** setter for `mass` property */
    void setMass(double mass);
    /** setter for `k` property */
    void setK(double k);
    /** setter for `c` property */
    void setC(double c);
    /** setter for `rhoInit` property */
    void setRhoInit(double rhoInit) {this->rhoInit = rhoInit;};
    /** setter for `rhoDotInit` property */
    void setRhoDotInit(double rhoDotInit) {this->rhoDotInit = rhoDotInit;};
    /** setter for `fHat_P` property */
    void setFHat_P(Eigen::Vector3d fHat_P);
    /** setter for `r_FcF_F` property */
    void setR_FcF_F(Eigen::Vector3d r_FcF_F) {this->r_FcF_F = r_FcF_F;};
    /** setter for `r_F0B_B` property */
    void setR_F0P_P(Eigen::Vector3d r_F0P_P) {this->r_F0P_P = r_F0P_P;};
    /** setter for `IPntFc_F` property */
    void setIPntFc_F(Eigen::Matrix3d IPntFc_F) {this->IPntFc_F = IPntFc_F;};
    /** setter for `dcm_FB` property */
    void setDCM_FB(Eigen::Matrix3d dcm_FB) {this->dcm_FB = dcm_FB;};

    /** setter for `mass` property */
    double getMass() const {return this->mass;};
    /** setter for `k` property */
    double getK() const {return this->k;};
    /** setter for `c` property */
    double getC() const {return this->c;};
    /** setter for `rhoInit` property */
    double getRhoInit() const {return this->rhoInit;};
    /** setter for `rhoDotInit` property */
    double getRhoDotInit() const {return this->rhoDotInit;};
    /** setter for `fHat_P` property */
    Eigen::Vector3d getFHat_P() const {return this->fHat_P;};
    /** setter for `r_FcF_F` property */
    Eigen::Vector3d getR_FcF_F() const {return this->r_FcF_F;};
    /** setter for `r_F0P_P` property */
    Eigen::Vector3d getR_F0P_P() const {return this->r_F0P_P;};
    /** setter for `IPntFc_F` property */
    Eigen::Matrix3d getIPntFc_F() const {return IPntFc_F;};
    /** setter for `dcm_FP` property */
    Eigen::Matrix3d getDCM_FP() const {return dcm_FP;};

private:
    friend class linearTranslationNDOFStateEffector;

    double mass = 0.0;
    Eigen::Matrix3d IPntFc_F = Eigen::Matrix3d::Identity();           //!< [kg-m^2] Inertia of body about point Pc in P frame components
    Eigen::Vector3d r_FcF_F = Eigen::Vector3d::Zero();           //!< [m] vector pointing from translating frame P origin to point Pc (center of mass of arm) in P frame components
    Eigen::Vector3d r_F0P_P = Eigen::Vector3d::Zero();
    Eigen::Vector3d fHat_P{1.0, 0.0, 0.0};         //!< -- translating axis in P frame components.
    double k = 0.0;                                           //!< [N-m/rad] torsional spring constant
    double c = 0.0;                                           //!< [N-m-s/rad] rotational damping coefficient
    double rhoInit = 0.0;                                     //!< [rad] initial spinning body angle
    double rhoDotInit = 0.0;                                  //!< [rad/s] initial spinning body angle rate
    Eigen::Matrix3d dcm_FP = Eigen::Matrix3d::Identity();            //!< -- DCM from P frame to body frame

    // Scalar Properties
    double rho = 0.0;
    double rhoDot = 0.0;
    double rhoRef = 0.0;
    double rhoDotRef = 0.0;
    double u;
    bool isAxisLocked = false;

    // Vector quantities
    Eigen::Vector3d r_FF0_B;
    Eigen::Vector3d r_F0P_B;
    Eigen::Vector3d fHat_B;                                     //!< -- translating axis in B frame components.
    Eigen::Vector3d r_FcF_B;            //!< [m] vector pointing from translating frame P origin to point Pc (center of mass of arm) in B frame components
    Eigen::Vector3d r_FB_B;             //!< [m] vector pointing from body frame B origin to P frame origin in B frame components
    Eigen::Vector3d r_FcB_B;              //!< [m] vector pointing from body frame B origin to Pc in B frame components
    Eigen::Vector3d rPrime_FB_B;     //!< [m/s] body frame time derivative of r_Sc1S1_B
    Eigen::Vector3d rPrime_FcF_B;     //!< [m/s] body frame time derivative of r_Sc1S1_B
    Eigen::Vector3d rPrime_FcB_B;      //!< [m/s] body frame time derivative of r_Sc1B_B
    Eigen::Vector3d rDot_FcB_B;        //!< [m/s] inertial frame time derivative of r_Sc1B_B
    Eigen::Vector3d r_FP_B;        //!< [m/s] vector from parent frame to current F frame
    Eigen::Vector3d r_FP_P;        //!< [m/s] vector from parent frame to current F frame
    Eigen::Vector3d rPrime_FP_B;        //!< [m/s] vector from parent frame to current F frame
    Eigen::Vector3d rPrime_FF0_B;

    Eigen::Vector3d omega_FN_B;        //!< [rad/s] angular velocity of the P frame wrt the N frame in B frame components
    Eigen::Vector3d omega_SB_B; // zero for all bodies
    Eigen::Matrix3d omegaTilde_FB_B;

    // Matrix quantities
    Eigen::Matrix3d dcm_FB;            //!< -- DCM from P frame to body frame
    Eigen::Matrix3d IPntFc_B;          //!< -- [kg-m^2] Inertia of body about point Pc in B frame components
    Eigen::Matrix3d IPrimePntFc_B;     //!< [kg-m^2] Inertia of body about point Pc in B frame components
    Eigen::Matrix3d rTilde_FcB_B;      //!< [m] tilde matrix of r_Sc2B_B

    // Inertial properties
    Eigen::Vector3d r_FcN_N;            //!< [m] position vector of translating body's center of mass Sc relative to the inertial frame origin N
    Eigen::Vector3d v_FcN_N;            //!< [m/s] inertial velocity vector of Sc relative to inertial frame
    Eigen::Vector3d sigma_FN;           //!< -- MRP attitude of frame S relative to inertial frame
    Eigen::Vector3d omega_FN_F;         //!< [rad/s] inertial translating body frame angular velocity vector

    BSKLogger bskLogger;
};

/*! @brief translating body state effector class */
class linearTranslationNDOFStateEffector: public StateEffector, public SysModel {
public:

    linearTranslationNDOFStateEffector();      //!< -- Contructor
    ~linearTranslationNDOFStateEffector() final;     //!< -- Destructor

    std::vector<Message<LinearTranslationRigidBodyMsgPayload>*> translatingBodyOutMsgs;       //!< vector of state output messages
    std::vector<Message<SCStatesMsgPayload>*> translatingBodyConfigLogOutMsgs;     //!< vector of spinning body state config log messages
    std::vector<ReadFunctor<LinearTranslationRigidBodyMsgPayload>> translatingBodyRefInMsgs;  //!< (optional) reference state input message
    ReadFunctor<ArrayMotorForceMsgPayload> motorForceInMsg;                   //!< -- (optional) motor force input message name
    ReadFunctor<ArrayEffectorLockMsgPayload> motorLockInMsg;                    //!< -- (optional) motor lock input message name

    void addTranslatingBody(translatingBody const& newBody); //!< class method

    void setNameOfRhoState(const std::string& nameOfRhoState) { this->nameOfRhoState = nameOfRhoState; };
    void setNameOfRhoDotState(const std::string& nameOfRhoDotState) { this->nameOfRhoDotState = nameOfRhoDotState; };
    std::string getNameOfRhoState() const { return this->nameOfRhoState; };
    std::string getNameOfRhoDotState() const { return this->nameOfRhoDotState; };

private:
    static uint64_t effectorID;     //!< [] ID number of this effector
    int N = 0;
    std::vector<translatingBody> translatingBodyVec;

    // Terms needed for back substitution
    Eigen::MatrixXd ARho;     //!< -- rDDot_BN term for back substitution
    Eigen::MatrixXd BRho;     //!< -- omegaDot_BN term for back substitution
    Eigen::VectorXd CRho;     //!< -- scalar term for back substitution

    // Hub properties
    Eigen::Vector3d omega_BN_B;         //!< [rad/s] angular velocity of the B frame wrt the N frame in B frame components
    Eigen::MRPd sigma_BN;               //!< -- body frame attitude wrt to the N frame in MRPs
    Eigen::Matrix3d dcm_BN;             //!< -- DCM from inertial frame to body frame

    // States
    Eigen::MatrixXd* inertialPositionProperty = nullptr;    //!< [m] r_N inertial position relative to system spice zeroBase/refBase
    Eigen::MatrixXd* inertialVelocityProperty = nullptr;    //!< [m] v_N inertial velocity relative to system spice zeroBase/refBase
    StateData* rhoState = nullptr;
    StateData* rhoDotState = nullptr;
    std::string nameOfRhoState;                               //!< -- identifier for the theta state data container
    std::string nameOfRhoDotState;                            //!< -- identifier for the thetaDot state data container

    // module functions
    void Reset(uint64_t CurrentClock) final;      //!< -- Method for reset
    void writeOutputStateMessages(uint64_t CurrentClock) final;   //!< -- Method for writing the output messages
    void UpdateState(uint64_t CurrentSimNanos) final;             //!< -- Method for updating information
    void registerStates(DynParamManager& statesIn) final;         //!< -- Method for registering the SB states
    void linkInStates(DynParamManager& states) final;             //!< -- Method for getting access to other states
    void updateContributions(double integTime,
                             BackSubMatrices& backSubContr,
                             Eigen::Vector3d sigma_BN,
                             Eigen::Vector3d omega_BN_B,
                             Eigen::Vector3d g_N) final;  //!< -- Method for back-substitution contributions
    void computeDerivatives(double integTime,
                            Eigen::Vector3d rDDot_BN_N,
                            Eigen::Vector3d omegaDot_BN_B,
                            Eigen::Vector3d sigma_BN) final;                         //!< -- Method for SB to compute its derivatives
    void updateEffectorMassProps(double integTime) final;         //!< -- Method for giving the s/c the HRB mass props and prop rates
    void updateEnergyMomContributions(double integTime,
                                      Eigen::Vector3d& rotAngMomPntCContr_B,
                                      double& rotEnergyContr,
                                      Eigen::Vector3d omega_BN_B) final;       //!< -- Method for computing energy and momentum for SBs

    void prependSpacecraftNameToStates() final;                   //!< Method used for multiple spacecraft
    void computeTranslatingBodyInertialStates();               //!< Method for computing the SB's states
};

#endif /* LINEAR_TRANSLATION_N_DOF_STATE_EFFECTOR_H */
