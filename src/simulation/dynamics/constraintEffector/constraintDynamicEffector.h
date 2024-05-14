/*
 ISC License

 Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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


#ifndef CONSTRAINT_DYNAMIC_EFFECTOR_H
#define CONSTRAINT_DYNAMIC_EFFECTOR_H

#include "simulation/dynamics/_GeneralModuleFiles/dynamicEffector.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"

#include "architecture/messaging/messaging.h"

#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/avsEigenMRP.h"
#include <Eigen/Dense>
#include <vector>

/*! @brief constraint dynamic effector class */
class ConstraintDynamicEffector: public SysModel, public DynamicEffector {
public:
    ConstraintDynamicEffector();
    ~ConstraintDynamicEffector();
    void Reset(uint64_t CurrentSimNanos);
    void linkInStates(DynParamManager& states);
    void computeForceTorque(double integTime, double timeStep);
    void UpdateState(uint64_t CurrentSimNanos);

    /** setter for `r_P2P1_B1Init` initial spacecraft separation */
    void setR_P2P1_B1Init(Eigen::Vector3d r_P2P1_B1Init);
    /** setter for `r_P1B1_B1` connection point position on spacecraft 1 */
    void setR_P1B1_B1(Eigen::Vector3d r_P1B1_B1);
    /** setter for `r_P2B2_B2` connection point position on spacecraft 2 */
    void setR_P2B2_B2(Eigen::Vector3d r_P2B2_B2);
    /** setter for `alpha` gain tuning parameter */
    void setAlpha(double alpha);
    /** setter for `beta` gain tuning parameter */
    void setBeta(double beta);
    /** setter for `k_d` gain */
    void setKd(double k_d);
    /** setter for `c_d` gain */
    void setCd(double c_d);
    /** setter for `k_a` gain */
    void setKa(double k_a);
    /** setter for `c_a` gain */
    void setCa(double c_a);

    /** getter for `r_P2P1_B1Init` initial spacecraft separation */
    Eigen::Vector3d getR_P2P1_B1Init() const {return this->r_P2P1_B1Init;};
    /** getter for `r_P1B1_B1` connection point position on spacecraft 1 */
    Eigen::Vector3d getR_P1B1_B1() const {return this->r_P1B1_B1;};
    /** getter for `r_P2B2_B2` connection point position on spacecraft 2 */
    Eigen::Vector3d getR_P2B2_B2() const {return this->r_P2B2_B2;};
    /** getter for `alpha` gain tuning parameter */
    double getAlpha() const {return this->alpha;};
    /** getter for `beta` gain tuning parameter */
    double getBeta() const {return this->beta;};
    /** getter for `k_d` gain */
    double getKd() const {return this->k_d;};
    /** getter for `c_d` gain */
    double getCd() const {return this->c_d;};
    /** getter for `k_a` gain */
    double getKa() const {return this->k_a;};
    /** getter for `c_a` gain */
    double getCa() const {return this->c_a;};

private:
    // Counters and flags
    int scInitCounter; //!< [] counter to kill simulation if more than two spacecraft initialized
    int scID; //!< [] 0,1 alternating spacecraft tracker to output appropriate force/torque

    // Constraint length and direction
    Eigen::Vector3d r_P1B1_B1; //!< [m] position vector from spacecraft 1 hub to its connection point P1
    Eigen::Vector3d r_P2B2_B2; //!< [m] position vector from spacecraft 2 hub to its connection point P2
    Eigen::Vector3d r_P2P1_B1Init; //!< [m] precribed position vector from spacecraft 1 connection point to spacecraft 2 connection point

    // Gains for PD controller
    double alpha; //!< [] Baumgarte stabilization gain tuning variable
    double beta; //!< [] Baumgarte stabilization gain tuning variable
    double k_d; //!< [] direction constraint proportional gain
    double c_d; //!< [] direction constraint derivative gain
    double k_a; //!< [] attitude constraint proportional gain
    double c_a; //!< [] attitude constraint derivative gain

    // Simulation variable pointers
    std::vector<StateData*> hubPosition;    //!< [m] parent position
    std::vector<StateData*> hubVelocity;    //!< [m/s] parent velocity
    std::vector<StateData*> hubSigma;       //!< [] parent attitude
    std::vector<StateData*> hubOmega;       //!< [rad/s] parent angular velocity

    // Constraint violations
    Eigen::Vector3d psi_N; //!< [m] direction constraint violation in inertial frame
    Eigen::Vector3d psiPrime_N; //!< [m/s] direction rate constraint violation in inertial frame
    Eigen::MRPd sigma_B2B1; //!< [] attitude constraint violation
    Eigen::Vector3d omega_B2B1_B2; //!< [rad/s] angular velocity constraint violation in spacecraft 2 body frame

    // Force and torque quantities stored to be assigned on the alternating call of computeForceTorque
    Eigen::Vector3d Fc_N; //!< [N] force applied on each spacecraft COM in the inertial frame
    Eigen::Vector3d L_B2; //!< [N-m] torque applied on spacecraft 2 in its body frame

    BSKLogger bskLogger;                    //!< BSK Logging
};


#endif /* CONSTRAINT_DYNAMIC_EFFECTOR_H */
