#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  tools = Tools();
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

VectorXd extractState(const double rho,const double phi,const double rho_dot)
{
    VectorXd state(4);

    double rho_sq = pow(rho, 2);
    double tan_phi_sq = pow(tan(phi), 2);

    double px = sqrt(rho_sq / (1 + tan_phi_sq));
    double py = tan(phi) * px;

    double vx = 0;
    double vy = 0;
    state << px, py, vx, vy;


    return state;
}



void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

      MatrixXd F_ = MatrixXd(4, 4);
      F_ << 1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;

      MatrixXd P_ = MatrixXd(4, 4);
      P_ << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;

      MatrixXd Q_ = MatrixXd(4, 4);
      Q_ << 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;


    // first measurement

    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates
      //         and initialize state.
     double rho = measurement_pack.raw_measurements_[0],
            phi = measurement_pack.raw_measurements_[1],
            rho_dot = measurement_pack.raw_measurements_[2];

     VectorXd x_ = extractState(rho, phi, rho_dot);
     /*
     Have : x_, P_, F_, R_, Q_
     Want : H_laser or Hj
     */
     Hj_ = tools.CalculateJacobian(x_);
     ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);

    cout << "rader Init " << endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.

     double init_px = measurement_pack.raw_measurements_[0];
     double init_py = measurement_pack.raw_measurements_[1];
     double init_vx = 0;
     double init_vy = 0;

     VectorXd x_(4);
     x_ << init_px, init_py, init_vx, init_vy;

     H_laser_ << 1, 0, 0, 0,
                 0, 1, 0, 0;

     ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

    }

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "Lidar Init " << endl;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // TODO: YOUR CODE HERE

  // 1. Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // 2. Set the process covariance matrix Q

  float dt_cube = pow(dt, 3);
  float dt_sq = pow(dt, 2);
  float dt_quad = pow(dt, 4);

  float noise_ax = 9, noise_ay = 9;

  ekf_.Q_ << dt_quad * noise_ax/ 4, 0, dt_cube * noise_ax/ 2, 0,
            0, dt_quad * noise_ay/ 4, 0, dt_cube * noise_ay/ 2,
            dt_cube * noise_ax/ 2, 0, dt_sq * noise_ax, 0,
            0, dt_cube * noise_ay/ 2, 0, dt_sq * noise_ay;
  // 3. Call the Kalman Filter predict() function



  ekf_.Predict();
    cout << "predict " << endl;

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    cout << "starting ekf update" << endl;
    ekf_.R_ = R_radar_;

    VectorXd z(3);
    z << measurement_pack.raw_measurements_[0],
        measurement_pack.raw_measurements_[1],
        measurement_pack.raw_measurements_[2];

    ekf_.UpdateEKF(z);
        cout << "ekf - UpdateEKF() " << endl;
  } else {
    // TODO: Laser updates
    cout << "starting normal update 2" << endl;
    ekf_.R_ = R_laser_;

    VectorXd z(2);
    z << measurement_pack.raw_measurements_[0],
        measurement_pack.raw_measurements_[1];

    ekf_.Update(z);
        cout << "ekf - Update() " << endl;
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
