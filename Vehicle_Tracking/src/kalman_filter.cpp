#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using std::cout;
using std::endl;

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;


}

void KalmanFilter::Predict() {

  //Predict state(estimation)
  x_ = F_ * x_;

  //Predict process co-variance matrix (estimation error)
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();

  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;


  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

VectorXd linearize(const VectorXd &x)
{

    double px = x[0];
    double py = x[1];
    double vx = x[2];
    double vy = x[3];

    double epsilon = pow(10, -15);

    double rho = sqrt(pow(px, 2) + pow(py, 2));

    //possible_bug - handle div by 0 && atan bw -pi and +pi
    double phi = atan2(py, px); //atan(py/ px);

    double rho_dot = (px*vx + py*vy + epsilon) / (rho + epsilon);


    VectorXd Hx(3);
    Hx << rho, phi, rho_dot;

    return Hx;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */


    Tools t_ = Tools();
    VectorXd Hx = linearize(x_);

    VectorXd y = z - Hx;
    y[1] = t_.normalize(y[1]);
    MatrixXd Hj = t_.CalculateJacobian(x_);
    MatrixXd Hj_t = Hj.transpose();


    MatrixXd S = Hj * P_ * Hj_t + R_;


    MatrixXd Si = S.inverse();


    MatrixXd PHt = P_ * Hj_t;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * Hj) * P_;


}
