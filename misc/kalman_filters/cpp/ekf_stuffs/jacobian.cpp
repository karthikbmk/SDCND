#include <iostream>
#include <vector>
#include "Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {
  /**
   * Compute the Jacobian Matrix
   */

  // predicted state example
  // px = 1, py = 2, vx = 0.2, vy = 0.4
  VectorXd x_predicted(4);
  x_predicted << 1, 2, 0.2, 0.4;

  MatrixXd Hj = CalculateJacobian(x_predicted);

  cout << "Hj:" << endl << Hj << endl;

  return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // TODO: YOUR CODE HERE
  VectorXd hx(3);

  try
  {
      float d1 = pow(px,2) + pow(py, 2);
      float d2 = sqrt(d1);
      float d3 = pow(d1, 1.5);

      Hj << px / d2 , py / d2, 0 , 0,
            -py/ d1, px / d1, 0, 0,
            py*(vx*py - vy*px)/ d3, px*(vy*px - vx*py)/ d3, px/ d2, py / d2;


      return Hj;

  }
  catch(const char* p)
  {
    cout << p << endl;
    return Hj;
  }
  // check division by zero

  // compute the Jacobian matrix
}
