#include "tools.h"
#include <iostream>
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if ( estimations.size() == 0)
    throw "no estimations observed";

  if (estimations.size() != ground_truth.size())
    throw "no of estimations != no of ground truths";

  VectorXd tmp ;
  for (int i=0; i < estimations.size(); ++i) {
    tmp = estimations[i].array() - ground_truth[i].array();
    tmp = tmp.array() * tmp.array();
    rmse += tmp;

  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);
  // recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double epsilon = pow(10, -20);
  VectorXd hx(3);
  try
  {
      double d1 = pow(px,2) + pow(py, 2);
      double d2 = sqrt(d1);
      double d3 = pow(d1, 1.5);

      Hj << (px + epsilon) / (d2 + epsilon) , (py +epsilon) / (d2 + epsilon), 0 , 0,
            (-py + epsilon)/ (d1 + epsilon), (px + epsilon) / (d1 + epsilon), 0, 0,
            (py*(vx*py - vy*px) + epsilon)/ (d3 + epsilon) , 
            (px*(vy*px - vx*py) + epsilon)/ (d3 + epsilon), (px + epsilon)/ (d2 + epsilon), (py + epsilon) / (d2 + epsilon);

      return Hj;

  }
 catch(const char* p)
  {
    cout << p << endl;
    return Hj;
  }
}




double Tools::normalize(double n)
{
    if (n > M_PI)
    {
        while (n > M_PI)
        {
            n = n - (2*M_PI);
        }
    }

    else if (n < -M_PI)
    {
        while (n < M_PI)
        {
            n = n + (2*M_PI);
        }
    }
    return n;
}

