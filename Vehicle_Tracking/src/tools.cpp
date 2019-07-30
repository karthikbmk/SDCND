#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

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
    /*
    cout << "est " << estimations[i] << endl;
    cout << "truth " << ground_truth[i] << endl;
    cout << "diff " << estimations[i] - ground_truth[i] << endl;
    break;
    */
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
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

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



}
