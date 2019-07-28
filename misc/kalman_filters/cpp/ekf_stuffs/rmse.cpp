#include <iostream>
#include <vector>
#include "eigen3/Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth);

int main() {
  /**
   * Compute RMSE
   */
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  // the input list of estimations
  VectorXd e(4);
  e << 1, 1, 0.2, 0.1;
  estimations.push_back(e);
  e << 2, 2, 0.3, 0.2;
  estimations.push_back(e);
  e << 3, 3, 0.4, 0.3;
  estimations.push_back(e);

  // the corresponding list of ground truth values
  VectorXd g(4);
  g << 1.1, 1.1, 0.3, 0.2;
  ground_truth.push_back(g);
  g << 2.1, 2.1, 0.4, 0.3;
  ground_truth.push_back(g);
  g << 3.1, 3.1, 0.5, 0.4;
  ground_truth.push_back(g);

  // call the CalculateRMSE and print out the result
  cout << CalculateRMSE(estimations, ground_truth) << endl;

  return 0;
}

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // TODO: YOUR CODE HERE
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if ( estimations.size() == 0)
    throw "no estimations observed";
  //  * the estimation vector size should equal ground truth vector size

  if (estimations.size() != ground_truth.size())
    throw "no of estimations != no of ground truths";

  // TODO: accumulate squared residual
  VectorXd tmp ;
  for (int i=0; i < estimations.size(); ++i) {
    // ... your code here
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

  // TODO: calculate the mean

  // TODO: calculate the squared root

  // return the result
  return rmse;
}

