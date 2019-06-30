#ifndef FORESTRYCPP_UTILS_H
#define FORESTRYCPP_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <RcppArmadillo.h>

void print_vector(
  std::vector<size_t> v
);


struct tree_info {
  std::vector< int > var_id;
  // contains the variable id for a splitting node and the negative number of
  // observations in a leaf for a leaf node
  std::vector< long double > split_val;
  // contains the split values for regular nodes
  std::vector< int > leafAveidx;
  // contains the indices of observations in a leaf.
  std::vector< int > leafSplidx;
  // contains the indices of observations in a leaf.
  std::vector< int > averagingSampleIndex;
  // contains the indices of the average set.
  std::vector< int > splittingSampleIndex;
  // contains the indices of the splitting set.
};

struct predict_info{
  // Booleans contain the desired prediction method
  bool isPredict;
  bool isWeightMatrix;
  bool isRidgeRF;
  bool isRFdistance;

  // contains the address of the weight matrix
  arma::Mat<float>* weightMatrix;
  // contains the overfit penalty (lambda)
  float overfitPenalty;
  // contains the power for computing the detachment indices
  float power;
  // contains the column number for the feature in which the rf distances
  // will be computed
  int distanceNumCol;
};
#endif //FORESTRYCPP_UTILS_H
