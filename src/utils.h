#ifndef FORESTRYCPP_UTILS_H
#define FORESTRYCPP_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>

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
  bool isPredict;
  bool isWeightMatrix;
  bool isRidgeRF;
  bool isRFdistance;
  // Booleans contain the desired prediction method

  arma::Mat<float>* weightMatrix;
  // contains the address of the weight matrix
  float overfitPenalty;
  // contains the overfit penalty (lambda)
  float power;
  // contains the power for the rf distances
  int distanceNumCol;
  // contains the column number for the feature in which the rf distances
  // will be computed
};
#endif //FORESTRYCPP_UTILS_H
