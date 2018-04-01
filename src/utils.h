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
  std::vector< size_t > var_id;
  std::vector< long double > split_val;
};

#endif //FORESTRYCPP_UTILS_H
