#include <Rcpp.h>
#include <vector>
#include <string>
#include <iostream>

void print_vector(
  std::vector<size_t> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
  }
  Rcpp::Rcout << std::endl;
}

int add_vector(
  std::vector<int>* v
) {
  int sum=0;
  for (size_t i = 0; i < v->size(); i++) {
    sum += (*v)[i];
  }
  return sum;
}
