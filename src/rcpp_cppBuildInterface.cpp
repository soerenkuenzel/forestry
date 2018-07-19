// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "DataFrame.h"
#include "forestryTree.h"
#include "RFNode.h"
#include "forestry.h"

void freeforestry(
  SEXP ptr
){
  if (NULL == R_ExternalPtrAddr(ptr))
    return;
  forestry* pm = (forestry*)(R_ExternalPtrAddr(ptr));
  delete(pm);
  R_ClearExternalPtr(ptr);
}

// [[Rcpp::export]]
SEXP rcpp_cppDataFrameInterface(
    Rcpp::List x,
    Rcpp::NumericVector y,
    Rcpp::NumericVector catCols,
    int numRows,
    int numColumns
){

  try {
    std::unique_ptr<std::vector< std::vector<float> > > featureDataRcpp (
        new std::vector< std::vector<float> >(
            Rcpp::as< std::vector< std::vector<float> > >(x)
        )
    );

    std::unique_ptr<std::vector<float>> outcomeDataRcpp (
        new std::vector<float>(
            Rcpp::as< std::vector<float> >(y)
        )
    );

    std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
        new std::vector<size_t>(
            Rcpp::as< std::vector<size_t> >(catCols)
        )
    );

    DataFrame* trainingData = new DataFrame(
        std::move(featureDataRcpp),
        std::move(outcomeDataRcpp),
        std::move(categoricalFeatureColsRcpp),
        (size_t) numRows,
        (size_t) numColumns
    );

    Rcpp::XPtr<DataFrame> ptr(trainingData, true) ;
    return ptr;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return NULL;
}


// [[Rcpp::export]]
SEXP rcpp_cppBuildInterface(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  int numRows,
  int numColumns,
  int ntree,
  bool replace,
  int sampsize,
  int mtry,
  float splitratio,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  bool ridgeRF,
  double overfitPenalty,
  bool doubleTree,
  bool existing_dataframe_flag,
  SEXP existing_dataframe
){

  if (existing_dataframe_flag) {

    try {
      Rcpp::XPtr< DataFrame > trainingData(existing_dataframe) ;

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        ridgeRF,
        (float) overfitPenalty,
        doubleTree
      );

      // delete(testFullForest);
      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;
    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }

  } else {

    try {
      std::unique_ptr<std::vector< std::vector<float> > > featureDataRcpp (
          new std::vector< std::vector<float> >(
              Rcpp::as< std::vector< std::vector<float> > >(x)
          )
      );

      std::unique_ptr<std::vector<float>> outcomeDataRcpp (
          new std::vector<float>(
              Rcpp::as< std::vector<float> >(y)
          )
      );

      std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
          new std::vector<size_t>(
              Rcpp::as< std::vector<size_t> >(catCols)
          )
      );

      DataFrame* trainingData = new DataFrame(
          std::move(featureDataRcpp),
          std::move(outcomeDataRcpp),
          std::move(categoricalFeatureColsRcpp),
          (size_t) numRows,
          (size_t) numColumns
      );

      forestry* testFullForest = new forestry(
        trainingData,
        (size_t) ntree,
        replace,
        (size_t) sampsize,
        splitratio,
        (size_t) mtry,
        (size_t) nodesizeSpl,
        (size_t) nodesizeAvg,
        (size_t) nodesizeStrictSpl,
        (size_t) nodesizeStrictAvg,
        (unsigned int) seed,
        (size_t) nthread,
        verbose,
        middleSplit,
        (size_t) maxObs,
        ridgeRF,
        (float) overfitPenalty,
        doubleTree
      );

      // delete(testFullForest);
      Rcpp::XPtr<forestry> ptr(testFullForest, true) ;
      R_RegisterCFinalizerEx(
        ptr,
        (R_CFinalizer_t) freeforestry,
        (Rboolean) TRUE
      );
      return ptr;

    } catch(std::runtime_error const& err) {
      forward_exception_to_r(err);
    } catch(...) {
      ::Rf_error("c++ exception (unknown reason)");
    }
  }
  return NULL;
}

// [[Rcpp::export]]
Rcpp::List rcpp_cppPredictInterface(
  SEXP forest,
  Rcpp::List x,
  std::string aggregation
){
  try {

    Rcpp::XPtr< forestry > testFullForest(forest) ;

    std::vector< std::vector<float> > featureData =
      Rcpp::as< std::vector< std::vector<float> > >(x);

    std::unique_ptr< std::vector<float> > testForestPrediction;
    // We always initialize the weightMatrix. If the aggregation is weightMatrix
    // then we inialize the empty weight matrix
    Eigen::MatrixXf weightMatrix;
    if(aggregation == "weightMatrix") {
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testFullForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix = Eigen::MatrixXf::Zero(nrow, ncol); // set it all to 0

      // The idea is that, if the weightMatrix is point to NULL it won't be
      // be updated, but otherwise it will be updated:
      testForestPrediction = (*testFullForest).predict(&featureData, &weightMatrix);
    } else {
      testForestPrediction = (*testFullForest).predict(&featureData, NULL);
    }

    std::vector<float>* testForestPrediction_ =
      new std::vector<float>(*testForestPrediction.get());

    Rcpp::NumericVector predictions = Rcpp::wrap(*testForestPrediction_);

    return Rcpp::List::create(Rcpp::Named("predictions") = predictions,
                              Rcpp::Named("weightMatrix") = weightMatrix);

    // return output;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return NULL;
}


// [[Rcpp::export]]
float rcpp_OBBPredictInterface(
    SEXP forest
){

  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    float OOBError = (*testFullForest).getOOBError();
    return OOBError;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na() ;
}


// [[Rcpp::export]]
float rcpp_getObservationSizeInterface(
    SEXP df
){

  try {
    Rcpp::XPtr< DataFrame > trainingData(df) ;
    float nrows = (float) (*trainingData).getNumRows();
    return nrows;
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return Rcpp::NumericVector::get_na() ;
}


// [[Rcpp::export]]
void rcpp_AddTreeInterface(
    SEXP forest,
    int ntree
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    (*testFullForest).addTrees(ntree);
  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
}
