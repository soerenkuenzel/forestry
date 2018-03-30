// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
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
Rcpp::NumericVector rcpp_cppPredictInterface(
  SEXP forest,
  Rcpp::List x,
  std::string aggregation
){
  try {

    Rcpp::XPtr< forestry > testFullForest(forest) ;

    std::vector< std::vector<float> > featureData =
      Rcpp::as< std::vector< std::vector<float> > >(x);

    std::unique_ptr< std::vector<float> > testForestPrediction (
      (*testFullForest).predict(&featureData, aggregation)
    );

    std::vector<float>* testForestPrediction_ =
      new std::vector<float>(*testForestPrediction.get());

    Rcpp::NumericVector output = Rcpp::wrap(*testForestPrediction_);

    return output;

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
