// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include "DataFrame.h"
#include "forestryTree.h"
#include "RFNode.h"
#include "forestry.h"
#include "utils.h"

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
      std::unique_ptr< std::vector< std::vector<float> > > featureDataRcpp (
          new std::vector< std::vector<float> >(
              Rcpp::as< std::vector< std::vector<float> > >(x)
          )
      );

      std::unique_ptr< std::vector<float> > outcomeDataRcpp (
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

// [[Rcpp::export]]
Rcpp::List rcpp_CppToR_translator(
    SEXP forest
){
  try {
    Rcpp::XPtr< forestry > testFullForest(forest) ;
    std::unique_ptr< std::vector<tree_info> > forest_dta(
      new std::vector<tree_info>
    );
    (*testFullForest).fillinTreeInfo(forest_dta);

    // Return the lis of list. For each tree an element in the first list:
    Rcpp::List list_to_return;
    for(size_t i=0; i!=forest_dta->size(); i++){
      Rcpp::NumericVector var_id = Rcpp::wrap(((*forest_dta)[i]).var_id);
      Rcpp::NumericVector split_val = Rcpp::wrap(((*forest_dta)[i]).split_val);
      Rcpp::NumericVector leaf_idx = Rcpp::wrap(((*forest_dta)[i]).leaf_idx);
      Rcpp::NumericVector averagingSampleIndex =
        Rcpp::wrap(((*forest_dta)[i]).averagingSampleIndex);
      Rcpp::NumericVector splittingSampleIndex =
        Rcpp::wrap(((*forest_dta)[i]).splittingSampleIndex);



      Rcpp::List list_i =
        Rcpp::List::create(
          Rcpp::Named("var_id") = var_id,
          Rcpp::Named("split_val") = split_val,
          Rcpp::Named("leaf_idx") = leaf_idx,
          Rcpp::Named("averagingSampleIndex") = averagingSampleIndex,
          Rcpp::Named("splittingSampleIndex") = splittingSampleIndex);

      list_to_return.push_back(list_i);
    }

    return list_to_return;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
}


////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
SEXP rcpp_reconstructree(
  SEXP dataframe,
  Rcpp::NumericVector catCols,
  Rcpp::List forest_R,
  bool replace,
  int sampsize,
  float splitratio,
  int mtry,
  int nodesizeSpl,
  int nodesizeAvg,
  int nodesizeStrictSpl,
  int nodesizeStrictAvg,
  int seed,
  int nthread,
  bool verbose,
  bool middleSplit,
  int maxObs,
  bool doubleTree
){
  // Decode catCols and forest_R
  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.

  // Decode the forest_R data and create appropriate pointers to pointers:
  std::unique_ptr< std::vector< std::vector<int> > > var_ids(
      new std::vector< std::vector<int> >
  );
  std::unique_ptr< std::vector< std::vector<double> > > split_vals(
      new  std::vector< std::vector<double> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > leaf_idxs(
      new  std::vector< std::vector<size_t> >
  );


  for(size_t i=0; i!=forest_R.size(); i++){
    var_ids->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(forest_R[i]))[0])
    );
    split_vals->push_back(
        Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(forest_R[i]))[1])
      );
    leaf_idxs->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(forest_R[i]))[2])
      );
  }

  // Setting up an empytforest with all parameters but without any trees as the
  // trees will be constructed from the R data set.
  forestry* testFullForest = new forestry(
    (DataFrame*) dataframe,
    (size_t) 0,
    (bool) replace,
    (size_t) sampsize,
    (float) splitratio,
    (size_t) mtry,
    (size_t) nodesizeSpl,
    (size_t) nodesizeAvg,
    (size_t) nodesizeStrictSpl,
    (size_t) nodesizeStrictAvg,
    (unsigned int) seed,
    (size_t) nthread,
    (bool) verbose,
    (bool) middleSplit,
    (size_t) maxObs,
    (bool) doubleTree
  );

  testFullForest->reconstructTrees(categoricalFeatureColsRcpp,
                                   var_ids,
                                   split_vals,
                                   leaf_idxs);


  //////////////////////////////////////////////////////////////////////////////

  // delete(testFullForest);
  Rcpp::XPtr<forestry> ptr(testFullForest, true);
  R_RegisterCFinalizerEx(
    ptr,
    (R_CFinalizer_t) freeforestry,
    (Rboolean) TRUE
  );
  return ptr;
}
