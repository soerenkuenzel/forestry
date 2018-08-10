// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
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
    arma::Mat<float> weightMatrix;
    if(aggregation == "weightMatrix") {
      size_t nrow = featureData[0].size(); // number of features to be predicted
      size_t ncol = (*testFullForest).getNtrain(); // number of train data
      weightMatrix.resize(nrow, ncol); // initialize the space for the matrix
      weightMatrix.zeros(nrow, ncol);// set it all to 0

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

    //   Print statements for debugging
    // std::cout << "hello\n";
    // std::cout.flush();

    // Return the lis of list. For each tree an element in the first list:
    Rcpp::List list_to_return;
    for(size_t i=0; i!=forest_dta->size(); i++){
      Rcpp::IntegerVector var_id = Rcpp::wrap(((*forest_dta)[i]).var_id);

      // std::cout << "var_id\n";
      // std::cout.flush();

      Rcpp::NumericVector split_val = Rcpp::wrap(((*forest_dta)[i]).split_val);

      // std::cout << "split_val\n";
      // std::cout.flush();


      Rcpp::IntegerVector leafAveidx = Rcpp::wrap(((*forest_dta)[i]).leafAveidx);

      // std::cout << "leafAveidx\n";
      // std::cout.flush();

      Rcpp::IntegerVector leafSplidx = Rcpp::wrap(((*forest_dta)[i]).leafSplidx);

      // std::cout << "leafSplidx\n";
      // std::cout.flush();

      Rcpp::IntegerVector averagingSampleIndex =
	Rcpp::wrap(((*forest_dta)[i]).averagingSampleIndex);

      // std::cout << "averagingSampleIndex\n";
      // std::cout.flush();

      Rcpp::IntegerVector splittingSampleIndex =
	Rcpp::wrap(((*forest_dta)[i]).splittingSampleIndex);

      // std::cout << "splittingSampleIndex\n";
      // std::cout.flush();

      Rcpp::List list_i =
        Rcpp::List::create(
			   Rcpp::Named("var_id") = var_id,
			   Rcpp::Named("split_val") = split_val,
			   Rcpp::Named("leafAveidx") = leafAveidx,
			   Rcpp::Named("leafSplidx") = leafSplidx,
			   Rcpp::Named("averagingSampleIndex") = averagingSampleIndex,
			   Rcpp::Named("splittingSampleIndex") = splittingSampleIndex);

      // std::cout << "finished list\n";
      // std::cout.flush();

      list_to_return.push_back(list_i);
    }

    // std::cout << "hello1\n";
    // std::cout.flush();

    return list_to_return;

  } catch(std::runtime_error const& err) {
    forward_exception_to_r(err);
  } catch(...) {
    ::Rf_error("c++ exception (unknown reason)");
  }
  return NULL;
}

// [[Rcpp::export]]
Rcpp::List rcpp_reconstructree(
  Rcpp::List x,
  Rcpp::NumericVector y,
  Rcpp::NumericVector catCols,
  int numRows,
  int numColumns,
  Rcpp::List R_forest,
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
  bool ridgeRF,
  double overfitPenalty,
  bool doubleTree
){

  // Decode the R_forest data and create appropriate pointers to pointers:
  std::unique_ptr< std::vector< std::vector<int> > > var_ids(
      new std::vector< std::vector<int> >
  );
  std::unique_ptr< std::vector< std::vector<double> > > split_vals(
      new  std::vector< std::vector<double> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > leafAveidxs(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > leafSplidxs(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > averagingSampleIndex(
      new  std::vector< std::vector<size_t> >
  );
  std::unique_ptr< std::vector< std::vector<size_t> > > splittingSampleIndex(
      new  std::vector< std::vector<size_t> >
  );


  for(size_t i=0; i!=R_forest.size(); i++){
    var_ids->push_back(
        Rcpp::as< std::vector<int> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[0])
      );
    split_vals->push_back(
        Rcpp::as< std::vector<double> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[1])
      );
    leafAveidxs->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[2])
      );
    leafSplidxs->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[3])
    );
    averagingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[4])
      );
    splittingSampleIndex->push_back(
        Rcpp::as< std::vector<size_t> > ((Rcpp::as<Rcpp::List>(R_forest[i]))[5])
      );
  }

  // Decode catCols and R_forest
  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.


  std::unique_ptr< std::vector<size_t> > categoricalFeatureColsRcpp_copy(
      new std::vector<size_t>
  );

  for(size_t i=0; i<(*categoricalFeatureColsRcpp).size(); i++){
    (*categoricalFeatureColsRcpp_copy).push_back(
        (*categoricalFeatureColsRcpp)[i]);
  }
  (
      new std::vector<size_t>(
          Rcpp::as< std::vector<size_t> >(catCols)
      )
  ); // contains the col indices of categorical features.

  std::unique_ptr<std::vector< std::vector<float> > > featureDataRcpp (
      new std::vector< std::vector<float> >(
          Rcpp::as< std::vector< std::vector<float> > >(x)
      )
  );

  std::unique_ptr< std::vector<float> > outcomeDataRcpp (
      new std::vector<float>(
          Rcpp::as< std::vector<float> >(y)
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
    (DataFrame*) trainingData,
    (int) 0,
    (bool) replace,
    (int) sampsize,
    (float) splitratio,
    (int) mtry,
    (int) nodesizeSpl,
    (int) nodesizeAvg,
    (int) nodesizeStrictSpl,
    (int) nodesizeStrictAvg,
    (unsigned int) seed,
    (int) nthread,
    (bool) verbose,
    (bool) middleSplit,
    (int) maxObs,
    (bool) ridgeRF,
    (double) overfitPenalty,
    doubleTree
  );

  testFullForest->reconstructTrees(categoricalFeatureColsRcpp_copy,
                                   var_ids,
                                   split_vals,
                                   leafAveidxs,
                                   leafSplidxs,
                                   averagingSampleIndex,
                                   splittingSampleIndex);

  // delete(testFullForest);
  Rcpp::XPtr<forestry> ptr(testFullForest, true);
  R_RegisterCFinalizerEx(
    ptr,
    (R_CFinalizer_t) freeforestry,
    (Rboolean) TRUE
  );
  Rcpp::XPtr<DataFrame> df_ptr(trainingData, true) ;
  return Rcpp::List::create(Rcpp::Named("forest_ptr") = ptr,
                            Rcpp::Named("data_frame_ptr") = df_ptr);
}
