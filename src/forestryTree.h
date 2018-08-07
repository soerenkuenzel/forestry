#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "DataFrame.h"
#include "RFNode.h"

class forestryTree {

public:
  forestryTree();
  virtual ~forestryTree();

  forestryTree(
    DataFrame* trainingData,
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty
  );

  // This tree is only for testing purpose
  void setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    float overfitPenalty
  );

  void predict(
    std::vector<float> &outputPrediction,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    arma::Mat<float>* weightMatrix = NULL,
    bool ridgeRF = false
  );

  void recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<float> gTotal,
    arma::Mat<float> sTotal
  );

  void selectBestFeature(
    size_t& bestSplitFeature,
    double& bestSplitValue,
    float& bestSplitLoss,
    arma::Mat<float> &bestSplitGL,
    arma::Mat<float> &bestSplitGR,
    arma::Mat<float> &bestSplitSL,
    arma::Mat<float> &bestSplitSR,
    std::vector<size_t>* featureList,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<float>& gTotal,
    arma::Mat<float>& sTotal
  );

  void initializeRidgeRF(
      DataFrame* trainingData,
      arma::Mat<float>& gTotal,
      arma::Mat<float>& sTotal,
      size_t numLinearFeatures,
      std::vector<size_t>* splitIndexes
  );

  void printTree();

  void trainTiming();

  void getOOBindex(
    std::vector<size_t> &outputOOBIndex,
    size_t nRows
  );

  void getOOBPrediction(
    std::vector<float> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData
  );

  size_t getMtry() {
    return _mtry;
  }

  size_t getMinNodeSizeSpt() {
    return _minNodeSizeSpt;
  }

  size_t getMinNodeSizeAvg() {
    return _minNodeSizeAvg;
  }

  size_t getMinNodeSizeToSplitSpt() {
    return _minNodeSizeToSplitSpt;
  }

  size_t getMinNodeSizeToSplitAvg() {
    return _minNodeSizeToSplitAvg;
  }

  std::vector<size_t>* getSplittingIndex() {
    return _splittingSampleIndex.get();
  }

  std::vector<size_t>* getAveragingIndex() {
    return _averagingSampleIndex.get();
  }

  RFNode* getRoot() {
    return _root.get();
  }

  float getOverfitPenalty() {
    return _overfitPenalty;
  }

  std::vector<double>* getBenchmark() {
    return _benchmark;
  }

private:
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  std::unique_ptr< RFNode > _root;
  float _overfitPenalty;
  std::vector<double>* _benchmark;
};


#endif //HTECPP_RFTREE_H
