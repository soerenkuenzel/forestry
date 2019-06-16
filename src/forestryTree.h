#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>

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
    float minSplitGain,
    size_t maxDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    float overfitPenalty
  );

  // This tree is only for testing purpose
  void setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    float minSplitGain,
    size_t maxDepth,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    float overfitPenalty
  );

  void predict(
    std::vector<float> &outputPrediction,
    std::vector< std::vector<float> > &outputCoefficients,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    arma::Mat<float>* weightMatrix = NULL,
    bool linear = false
  );

  std::unique_ptr<tree_info> getTreeInfo(
      DataFrame* trainingData
  );

  void reconstruct_tree(
      size_t mtry,
      size_t minNodeSizeSpt,
      size_t minNodeSizeAvg,
      size_t minNodeSizeToSplitSpt,
      size_t minNodeSizeToSplitAvg,
      float minSplitGain,
      size_t maxDepth,
      bool linear,
      float overfitPenalty,
      std::vector<size_t> categoricalFeatureColsRcpp,
      std::vector<int> var_ids,
      std::vector<double> split_vals,
      std::vector<size_t> leafAveidxs,
      std::vector<size_t> leafSplidxs,
      std::vector<size_t> averagingSampleIndex,
      std::vector<size_t> splittingSampleIndex);

  void recursive_reconstruction(
      RFNode* currentNode,
      std::vector<int> * var_ids,
      std::vector<double> * split_vals,
      std::vector<size_t> * leafAveidxs,
      std::vector<size_t> * leafSplidxs
  );

  void recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    size_t depth,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<double> gTotal,
    arma::Mat<double> sTotal
  );

  void selectBestFeature(
    size_t& bestSplitFeature,
    double& bestSplitValue,
    float& bestSplitLoss,
    arma::Mat<double> &bestSplitGL,
    arma::Mat<double> &bestSplitGR,
    arma::Mat<double> &bestSplitSL,
    arma::Mat<double> &bestSplitSR,
    std::vector<size_t>* featureList,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<double>& gTotal,
    arma::Mat<double>& sTotal
  );

  void initializelinear(
      DataFrame* trainingData,
      arma::Mat<double>& gTotal,
      arma::Mat<double>& sTotal,
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

  void getShuffledOOBPrediction(
      std::vector<float> &outputOOBPrediction,
      std::vector<size_t> &outputOOBCount,
      DataFrame* trainingData,
      size_t shuffleFeature
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

  float getMinSplitGain() {
    return _minSplitGain;
  }

  size_t getMaxDepth() {
    return _maxDepth;
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
  float _minSplitGain;
  size_t _maxDepth;
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  std::unique_ptr< RFNode > _root;
  bool _linear;
  float _overfitPenalty;
  std::vector<double>* _benchmark;
};


#endif //HTECPP_RFTREE_H
