#ifndef HTECPP_RFTREE_H
#define HTECPP_RFTREE_H

#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"

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
    size_t maxObs
  );

  // This tree is only for testing purpose
  void setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex
  );

  void predict(
    std::vector<float> &outputPrediction,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    Eigen::MatrixXf* weightMatrix = NULL
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
    bool splitMiddle,
    size_t maxObs
  );

  void selectBestFeature(
    size_t& bestSplitFeature,
    double& bestSplitValue,
    float& bestSplitLoss,
    std::vector<size_t>* featureList,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs
  );

  void printTree();

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

private:
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  std::unique_ptr< RFNode > _root;
};


#endif //HTECPP_RFTREE_H
