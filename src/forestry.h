#ifndef HTECPP_RF_H
#define HTECPP_RF_H

#include "forestryTree.h"
#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <string>
#include "DataFrame.h"


class forestry {

public:
  forestry();
  virtual ~forestry();

  forestry(
    DataFrame* trainingData,
    size_t ntree,
    bool replace,
    size_t sampSize,
    float splitRatio,
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    unsigned int seed,
    size_t nthread,
    bool verbose,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    bool doubleTree
  );

  std::unique_ptr< std::vector<float> > predict(
    std::vector< std::vector<float> >* xNew,
    arma::Mat<float>* weightMatrix
  );

  void calculateOOBError();

  float getOOBError() {
    calculateOOBError();
    return _OOBError;
  }

  void addTrees(size_t ntree);

  DataFrame* getTrainingData() {
    return _trainingData;
  }

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

  size_t getNtree() {
    return _ntree;
  }

  size_t getNtrain(){
    return (*_trainingData).getNumRows();
  }

  size_t getSampleSize() {
    // This is the sample size used for each tree in the bootstrap not ntrain
    return _sampSize;
  }

  float getSplitRatio() {
    return _splitRatio;
  }

  bool isReplacement() {
    return _replace;
  }

  unsigned int getSeed() {
    return _seed;
  }

  std::vector< std::unique_ptr< forestryTree > >* getForest() {
    return _forest.get();
  }

  bool isVerbose() {
    return _verbose;
  }

  size_t getNthread(){
    return _nthread;
  }

  bool getSplitMiddle(){
    return _splitMiddle;
  }

  size_t getMaxObs() {
    return maxObs;
  }

  bool getRidgeRF() {
    return _ridgeRF;
  }

  float getOverfitPenalty() {
    return _overfitPenalty;
  }

private:
  DataFrame* _trainingData;
  size_t _ntree;
  bool _replace;
  size_t _sampSize;
  float _splitRatio;
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  std::unique_ptr< std::vector< std::unique_ptr< forestryTree > > > _forest;
  unsigned int _seed;
  bool _verbose;
  size_t _nthread;
  float _OOBError;
  bool _splitMiddle;
  size_t maxObs;
  bool _ridgeRF;
  float _overfitPenalty;
  bool _doubleTree;
};

#endif //HTECPP_RF_H
