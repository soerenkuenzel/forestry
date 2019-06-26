#ifndef FORESTRYCPP_MULTILAYERRF_H
#define FORESTRYCPP_MULTILAYERRF_H

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include "DataFrame.h"
#include "forestry.h"
#include "forestryTree.h"
#include "utils.h"
#include <RcppArmadillo.h>

class multilayerForestry {

public:
  multilayerForestry();
  virtual ~multilayerForestry();

  multilayerForestry(
    DataFrame* trainingData,
    size_t ntree,
    size_t nrounds,
    float eta,
    bool replace,
    size_t sampSize,
    float splitRatio,
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    float minSplitGain,
    size_t maxDepth,
    unsigned int seed,
    size_t nthread,
    bool verbose,
    bool splitMiddle,
    size_t maxObs,
    bool linear,
    float overfitPenalty,
    bool doubleTree
  );

  void addForests(size_t ntree);

  std::unique_ptr< std::vector<float> > predict(
      std::vector< std::vector<float> >* xNew,
      arma::Mat<float>* weightMatrix
  );

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

  float getMinSplitGain() {
    return _minSplitGain;
  }

  size_t getMaxDepth() {
    return _maxDepth;
  }

  size_t getNtree() {
    return _ntree;
  }

  size_t getNrounds() {
    return _nrounds;
  }

  size_t getEta() {
    return _eta;
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
    return _maxObs;
  }

  bool getlinear() {
    return _linear;
  }

  float getOverfitPenalty() {
    return _overfitPenalty;
  }


  bool isDoubleTree() {
    return _doubleTree;
  }

  std::vector< forestry* >* getMultilayerForests() {
    return _multilayerForests.get();
  }

  std::vector<float> getGammas() {
    return _gammas;
  }

  float getMeanOutcome() {
    return _meanOutcome;
  }


private:
  DataFrame* _trainingData;
  size_t _ntree;
  size_t _nrounds;
  float _eta;
  bool _replace;
  size_t _sampSize;
  float _splitRatio;
  size_t _mtry;
  size_t _minNodeSizeSpt;
  size_t _minNodeSizeAvg;
  size_t _minNodeSizeToSplitSpt;
  size_t _minNodeSizeToSplitAvg;
  float _minSplitGain;
  size_t _maxDepth;
  unsigned int _seed;
  bool _verbose;
  size_t _nthread;
  std::unique_ptr< std::vector<float> > _variableImportance;
  bool _splitMiddle;
  size_t _maxObs;
  bool _linear;
  float _overfitPenalty;
  bool _doubleTree;

  std::unique_ptr<std::vector< forestry* > > _multilayerForests;
  std::vector<float> _gammas;
  float _meanOutcome;
};


#endif //FORESTRYCPP_MULTILAYERRF_H
