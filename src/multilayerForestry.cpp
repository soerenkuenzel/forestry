#include "multilayerForestry.h"
#include <RcppArmadillo.h>
#include <random>
#include <thread>
#include <mutex>
#include "DataFrame.h"
#include "forestry.h"
#include "utils.h"

multilayerForestry::multilayerForestry():
  _multilayerForests(nullptr), _gammas(0) {}

multilayerForestry::~multilayerForestry() {};

multilayerForestry::multilayerForestry(
  DataFrame* trainingData,
  size_t ntree,
  size_t nrounds,
  bool replace,
  size_t sampSize,
  float splitRatio,
  size_t mtry,
  size_t minNodeSizeSpt,
  size_t minNodeSizeAvg,
  size_t minNodeSizeToSplitSpt,
  size_t minNodeSizeToSplitAvg,
  size_t maxDepth,
  unsigned int seed,
  size_t nthread,
  bool verbose,
  bool splitMiddle,
  size_t maxObs,
  bool ridgeRF,
  float overfitPenalty,
  bool doubleTree
){
  this->_trainingData = trainingData;
  this->_ntree = ntree;
  this->_nrounds= nrounds;
  this->_replace = replace;
  this->_sampSize = sampSize;
  this->_splitRatio = splitRatio;
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_maxDepth = maxDepth;
  this->_seed = seed;
  this->_nthread = nthread;
  this->_verbose = verbose;
  this->_splitMiddle = splitMiddle;
  this->_maxObs = maxObs;
  this->_ridgeRF = ridgeRF;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;

  addForests(ntree);
}

static inline float computeSquare (float x) { return x*x; }

void multilayerForestry::addForests(size_t ntree) {
  std::cout << "Num rounds are " << (_nrounds) << std::endl;

  // Create vectors to store gradient boosted forests and gamma values
  std::vector< forestry* > multilayerForests(_nrounds);
  std::vector<float> gammas(_nrounds);
  std::unique_ptr< std::vector<float> > predictedOutcome;
  float eta = 0.3;

  // Save initial forestry object
  forestry *initialForest = new forestry(
      _trainingData,
      _ntree,
      _replace,
      _sampSize,
      _splitRatio,
      _mtry,
      _minNodeSizeSpt,
      _minNodeSizeAvg,
      _minNodeSizeToSplitSpt,
      _minNodeSizeToSplitAvg,
      _maxDepth,
      _seed,
      _nthread,
      _verbose,
      _splitMiddle,
      _maxObs,
      _ridgeRF,
      _overfitPenalty,
      _doubleTree
  );

  multilayerForests.push_back(initialForest);
  gammas.push_back(1);

  // Calculate initial residuals
  DataFrame *trainingData = getTrainingData();
  std::vector<float> *outcomeData = trainingData->getOutcomeData();

  std::vector<float> residuals;
  predictedOutcome =
    initialForest->predict(getTrainingData()->getAllFeatureData(), NULL);
  std::transform(outcomeData->begin(), outcomeData->end(),
                 predictedOutcome->begin(), residuals.begin(), std::minus<float>());

  // Apply gradient boosting using forestry to predict residuals
  for (int i = 1; i < getNrounds(); i++) {
    trainingData->setOutcomeData(residuals);
    forestry *residualForest = new forestry(
      _trainingData,
      _ntree,
      _replace,
      _sampSize,
      _splitRatio,
      _mtry,
      _minNodeSizeSpt,
      _minNodeSizeAvg,
      _minNodeSizeToSplitSpt,
      _minNodeSizeToSplitAvg,
      _maxDepth,
      _seed,
      _nthread,
      _verbose,
      _splitMiddle,
      _maxObs,
      _ridgeRF,
      _overfitPenalty,
      _doubleTree
    );
    multilayerForests.push_back(residualForest);
    std::unique_ptr< std::vector<float> > predictedResiduals =
      residualForest->predict(getTrainingData()->getAllFeatureData(), NULL);

    // Calculate and store best gamma value
    std::vector<float> gammaPredictedResiduals;

    std::vector<float> bestPredictedResiduals;
    std::vector<float> gammaError;
    float minMeanSquaredError = INFINITY;

    for (float gamma = -1; gamma <= 1; gamma += 0.02) {

      // Find gamma with smallest mean squared error
      std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                     gammaPredictedResiduals.begin(), std::bind1st(std::multiplies<float>(), gamma));
      std::transform(predictedOutcome->begin(), predictedOutcome->end(),
                     gammaPredictedResiduals.begin(), gammaError.begin(), std::plus<float>());
      std::transform(outcomeData->begin(), outcomeData->end(),
                     gammaError.begin(), gammaError.begin(), std::minus<float>());
      std::transform(gammaError.begin(), gammaError.end(), gammaError.begin(), computeSquare);
      float gammaMeanSquaredError =
        accumulate(gammaError.begin(), gammaError.end(), 0.0)/gammaError.size();

      if (gammaMeanSquaredError < minMeanSquaredError) {
        gammas.push_back(gamma * eta);
        minMeanSquaredError = gammaMeanSquaredError;
        bestPredictedResiduals = gammaPredictedResiduals;
      }
    }

    // Update prediction after each round of gradient boosting
    std::transform(predictedOutcome->begin(), predictedOutcome->end(),
                   bestPredictedResiduals.begin(), predictedOutcome->begin(), std::plus<float>());
  }

  // Save vector of forestry objects and gamma values
  trainingData->setOutcomeData(*outcomeData);

  std::unique_ptr<std::vector< forestry* > > multilayerForests_(
    new std::vector< forestry* >(multilayerForests)
  );
  this->_multilayerForests = std::move(multilayerForests_);

  this->_gammas = std::move(gammas);
}

std::unique_ptr< std::vector<float> > multilayerForestry::predict(
    std::vector< std::vector<float> >* xNew,
    arma::Mat<float>* weightMatrix
) {
  std::unique_ptr< std::vector<float> > prediction;
  std::vector< forestry* > multilayerForests = *getMultilayerForests();
  std::vector<float> gammas = getGammas();

  // Use forestry objects and gamma values to make prediction
  prediction = multilayerForests[0]->predict(xNew, weightMatrix);
  for (int i = 1; i < getNrounds(); i ++) {
    std::unique_ptr< std::vector<float> > predictedResiduals =
      multilayerForests[i]->predict(xNew, weightMatrix);
    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind1st(std::multiplies<float>(), gammas[i]));
    std::transform(prediction->begin(), prediction->end(),
                   predictedResiduals->begin(), prediction->begin(), std::plus<float>());
  }

  return prediction;
}

