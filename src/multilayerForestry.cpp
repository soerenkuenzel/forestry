#include "multilayerForestry.h"
#include <random>
#include <thread>
#include <mutex>
#include "DataFrame.h"
#include "forestry.h"
#include "utils.h"
#include <RcppArmadillo.h>

multilayerForestry::multilayerForestry():
  _multilayerForests(nullptr), _gammas(0) {}

multilayerForestry::~multilayerForestry() {};

multilayerForestry::multilayerForestry(
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
  bool ridgeRF,
  float overfitPenalty,
  bool doubleTree
){
  this->_trainingData = trainingData;
  this->_ntree = ntree;
  this->_nrounds= nrounds;
  this->_eta = eta;
  this->_replace = replace;
  this->_sampSize = sampSize;
  this->_splitRatio = splitRatio;
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
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

void multilayerForestry::addForests(size_t ntree) {

  // Create vectors to store gradient boosted forests and gamma values
  std::vector< forestry* > multilayerForests(_nrounds);
  std::vector<float> gammas(_nrounds);

  // Calculate initial prediction
  DataFrame *trainingData = getTrainingData();
  std::vector<float> outcomeData = *(trainingData->getOutcomeData());
  float meanOutcome =
    accumulate(outcomeData.begin(), outcomeData.end(), 0.0) / outcomeData.size();
  this->_meanOutcome = meanOutcome;

  std::vector<float> predictedOutcome(trainingData->getNumRows(), meanOutcome);

  // Apply gradient boosting to predict residuals
  std::vector<float> residuals(trainingData->getNumRows());
  for (int i = 0; i < getNrounds(); i++) {
    std::transform(outcomeData.begin(), outcomeData.end(),
                   predictedOutcome.begin(), residuals.begin(), std::minus<float>());

    std::shared_ptr< std::vector< std::vector<float> > > residualFeatureData_(
        new std::vector< std::vector<float> >(*(trainingData->getAllFeatureData()))
    );
    std::unique_ptr< std::vector<float> > residuals_(
        new std::vector<float>(residuals)
    );
    std::unique_ptr< std::vector<size_t> > residualCatCols_(
        new std::vector<size_t>(*(trainingData->getCatCols()))
    );
    std::unique_ptr< std::vector<size_t> > residualSplitCols_(
        new std::vector<size_t>(*(trainingData->getLinCols()))
    );
    std::unique_ptr< std::vector<size_t> > residualLinCols_(
        new std::vector<size_t>(*(trainingData->getLinCols()))
    );

    DataFrame* residualDataFrame = new DataFrame(
      residualFeatureData_,
      std::move(residuals_),
      std::move(residualCatCols_),
      std::move(residualSplitCols_),
      std::move(residualLinCols_),
      trainingData->getNumRows(),
      trainingData->getNumColumns()
    );

    forestry *residualForest = new forestry(
      residualDataFrame,
      _ntree,
      _replace,
      _sampSize,
      _splitRatio,
      _mtry,
      _minNodeSizeSpt,
      _minNodeSizeAvg,
      _minNodeSizeToSplitSpt,
      _minNodeSizeToSplitAvg,
      _minSplitGain,
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

    multilayerForests[i] = residualForest;
    std::unique_ptr< std::vector<float> > predictedResiduals =
      residualForest->predict(getTrainingData()->getAllFeatureData(), NULL);

    // Calculate and store best gamma value
    // std::vector<float> bestPredictedResiduals(trainingData->getNumRows());
    // float minMeanSquaredError = INFINITY;
    // static inline float computeSquare (float x) { return x*x; }

    // for (float gamma = -1; gamma <= 1; gamma += 0.02) {
    //   std::vector<float> gammaPredictedResiduals(trainingData->getNumRows());
    //   std::vector<float> gammaError(trainingData->getNumRows());
    //
    //   // Find gamma with smallest mean squared error
    //   std::transform(predictedResiduals->begin(), predictedResiduals->end(),
    //                  gammaPredictedResiduals.begin(), std::bind1st(std::multiplies<float>(), gamma));
    //   std::transform(predictedOutcome->begin(), predictedOutcome->end(),
    //                  gammaPredictedResiduals.begin(), gammaError.begin(), std::plus<float>());
    //   std::transform(outcomeData->begin(), outcomeData->end(),
    //                  gammaError.begin(), gammaError.begin(), std::minus<float>());
    //   std::transform(gammaError.begin(), gammaError.end(), gammaError.begin(), computeSquare);
    //   float gammaMeanSquaredError =
    //     accumulate(gammaError.begin(), gammaError.end(), 0.0)/gammaError.size();
    //   std::cout << gammaMeanSquaredError << std::endl;
    //
    //   if (gammaMeanSquaredError < minMeanSquaredError) {
    //
    //     gammas[i] = (gamma * eta);
    //     minMeanSquaredError = gammaMeanSquaredError;
    //     bestPredictedResiduals = gammaPredictedResiduals;
    //   }
    // }

    gammas[i] = 1 * _eta;
    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind1st(std::multiplies<float>(), gammas[i]));

    // Update prediction after each round of gradient boosting
    std::transform(predictedOutcome.begin(), predictedOutcome.end(),
                   predictedResiduals->begin(), predictedOutcome.begin(), std::plus<float>());
  }

  // Save vector of forestry objects and gamma values
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
  std::vector< forestry* > multilayerForests = *getMultilayerForests();
  std::vector<float> gammas = getGammas();

  std::unique_ptr< std::vector<float> > initialPrediction =
    multilayerForests[0]->predict(xNew, NULL);

  std::vector<float> prediction(initialPrediction->size(), getMeanOutcome());

  // Use forestry objects and gamma values to make prediction
  for (int i = 0; i < getNrounds(); i ++) {
    std::unique_ptr< std::vector<float> > predictedResiduals =
      multilayerForests[i]->predict(xNew, NULL);

    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind1st(std::multiplies<float>(), gammas[i]));

    std::transform(prediction.begin(), prediction.end(),
                   predictedResiduals->begin(), prediction.begin(), std::plus<float>());
  }

  std::unique_ptr< std::vector<float> > prediction_ (
      new std::vector<float>(prediction)
  );

  return prediction_;
}

