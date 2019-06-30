#include "forestry.h"
#include "utils.h"
#include <random>
#include <thread>
#include <mutex>
#include <RcppArmadillo.h>
#define DOPARELLEL true

forestry::forestry():
  _trainingData(nullptr), _ntree(0), _replace(0), _sampSize(0),
  _splitRatio(0), _mtry(0), _minNodeSizeSpt(0), _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0), _minNodeSizeToSplitAvg(0), _minSplitGain(0),
  _maxDepth(0), _forest(nullptr), _seed(0), _verbose(0), _nthread(0),
  _OOBError(0), _splitMiddle(0), _doubleTree(0){};

forestry::~forestry(){
//  for (std::vector<forestryTree*>::iterator it = (*_forest).begin();
//       it != (*_forest).end();
//       ++it) {
//    delete(*it);
//  }
//  std::cout << "forestry() destructor is called." << std::endl;
};

forestry::forestry(
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
){
  this->_trainingData = trainingData;
  this->_ntree = 0;
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
  this->_linear = linear;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;

  if (splitRatio > 1 || splitRatio < 0) {
    throw std::runtime_error("splitRatio shoule be between 0 and 1.");
  }

  size_t splitSampleSize = (size_t) (getSplitRatio() * sampSize);
  size_t averageSampleSize;
  if (splitRatio == 1 || splitRatio == 0) {
    averageSampleSize = splitSampleSize;
  } else {
    averageSampleSize = sampSize - splitSampleSize;
  }

  if (
    splitSampleSize < minNodeSizeToSplitSpt ||
    averageSampleSize < minNodeSizeToSplitAvg
  ) {
    throw std::runtime_error("splitRatio too big or too small.");
  }

  if (
    overfitPenalty < 0
  ) {
    throw std::runtime_error("overfitPenalty cannot be negative");
  }


  std::unique_ptr< std::vector< std::unique_ptr< forestryTree > > > forest (
    new std::vector< std::unique_ptr< forestryTree > >
  );
  this->_forest = std::move(forest);

  // Create initial trees
  addTrees(ntree);
}

void forestry::addTrees(size_t ntree) {

  int newStartingTreeNumber = (int) getNtree();
  int newEndingTreeNumber = newStartingTreeNumber + (int) ntree;

  Rcpp::checkUserInterrupt();

  size_t nthreadToUse = getNthread();
  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }

  size_t splitSampleSize = (size_t) (getSplitRatio() * getSampleSize());

  #if DOPARELLEL
  if (isVerbose()) {
    Rcpp::Rcout << "Training parallel using " << nthreadToUse << " threads"
              << std::endl;
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (size_t t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const int iStart, const int iEnd, const int t_) {

        // loop over al assigned trees, iStart is the starting tree number
        // and iEnd is the ending tree number
        for (int i = iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for (int i=newStartingTreeNumber; i<newEndingTreeNumber; i++) {
  #endif
          unsigned int myseed = getSeed() * (i + 1);
          std::mt19937_64 random_number_generator;
          random_number_generator.seed(myseed);

          // Generate a sample index for each tree
          std::vector<size_t> sampleIndex;

          if (isReplacement()) {
            std::uniform_int_distribution<size_t> unif_dist(
              0, (size_t) (*getTrainingData()).getNumRows() - 1
            );
            // Generate index with replacement
            while (sampleIndex.size() < getSampleSize()) {
              size_t randomIndex = unif_dist(random_number_generator);
              sampleIndex.push_back(randomIndex);
            }
          } else {
            std::uniform_int_distribution<size_t> unif_dist(
              0, (size_t) (*getTrainingData()).getNumRows() - 1
            );
            // Generate index without replacement
            while (sampleIndex.size() < getSampleSize()) {
              size_t randomIndex = unif_dist(random_number_generator);

              if (
                sampleIndex.size() == 0 ||
                std::find(
                  sampleIndex.begin(),
                  sampleIndex.end(),
                  randomIndex
                ) == sampleIndex.end()
              ) {
                sampleIndex.push_back(randomIndex);
              }
            }
          }

          std::unique_ptr<std::vector<size_t> > splitSampleIndex;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex;

          std::unique_ptr<std::vector<size_t> > splitSampleIndex2;
          std::unique_ptr<std::vector<size_t> > averageSampleIndex2;

          if (getSplitRatio() == 1 || getSplitRatio() == 0) {

            // Treat it as normal RF
            splitSampleIndex.reset(new std::vector<size_t>(sampleIndex));
            averageSampleIndex.reset(new std::vector<size_t>(sampleIndex));

          } else {

            // Generate sample index based on the split ratio
            std::vector<size_t> splitSampleIndex_;
            std::vector<size_t> averageSampleIndex_;
            for (
              std::vector<size_t>::iterator it = sampleIndex.begin();
              it != sampleIndex.end();
              ++it
            ) {
              if (splitSampleIndex_.size() < splitSampleSize) {
                splitSampleIndex_.push_back(*it);
              } else {
                averageSampleIndex_.push_back(*it);
              }
            }

            splitSampleIndex.reset(
              new std::vector<size_t>(splitSampleIndex_)
            );
            averageSampleIndex.reset(
              new std::vector<size_t>(averageSampleIndex_)
            );

            if (_doubleTree) {
              splitSampleIndex2.reset(
                new std::vector<size_t>(splitSampleIndex_)
              );
              averageSampleIndex2.reset(
                new std::vector<size_t>(averageSampleIndex_)
              );
            }
          }

          try{
            forestryTree *oneTree(
              new forestryTree(
                getTrainingData(),
                getMtry(),
                getMinNodeSizeSpt(),
                getMinNodeSizeAvg(),
                getMinNodeSizeToSplitSpt(),
                getMinNodeSizeToSplitAvg(),
                getMinSplitGain(),
                getMaxDepth(),
                std::move(splitSampleIndex),
                std::move(averageSampleIndex),
                random_number_generator,
                getSplitMiddle(),
                getMaxObs(),
                getlinear(),
                getOverfitPenalty()
              )
            );

            forestryTree *anotherTree;
            if (_doubleTree) {
              anotherTree =
                new forestryTree(
                    getTrainingData(),
                    getMtry(),
                    getMinNodeSizeSpt(),
                    getMinNodeSizeAvg(),
                    getMinNodeSizeToSplitSpt(),
                    getMinNodeSizeToSplitAvg(),
                    getMinSplitGain(),
                    getMaxDepth(),
                    std::move(averageSampleIndex2),
                    std::move(splitSampleIndex2),
                    random_number_generator,
                    getSplitMiddle(),
                    getMaxObs(),
                    getlinear(),
                    getOverfitPenalty()
                 );
            }

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            if (isVerbose()) {
              std::cout << "Finish training tree # " << (i + 1) << std::endl;
            }
            (*getForest()).emplace_back(oneTree);
            _ntree = _ntree + 1;
            if (_doubleTree) {
              (*getForest()).emplace_back(anotherTree);
              _ntree = _ntree + 1;
            } else {
              // delete anotherTree;
            }

          } catch (std::runtime_error &err) {
            // Rcpp::Rcerr << err.what() << std::endl;
          }

        }
  #if DOPARELLEL
      },
      newStartingTreeNumber + t * ntree / nthreadToUse,
      (t + 1) == nthreadToUse ?
        (size_t) newEndingTreeNumber :
        newStartingTreeNumber + (t + 1) * ntree / nthreadToUse,
      t
    );

    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x){x.join();}
  );
  #endif
}

std::unique_ptr< std::vector<float> > forestry::predict(
  std::vector< std::vector<float> >* xNew,
  arma::Mat<float>* localVIMatrix,
  arma::Mat<float>* coefficients,
  predict_info predictInfo
){
  // Update isRidgeRF in predictInfo
  predictInfo.isRidgeRF = getlinear();

  std::vector<float> prediction;
  size_t numObservations = (*xNew)[0].size();
  for (size_t j=0; j<numObservations; j++) {
    prediction.push_back(0);
  }

  if (coefficients) {
    // Create coefficient vector of vectors of zeros
    std::vector< std::vector<float> > coef;
    size_t numObservations = (*xNew)[0].size();
    size_t numCol = (*coefficients).n_cols;
    for (size_t i=0; i<numObservations; i++) {
      std::vector<float> row;
      for (size_t j = 0; j<numCol; j++) {
        row.push_back(0);
      }
      coef.push_back(row);
    }
  }

  #if DOPARELLEL
  size_t nthreadToUse = getNthread();
  if (getNthread() == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }

  if (isVerbose()) {
    Rcpp::Rcout << "Prediction parallel using " << nthreadToUse << " threads"
              << std::endl;
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (size_t t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const int iStart, const int iEnd, const int t_) {

        // loop over al assigned trees, iStart is the starting tree number
        // and iEnd is the ending tree number
        for (int i=iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for(int i=0; i<((int) getNtree()); i++ ) {
  #endif
          try {
            std::vector<float> currentTreePrediction(numObservations);
            std::vector< std::vector<float> > currentTreeCoefficients(numObservations);
            forestryTree *currentTree = (*getForest())[i].get();

            //Return coefficients and predictions
            if (coefficients) {
              for (size_t l=0; l<numObservations; l++) {
                currentTreeCoefficients[l] = std::vector<float>(coefficients->n_cols);
              }
            }

            (*currentTree).predict(
                currentTreePrediction,
                currentTreeCoefficients,
                xNew,
                getTrainingData(),
                predictInfo
            );

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            # endif

            for (size_t j = 0; j < numObservations; j++) {
              prediction[j] += currentTreePrediction[j];
            }

            if (coefficients) {
              for (size_t k = 0; k < numObservations; k++) {
                for (size_t l = 0; l < coefficients->n_cols; l++) {
                  (*coefficients)(k,l) += currentTreeCoefficients[k][l];
                }
              }
            }

          } catch (std::runtime_error &err) {
            Rcpp::Rcerr << err.what() << std::endl;
          }
      }
  #if DOPARELLEL
      },
      t * getNtree() / nthreadToUse,
      (t + 1) == nthreadToUse ?
        getNtree() :
        (t + 1) * getNtree() / nthreadToUse,
      t
    );
    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x) { x.join(); }
  );
  #endif

  for (size_t j=0; j<numObservations; j++){
    prediction[j] /= getNtree();
    if(predictInfo.isRFdistance){
      prediction[j] = pow(prediction[j], 1 / predictInfo.power);
    }
  }

  // Average coefficients across number of trees after aggregating
  if (coefficients) {
    for (size_t k = 0; k < numObservations; k++) {
      for (size_t l = 0; l < coefficients->n_cols; l++) {
        (*coefficients)(k,l) /= getNtree();;
      }
    }
  }

  std::unique_ptr< std::vector<float> > prediction_ (
    new std::vector<float>(prediction)
  );

  // If we also update the weight matrix, we now have to divide every entry
  // by the number of trees:
  if (predictInfo.isWeightMatrix) {
    size_t nrow = (*xNew)[0].size(); // number of features to be predicted
    size_t ncol = getNtrain(); // number of train data
    for ( size_t i = 0; i < nrow; i++){
      for (size_t j = 0; j < ncol; j++){
        (*predictInfo.weightMatrix)(i,j) = (*predictInfo.weightMatrix)(i,j) / _ntree;
      }
    }
    if (localVIMatrix) {
      calculateLocalVariableImportance(
        xNew,
        localVIMatrix,
        prediction,
        predictInfo
      );
    }
  }



  return prediction_;
}

void forestry::calculateVariableImportance() {
  // For all variables, shuffle + get OOB Error, record in

  size_t numObservations = getTrainingData()->getNumRows();
  std::vector<float> variableImportances;

  std::vector<float> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

  //Loop through all features and populate variableImportances with shuffled OOB
  for (size_t featNum = 0; featNum < getTrainingData()->getNumColumns(); featNum++) {

    // Initialize MSEs/counts
    for (size_t i=0; i<numObservations; i++) {
      outputOOBPrediction[i] = 0;
      outputOOBCount[i] = 0;
    }
    //Use same parallelization scheme as before

    #if DOPARELLEL
    size_t nthreadToUse = getNthread();
    if (nthreadToUse == 0) {
      nthreadToUse = std::thread::hardware_concurrency();
    }
    if (isVerbose()) {
      Rcpp::Rcout << "Calculating OOB parallel using " << nthreadToUse << " threads"
                << std::endl;
    }

    std::vector<std::thread> allThreads(nthreadToUse);
    std::mutex threadLock;

    // For each thread, assign a sequence of tree numbers that the thread
    // is responsible for handling
    for (size_t t = 0; t < nthreadToUse; t++) {
      auto dummyThread = std::bind(
        [&](const int iStart, const int iEnd, const int t_) {

          // loop over all items
          for (int i=iStart; i < iEnd; i++) {
    #else
    // For non-parallel version, just simply iterate all trees serially
    for(int i=0; i<((int) getNtree()); i++ ) {
    #endif
        try {
          std::vector<float> outputOOBPrediction_iteration(numObservations);
          std::vector<size_t> outputOOBCount_iteration(numObservations);
          for (size_t j=0; j<numObservations; j++) {
            outputOOBPrediction_iteration[j] = 0;
            outputOOBCount_iteration[j] = 0;
          }
          forestryTree *currentTree = (*getForest())[i].get();
          (*currentTree).getShuffledOOBPrediction(
              outputOOBPrediction_iteration,
              outputOOBCount_iteration,
              getTrainingData(),
              featNum
          );
          #if DOPARELLEL
          std::lock_guard<std::mutex> lock(threadLock);
          #endif
          for (size_t j=0; j < numObservations; j++) {
            outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
            outputOOBCount[j] += outputOOBCount_iteration[j];
          }
        } catch (std::runtime_error &err) {
          Rcpp::Rcerr << err.what() << std::endl;
        }
      }
    #if DOPARELLEL
      },
      t * getNtree() / nthreadToUse,
      (t + 1) == nthreadToUse ?
        getNtree() :
        (t + 1) * getNtree() / nthreadToUse,
          t
        );
        allThreads[t] = std::thread(dummyThread);
      }

      std::for_each(
        allThreads.begin(),
        allThreads.end(),
        [](std::thread& x) { x.join(); }
      );
      #endif

      float current_MSE = 0;
      for (size_t j = 0; j < numObservations; j++){
        float trueValue = getTrainingData()->getOutcomePoint(j);
        if (outputOOBCount[j] != 0) {
          current_MSE +=
            pow(trueValue - outputOOBPrediction[j] / outputOOBCount[j], 2);
        }
      }
      variableImportances.push_back(current_MSE);
  }

  std::unique_ptr<std::vector<float> > variableImportances_(
      new std::vector<float>(variableImportances)
  );

  // Populate forest's variable importance with all shuffled MSE's
  this-> _variableImportance = std::move(variableImportances_);
}


void forestry::calculateLocalVariableImportance(
    std::vector< std::vector<float> >* xNew,
    arma::Mat<float>* localVIMatrix,
    std::vector<float> prediction,
    predict_info predictInfo
  ) {
  // Get predicted outcomes for training data
  size_t numNewObs = (*xNew)[0].size();
  size_t numTrainingObs = getTrainingData()->getNumRows();
  std::vector<float> predictedTrainingOutcome(numTrainingObs, 0);
  std::vector<size_t> predictedTrainingCount(numTrainingObs, 0);
  for (int treeIndex = 0; treeIndex < ((int) getNtree()); treeIndex++ ) {
    try {
      std::vector<float> predictedTrainingOutcome_iteration(numTrainingObs);
      std::vector<size_t> predictedTrainingCount_iteration(numTrainingObs);
      forestryTree *currentTree = (*getForest())[treeIndex].get();
      (*currentTree).getOOBPrediction(
          predictedTrainingOutcome_iteration,
          predictedTrainingCount_iteration,
          getTrainingData()
      );
      for (size_t obsIndex = 0; obsIndex < numTrainingObs; obsIndex++) {
        predictedTrainingOutcome[obsIndex] += predictedTrainingOutcome_iteration[obsIndex];
        predictedTrainingCount[obsIndex] += predictedTrainingCount_iteration[obsIndex];
      }
    } catch (std::runtime_error &err) {
      // Rcpp::Rcerr << err.what() << std::endl;
    }
  }
  std::vector<float> predictedTrainingY(predictedTrainingOutcome.size(), 0);
  for (size_t i = 0; i < predictedTrainingOutcome.size(); i++ ) {
    if (predictedTrainingCount[i] != 0 ) {
      predictedTrainingY[i] = predictedTrainingOutcome[i] / predictedTrainingCount[i];
    }
  }

  // Loop over all new observations
  for (size_t observationIndex = 0; observationIndex < numNewObs; observationIndex++) {
    std::vector<float> observationWeight =
      arma::conv_to< std::vector<float> >::from(predictInfo.weightMatrix->row(observationIndex));

    // 1. Calculate weightedMSE and weightedVariance
    float weightedMSE = 0;
    float weightedVariance = 0;
    float weightedOutcome = std::inner_product(predictedTrainingY.begin(), predictedTrainingY.end(),
                                               observationWeight.begin(), 0.0);
    for (size_t i = 0; i < predictedTrainingOutcome.size(); i++) {
      if (predictedTrainingCount[i] != 0) {
        float trueOutcome = getTrainingData()->getOutcomePoint(i);
        weightedMSE += observationWeight[i]*pow((trueOutcome - predictedTrainingY[i]), 2);
        weightedVariance += observationWeight[i]*pow((trueOutcome - weightedOutcome), 2);
      }
    }

    std::vector<float> outputTrainingPrediction(numTrainingObs);
    std::vector<size_t> outputTrainingCount(numTrainingObs);

    // Loop over all features
    for (size_t featNum = 0; featNum < getTrainingData()->getNumColumns(); featNum++) {
      for (int i = 0; i < ((int) getNtree()); i++) {
        try {
          std::vector<float> outputOOBPrediction_iteration(numTrainingObs, 0);
          std::vector<size_t> outputOOBCount_iteration(numTrainingObs, 0);
          forestryTree *currentTree = (*getForest())[i].get();
          (*currentTree).getShuffledOOBPrediction(
              outputOOBPrediction_iteration,
              outputOOBCount_iteration,
              getTrainingData(),
              featNum
          );
          for (size_t j=0; j < numTrainingObs; j++) {
            outputTrainingPrediction[j] += outputOOBPrediction_iteration[j];
            outputTrainingCount[j] += outputOOBCount_iteration[j];
          }
        } catch (std::runtime_error &err) {
          Rcpp::Rcerr << err.what() << std::endl;
        }
      }

      float weightedPermutedMSE = 0;
      for (size_t i = 0; i < outputTrainingPrediction.size(); i++) {
        if (outputTrainingCount[i] != 0) {
          float trueOutcome = getTrainingData()->getOutcomePoint(i);
          weightedPermutedMSE += observationWeight[i]
            *pow(trueOutcome - (outputTrainingPrediction[i]/outputTrainingCount[i]), 2);
        }
      }
      (*localVIMatrix)(observationIndex, featNum) =
        ((weightedPermutedMSE-weightedMSE)/weightedVariance);
    }
  }
}


void forestry::calculateOOBError() {

  size_t numObservations = getTrainingData()->getNumRows();

  std::vector<float> outputOOBPrediction(numObservations);
  std::vector<size_t> outputOOBCount(numObservations);

  for (size_t i=0; i<numObservations; i++) {
    outputOOBPrediction[i] = 0;
    outputOOBCount[i] = 0;
  }

  #if DOPARELLEL
  size_t nthreadToUse = getNthread();
  if (nthreadToUse == 0) {
    // Use all threads
    nthreadToUse = std::thread::hardware_concurrency();
  }
  if (isVerbose()) {
    Rcpp::Rcout << "Calculating OOB parallel using " << nthreadToUse << " threads"
              << std::endl;
  }

  std::vector<std::thread> allThreads(nthreadToUse);
  std::mutex threadLock;

  // For each thread, assign a sequence of tree numbers that the thread
  // is responsible for handling
  for (size_t t = 0; t < nthreadToUse; t++) {
    auto dummyThread = std::bind(
      [&](const int iStart, const int iEnd, const int t_) {

        // loop over all items
        for (int i=iStart; i < iEnd; i++) {
  #else
  // For non-parallel version, just simply iterate all trees serially
  for(int i=0; i<((int) getNtree()); i++ ) {
  #endif
          try {
            std::vector<float> outputOOBPrediction_iteration(numObservations);
            std::vector<size_t> outputOOBCount_iteration(numObservations);
            for (size_t j=0; j<numObservations; j++) {
              outputOOBPrediction_iteration[j] = 0;
              outputOOBCount_iteration[j] = 0;
            }
            forestryTree *currentTree = (*getForest())[i].get();
            (*currentTree).getOOBPrediction(
              outputOOBPrediction_iteration,
              outputOOBCount_iteration,
              getTrainingData()
            );

            #if DOPARELLEL
            std::lock_guard<std::mutex> lock(threadLock);
            #endif

            for (size_t j=0; j < numObservations; j++) {
              outputOOBPrediction[j] += outputOOBPrediction_iteration[j];
              outputOOBCount[j] += outputOOBCount_iteration[j];
            }

          } catch (std::runtime_error &err) {
            // Rcpp::Rcerr << err.what() << std::endl;
          }
        }
  #if DOPARELLEL
        },
        t * getNtree() / nthreadToUse,
        (t + 1) == nthreadToUse ?
          getNtree() :
          (t + 1) * getNtree() / nthreadToUse,
        t
    );
    allThreads[t] = std::thread(dummyThread);
  }

  std::for_each(
    allThreads.begin(),
    allThreads.end(),
    [](std::thread& x) { x.join(); }
  );
  #endif

  float OOB_MSE = 0;
  for (size_t j=0; j<numObservations; j++){
    float trueValue = getTrainingData()->getOutcomePoint(j);
    if (outputOOBCount[j] != 0) {
      OOB_MSE +=
        pow(trueValue - outputOOBPrediction[j] / outputOOBCount[j], 2);
    }
  }

  this->_OOBError = OOB_MSE;
};


// -----------------------------------------------------------------------------

void forestry::fillinTreeInfo(
    std::unique_ptr< std::vector< tree_info > > & forest_dta
){

  if (isVerbose()) {
    Rcpp::Rcout << "Starting to translate Forest to R.\n";
  }

  for(int i=0; i<((int) getNtree()); i++ ) {
    // read out each tree and add it to the forest_dta:
    try {
      forestryTree *currentTree = (*getForest())[i].get();
      std::unique_ptr<tree_info> treeInfo_i =
        (*currentTree).getTreeInfo(_trainingData);

      forest_dta->push_back(*treeInfo_i);

    } catch (std::runtime_error &err) {
      Rcpp::Rcerr << err.what() << std::endl;

    }

    if (isVerbose()) {
      Rcpp::Rcout << "Done with tree " << i + 1 << " of " << getNtree() << ".\n";
    }

  }

  if (isVerbose()) {
    Rcpp::Rcout << "Translation done.\n";
  }

  return ;
};

void forestry::reconstructTrees(
    std::unique_ptr< std::vector<size_t> > & categoricalFeatureColsRcpp,
    std::unique_ptr< std::vector< std::vector<int> >  > & var_ids,
    std::unique_ptr< std::vector< std::vector<double> >  > & split_vals,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & leafAveidxs,
    std::unique_ptr< std::vector< std::vector<size_t> >  > & leafSplidxs,
    std::unique_ptr< std::vector< std::vector<size_t> >  > &
      averagingSampleIndex,
    std::unique_ptr< std::vector< std::vector<size_t> >  > &
      splittingSampleIndex){

    for (int i=0; i<split_vals->size(); i++) {
      try{
        forestryTree *oneTree = new forestryTree();

        oneTree->reconstruct_tree(
                getMtry(),
                getMinNodeSizeSpt(),
                getMinNodeSizeAvg(),
                getMinNodeSizeToSplitSpt(),
                getMinNodeSizeToSplitAvg(),
                getMinSplitGain(),
                getMaxDepth(),
                getlinear(),
                getOverfitPenalty(),
                (*categoricalFeatureColsRcpp),
                (*var_ids)[i],
                (*split_vals)[i],
                (*leafAveidxs)[i],
                (*leafSplidxs)[i],
                (*averagingSampleIndex)[i],
                (*splittingSampleIndex)[i]);

        (*getForest()).emplace_back(oneTree);
        _ntree = _ntree + 1;
      } catch (std::runtime_error &err) {
        Rcpp::Rcerr << err.what() << std::endl;
      }

  }

  return;
}



