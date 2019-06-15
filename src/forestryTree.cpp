#include "forestryTree.h"
#include "utils.h"
#include <RcppArmadillo.h>
#include <math.h>
#include <set>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
// [[Rcpp::plugins(cpp11)]]

forestryTree::forestryTree():
  _mtry(0),
  _minNodeSizeSpt(0),
  _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0),
  _minNodeSizeToSplitAvg(0),
  _minSplitGain(0),
  _maxDepth(0),
  _averagingSampleIndex(nullptr),
  _splittingSampleIndex(nullptr),
  _root(nullptr) {};

forestryTree::~forestryTree() {};

forestryTree::forestryTree(
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
  bool ridgeRF,
  float overfitPenalty
){
  /**
  * @brief Honest random forest tree constructor
  * @param trainingData    A DataFrame object
  * @param mtry    The total number of features to use for each split
  * @param minNodeSizeSpt    Minimum splitting size of leaf node
  * @param minNodeSizeAvg    Minimum averaging size of leaf node
  * @param minNodeSizeToSplitSpt    Minimum splitting size of a splitting node
  * @param minNodeSizeToSplitAvg    Minimum averaging size of a splitting node
  * @param minSplitGain    Minimum loss reduction to split a node.
  * @param maxDepth    Max depth of a tree
  * @param splittingSampleIndex    A vector with index of splitting samples
  * @param averagingSampleIndex    A vector with index of averaging samples
  * @param random_number_generator    A mt19937 random generator
  * @param splitMiddle    Boolean to indicate if new feature value is
  *    determined at a random position between two feature values
  * @param maxObs    Max number of observations to split on
  */

  /* Sanity Check */
  if (minNodeSizeAvg == 0) {
    throw std::runtime_error("minNodeSizeAvg cannot be set to 0.");
  }
  if (minNodeSizeSpt == 0) {
    throw std::runtime_error("minNodeSizeSpt cannot be set to 0.");
  }
  if (minNodeSizeToSplitSpt == 0) {
    throw std::runtime_error("minNodeSizeToSplitSpt cannot be set to 0.");
  }
  if (minNodeSizeToSplitAvg == 0) {
    throw std::runtime_error("minNodeSizeToSplitAvg cannot be set to 0.");
  }
  if (minNodeSizeToSplitAvg > (*averagingSampleIndex).size()) {
    std::ostringstream ostr;
    ostr << "minNodeSizeToSplitAvg cannot exceed total elements in the "
    "averaging samples: minNodeSizeToSplitAvg=" << minNodeSizeToSplitAvg <<
      ", averagingSampleSize=" << (*averagingSampleIndex).size() << ".";
    throw std::runtime_error(ostr.str());
  }
  if (minNodeSizeToSplitSpt > (*splittingSampleIndex).size()) {
    std::ostringstream ostr;
    ostr << "minNodeSizeToSplitSpt cannot exceed total elements in the "
    "splitting samples: minNodeSizeToSplitSpt=" << minNodeSizeToSplitSpt <<
      ", splittingSampleSize=" << (*splittingSampleIndex).size() << ".";
    throw std::runtime_error(ostr.str());
  }
  if (maxDepth == 0) {
    throw std::runtime_error("maxDepth cannot be set to 0.");
  }
  if (minSplitGain != 0 && !ridgeRF) {
    throw std::runtime_error("minSplitGain cannot be set without setting ridgeRF to be true.");
  }
  if ((*averagingSampleIndex).size() == 0) {
    throw std::runtime_error("averagingSampleIndex size cannot be set to 0.");
  }
  if ((*splittingSampleIndex).size() == 0) {
    throw std::runtime_error("splittingSampleIndex size cannot be set to 0.");
  }
  if (mtry == 0) {
    throw std::runtime_error("mtry cannot be set to 0.");
  }
  if (mtry > (*trainingData).getNumColumns()) {
    std::ostringstream ostr;
    ostr << "mtry cannot exceed total amount of features: mtry=" << mtry
         << ", totalNumFeatures=" << (*trainingData).getNumColumns() << ".";
    throw std::runtime_error(ostr.str());
  }

  /* Move all pointers to the current object */
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_maxDepth = maxDepth;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
  std::unique_ptr< RFNode > root ( new RFNode() );
  this->_root = std::move(root);
  this->_benchmark = new std::vector<double>;
  //for (size_t i = 0; i < 8; i++) {
  //  _benchmark->push_back(0);
  //}


  /* If ridge splitting, initialize RSS components to pass to leaves*/

  std::vector<size_t>* splitIndexes = getSplittingIndex();
  size_t numLinearFeatures;
  std::vector<float> firstOb = trainingData->getLinObsData((*splitIndexes)[0]);
  numLinearFeatures = firstOb.size();
  firstOb.push_back(1.0);
  arma::Mat<double> sTotal(firstOb.size(),
                               1);
  sTotal.col(0) = arma::conv_to<arma::Col<double> >::from(firstOb);
  arma::Mat<double> gTotal(numLinearFeatures + 1,
                          numLinearFeatures + 1);
  if (ridgeRF) {
    this->initializeRidgeRF(trainingData,
                                  gTotal,
                                  sTotal,
                                  numLinearFeatures,
                                  getSplittingIndex());
  }


  /* Recursively grow the tree */
  recursivePartition(
    getRoot(),
    getAveragingIndex(),
    getSplittingIndex(),
    trainingData,
    random_number_generator,
    0,
    splitMiddle,
    maxObs,
    ridgeRF,
    overfitPenalty,
    getBenchmark(),
    gTotal,
    sTotal
  );

  //this->_root->printSubtree();
  //this->trainTiming();
}

void forestryTree::setDummyTree(
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
){
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_maxDepth = maxDepth;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
}

void forestryTree::predict(
    std::vector<float> &outputPrediction,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    predict_info predictInfo
){
  struct rangeGenerator {
    size_t currentNumber;
    rangeGenerator(size_t startNumber): currentNumber(startNumber) {};
    size_t operator()() {return currentNumber++; }
  };

  std::vector<size_t> updateIndex(outputPrediction.size());
  rangeGenerator _rangeGenerator(0);
  std::generate(updateIndex.begin(), updateIndex.end(), _rangeGenerator);
  predictInfo.overfitPenalty = getOverfitPenalty();
  (*getRoot()).predict(outputPrediction,
   &updateIndex,
   xNew,
   trainingData,
   predictInfo);
}


std::vector<size_t> sampleFeatures(
    size_t mtry,
    std::mt19937_64& random_number_generator,
    bool numFeaturesOnly,
    std::vector<size_t>* splitCols,
    std::vector<size_t>* numCols
){
  // Sample features without replacement
  std::vector<size_t> featureList;
  if (numFeaturesOnly) {
    // TODO: Set numericCols to be the intersection of numCols and splitCols
    std::vector<size_t> numericCols;
    std::set_intersection(numCols->begin(), numCols->end(),
                          splitCols->begin(), splitCols->end(), back_inserter(numericCols));
    size_t mtry_split = std::min(mtry, numericCols.size());
    while (featureList.size() < mtry_split) {
      std::uniform_int_distribution<size_t> unif_dist(
          0, (size_t) numericCols.size() - 1
      );

      size_t index = unif_dist(random_number_generator);

      if (featureList.size() == 0 ||
          std::find(
            featureList.begin(),
            featureList.end(),
            (numericCols)[index]
          ) == featureList.end()
      ) {
        featureList.push_back(numericCols[index]);
      }
    }

  } else {
    size_t mtry_split = std::min(mtry, splitCols->size());
    while (featureList.size() < mtry_split) {
      std::uniform_int_distribution<size_t> unif_dist(
          0, (size_t) splitCols->size() - 1
      );
      size_t randomIndex = unif_dist(random_number_generator);
      if (featureList.size() == 0 ||
          std::find(
            featureList.begin(),
            featureList.end(),
            (*splitCols)[randomIndex]
          ) == featureList.end()
      ) {
        featureList.push_back((*splitCols)[randomIndex]);
      }
    }
  }
  return featureList;
}


void splitDataIntoTwoParts(
    DataFrame* trainingData,
    std::vector<size_t>* sampleIndex,
    size_t splitFeature,
    float splitValue,
    std::vector<size_t>* leftPartitionIndex,
    std::vector<size_t>* rightPartitionIndex,
    bool categoical
){
  for (
      std::vector<size_t>::iterator it = (*sampleIndex).begin();
      it != (*sampleIndex).end();
      ++it
  ) {
    if (categoical) {
      // categorical, split by (==) or (!=)
      if ((*trainingData).getPoint(*it, splitFeature) == splitValue) {
        (*leftPartitionIndex).push_back(*it);
      } else {
        (*rightPartitionIndex).push_back(*it);
      }
    } else {
      // Non-categorical, split to left (<) and right (>=) according to the
      if ((*trainingData).getPoint(*it, splitFeature) < splitValue) {
        (*leftPartitionIndex).push_back(*it);
      } else {
        (*rightPartitionIndex).push_back(*it);
      }
    }
  }
}

void splitData(
    DataFrame* trainingData,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t splitFeature,
    float splitValue,
    std::vector<size_t>* averagingLeftPartitionIndex,
    std::vector<size_t>* averagingRightPartitionIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    bool categoical
){
  // averaging data
  splitDataIntoTwoParts(
    trainingData,
    averagingSampleIndex,
    splitFeature,
    splitValue,
    averagingLeftPartitionIndex,
    averagingRightPartitionIndex,
    categoical
  );
  // splitting data
  splitDataIntoTwoParts(
    trainingData,
    splittingSampleIndex,
    splitFeature,
    splitValue,
    splittingLeftPartitionIndex,
    splittingRightPartitionIndex,
    categoical
  );
}

float calculateRSS(
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    float overfitPenalty
) {
  // Get cross validation folds
  std::vector< std::vector< size_t > > cvFolds(10);
  if (splittingSampleIndex->size() >= 10) {
    std::random_shuffle(splittingSampleIndex->begin(), splittingSampleIndex->end());
    size_t foldIndex = 0;
    for (size_t sampleIndex : *splittingSampleIndex) {
      cvFolds.at(foldIndex).push_back(sampleIndex);
      foldIndex++;
      foldIndex = foldIndex % 10;
    }
  }

  float residualSumSquares = 0;
  size_t numFolds = cvFolds.size();
  if (splittingSampleIndex->size() < 10) {
    numFolds = 1;
  }

  for (size_t i = 0; i < numFolds; i++) {
    std::vector<size_t> trainIndex;
    std::vector<size_t> testIndex;

    if (splittingSampleIndex->size() < 10) {
      trainIndex = *splittingSampleIndex;
      testIndex = *splittingSampleIndex;
    }
    for (size_t j = 0; j < numFolds; j++) {
      if (j == i) {
        testIndex = cvFolds.at(j);
      } else {
        trainIndex.insert(trainIndex.end(), cvFolds.at(j).begin(), cvFolds.at(j).end());
      }
    }

    //Number of linear features in training data
    size_t dimension = (trainingData->getLinObsData(trainIndex[0])).size();
    arma::Mat<double> identity(dimension + 1, dimension + 1);
    identity.eye();
    arma::Mat<double> xTrain(trainIndex.size(), dimension + 1);

    //Don't penalize intercept
    identity(dimension, dimension) = 0.0;

    std::vector<float> outcomePoints;
    std::vector<float> currentObservation;

    // Contruct X and outcome vector
    for (size_t i = 0; i < trainIndex.size(); i++) {
      currentObservation = trainingData->getLinObsData((trainIndex)[i]);
      currentObservation.push_back(1.0);
      xTrain.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);
      outcomePoints.push_back(trainingData->getOutcomePoint((trainIndex)[i]));
    }

    arma::Mat<double> y(outcomePoints.size(), 1);
    y.col(0) = arma::conv_to<arma::Col<double> >::from(outcomePoints);

    // Compute XtX + lambda * I * Y = C
    arma::Mat<double> coefficients = (xTrain.t() * xTrain +
      identity * overfitPenalty).i() * xTrain.t() * y;

    // Compute test matrix
    arma::Mat<double> xTest(testIndex.size(), dimension + 1);

    for (size_t i = 0; i < testIndex.size(); i++) {
      currentObservation = trainingData->getLinObsData((testIndex)[i]);
      currentObservation.push_back(1.0);
      xTest.row(i) = arma::conv_to<arma::Row<double> >::from(currentObservation);
    }

    arma::Mat<double> predictions = xTest * coefficients;
    for (size_t i = 0; i < predictions.size(); i++) {
      float residual = (trainingData->getOutcomePoint((testIndex)[i])) - predictions(i, 0);
      residualSumSquares += residual * residual;
    }
  }
  return residualSumSquares;
}


std::pair<float, float> calculateRSquaredSplit (
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    float overfitPenalty
) {
  // Get residual sum of squares for parent, left child, and right child nodes
  float rssParent, rssLeft, rssRight;
  rssParent = calculateRSS(trainingData,
                           splittingSampleIndex,
                           overfitPenalty);
  rssLeft = calculateRSS(trainingData,
                         splittingLeftPartitionIndex,
                         overfitPenalty);
  rssRight = calculateRSS(trainingData,
                          splittingRightPartitionIndex,
                          overfitPenalty);

  // Calculate total sum of squares
  float outcomeSum = 0;
  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    outcomeSum += trainingData->getOutcomePoint((*splittingSampleIndex)[i]);
  }
  float outcomeMean = outcomeSum/(splittingSampleIndex->size());

  float totalSumSquares = 0;
  float meanDifference;
  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    meanDifference =
      (trainingData->getOutcomePoint((*splittingSampleIndex)[i]) - outcomeMean);
    totalSumSquares += meanDifference * meanDifference;
  }

  // Use TSS and RSS to calculate r^2 values for parent and children
  float rSquaredParent = (1 - (rssParent/totalSumSquares));
  float rSquaredChildren = (1 - ((rssLeft + rssRight)/totalSumSquares));
  return std::make_pair(rSquaredParent, rSquaredChildren);
}

float crossValidatedRSquared (
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    std::vector<size_t>* splittingLeftPartitionIndex,
    std::vector<size_t>* splittingRightPartitionIndex,
    float overfitPenalty,
    size_t numTimesCV
) {
  // Apply 5 times 10-fold cross-validation
  float rSquaredParent, rSquaredChildren;
  float totalRSquaredParent = 0;
  float totalRSquaredChildren = 0;

  for (size_t i = 0; i < numTimesCV; i++) {
    std::tie(rSquaredParent, rSquaredChildren) =
      calculateRSquaredSplit(
        trainingData,
        splittingSampleIndex,
        splittingLeftPartitionIndex,
        splittingRightPartitionIndex,
        overfitPenalty
      );
    totalRSquaredParent += rSquaredParent;
    totalRSquaredChildren += rSquaredChildren;
  }

  return (totalRSquaredChildren/numTimesCV) - (totalRSquaredParent/numTimesCV);
}


void forestryTree::recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    size_t depth,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<double> gTotal,
    arma::Mat<double> sTotal
){
  if ((*averagingSampleIndex).size() < getMinNodeSizeAvg() ||
      (*splittingSampleIndex).size() < getMinNodeSizeSpt() ||
      (depth == getMaxDepth())) {
    // Create two lists on heap and transfer the owernship to the node
    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>(*averagingSampleIndex)
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>(*splittingSampleIndex)
    );
    (*rootNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_)
    );
    return;
  }

  // Sample mtry amounts of features
  std::vector<size_t> featureList;
  featureList = sampleFeatures(
    getMtry(),
    random_number_generator,
    false,
    trainingData->getSplitCols(),
    trainingData->getNumCols()
  );


  // Select best feature
  size_t bestSplitFeature;
  double bestSplitValue;
  float bestSplitLoss;
  arma::Mat<double> bestSplitGL(size(gTotal));
  arma::Mat<double> bestSplitGR(size(gTotal));
  arma::Mat<double> bestSplitSL(size(sTotal));
  arma::Mat<double> bestSplitSR(size(sTotal));

  selectBestFeature(
    bestSplitFeature,
    bestSplitValue,
    bestSplitLoss,
    bestSplitGL,
    bestSplitGR,
    bestSplitSL,
    bestSplitSR,
    &featureList,
    averagingSampleIndex,
    splittingSampleIndex,
    trainingData,
    random_number_generator,
    splitMiddle,
    maxObs,
    ridgeRF,
    overfitPenalty,
    benchmark,
    gTotal,
    sTotal
  );

  // Create a leaf node if the current bestSplitValue is NA
  if (std::isnan(bestSplitValue)) {
    // Create two lists on heap and transfer the owernship to the node
    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>(*averagingSampleIndex)
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>(*splittingSampleIndex)
    );
    (*rootNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_)
    );

  } else {
    // Test if the current feature is categorical
    std::vector<size_t> averagingLeftPartitionIndex;
    std::vector<size_t> averagingRightPartitionIndex;
    std::vector<size_t> splittingLeftPartitionIndex;
    std::vector<size_t> splittingRightPartitionIndex;
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();

    // Create split for both averaging and splitting dataset based on
    // categorical feature or not
    splitData(
      trainingData,
      averagingSampleIndex,
      splittingSampleIndex,
      bestSplitFeature,
      bestSplitValue,
      &averagingLeftPartitionIndex,
      &averagingRightPartitionIndex,
      &splittingLeftPartitionIndex,
      &splittingRightPartitionIndex,
      std::find(
        categorialCols.begin(),
        categorialCols.end(),
        bestSplitFeature
      ) != categorialCols.end()
    );

    // Stopping-criteria
    if (getMinSplitGain() > 0) {
      float rSquaredDifference = crossValidatedRSquared(
        trainingData,
        splittingSampleIndex,
        &splittingLeftPartitionIndex,
        &splittingRightPartitionIndex,
        overfitPenalty,
        1
      );

      if (rSquaredDifference < getMinSplitGain()) {
        std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
            new std::vector<size_t>(*averagingSampleIndex)
        );
        std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
            new std::vector<size_t>(*splittingSampleIndex)
        );
        (*rootNode).setLeafNode(
            std::move(averagingSampleIndex_),
            std::move(splittingSampleIndex_)
        );
        return;
      }
    }

    // Update sample index for both left and right partitions
    // Recursively grow the tree
    std::unique_ptr< RFNode > leftChild ( new RFNode() );
    std::unique_ptr< RFNode > rightChild ( new RFNode() );

    size_t childDepth = depth + 1;

    recursivePartition(
      leftChild.get(),
      &averagingLeftPartitionIndex,
      &splittingLeftPartitionIndex,
      trainingData,
      random_number_generator,
      childDepth,
      splitMiddle,
      maxObs,
      ridgeRF,
      overfitPenalty,
      benchmark,
      bestSplitGL,
      bestSplitSL
    );
    recursivePartition(
      rightChild.get(),
      &averagingRightPartitionIndex,
      &splittingRightPartitionIndex,
      trainingData,
      random_number_generator,
      childDepth,
      splitMiddle,
      maxObs,
      ridgeRF,
      overfitPenalty,
      benchmark,
      bestSplitGR,
      bestSplitSR
    );

    (*rootNode).setSplitNode(
        bestSplitFeature,
        bestSplitValue,
        std::move(leftChild),
        std::move(rightChild)
    );
  }
}

void forestryTree::initializeRidgeRF(
    DataFrame* trainingData,
    arma::Mat<double>& gTotal,
    arma::Mat<double>& sTotal,
    size_t numLinearFeatures,
    std::vector<size_t>* splitIndexes
) {
  gTotal = sTotal * (sTotal.t());
  sTotal = trainingData->getOutcomePoint((*splitIndexes)[0]) * sTotal;

  std::vector<float> temp(numLinearFeatures + 1);
  arma::Mat<double> tempOb(numLinearFeatures + 1, 1);
  /* Sum up sTotal and gTotal once on every observation in splitting set*/
  for (size_t i = 1; i < splitIndexes->size(); i++) {
    temp = trainingData->getLinObsData((*splitIndexes)[i]);
    temp.push_back(1.0);
    tempOb.col(0) = arma::conv_to<arma::Col<double> >::from(temp);
    gTotal = gTotal + (tempOb * (tempOb.t()));
    sTotal = sTotal + trainingData->getOutcomePoint((*splitIndexes)[i])
      * tempOb;
  }
}

void updateBestSplit(
    float* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    float currentSplitLoss,
    double currentSplitValue,
    size_t currentFeature,
    size_t bestSplitTableIndex,
    std::mt19937_64& random_number_generator
) {

  // Update the value if a higher value has been seen
  if (currentSplitLoss > bestSplitLossAll[bestSplitTableIndex]) {
    bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
    bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
    bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
    bestSplitCountAll[bestSplitTableIndex] = 1;
  } else {

    //If we are as good as the best split
    if (currentSplitLoss == bestSplitLossAll[bestSplitTableIndex]) {
      bestSplitCountAll[bestSplitTableIndex] =
        bestSplitCountAll[bestSplitTableIndex] + 1;

      // Only update with probability 1/nseen
      std::uniform_real_distribution<float> unif_dist;
      float tmp_random = unif_dist(random_number_generator);
      if (tmp_random * bestSplitCountAll[bestSplitTableIndex] <= 1) {
        bestSplitLossAll[bestSplitTableIndex] = currentSplitLoss;
        bestSplitFeatureAll[bestSplitTableIndex] = currentFeature;
        bestSplitValueAll[bestSplitTableIndex] = currentSplitValue;
      }
    }
  }
}

void updateBestSplitS(
  arma::Mat<double> &bestSplitSL,
  arma::Mat<double> &bestSplitSR,
  arma::Mat<double> &sTotal,
  DataFrame* trainingData,
  std::vector<size_t>* splittingSampleIndex,
  size_t bestSplitFeature,
  double bestSplitValue
) {
  //Get splitfeaturedata
  //sort splitindicesby splitfeature
  //while currentoutcome (getPoint(currentindex, splitfeature)) < splitValue
  //Add up outcome(i)*feat+1(i) ------ This is sL
  //sR = sTotal - sL
  //Get indexes of observations
  std::vector<size_t> splittingIndices;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndices.push_back((*splittingSampleIndex)[i]);
  }

  //Sort indices of observations ascending by currentFeature
  std::vector<float>* featureData = trainingData->getFeatureData(bestSplitFeature);

  std::sort(splittingIndices.begin(),
       splittingIndices.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  std::vector<size_t>::iterator featIter = splittingIndices.begin();
  float currentValue = trainingData->getPoint(*featIter, bestSplitFeature);


  std::vector<float> observation;
  arma::Mat<double> crossingObservation = arma::Mat<double>(size(sTotal)).zeros();
  arma::Mat<double> sTemp = arma::Mat<double>(size(sTotal)).zeros();

  while (featIter != splittingIndices.end() &&
         currentValue < bestSplitValue
  ) {
    //Update Matriices
    observation = trainingData->getLinObsData(*featIter);
    observation.push_back(1);

    crossingObservation.col(0) =
          arma::conv_to<arma::Col<double> >::from(observation);
    crossingObservation = crossingObservation *
                          trainingData->getOutcomePoint(*featIter);
    sTemp = sTemp + crossingObservation;

    ++featIter;
    currentValue = trainingData->getPoint(*featIter, bestSplitFeature);
  }

  bestSplitSL = sTemp;
  bestSplitSR = sTotal - sTemp;
}

void updateBestSplitG(
    arma::Mat<double> &bestSplitGL,
    arma::Mat<double> &bestSplitGR,
    arma::Mat<double> &gTotal,
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitFeature,
    double bestSplitValue
) {

  std::vector<size_t> splittingIndices;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndices.push_back((*splittingSampleIndex)[i]);
  }

  //Sort indices of observations ascending by currentFeature
  std::vector<float>* featureData = trainingData->getFeatureData(bestSplitFeature);

  std::sort(splittingIndices.begin(),
            splittingIndices.end(),
            [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  std::vector<size_t>::iterator featIter = splittingIndices.begin();
  float currentValue = trainingData->getPoint(*featIter, bestSplitFeature);


  std::vector<float> observation;
  arma::Mat<double> crossingObservation = arma::Mat<double>(size(gTotal)).zeros();
  arma::Mat<double> gTemp = arma::Mat<double>(size(gTotal)).zeros();

  while (featIter != splittingIndices.end() &&
         currentValue < bestSplitValue
  ) {
    //Update Matriices
    observation = trainingData->getLinObsData(*featIter);
    observation.push_back(1);

    crossingObservation.col(0) =
      arma::conv_to<arma::Col<double> >::from(observation);

    gTemp = gTemp + (crossingObservation * crossingObservation.t());

    ++featIter;
    currentValue = trainingData->getPoint(*featIter, bestSplitFeature);
  }

  bestSplitGL = gTemp;
  bestSplitGR = gTotal - gTemp;
}

void updateBestSplitS(
    arma::Mat<double> &bestSplitS,
    size_t bestSplitFeature,
    double bestSplitValue,
    bool left
) {

}

void updateAArmadillo(
    arma::Mat<double>& a_k,
    arma::Mat<double>& new_x,
    bool leftNode
){
  //Initilize z_K
  arma::Mat<double> z_K = a_k * new_x;

  //Update A using Sherman–Morrison formula corresponding to right or left side
  if (leftNode) {
    a_k = a_k - ((z_K) * (z_K).t()) /
      (1 + as_scalar(new_x.t() * z_K));
  } else {
    a_k = a_k + ((z_K) * (z_K).t()) /
      (1 - as_scalar(new_x.t() * z_K));
  }
}

void updateSkArmadillo(
    arma::Mat<double>& s_k,
    arma::Mat<double>& next,
    float next_y,
    bool left
){
  if (left) {
    s_k = s_k + (next_y * (next));
  } else {
    s_k = s_k - (next_y * (next));
  }
}

float computeRSSArmadillo(
    arma::Mat<double>& A_r,
    arma::Mat<double>& A_l,
    arma::Mat<double>& S_r,
    arma::Mat<double>& S_l,
    arma::Mat<double>& G_r,
    arma::Mat<double>& G_l
){
  return (as_scalar((S_l.t() * A_l) * (G_l * (A_l * S_l))) +
          as_scalar((S_r.t() * A_r) * (G_r * (A_r * S_r))) -
          as_scalar(2.0 * S_l.t() * (A_l * S_l)) -
          as_scalar(2.0 * S_r.t() * (A_r * S_r)));
}

float computeRSSArmadillo2(
    float x
){
  return (x);
}

void updateRSSComponents(
    DataFrame* trainingData,
    size_t nextIndex,
    arma::Mat<double>& aLeft,
    arma::Mat<double>& aRight,
    arma::Mat<double>& sLeft,
    arma::Mat<double>& sRight,
    arma::Mat<double>& gLeft,
    arma::Mat<double>& gRight,
    arma::Mat<double>& crossingObservation,
    arma::Mat<double>& obOuter
) {
  //Get observation that will cross the partition
  std::vector<float> newLeftObservation =
    trainingData->getLinObsData(nextIndex);

  newLeftObservation.push_back(1.0);

  crossingObservation.col(0) =
    arma::conv_to<arma::Col<double> >::from(newLeftObservation);

  float crossingOutcome = trainingData->getOutcomePoint(nextIndex);

  //Use to update RSS components
  updateSkArmadillo(sLeft, crossingObservation, crossingOutcome, true);
  updateSkArmadillo(sRight, crossingObservation, crossingOutcome, false);

  obOuter = crossingObservation * crossingObservation.t();
  gLeft = gLeft + obOuter;
  gRight = gRight - obOuter;

  updateAArmadillo(aLeft, crossingObservation, true);
  updateAArmadillo(aRight, crossingObservation, false);
}

void initializeRSSComponents(
    DataFrame* trainingData,
    size_t index,
    size_t numLinearFeatures,
    float overfitPenalty,
    arma::Mat<double>& gTotal,
    arma::Mat<double>& sTotal,
    arma::Mat<double>& aLeft,
    arma::Mat<double>& aRight,
    arma::Mat<double>& sLeft,
    arma::Mat<double>& sRight,
    arma::Mat<double>& gLeft,
    arma::Mat<double>& gRight,
    arma::Mat<double>& crossingObservation
) {
  //Initialize sLeft
  sLeft = trainingData->getOutcomePoint(index) *crossingObservation;

  sRight = sTotal - sLeft;

  //Initialize gLeft
  gLeft = crossingObservation * (crossingObservation.t());

  gRight = gTotal - gLeft;
  //Initialize sRight, gRight

  arma::Mat<double> identity(numLinearFeatures + 1,
                            numLinearFeatures + 1);
  identity.eye();

  //Don't penalize intercept
  identity(numLinearFeatures, numLinearFeatures) = 0.0;
  identity = overfitPenalty * identity;

  //Initialize aLeft
  aLeft = (gLeft + identity).i();

  //Initialize aRight
  aRight = (gRight + identity).i();
}

void findBestSplitRidgeCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    float* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    float overfitPenalty,
    arma::Mat<double>& gTotal,
    arma::Mat<double>& sTotal
) {
  /* Put all categories in a set
   * aggregate G_k matrices to put in left node when splitting
   * aggregate S_k and G_k matrices at each step
   *
   * linearly iterate through averaging indices adding count to total set count
   *
   * linearly iterate thought splitting indices and add G_k to the matrix mapped
   * to each index, then put in the all categories set
   *
   * Left is aggregated, right is total - aggregated
   * subtract and feed to RSS calculator for each partition
   * call updateBestSplitRidge with correct G_k matrices
   */

  // Set to hold all different categories
  std::set<float> all_categories;
  std::vector<float> temp;

  // temp matrices for RSS components
  arma::Mat<double> gRightTemp(size(gTotal));
  arma::Mat<double> sRightTemp(size(sTotal));
  arma::Mat<double> aRightTemp(size(gTotal));
  arma::Mat<double> aLeftTemp(size(gTotal));
  arma::Mat<double> crossingObservation(size(sTotal));
  arma::Mat<double> identity(size(gTotal));

  identity.eye();
  identity(identity.n_rows-1, identity.n_cols-1) = 0.0;
  size_t splitTotalCount = 0;
  size_t averageTotalCount = 0;

  // Create map to track the count and RSS components
  std::map<float, size_t> splittingCategoryCount;
  std::map<float, size_t> averagingCategoryCount;
  std::map<float, arma::Mat<double> > gMatrices;
  std::map<float, arma::Mat<double> > sMatrices;

  for (size_t j=0; j<averagingSampleIndex->size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint((*averagingSampleIndex)[j], currentFeature)
    );
    averageTotalCount++;
  }

  for (size_t j=0; j<splittingSampleIndex->size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint((*splittingSampleIndex)[j], currentFeature)
    );
    splitTotalCount++;
  }

  for (
      std::set<float>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    splittingCategoryCount[*it] = 0;
    averagingCategoryCount[*it] = 0;
    gMatrices[*it] = arma::Mat<double>(size(gTotal)).zeros();
    sMatrices[*it] = arma::Mat<double>(size(sTotal)).zeros();
  }

  // Put all matrices in map
  for (size_t j = 0; j<splittingSampleIndex->size(); j++) {
    // Add each observation to correct matrix in map
    float currentCategory = trainingData->getPoint((*splittingSampleIndex)[j],
                                                   currentFeature);
    float currentOutcome =
      trainingData->getOutcomePoint((*splittingSampleIndex)[j]);

    temp = trainingData->getLinObsData((*splittingSampleIndex)[j]);
    temp.push_back(1);
    crossingObservation.col(0) = arma::conv_to<arma::Col<double> >::from(temp);

    updateSkArmadillo(sMatrices[currentCategory],
                      crossingObservation,
                      currentOutcome,
                      true);

    gMatrices[currentCategory] = gMatrices[currentCategory]
                                 +crossingObservation * crossingObservation.t();
    splittingCategoryCount[currentCategory]++;
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    float currentCategory = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    averagingCategoryCount[currentCategory]++;
  }

  // Evaluate possible splits using associated RSS components
  for (
      std::set<float>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    // Check leaf size at least nodesize
    if (
        std::min(
          splittingCategoryCount[*it],
          splitTotalCount - splittingCategoryCount[*it]
        ) < splitNodeSize ||
          std::min(
            averagingCategoryCount[*it],
            averageTotalCount - averagingCategoryCount[*it]
          ) < averageNodeSize
    ) {
      continue;
    }
    gRightTemp = gTotal - gMatrices[*it];
    sRightTemp = sTotal - sMatrices[*it];

    aRightTemp = (gRightTemp + overfitPenalty * identity).i();
    aLeftTemp = (gMatrices[*it] + overfitPenalty * identity).i();

    float currentSplitLoss = computeRSSArmadillo(aRightTemp,
                                                 aLeftTemp,
                                                 sRightTemp,
                                                 sMatrices[*it],
                                                 gRightTemp,
                                                 gMatrices[*it]);

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      -currentSplitLoss,
      (double) *it,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
  }
}

void findBestSplitValueCategorical(
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitTableIndex,
    size_t currentFeature,
    float* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    DataFrame* trainingData,
    size_t splitNodeSize,
    size_t averageNodeSize,
    std::mt19937_64& random_number_generator,
    size_t maxObs
){

  // Count total number of observations for different categories
  std::set<float> all_categories;
  float splitTotalSum = 0;
  size_t splitTotalCount = 0;
  size_t averageTotalCount = 0;

  //EDITED
  //Move indices to vectors so we can downsample if needed
  std::vector<size_t> splittingIndices;
  std::vector<size_t> averagingIndices;

  for (size_t y = 0; y < (*splittingSampleIndex).size(); y++) {
    splittingIndices.push_back((*splittingSampleIndex)[y]);
  }

  for (size_t y = 0; y < (*averagingSampleIndex).size(); y++) {
    averagingIndices.push_back((*averagingSampleIndex)[y]);
  }


  //If maxObs is smaller, randomly downsample
  if (maxObs < (*splittingSampleIndex).size()) {
    std::vector<size_t> newSplittingIndices;
    std::vector<size_t> newAveragingIndices;

    std::shuffle(splittingIndices.begin(), splittingIndices.end(),
                 random_number_generator);
    std::shuffle(averagingIndices.begin(), averagingIndices.end(),
                 random_number_generator);

    for (int q = 0; q < maxObs; q++) {
      newSplittingIndices.push_back(splittingIndices[q]);
      newAveragingIndices.push_back(averagingIndices[q]);
    }

    std::swap(newSplittingIndices, splittingIndices);
    std::swap(newAveragingIndices, averagingIndices);
  }

  for (size_t j=0; j<splittingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(splittingIndices[j], currentFeature)
    );
    splitTotalSum +=
      (*trainingData).getOutcomePoint(splittingIndices[j]);
    splitTotalCount++;
  }
  for (size_t j=0; j<averagingIndices.size(); j++) {
    all_categories.insert(
      (*trainingData).getPoint(averagingIndices[j], currentFeature)
    );
    averageTotalCount++;
  }

  // Create map to track the count and sum of y squares
  std::map<float, size_t> splittingCategoryCount;
  std::map<float, size_t> averagingCategoryCount;
  std::map<float, float> splittingCategoryYSum;

  for (
      std::set<float>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    splittingCategoryCount[*it] = 0;
    averagingCategoryCount[*it] = 0;
    splittingCategoryYSum[*it] = 0;
  }

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
    float currentXValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    float currentYValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);
    splittingCategoryCount[currentXValue] += 1;
    splittingCategoryYSum[currentXValue] += currentYValue;
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
    float currentXValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    averagingCategoryCount[currentXValue] += 1;
  }

  // Go through the sums and determine the best partition
  for (
      std::set<float>::iterator it=all_categories.begin();
      it != all_categories.end();
      ++it
  ) {
    // Check leaf size at least nodesize
    if (
        std::min(
          splittingCategoryCount[*it],
                                splitTotalCount - splittingCategoryCount[*it]
        ) < splitNodeSize ||
          std::min(
            averagingCategoryCount[*it],
                                averageTotalCount - averagingCategoryCount[*it]
          ) < averageNodeSize
    ) {
      continue;
    }

    float leftPartitionMean = splittingCategoryYSum[*it] /
      splittingCategoryCount[*it];
    float rightPartitionMean = (splitTotalSum -
                                splittingCategoryYSum[*it]) /
                                  (splitTotalCount - splittingCategoryCount[*it]);
    float currentSplitLoss = splittingCategoryCount[*it] *
      leftPartitionMean * leftPartitionMean +
      (splitTotalCount - splittingCategoryCount[*it]) *
      rightPartitionMean * rightPartitionMean;

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      currentSplitLoss,
      *it,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
  }
}

void findBestSplitRidge(
  std::vector<size_t>* averagingSampleIndex,
  std::vector<size_t>* splittingSampleIndex,
  size_t bestSplitTableIndex,
  size_t currentFeature,
  float* bestSplitLossAll,
  double* bestSplitValueAll,
  size_t* bestSplitFeatureAll,
  size_t* bestSplitCountAll,
  DataFrame* trainingData,
  size_t splitNodeSize,
  size_t averageNodeSize,
  std::mt19937_64& random_number_generator,
  bool splitMiddle,
  size_t maxObs,
  float overfitPenalty,
  std::vector<double>* benchmark,
  arma::Mat<double>& gTotal,
  arma::Mat<double>& sTotal
){

  //Get indexes of observations
  std::vector<size_t> splittingIndexes;
  std::vector<size_t> averagingIndexes;

  for (size_t i = 0; i < splittingSampleIndex->size(); i++) {
    splittingIndexes.push_back((*splittingSampleIndex)[i]);
  }

  for (size_t j = 0; j < averagingSampleIndex->size(); j++) {
    averagingIndexes.push_back((*averagingSampleIndex)[j]);
  }

  //Sort indexes of observations ascending by currentFeature
  std::vector<float>* featureData = trainingData->getFeatureData(currentFeature);

  sort(splittingIndexes.begin(),
       splittingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  sort(averagingIndexes.begin(),
       averagingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  size_t splitLeftCount = 0;
  size_t averageLeftCount = 0;
  size_t splitTotalCount = splittingIndexes.size();
  size_t averageTotalCount = averagingIndexes.size();

  std::vector<size_t>::iterator splitIter = splittingIndexes.begin();
  std::vector<size_t>::iterator averageIter = averagingIndexes.begin();
  /* Increment splitIter because we have initialized RSS components with
   * observation from splitIter.begin(), so we need to avoid duplicate 1st obs
   */


  //Now begin splitting
  size_t currentIndex;

  /* Need at least one splitOb to evaluate RSS */
  currentIndex = (*splitIter);
  ++splitIter;
  splitLeftCount++;

  /* Move appropriate averagingObs to left */

  while (
    averageIter < averagingIndexes.end() && (
    trainingData->getPoint((*averageIter), currentFeature) <=
    trainingData->getPoint(currentIndex, currentFeature))
  ) {
    ++averageIter;
    averageLeftCount++;
  }

  float currentValue = trainingData->getPoint(currentIndex, currentFeature);

  size_t newIndex;
  size_t numLinearFeatures;
  bool oneDistinctValue = true;

  //Initialize RSS components
  //TODO: think about completely duplicate observations

  std::vector<float> firstOb = trainingData->getLinObsData(currentIndex);

  numLinearFeatures = firstOb.size();
  firstOb.push_back(1.0);

  //Initialize crossingObs for body of loop
  arma::Mat<double> crossingObservation(firstOb.size(),
                                       1);

  arma::Mat<double> obOuter(numLinearFeatures + 1,
                           numLinearFeatures + 1);

  crossingObservation.col(0) = arma::conv_to<arma::Col<double> >::from(firstOb);

  arma::Mat<double> aLeft(numLinearFeatures + 1, numLinearFeatures + 1),
                   aRight(numLinearFeatures + 1, numLinearFeatures + 1),
                   gLeft(numLinearFeatures + 1, numLinearFeatures + 1),
                   gRight(numLinearFeatures + 1, numLinearFeatures + 1),
                   sLeft(numLinearFeatures + 1, 1),
                   sRight(numLinearFeatures + 1, 1);

  initializeRSSComponents(
    trainingData,
    currentIndex,
    numLinearFeatures,
    overfitPenalty,
    gTotal,
    sTotal,
    aLeft,
    aRight,
    sLeft,
    sRight,
    gLeft,
    gRight,
    crossingObservation
  );

  while (
      splitIter < splittingIndexes.end() ||
        averageIter < averagingIndexes.end()
  ) {

    currentValue = trainingData->getPoint(currentIndex, currentFeature);
    //Move iterators forward
    while (
        splitIter < splittingIndexes.end() &&
          trainingData->getPoint((*splitIter), currentFeature) <= currentValue
    ) {
      //UPDATE RSS pieces with current splitIter index
      updateRSSComponents(
        trainingData,
        (*splitIter),
        aLeft,
        aRight,
        sLeft,
        sRight,
        gLeft,
        gRight,
        crossingObservation,
        obOuter
      );

      splitLeftCount++;
      ++splitIter;
    }

    while (
            averageIter < averagingIndexes.end() &&
            trainingData->getPoint((*averageIter), currentFeature) <=
            currentValue
            ) {
      averageLeftCount++;
      ++averageIter;
    }

    //Test if we only have one feature value to be considered
    if (oneDistinctValue) {
      oneDistinctValue = false;
      if (
          splitIter == splittingIndexes.end() &&
            averageIter == averagingIndexes.end()
      ) {
        break;
      }
    }

    //Set newIndex to index iterator with the minimum currentFeature value
    if (
        splitIter == splittingIndexes.end() &&
          averageIter == averagingIndexes.end()
    ) {
      break;
    } else if (
        splitIter == splittingIndexes.end()
    ) {
      /* Can't pass down matrix if we split past last splitting index */
      break;
    } else if (
        averageIter == averagingIndexes.end()
    ) {
      newIndex = (*splitIter);
    } else if (
        trainingData->getPoint((*averageIter), currentFeature) <
          trainingData->getPoint((*splitIter), currentFeature)
    ) {
      newIndex = (*averageIter);
    } else {
      newIndex = (*splitIter);
    }

    //Check if split would create a node too small
    if (
        std::min(
        splitLeftCount,
        splitTotalCount - splitLeftCount
        ) < splitNodeSize ||
        std::min(
          averageLeftCount,
          averageTotalCount - averageLeftCount
        ) < averageNodeSize
        ) {
        currentIndex = newIndex;
        continue;
    }

    //Sum of RSS's of models fit on left and right partitions
    float currentRSS = computeRSSArmadillo(aRight,
                                           aLeft,
                                           sRight,
                                           sLeft,
                                           gRight,
                                           gLeft);

    double currentSplitValue;

    float featureValue = trainingData->getPoint(currentIndex, currentFeature);

    float newFeatureValue = trainingData->getPoint(newIndex, currentFeature);

    if (splitMiddle) {
      currentSplitValue = (featureValue + newFeatureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }
    //Rcpp::Rcout << currentRSS << " " << currentSplitValue << "\n";
    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      -currentRSS,
      currentSplitValue,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );
    currentIndex = newIndex;
  }
}


void findBestSplitValueNonCategorical(
  std::vector<size_t>* averagingSampleIndex,
  std::vector<size_t>* splittingSampleIndex,
  size_t bestSplitTableIndex,
  size_t currentFeature,
  float* bestSplitLossAll,
  double* bestSplitValueAll,
  size_t* bestSplitFeatureAll,
  size_t* bestSplitCountAll,
  DataFrame* trainingData,
  size_t splitNodeSize,
  size_t averageNodeSize,
  std::mt19937_64& random_number_generator,
  bool splitMiddle,
  size_t maxObs
) {

  // Create specific vectors to holddata
  typedef std::tuple<float,float> dataPair;
  std::vector<dataPair> splittingData;
  std::vector<dataPair> averagingData;
  float splitTotalSum = 0;
  for (size_t j=0; j<(*splittingSampleIndex).size(); j++){
    // Retrieve the current feature value
    float tmpFeatureValue = (*trainingData).
    getPoint((*splittingSampleIndex)[j], currentFeature);
    float tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*splittingSampleIndex)[j]);
    splitTotalSum += tmpOutcomeValue;

    // Adding data to the internal data vector (Note: R index)
    splittingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        tmpOutcomeValue
      )
    );
  }

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++){
    // Retrieve the current feature value
    float tmpFeatureValue = (*trainingData).
    getPoint((*averagingSampleIndex)[j], currentFeature);
    float tmpOutcomeValue = (*trainingData).
    getOutcomePoint((*averagingSampleIndex)[j]);

    // Adding data to the internal data vector (Note: R index)
    averagingData.push_back(
      std::make_tuple(
        tmpFeatureValue,
        tmpOutcomeValue
      )
    );
  }
  // If there are more than maxSplittingObs, randomly downsample maxObs samples
  if (maxObs < splittingData.size()) {

    std::vector<dataPair> newSplittingData;
    std::vector<dataPair> newAveragingData;

    std::shuffle(splittingData.begin(), splittingData.end(),
                 random_number_generator);
    std::shuffle(averagingData.begin(), averagingData.end(),
                 random_number_generator);

    for (int q = 0; q < maxObs; q++) {
      newSplittingData.push_back(splittingData[q]);
      newAveragingData.push_back(averagingData[q]);
    }

    std::swap(newSplittingData, splittingData);
    std::swap(newAveragingData, averagingData);

  }

  // Sort both splitting and averaging dataset
  sort(
    splittingData.begin(),
    splittingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );
  sort(
    averagingData.begin(),
    averagingData.end(),
    [](const dataPair &lhs, const dataPair &rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    }
  );

  size_t splitLeftPartitionCount = 0;
  size_t averageLeftPartitionCount = 0;
  size_t splitTotalCount = splittingData.size();
  size_t averageTotalCount = averagingData.size();

  float splitLeftPartitionRunningSum = 0;

  std::vector<dataPair>::iterator splittingDataIter = splittingData.begin();
  std::vector<dataPair>::iterator averagingDataIter = averagingData.begin();

  // Initialize the split value to be minimum of first value in two datsets
  float featureValue = std::min(
    std::get<0>(*splittingDataIter),
    std::get<0>(*averagingDataIter)
  );

  float newFeatureValue;
  bool oneValueDistinctFlag = true;

  while (
      splittingDataIter < splittingData.end() ||
        averagingDataIter < averagingData.end()
  ){

    // Exhaust all current feature value in both dataset as partitioning
    while (
        splittingDataIter < splittingData.end() &&
          std::get<0>(*splittingDataIter) == featureValue
    ) {
      splitLeftPartitionCount++;
      splitLeftPartitionRunningSum += std::get<1>(*splittingDataIter);
      splittingDataIter++;
    }

    while (
        averagingDataIter < averagingData.end() &&
          std::get<0>(*averagingDataIter) == featureValue
    ) {
      averagingDataIter++;
      averageLeftPartitionCount++;
    }

    // Test if the all the values for the feature are the same, then proceed
    if (oneValueDistinctFlag) {
      oneValueDistinctFlag = false;
      if (
          splittingDataIter == splittingData.end() &&
            averagingDataIter == averagingData.end()
      ) {
        break;
      }
    }

    // Make partitions on the current feature and value in both splitting
    // and averaging dataset. `averageLeftPartitionCount` and
    // `splitLeftPartitionCount` already did the partition after we sort the
    // array.
    // Get new feature value
    if (
        splittingDataIter == splittingData.end() &&
          averagingDataIter == averagingData.end()
    ) {
      break;
    } else if (splittingDataIter == splittingData.end()) {
      newFeatureValue = std::get<0>(*averagingDataIter);
    } else if (averagingDataIter == averagingData.end()) {
      newFeatureValue = std::get<0>(*splittingDataIter);
    } else {
      newFeatureValue = std::min(
        std::get<0>(*splittingDataIter),
        std::get<0>(*averagingDataIter)
      );
    }

    // Check leaf size at least nodesize
    if (
        std::min(
          splitLeftPartitionCount,
          splitTotalCount - splitLeftPartitionCount
        ) < splitNodeSize||
          std::min(
            averageLeftPartitionCount,
            averageTotalCount - averageLeftPartitionCount
          ) < averageNodeSize
    ) {
      // Update the oldFeature value before proceeding
      featureValue = newFeatureValue;
      continue;
    }

    // Calculate sample mean in both splitting partitions
    float leftPartitionMean =
      splitLeftPartitionRunningSum / splitLeftPartitionCount;
    float rightPartitionMean =
      (splitTotalSum - splitLeftPartitionRunningSum)
      / (splitTotalCount - splitLeftPartitionCount);

    // Calculate the variance of the splitting
    float muBarSquareSum =
    splitLeftPartitionCount * leftPartitionMean * leftPartitionMean +
    (splitTotalCount - splitLeftPartitionCount) * rightPartitionMean
      * rightPartitionMean;

    double currentSplitValue;
    if (splitMiddle) {
      currentSplitValue = (newFeatureValue + featureValue) / 2.0;
    } else {
      std::uniform_real_distribution<double> unif_dist;
      double tmp_random = unif_dist(random_number_generator) *
        (newFeatureValue - featureValue);
      double epsilon_lower = std::nextafter(featureValue, newFeatureValue);
      double epsilon_upper = std::nextafter(newFeatureValue, featureValue);
      currentSplitValue = tmp_random + featureValue;
      if (currentSplitValue > epsilon_upper) {
        currentSplitValue = epsilon_upper;
      }
      if (currentSplitValue < epsilon_lower) {
        currentSplitValue = epsilon_lower;
      }
    }

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      muBarSquareSum,
      currentSplitValue,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );

    // Update the old feature value
    featureValue = newFeatureValue;
  }
}

void determineBestSplit(
    size_t &bestSplitFeature,
    double &bestSplitValue,
    float &bestSplitLoss,
    size_t mtry,
    float* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    std::mt19937_64& random_number_generator
){

  // Get the best split values among all features
  float bestSplitLoss_ = -std::numeric_limits<float>::infinity();
  std::vector<size_t> bestFeatures;

  for (size_t i=0; i<mtry; i++) {
    if (bestSplitLossAll[i] > bestSplitLoss_) {
      bestSplitLoss_ = bestSplitLossAll[i];
    }
  }

  for (size_t i=0; i<mtry; i++) {
    if (bestSplitLossAll[i] == bestSplitLoss_) {
      for (size_t j=0; j<bestSplitCountAll[i]; j++) {
        bestFeatures.push_back(i);
      }
    }
  }

  // If we found a feasible splitting point
  if (bestFeatures.size() > 0) {

    // If there are multiple best features, sample one according to their
    // frequency of occurence
    std::uniform_int_distribution<size_t> unif_dist(
        0, bestFeatures.size() - 1
    );
    size_t tmp_random = unif_dist(random_number_generator);
    size_t bestFeatureIndex = bestFeatures.at(tmp_random);
    // Return the best splitFeature and splitValue
    bestSplitFeature = bestSplitFeatureAll[bestFeatureIndex];
    bestSplitValue = bestSplitValueAll[bestFeatureIndex];
    bestSplitLoss = bestSplitLoss_;
  } else {
    // If none of the features are possible, return NA
    bestSplitFeature = std::numeric_limits<size_t>::quiet_NaN();
    bestSplitValue = std::numeric_limits<double>::quiet_NaN();
    bestSplitLoss = std::numeric_limits<float>::quiet_NaN();
  }

}


void forestryTree::selectBestFeature(
    size_t &bestSplitFeature,
    double &bestSplitValue,
    float &bestSplitLoss,
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
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark,
    arma::Mat<double> &gTotal,
    arma::Mat<double> &sTotal
){

  // Get the number of total features
  size_t mtry = (*featureList).size();

  // Initialize the minimum loss for each feature
  float* bestSplitLossAll = new float[mtry];
  double* bestSplitValueAll = new double[mtry];
  size_t* bestSplitFeatureAll = new size_t[mtry];
  size_t* bestSplitCountAll = new size_t[mtry];

  for (size_t i=0; i<mtry; i++) {
    bestSplitLossAll[i] = -std::numeric_limits<float>::infinity();
    bestSplitValueAll[i] = std::numeric_limits<double>::quiet_NaN();
    bestSplitFeatureAll[i] = std::numeric_limits<size_t>::quiet_NaN();
    bestSplitCountAll[i] = 0;
  }

  // Iterate each selected features
  for (size_t i=0; i<mtry; i++) {
    size_t currentFeature = (*featureList)[i];
    // Test if the current feature is in the categorical list
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();
    if (
        std::find(
          categorialCols.begin(),
          categorialCols.end(),
          currentFeature
        ) != categorialCols.end()
    ){
      if (ridgeRF) {
        findBestSplitRidgeCategorical(
          averagingSampleIndex,
          splittingSampleIndex,
          i,
          currentFeature,
          bestSplitLossAll,
          bestSplitValueAll,
          bestSplitFeatureAll,
          bestSplitCountAll,
          trainingData,
          getMinNodeSizeToSplitSpt(),
          getMinNodeSizeToSplitAvg(),
          random_number_generator,
          overfitPenalty,
          gTotal,
          sTotal
        );
      } else {
        findBestSplitValueCategorical(
          averagingSampleIndex,
          splittingSampleIndex,
          i,
          currentFeature,
          bestSplitLossAll,
          bestSplitValueAll,
          bestSplitFeatureAll,
          bestSplitCountAll,
          trainingData,
          getMinNodeSizeToSplitSpt(),
          getMinNodeSizeToSplitAvg(),
          random_number_generator,
          maxObs
        );
      }
    } else if (ridgeRF) {
      findBestSplitRidge(
        averagingSampleIndex,
        splittingSampleIndex,
        i,
        currentFeature,
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        trainingData,
        getMinNodeSizeToSplitSpt(),
        getMinNodeSizeToSplitAvg(),
        random_number_generator,
        splitMiddle,
        maxObs,
        overfitPenalty,
        benchmark,
        gTotal,
        sTotal
      );
    } else {
      findBestSplitValueNonCategorical(
        averagingSampleIndex,
        splittingSampleIndex,
        i,
        currentFeature,
        bestSplitLossAll,
        bestSplitValueAll,
        bestSplitFeatureAll,
        bestSplitCountAll,
        trainingData,
        getMinNodeSizeToSplitSpt(),
        getMinNodeSizeToSplitAvg(),
        random_number_generator,
        splitMiddle,
        maxObs
      );
    }
  }

  determineBestSplit(
    bestSplitFeature,
    bestSplitValue,
    bestSplitLoss,
    mtry,
    bestSplitLossAll,
    bestSplitValueAll,
    bestSplitFeatureAll,
    bestSplitCountAll,
    random_number_generator
  );

  // If ridge splitting, need to update RSS components to pass down
  if (ridgeRF) {
    updateBestSplitG(bestSplitGL,
                     bestSplitGR,
                     gTotal,
                     trainingData,
                     splittingSampleIndex,
                     bestSplitFeature,
                     bestSplitValue);

    updateBestSplitS(bestSplitSL,
                     bestSplitSR,
                     sTotal,
                     trainingData,
                     splittingSampleIndex,
                     bestSplitFeature,
                     bestSplitValue);
  }

  delete[](bestSplitLossAll);
  delete[](bestSplitValueAll);
  delete[](bestSplitFeatureAll);
  delete[](bestSplitCountAll);
}

void forestryTree::printTree(){
  (*getRoot()).printSubtree();
}

void forestryTree::trainTiming(){

  /* Record timing for tree construction */
  double total = (*getBenchmark())[7];
  double splittime = 0;
  for (size_t i = 0; i < 7; i++) {
    splittime += (*getBenchmark())[i];
  }

  Rcpp::Rcout << "Sorting: " << 100*((*getBenchmark())[0] / total) << "% "
              << " InitG/S: " << 100*((*getBenchmark())[1] / total) << "% "
              << " InvertA: " << 100*((*getBenchmark())[2] / total) << "% "
              << " UpdateA: " << 100*((*getBenchmark())[3] / total) << "% "
              << " UpdateG: " << 100*((*getBenchmark())[4] / total) << "% "
              << " UpdateS: " << 100*((*getBenchmark())[5] / total) << "% "
              << " getRSS: " << 100*((*getBenchmark())[6] / total) << "% "
              << " Other: " << 100*((total - splittime) / total) << "% "
              << "Time: " << total << " seconds";

  Rcpp::Rcout << "\n";
}

void forestryTree::getOOBindex(
    std::vector<size_t> &outputOOBIndex,
    size_t nRows
){

  // Generate union of splitting and averaging dataset
  std::sort(
    (*getSplittingIndex()).begin(),
    (*getSplittingIndex()).end()
  );
  std::sort(
    (*getAveragingIndex()).begin(),
    (*getAveragingIndex()).end()
  );

  std::vector<size_t> allSampledIndex(
      (*getSplittingIndex()).size() + (*getAveragingIndex()).size()
  );

  std::vector<size_t>::iterator it= std::set_union(
    (*getSplittingIndex()).begin(),
    (*getSplittingIndex()).end(),
    (*getAveragingIndex()).begin(),
    (*getAveragingIndex()).end(),
    allSampledIndex.begin()
  );

  allSampledIndex.resize((unsigned long) (it - allSampledIndex.begin()));

  // Generate a vector of all index based on nRows
  struct IncGenerator {
    size_t current_;
    IncGenerator(size_t start): current_(start) {}
    size_t operator()() { return current_++; }
  };
  std::vector<size_t> allIndex(nRows);
  IncGenerator g(0);
  std::generate(allIndex.begin(), allIndex.end(), g);

  // OOB index is the set difference between sampled index and all index
  std::vector<size_t> OOBIndex(nRows);

  it = std::set_difference (
    allIndex.begin(),
    allIndex.end(),
    allSampledIndex.begin(),
    allSampledIndex.end(),
    OOBIndex.begin()
  );
  OOBIndex.resize((unsigned long) (it - OOBIndex.begin()));

  for (
      std::vector<size_t>::iterator it_ = OOBIndex.begin();
      it_ != OOBIndex.end();
      ++it_
  ) {
    outputOOBIndex.push_back(*it_);
  }

}

void forestryTree::getOOBPrediction(
    std::vector<float> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData
){

  std::vector<size_t> OOBIndex;
  getOOBindex(OOBIndex, trainingData->getNumRows());

  for (
      std::vector<size_t>::iterator it=OOBIndex.begin();
      it!=OOBIndex.end();
      ++it
  ) {

    size_t OOBSampleIndex = *it;

    // Predict current oob sample
    std::vector<float> currentTreePrediction(1);
    std::vector<float> OOBSampleObservation((*trainingData).getNumColumns());
    (*trainingData).getObservationData(OOBSampleObservation, OOBSampleIndex);

    std::vector< std::vector<float> > OOBSampleObservation_;
    for (size_t k=0; k<(*trainingData).getNumColumns(); k++){
      std::vector<float> OOBSampleObservation_iter(1);
      OOBSampleObservation_iter[0] = OOBSampleObservation[k];
      OOBSampleObservation_.push_back(OOBSampleObservation_iter);
    }

    predict(
      currentTreePrediction,
      &OOBSampleObservation_,
      trainingData
    );

    // Update the global OOB vector
    outputOOBPrediction[OOBSampleIndex] += currentTreePrediction[0];
    outputOOBCount[OOBSampleIndex] += 1;
  }
}

void forestryTree::getShuffledOOBPrediction(
    std::vector<float> &outputOOBPrediction,
    std::vector<size_t> &outputOOBCount,
    DataFrame* trainingData,
    size_t shuffleFeature
){
  // Gives OOB prediction with shuffleFeature premuted randomly
  // For use in determining variable importance

  std::vector<size_t> OOBIndex;
  getOOBindex(OOBIndex, trainingData->getNumRows());

  std::vector<size_t> shuffledOOBIndex = OOBIndex;
  std::random_shuffle(shuffledOOBIndex.begin(), shuffledOOBIndex.end());
  size_t currentIndex = 0;

  for (
      std::vector<size_t>::iterator it=OOBIndex.begin();
      it!=OOBIndex.end();
      ++it
  ) {

    size_t OOBSampleIndex = *it;

    // Predict current oob sample
    std::vector<float> currentTreePrediction(1);
    std::vector<float> OOBSampleObservation((*trainingData).getNumColumns());
    (*trainingData).getShuffledObservationData(OOBSampleObservation,
                                               OOBSampleIndex,
                                               shuffleFeature,
                                               shuffledOOBIndex[currentIndex]);

    std::vector< std::vector<float> > OOBSampleObservation_;
    for (size_t k=0; k<(*trainingData).getNumColumns(); k++){
      std::vector<float> OOBSampleObservation_iter(1);
      OOBSampleObservation_iter[0] = OOBSampleObservation[k];
      OOBSampleObservation_.push_back(OOBSampleObservation_iter);
    }

    predict(
      currentTreePrediction,
      &OOBSampleObservation_,
      trainingData
    );

    // Update the global OOB vector
    outputOOBPrediction[OOBSampleIndex] += currentTreePrediction[0];
    outputOOBCount[OOBSampleIndex] += 1;
    currentIndex++;
  }
}

// -----------------------------------------------------------------------------
std::unique_ptr<tree_info> forestryTree::getTreeInfo(
    DataFrame* trainingData
){
  std::unique_ptr<tree_info> treeInfo(
    new tree_info
  );
  (*getRoot()).write_node_info(treeInfo, trainingData);

  for (size_t i = 0; i<_averagingSampleIndex->size(); i++) {
    treeInfo->averagingSampleIndex.push_back((*_averagingSampleIndex)[i] + 1);
  }
  for (size_t i = 0; i<_splittingSampleIndex->size(); i++) {
    treeInfo->splittingSampleIndex.push_back((*_splittingSampleIndex)[i] + 1);
  }

  return treeInfo;
}

void forestryTree::reconstruct_tree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    float minSplitGain,
    size_t maxDepth,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<size_t> categoricalFeatureColsRcpp,
    std::vector<int> var_ids,
    std::vector<double> split_vals,
    std::vector<size_t> leafAveidxs,
    std::vector<size_t> leafSplidxs,
    std::vector<size_t> averagingSampleIndex,
    std::vector<size_t> splittingSampleIndex){

  // Setting all the parameters:
  _mtry = mtry;
  _minNodeSizeSpt = minNodeSizeSpt;
  _minNodeSizeAvg = minNodeSizeAvg;
  _minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  _minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  _minSplitGain = minSplitGain;
  _maxDepth = maxDepth;
  _ridgeRF = ridgeRF;
  _overfitPenalty = overfitPenalty;

  _averagingSampleIndex = std::unique_ptr< std::vector<size_t> > (
    new std::vector<size_t>
  );
  for(size_t i=0; i<averagingSampleIndex.size(); i++){
    (*_averagingSampleIndex).push_back(averagingSampleIndex[i] - 1);
  }
  _splittingSampleIndex = std::unique_ptr< std::vector<size_t> > (
    new std::vector<size_t>
  );
  for(size_t i=0; i<splittingSampleIndex.size(); i++){
    (*_splittingSampleIndex).push_back(splittingSampleIndex[i] - 1);
  }

  std::unique_ptr< RFNode > root ( new RFNode() );
  this->_root = std::move(root);

  recursive_reconstruction(
    _root.get(),
    &var_ids,
    &split_vals,
    &leafAveidxs,
    &leafSplidxs
  );

  return ;
}


void forestryTree::recursive_reconstruction(
  RFNode* currentNode,
  std::vector<int> * var_ids,
  std::vector<double> * split_vals,
  std::vector<size_t> * leafAveidxs,
  std::vector<size_t> * leafSplidxs
) {
  int var_id = (*var_ids)[0];
    (*var_ids).erase((*var_ids).begin());
  double  split_val = (*split_vals)[0];
    (*split_vals).erase((*split_vals).begin());

  if(var_id < 0){
    // This is a terminal node
    int nAve = abs(var_id);
    int nSpl = abs((*var_ids)[0]);
    (*var_ids).erase((*var_ids).begin());

    std::unique_ptr<std::vector<size_t> > averagingSampleIndex_(
        new std::vector<size_t>
    );
    std::unique_ptr<std::vector<size_t> > splittingSampleIndex_(
        new std::vector<size_t>
    );

    for(int i=0; i<nAve; i++){
      averagingSampleIndex_->push_back((*leafAveidxs)[0] - 1);
      (*leafAveidxs).erase((*leafAveidxs).begin());
    }

    for(int i=0; i<nSpl; i++){
      splittingSampleIndex_->push_back((*leafSplidxs)[0] - 1);
      (*leafSplidxs).erase((*leafSplidxs).begin());
    }

    (*currentNode).setLeafNode(
        std::move(averagingSampleIndex_),
        std::move(splittingSampleIndex_)
    );
    return;
  } else {
    // This is a normal splitting node
    std::unique_ptr< RFNode > leftChild ( new RFNode() );
    std::unique_ptr< RFNode > rightChild ( new RFNode() );

    recursive_reconstruction(
      leftChild.get(),
      var_ids,
      split_vals,
      leafAveidxs,
      leafSplidxs
      );

    recursive_reconstruction(
      rightChild.get(),
      var_ids,
      split_vals,
      leafAveidxs,
      leafSplidxs
    );

    (*currentNode).setSplitNode(
      (size_t) var_id - 1,
      split_val,
      std::move(leftChild),
      std::move(rightChild)
      );
    return;
  }
}
