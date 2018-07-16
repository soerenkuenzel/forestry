#include <RcppArmadillo.h>
#include "forestryTree.h"
#include <RcppEigen.h>
#include <math.h>
#include <set>
#include <map>
#include <random>
#include <sstream>
#include <chrono>
#include "utils.h"
// [[Rcpp::plugins(cpp11)]]

forestryTree::forestryTree():
  _mtry(0),
  _minNodeSizeSpt(0),
  _minNodeSizeAvg(0),
  _minNodeSizeToSplitSpt(0),
  _minNodeSizeToSplitAvg(0),
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
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
  std::unique_ptr< RFNode > root ( new RFNode() );
  this->_root = std::move(root);
  this->_benchmark = new std::vector<double>;
  for (size_t i = 0; i < 8; i++) {
    _benchmark->push_back(0);
  }


  /* Recursively grow the tree */
  recursivePartition(
    getRoot(),
    getAveragingIndex(),
    getSplittingIndex(),
    trainingData,
    random_number_generator,
    splitMiddle,
    maxObs,
    ridgeRF,
    overfitPenalty,
    getBenchmark()
  );

  //this->_root->printSubtree();
  this->trainTiming();
}

void forestryTree::setDummyTree(
    size_t mtry,
    size_t minNodeSizeSpt,
    size_t minNodeSizeAvg,
    size_t minNodeSizeToSplitSpt,
    size_t minNodeSizeToSplitAvg,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex,
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    float overfitPenalty
){
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_averagingSampleIndex = std::move(averagingSampleIndex);
  this->_splittingSampleIndex = std::move(splittingSampleIndex);
  this->_overfitPenalty = overfitPenalty;
}

void forestryTree::predict(
    std::vector<float> &outputPrediction,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    Eigen::MatrixXf* weightMatrix,
    bool ridgeRF
){
  // If we are estimating the average in each leaf:
  struct rangeGenerator {
    size_t currentNumber;
    rangeGenerator(size_t startNumber): currentNumber(startNumber) {};
    size_t operator()() {return currentNumber++; }
  };

  std::vector<size_t> updateIndex(outputPrediction.size());
  rangeGenerator _rangeGenerator(0);
  std::generate(updateIndex.begin(), updateIndex.end(), _rangeGenerator);
  (*getRoot()).predict(outputPrediction, &updateIndex, xNew, trainingData,
   weightMatrix, ridgeRF, getOverfitPenalty());
}


std::vector<size_t> sampleFeatures(
    size_t mtry,
    std::mt19937_64& random_number_generator,
    int totalColumns,
    bool numFeaturesOnly,
    std::vector<size_t>* numCols
){
  // Sample features without replacement
  std::vector<size_t> featureList;
  if (numFeaturesOnly) {

    while (featureList.size() < mtry) {
      std::uniform_int_distribution<size_t> unif_dist(
          0, (size_t) numCols->size() - 1
      );

      size_t index = unif_dist(random_number_generator);

      if (featureList.size() == 0 ||
          std::find(featureList.begin(),
                    featureList.end(),
                    (*numCols)[index]) == featureList.end()
      ) {
        featureList.push_back((*numCols)[index]);
      }
    }

  } else {
    while (featureList.size() < mtry) {
      std::uniform_int_distribution<size_t> unif_dist(
          0, (size_t) totalColumns - 1
      );
      size_t randomIndex = unif_dist(random_number_generator);
      if (
          featureList.size() == 0 ||
            std::find(
              featureList.begin(),
              featureList.end(),
              randomIndex
            ) == featureList.end()
      ) {
        featureList.push_back(randomIndex);
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


void forestryTree::recursivePartition(
    RFNode* rootNode,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark
){

  if ((*averagingSampleIndex).size() < getMinNodeSizeAvg() ||
      (*splittingSampleIndex).size() < getMinNodeSizeSpt()) {
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
  if (ridgeRF) {
    featureList = sampleFeatures(
      getMtry(),
      random_number_generator,
      ((int) (*trainingData).getNumColumns()),
      true,
      trainingData->getNumCols()
    );
  } else {
    featureList = sampleFeatures(
      getMtry(),
      random_number_generator,
      ((int) (*trainingData).getNumColumns()),
      false,
      trainingData->getNumCols()
    );
  }

  // Select best feature
  size_t bestSplitFeature;
  double bestSplitValue;
  float bestSplitLoss;

  selectBestFeature(
    bestSplitFeature,
    bestSplitValue,
    bestSplitLoss,
    &featureList,
    averagingSampleIndex,
    splittingSampleIndex,
    trainingData,
    random_number_generator,
    splitMiddle,
    maxObs,
    ridgeRF,
    overfitPenalty,
    benchmark
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
    // Update sample index for both left and right partitions
    // Recursively grow the tree
    std::unique_ptr< RFNode > leftChild ( new RFNode() );
    std::unique_ptr< RFNode > rightChild ( new RFNode() );

    recursivePartition(
      leftChild.get(),
      &averagingLeftPartitionIndex,
      &splittingLeftPartitionIndex,
      trainingData,
      random_number_generator,
      splitMiddle,
      maxObs,
      ridgeRF,
      overfitPenalty,
      benchmark
    );
    recursivePartition(
      rightChild.get(),
      &averagingRightPartitionIndex,
      &splittingRightPartitionIndex,
      trainingData,
      random_number_generator,
      splitMiddle,
      maxObs,
      ridgeRF,
      overfitPenalty,
      benchmark
    );

    (*rootNode).setSplitNode(
        bestSplitFeature,
        bestSplitValue,
        std::move(leftChild),
        std::move(rightChild)
    );
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

void updateAArmadillo(
    arma::Mat<float>& a_k,
    arma::Mat<float>& new_x,
    bool leftNode
){


  //Initilize z_K
  arma::Mat<float> z_K = a_k * new_x;

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
    arma::Mat<float>& s_k,
    arma::Mat<float>& next,
    float next_y,
    bool left
){
  if (left) {
    s_k = s_k + (next_y * (next));
  } else {
    s_k = s_k - (next_y * (next));
  }
}

void updateGkArmadillo(
    arma::Mat<float>& g_k,
    arma::Mat<float>& next,
    bool left
){
  if (left) {
    g_k = g_k + (next * next.t());
  } else {
    g_k = g_k - (next * next.t());
  }
}

float computeRSSArmadillo(
    arma::Mat<float>& A_r,
    arma::Mat<float>& A_l,
    arma::Mat<float>& S_r,
    arma::Mat<float>& S_l,
    arma::Mat<float>& G_r,
    arma::Mat<float>& G_l
){
  return (as_scalar((S_l.t() * A_l) * (G_l * (A_l * S_l))) +
          as_scalar((S_r.t() * A_r) * (G_r * (A_r * S_r))) -
          as_scalar(2.0 * S_l.t() * (A_l * S_l)) -
          as_scalar(2.0 * S_r.t() * (A_r * S_r)));
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
  std::vector<double>* benchmark
){

  auto startTotal = std::chrono::high_resolution_clock::now();

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

  auto startSort = std::chrono::high_resolution_clock::now();

  sort(splittingIndexes.begin(),
       splittingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  sort(averagingIndexes.begin(),
       averagingIndexes.end(),
       [&](int fi, int si){return (*featureData)[fi] < (*featureData)[si];});

  auto endSort = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedSort = endSort - startSort;
  (*benchmark)[0] += elapsedSort.count();


  size_t splitLeftCount = 0;
  size_t averageLeftCount = 0;
  size_t splitTotalCount = splittingIndexes.size();
  size_t averageTotalCount = averagingIndexes.size();

  std::vector<size_t>::iterator splitIter = splittingIndexes.begin();
  std::vector<size_t>::iterator averageIter = averagingIndexes.begin();


  //Now begin splitting
  size_t currentIndex;

  if (
    trainingData->getPoint((*averageIter), currentFeature) <
    trainingData->getPoint((*splitIter), currentFeature)
  ) {
    currentIndex = (*averageIter);
  } else {
    currentIndex = (*splitIter);
  }

  float currentValue = trainingData->getPoint(currentIndex, currentFeature);

  size_t newIndex;
  size_t numLinearFeatures;
  bool oneDistinctValue = true;

  //Initialize RSS components
  //TODO: think about completely duplicate observations

  std::vector<float> firstOb = trainingData->getLinObsData(splittingIndexes[0]);

  numLinearFeatures = firstOb.size();
  firstOb.push_back(1.0);
  arma::Mat<float> appendedFOb(firstOb.size(),
                               1);
  appendedFOb.col(0) = arma::conv_to<arma::Col<float> >::from(firstOb);

  std::vector<float> nextOb = trainingData->getLinObsData(splittingIndexes[1]);

  nextOb.push_back(1.0);
  arma::Mat<float> appendedSOb(nextOb.size(),
                               1);
  appendedSOb.col(0) = arma::conv_to<arma::Col<float> >::from(nextOb);

  //Initialize crossingObs for body of loop
  arma::Mat<float> crossingObservation(firstOb.size(),
                                       1);
  crossingObservation.col(0) = arma::conv_to<arma::Col<float> >::from(firstOb);


  //Initialize sLeft
  arma::Mat<float> sLeft = trainingData->getOutcomePoint(splittingIndexes[0]) *
                          appendedFOb;

  arma::Mat<float> sRight = trainingData->getOutcomePoint(splittingIndexes[1]) *
                          appendedSOb;

  //Initialize gLeft

  auto startSG = std::chrono::high_resolution_clock::now();
  arma::Mat<float> gLeft = appendedFOb * (appendedFOb.t());

  arma::Mat<float> gRight = appendedSOb * (appendedSOb.t());


  //Initialize sRight, gRight
  //?MAPPING PROBLEM

  //Todo: clean this up
  std::vector<float> temp = trainingData->getLinObsData(splittingIndexes[2]);
  temp.push_back(1.0);
  arma::Mat<float> appendedTemp(temp.size(),
                                1);
  appendedTemp.col(0) = arma::conv_to<arma::Col<float> >::from(temp);

  sRight = sRight + trainingData->getOutcomePoint(splittingIndexes[2]) * appendedTemp;
  gRight = gRight + appendedTemp * appendedTemp.t();

  for (size_t d = 3; d < splittingIndexes.size(); d++) {
    temp = trainingData->getLinObsData(splittingIndexes[d]);
    temp.push_back(1.0);
    appendedTemp.col(0) = arma::conv_to<arma::Col<float> >::from(temp);

    sRight = sRight + trainingData->getOutcomePoint(splittingIndexes[d]) * appendedTemp;
    gRight = gRight + appendedTemp * appendedTemp.t();
  }

  auto endSG = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedSG = endSG - startSG;
  (*benchmark)[1] += elapsedSG.count();

  arma::Mat<float> identity(numLinearFeatures + 1,
                            numLinearFeatures + 1);
  identity.eye();

  //Don't penalize intercept
  identity(numLinearFeatures, numLinearFeatures) = 0.0;

  //Initialize aLeft
  auto startIA = std::chrono::high_resolution_clock::now();
  arma::Mat<float> aLeft = (gLeft + overfitPenalty * identity).i();

  //Initialize aRight
  arma::Mat<float> aRight = (gRight + overfitPenalty * identity). i();

  auto endIA = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedIA = endIA - startIA;
  (*benchmark)[2] += elapsedIA.count();


  arma::Mat<float> obOuter(numLinearFeatures+1,
                          numLinearFeatures+1);


  while (
      splitIter < splittingIndexes.end() ||
        averageIter < averagingIndexes.end()
  ) {


    //TODO: HANDLE SINGLE DISTINCT VALUE CASE ////DONE
    //TODO: MORE ELEGANT HANDLING


    currentValue = trainingData->getPoint(currentIndex, currentFeature);
    //Move iterators forward
    while (
        splitIter < splittingIndexes.end() &&
          trainingData->getPoint((*splitIter), currentFeature) == currentValue
    ) {
      splitLeftCount++;
      //UPDATE MATRIX pieces with current splitIter index
      //MAPPING PROBLEM

      //Get observation that will cross the partition
      std::vector<float> newLeftObservation = trainingData->getLinObsData((*splitIter));

      newLeftObservation.push_back(1.0);

      crossingObservation.col(0) = arma::conv_to<arma::Col<float> >::from(newLeftObservation);



      float crossingOutcome = trainingData->getOutcomePoint((*splitIter));


      //Use to update RSS components

      auto startUA = std::chrono::high_resolution_clock::now();

      updateAArmadillo(aLeft, crossingObservation, true);
      updateAArmadillo(aRight, crossingObservation, false);

      auto endUA = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsedUA = endUA - startUA;
      (*benchmark)[3] += elapsedUA.count();

      auto startUS = std::chrono::high_resolution_clock::now();

      updateSkArmadillo(sLeft, crossingObservation, crossingOutcome, true);
      updateSkArmadillo(sRight, crossingObservation, crossingOutcome, false);

      auto endUS = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsedUS = endUS - startUS;
      (*benchmark)[5] += elapsedUS.count();

      auto startUG = std::chrono::high_resolution_clock::now();

      obOuter = crossingObservation * crossingObservation.t();
      gLeft = gLeft - obOuter;
      gRight = gRight + obOuter;

      auto endUG = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsedUG = endUG - startUG;
      (*benchmark)[4] += elapsedUG.count();



      ++splitIter;
    }

    while (
            averageIter < averagingIndexes.end() &&
            trainingData->getPoint((*averageIter), currentFeature) ==
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
      newIndex = (*averageIter);
    } else if (
        averageIter == averagingIndexes.end()
    ) {
      newIndex = (*splitIter);
    } else {
      if (
          trainingData->getPoint((*averageIter), currentFeature) <
            trainingData->getPoint((*splitIter), currentFeature)
      ) {
        newIndex = (*averageIter);
      } else {
        newIndex = (*splitIter);
      }
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

    auto startRss = std::chrono::high_resolution_clock::now();

    //Sum of RSS's of models fit on left and right partitions
    float currentRSS = computeRSSArmadillo(aRight,
                                           aLeft,
                                           sRight,
                                           sLeft,
                                           gRight,
                                           gLeft);

    auto endRss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedRss = endRss - startRss;
    (*benchmark)[6] += elapsedRss.count();

    double currentSplitValue;

    float featureValue = trainingData->getPoint(currentIndex,
                                                currentFeature);
    float newFeatureValue = trainingData->getPoint(newIndex,
                                                   currentFeature);
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

    updateBestSplit(
      bestSplitLossAll,
      bestSplitValueAll,
      bestSplitFeatureAll,
      bestSplitCountAll,
      currentRSS,
      currentSplitValue,
      currentFeature,
      bestSplitTableIndex,
      random_number_generator
    );

    currentIndex = newIndex;
  }

  auto endTotal = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedTotal = endTotal - startTotal;
  (*benchmark)[7] += elapsedTotal.count();


//Rcpp::Rcout << "Initialization " << benchmark[0] << " A inversion took: " << benchmark[1] << " InitGk: " << benchmark[3] << " other init took " <<benchmark[0] - benchmark[1] - benchmark[3] << " seconds Total time took: " << benchmark[2] << "seconds \n";
//Rcpp::Rcout << "Armadillo Splitting on: " << *bestSplitValueAll << " in feature " << currentFeature << "\n \n";
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
    std::vector<size_t>* featureList,
    std::vector<size_t>* averagingSampleIndex,
    std::vector<size_t>* splittingSampleIndex,
    DataFrame* trainingData,
    std::mt19937_64& random_number_generator,
    bool splitMiddle,
    size_t maxObs,
    bool ridgeRF,
    float overfitPenalty,
    std::vector<double>* benchmark
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
        benchmark
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

  delete[](bestSplitLossAll);
  delete[](bestSplitValueAll);
  delete[](bestSplitFeatureAll);
  delete[](bestSplitCountAll);

}

void forestryTree::printTree(){
  (*getRoot()).printSubtree();
}

void forestryTree::trainTiming(){
  double total = (*getBenchmark())[7];

  Rcpp::Rcout << " Sorting: " << 100*((*getBenchmark())[0] / total) << " % "
              << " InitG/S: " << 100*((*getBenchmark())[1] / total) << " % "
              << " InvertA: " << 100*((*getBenchmark())[2] / total) << " % "
              << " UpdateA: " << 100*((*getBenchmark())[3] / total) << " % "
              << " UpdateG: " << 100*((*getBenchmark())[4] / total) << " % "
              << " UpdateS: " << 100*((*getBenchmark())[5] / total) << " % "
              << " getRSS: " << 100*((*getBenchmark())[6] / total) << " %";

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
