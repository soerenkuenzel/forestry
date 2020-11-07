#include "treeSplitting.h"
#include "forestryTree.h"
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>
#include <math.h>
#include <set>
#include <map>
#include <random>
#include <algorithm>
#include <sstream>
#include <tuple>
// [[Rcpp::plugins(cpp11)]]

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
    const arma::Mat<double> &sTotal,
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
    const arma::Mat<double> &gTotal,
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

void updateAArmadillo(
    arma::Mat<double>& a_k,
    arma::Mat<double>& new_x,
    bool leftNode
) {
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
) {
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
) {
  return (as_scalar((S_l.t() * A_l) * (G_l * (A_l * S_l))) +
          as_scalar((S_r.t() * A_r) * (G_r * (A_r * S_r))) -
          as_scalar(2.0 * S_l.t() * (A_l * S_l)) -
          as_scalar(2.0 * S_r.t() * (A_r * S_r)));
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
    const arma::Mat<double>& gTotal,
    const arma::Mat<double>& sTotal,
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
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal
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
  arma::Mat<double> gRightTemp(size((*gtotal)));
  arma::Mat<double> sRightTemp(size((*stotal)));
  arma::Mat<double> aRightTemp(size((*gtotal)));
  arma::Mat<double> aLeftTemp(size((*gtotal)));
  arma::Mat<double> crossingObservation(size((*stotal)));
  arma::Mat<double> identity(size((*gtotal)));

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
    gMatrices[*it] = arma::Mat<double>(size(*gtotal)).zeros();
    sMatrices[*it] = arma::Mat<double>(size(*stotal)).zeros();
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
    gRightTemp = (*gtotal) - gMatrices[*it];
    sRightTemp = (*stotal) - sMatrices[*it];

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
    std::shared_ptr< arma::Mat<double> > gtotal,
    std::shared_ptr< arma::Mat<double> > stotal
) {

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
    (*gtotal),
    (*stotal),
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
) {

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
  // if (maxObs < (*splittingSampleIndex).size()) {
  //   std::vector<size_t> newSplittingIndices;
  //   std::vector<size_t> newAveragingIndices;
  //
  //   std::shuffle(splittingIndices.begin(), splittingIndices.end(),
  //                random_number_generator);
  //   std::shuffle(averagingIndices.begin(), averagingIndices.end(),
  //                random_number_generator);
  //
  //   for (int q = 0; q < maxObs; q++) {
  //     newSplittingIndices.push_back(splittingIndices[q]);
  //     newAveragingIndices.push_back(averagingIndices[q]);
  //   }
  //
  //   std::swap(newSplittingIndices, splittingIndices);
  //   std::swap(newAveragingIndices, averagingIndices);
  // }

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
    size_t maxObs,
    float maxProp,
    bool monotone_splits,
    monotonic_info monotone_details
) {

  // Create specific vectors to holddata
  typedef std::tuple<float,float> dataPair;
  std::vector<dataPair> splittingData;
  std::vector<dataPair> averagingData;
  float splitTotalSum = 0;

  for (size_t j=0; j<(*splittingSampleIndex).size(); j++) {
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

  for (size_t j=0; j<(*averagingSampleIndex).size(); j++) {
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
  bool useMaxProp = maxProp < 1 && (splittingData.size() > 15);
  std::vector<dataPair> newSplittingData;
  std::vector<dataPair> newAveragingData;

  // Max proportion takes precedent over maxObs, so if maxProp < 1, we
  // downsample by that, otherwise we downsample by
  if (useMaxProp) {
    size_t nSample = (size_t) ceil(maxProp * (float) splittingData.size());

    // for now take at least 10 samples- this should be nodesizeSpl
    nSample = std::max(nSample, (size_t) 15);

    std::shuffle(splittingData.begin(), splittingData.end(),
                 random_number_generator);
    std::shuffle(averagingData.begin(), averagingData.end(),
                 random_number_generator);

    for (size_t q = 0; q < nSample; q++) {
      newSplittingData.push_back(splittingData[q]);
      newAveragingData.push_back(averagingData[q]);
    }

  } else if (maxObs < splittingData.size()) {

    std::shuffle(splittingData.begin(), splittingData.end(),
                 random_number_generator);
    std::shuffle(averagingData.begin(), averagingData.end(),
                 random_number_generator);

    for (size_t q = 0; q < maxObs; q++) {
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

  float newFeatureValue;
  bool oneValueDistinctFlag = true;

  float splitLeftPartitionRunningSum = 0;

  std::vector<dataPair>::iterator splittingDataIter = splittingData.begin();
  std::vector<dataPair>::iterator averagingDataIter = averagingData.begin();

  // Initialize the split value to be minimum of first value in two datasets
  float featureValue = std::min(
    std::get<0>(*splittingDataIter),
    std::get<0>(*averagingDataIter)
  );

  // If we are to sample using maxProp, we will only use the randomly selected
  // splitting observations which we have selected above.
  if (useMaxProp) {

    // Now as we have downsampled, we only iterate through the selectedSplits
    std::vector<dataPair>::iterator SelectSplittingDataIter = newSplittingData.begin();
    featureValue = std::get<0>(*SelectSplittingDataIter);

    while (
        SelectSplittingDataIter < newSplittingData.end()
    ) {

      //Now have to iterate select point iterator too
      while (
          SelectSplittingDataIter < newSplittingData.end() &&
            std::get<0>(*SelectSplittingDataIter) == featureValue
      ) {
        SelectSplittingDataIter++;
      }

      // Now only look at the splitting points
      while (
          splittingDataIter < splittingData.end() &&
            std::get<0>(*splittingDataIter) <= featureValue
      ) {
        splitLeftPartitionCount++;
        splitLeftPartitionRunningSum += std::get<1>(*splittingDataIter);
        splittingDataIter++;
      }

      while (
          averagingDataIter < averagingData.end() &&
            std::get<0>(*averagingDataIter) <= featureValue
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
          SelectSplittingDataIter == newSplittingData.end()
      ) {
        break;
      } else {
        newFeatureValue = std::get<0>(*SelectSplittingDataIter);
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

      // If we are using monotonic constraints, we need to work out whether
      // the monotone constraints will reject a split
      if (monotone_splits && (monotone_details.monotonic_constraints[currentFeature] != 0)) {
        bool keepMonotoneSplit = acceptMonotoneSplit(monotone_details,
                                                     currentFeature,
                                                     leftPartitionMean,
                                                     rightPartitionMean);

        if (!keepMonotoneSplit) {
          // Update the oldFeature value before proceeding
          featureValue = newFeatureValue;
          continue;
        }
      }

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

    // If we are not using maxProp, we iterate through splitting points normally
  } else {
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

      // If we are using monotonic constraints, we need to work out whether
      // the monotone constraints will reject a split
      if (monotone_splits && (monotone_details.monotonic_constraints[currentFeature] != 0)) {
        bool keepMonotoneSplit = acceptMonotoneSplit(monotone_details,
                                                     currentFeature,
                                                     leftPartitionMean,
                                                     rightPartitionMean);
        if (!keepMonotoneSplit) {
          // Update the oldFeature value before proceeding
          featureValue = newFeatureValue;
          continue;
        }
      }

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
) {

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

bool acceptMonotoneSplit(
  monotonic_info &monotone_details,
  size_t currentFeature,
  float leftPartitionMean,
  float rightPartitionMean
) {
    // If we have the uncle mean equal to infinity, then we enforce a simple
    // monotone split without worrying about the uncle bounds
    int monotone_direction = monotone_details.monotonic_constraints[currentFeature];
    float upper_bound = monotone_details.upper_bound;
    float lower_bound = monotone_details.lower_bound;

    // This is not right. I should check the split is correctly monotonic and then
    // check that neither node violates the upper and lower bounds

    // Monotone increasing
    if ((monotone_direction == 1) && (leftPartitionMean > rightPartitionMean)) {
      return false;
    } else if ((monotone_direction == -1) && (rightPartitionMean > leftPartitionMean)) {
      // Monotone decreasing
      return false;
    } else if ((monotone_direction == 1) && (rightPartitionMean > upper_bound)) {
      return false;
    } else if ((monotone_direction == 1) && (leftPartitionMean < lower_bound)) {
      return false;
    } else if ((monotone_direction == -1) && (rightPartitionMean < lower_bound)) {
      return false;
    } else if ((monotone_direction == -1) && (leftPartitionMean > upper_bound)) {
      return false;
    } else {
      return true;
    }
}

float calculateMonotonicBound(
  float node_mean,
  monotonic_info& monotone_details
) {
  if (node_mean < monotone_details.lower_bound) {
    return monotone_details.lower_bound;
  } else if (node_mean > monotone_details.upper_bound) {
    return monotone_details.upper_bound;
  } else {
    return node_mean;
  }
}

