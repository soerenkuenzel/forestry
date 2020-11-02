#ifndef FORESTRYCPP_TREESPLIT_H
#define FORESTRYCPP_TREESPLIT_H

#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include "forestryTree.h"
#include "DataFrame.h"
#include "RFNode.h"
#include "utils.h"
#include <RcppArmadillo.h>

float calculateRSS(
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    float overfitPenalty
);

void updateBestSplit(
    float* bestSplitLossAll,
    double* bestSplitValueAll,
    size_t* bestSplitFeatureAll,
    size_t* bestSplitCountAll,
    float* bestSplitLmeanAll,
    float* bestSplitRmeanAll,
    float currentSplitLoss,
    double currentSplitValue,
    size_t currentFeature,
    size_t bestSplitTableIndex,
    std::mt19937_64& random_number_generator,
    bool monotone_splits,
    float currentSplitLmean,
    float currentSplitRmean
);

void updateBestSplitS(
    arma::Mat<double> &bestSplitSL,
    arma::Mat<double> &bestSplitSR,
    const arma::Mat<double> &sTotal,
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitFeature,
    double bestSplitValue
);

void updateBestSplitG(
    arma::Mat<double> &bestSplitGL,
    arma::Mat<double> &bestSplitGR,
    const arma::Mat<double> &gTotal,
    DataFrame* trainingData,
    std::vector<size_t>* splittingSampleIndex,
    size_t bestSplitFeature,
    double bestSplitValue
);

void updateAArmadillo(
    arma::Mat<double>& a_k,
    arma::Mat<double>& new_x,
    bool leftNode
);

void updateSkArmadillo(
    arma::Mat<double>& s_k,
    arma::Mat<double>& next,
    float next_y,
    bool left
);

float computeRSSArmadillo(
    arma::Mat<double>& A_r,
    arma::Mat<double>& A_l,
    arma::Mat<double>& S_r,
    arma::Mat<double>& S_l,
    arma::Mat<double>& G_r,
    arma::Mat<double>& G_l
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

bool acceptMonotoneSplit(
        monotonic_info &monotone_details,
        size_t currentFeature,
        float leftPartitionMean,
        float rightPartitionMean
);

#endif //FORESTRYCPP_TREESPLIT_H
