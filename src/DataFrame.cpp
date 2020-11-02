#include "DataFrame.h"

DataFrame::DataFrame():
  _featureData(nullptr), _outcomeData(nullptr), _rowNumbers(nullptr),
  _categoricalFeatureCols(nullptr), _numericalFeatureCols(nullptr),
  _splitFeatureCols(nullptr), _linearFeatureCols(nullptr),
  _numRows(0), _numColumns(0), _sampleWeights(nullptr) {}

DataFrame::~DataFrame() {
//  std::cout << "DataFrame() destructor is called." << std::endl;
}

DataFrame::DataFrame(
  std::shared_ptr< std::vector< std::vector<float> > > featureData,
  std::unique_ptr< std::vector<float> > outcomeData,
  std::unique_ptr< std::vector<size_t> > categoricalFeatureCols,
  std::unique_ptr< std::vector<size_t> > splitFeatureCols,
  std::unique_ptr< std::vector<size_t> > linearFeatureCols,
  std::size_t numRows,
  std::size_t numColumns,
  std::unique_ptr< std::vector<float> > sampleWeights,
  std::shared_ptr< std::vector<int> > monotonicConstraints
) {
  this->_featureData = std::move(featureData);
  this->_outcomeData = std::move(outcomeData);
  this->_categoricalFeatureCols = std::move(categoricalFeatureCols);
  this->_splitFeatureCols = std::move(splitFeatureCols);
  this->_linearFeatureCols = std::move(linearFeatureCols);
  this->_numRows = numRows;
  this->_numColumns = numColumns;
  this->_sampleWeights = std::move(sampleWeights);
  this->_monotonicConstraints = std::move(monotonicConstraints);

  // define the row numbers to be the numbers from 1 to nrow:
  std::vector<size_t> rowNumberss;
  for(size_t j=0; j<numRows; j++){
    rowNumberss.push_back(j+1);
  }
  std::unique_ptr< std::vector<size_t> > rowNumbers (
      new std::vector<size_t>(rowNumberss));
  this->_rowNumbers = std::move(rowNumbers);

  // Add numericalFeatures
  std::vector<size_t> numericalFeatureColss;
  std::vector<size_t>* catCols = this->getCatCols();

  for (size_t i = 0; i < numColumns; i++) {
    if (std::find(catCols->begin(), catCols->end(), i) == catCols->end()) {
      numericalFeatureColss.push_back(i);
    }
  }

  std::unique_ptr< std::vector<size_t> > numericalFeatureCols (
      new std::vector<size_t>(numericalFeatureColss));
  this->_numericalFeatureCols = std::move(numericalFeatureCols);
}

float DataFrame::getPoint(size_t rowIndex, size_t colIndex) {
  // Check if rowIndex and colIndex are valid
  if (rowIndex < getNumRows() && colIndex < getNumColumns()) {
    return (*getAllFeatureData())[colIndex][rowIndex];
  } else {
    throw std::runtime_error("Invalid rowIndex or colIndex.");
  }
}

float DataFrame::getOutcomePoint(size_t rowIndex) {
  // Check if rowIndex is valid
  if (rowIndex < getNumRows()) {
    return (*getOutcomeData())[rowIndex];
  } else {
    throw std::runtime_error("Invalid rowIndex.");
  }
}

std::vector<float>* DataFrame::getFeatureData(
  size_t colIndex
) {
  if (colIndex < getNumColumns()) {
    return &(*getAllFeatureData())[colIndex];
  } else {
    throw std::runtime_error("Invalid colIndex.");
  }
}

std::vector<float> DataFrame::getLinObsData(
  size_t rowIndex
) {
  if (rowIndex < getNumRows()) {
    std::vector<float> feat;
    for (size_t i = 0; i < getLinCols()->size(); i++) {
      feat.push_back(getPoint(rowIndex, (*getLinCols())[i]));
    }
    return feat;
  } else {
    throw std::runtime_error("Invalid rowIndex");
  }
}

void DataFrame::getObservationData(
  std::vector<float> &rowData,
  size_t rowIndex
) {
  if (rowIndex < getNumRows()) {
    for (size_t i=0; i < getNumColumns(); i++) {
      rowData[i] = (*getAllFeatureData())[i][rowIndex];
    }
  } else {
    throw std::runtime_error("Invalid rowIndex.");
  }
}

void DataFrame::getShuffledObservationData(
  std::vector<float> &rowData,
  size_t rowIndex,
  size_t swapFeature,
  size_t swapIndex
) {
  // Helper that gives rowIndex observation with
  // swapFeature value of swapIndex observation
  if (rowIndex < getNumRows() && swapFeature < getNumColumns()) {
    for (size_t i=0; i < getNumColumns(); i++) {
      rowData[i] = (*getAllFeatureData())[i][rowIndex];
    }
    rowData[swapFeature] = getPoint(swapIndex, swapFeature);
  } else {
    throw std::runtime_error("Invalid row/colIndex.");
  }
}

float DataFrame::partitionMean(
  std::vector<size_t>* sampleIndex
){
  size_t totalSampleSize = (*sampleIndex).size();
  float accummulatedSum = 0;
  for (
    std::vector<size_t>::iterator it = (*sampleIndex).begin();
    it != (*sampleIndex).end();
    ++it
  ) {
    accummulatedSum += getOutcomePoint(*it);
  }
  return accummulatedSum / totalSampleSize;
}

std::vector<size_t> DataFrame::get_all_row_idx(
    std::vector<size_t>* sampleIndex
){
  std::vector<size_t> idx;
  for (
      std::vector<size_t>::iterator it = (*sampleIndex).begin();
      it != (*sampleIndex).end();
      ++it
  ) {
    idx.push_back(get_row_idx(*it));
  }
  return idx;
}

size_t DataFrame::get_row_idx(size_t rowIndex) {
  // Check if rowIndex is valid
  if (rowIndex < getNumRows()) {
    return (*getRowNumbers())[rowIndex];
  } else {
    throw std::runtime_error("rowIndex is too large");
  }
}

void DataFrame::setOutcomeData(std::vector<float> outcomeData) {
  std::unique_ptr<std::vector<float> > outcomeData_(
      new std::vector<float>(outcomeData)
  );
  this->_outcomeData =std::move(outcomeData_);
}
