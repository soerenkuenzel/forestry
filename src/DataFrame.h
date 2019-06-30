#ifndef FORESTRYCPP_DATAFRAME_H
#define FORESTRYCPP_DATAFRAME_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

class DataFrame {

public:
  DataFrame();
  virtual ~DataFrame();

  DataFrame(
    std::shared_ptr< std::vector< std::vector<float> > > featureData,
    std::unique_ptr< std::vector<float> > outcomeData,
    std::unique_ptr< std::vector<size_t> > categoricalFeatureCols,
    std::unique_ptr< std::vector<size_t> > splitFeatureCols,
    std::unique_ptr< std::vector<size_t> > linearCols,
    std::size_t numRows,
    std::size_t numColumns,
    std::unique_ptr< std::vector<float> > sampleWeights
  );

  float getPoint(size_t rowIndex, size_t colIndex);

  float getOutcomePoint(size_t rowIndex);

  float getFeaturePoint(size_t rowIndex, size_t colIndex);

  std::vector<float>* getFeatureData(size_t colIndex);

  std::vector<float> getLinObsData(size_t rowIndex);

  void getObservationData(std::vector<float> &rowData, size_t rowIndex);

  void getShuffledObservationData(std::vector<float> &rowData, size_t rowIndex,
                                  size_t swapFeature, size_t swapIndex);

  float partitionMean(std::vector<size_t>* sampleIndex);

  void computeTreeDistances(std::vector<size_t>* sampleIndex,
                            float power,
                            size_t distColIndex,
                            std::vector<size_t>* updateIndex,
                            std::vector< std::vector<float> >* xNew,
                            std::vector<float>* outputPrediction);

  std::vector< std::vector<float> >* getAllFeatureData() {
    return _featureData.get();
  }

  std::vector<float>* getOutcomeData() {
    return _outcomeData.get();
  }

  size_t getNumColumns() {
    return _numColumns;
  }

  size_t getNumRows() {
    return _numRows;
  }

  std::vector<size_t>* getCatCols() {
    return _categoricalFeatureCols.get();
  }

  std::vector<size_t>* getNumCols() {
    return _numericalFeatureCols.get();
  }

  std::vector<size_t>* getSplitCols() {
    return _splitFeatureCols.get();
  }

  std::vector<size_t>* getLinCols() {
    return _linearFeatureCols.get();
  }

  std::vector<float>* getSampleWeights() {
    return _sampleWeights.get();
  }

  std::vector<size_t>* getRowNumbers() {
    return _rowNumbers.get();
  }

  std::vector<size_t> get_all_row_idx(std::vector<size_t>* sampleIndex);

  size_t get_row_idx(size_t rowIndex);

  void setOutcomeData(std::vector<float> outcomeData);

private:
  std::shared_ptr< std::vector< std::vector<float> > > _featureData;
  std::unique_ptr< std::vector<float> > _outcomeData;
  std::unique_ptr< std::vector<size_t> > _rowNumbers;
  std::unique_ptr< std::vector<size_t> > _categoricalFeatureCols;
  std::unique_ptr< std::vector<size_t> > _numericalFeatureCols;
  std::unique_ptr< std::vector<size_t> > _splitFeatureCols;
  std::unique_ptr< std::vector<size_t> > _linearFeatureCols;
  std::size_t _numRows;
  std::size_t _numColumns;
  std::unique_ptr< std::vector<float> > _sampleWeights;
};


#endif //FORESTRYCPP_DATAFRAME_H
