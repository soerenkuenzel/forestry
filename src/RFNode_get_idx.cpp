#include "RFNode.h"

void RFNode::get_idx_in_leaf(
    std::vector<float> &outputPrediction,
    std::vector<size_t>* updateIndex,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData
) {

  // If the node is a leaf, aggregate all its averaging data samples
  if (is_leaf()) {

    // Calculate the mean of current node
    float predictedMean = (*trainingData).partitionMean(getAveragingIndex());

    // Give all updateIndex the mean of the node as prediction values
    for (
        std::vector<size_t>::iterator it = (*updateIndex).begin();
        it != (*updateIndex).end();
        ++it
    ) {
      outputPrediction[*it] = predictedMean - 1;
    }
    std::cout << (*trainingData).getNumRows() << " > ";
    std::cout << (*trainingData).get_all_row_idx(getAveragingIndex()) << "\n";
    // (*trainingData).get_row_idx(getAveragingIndex())
  } else {

    // Separate prediction tasks to two children
    std::vector<size_t>* leftPartitionIndex = new std::vector<size_t>();
    std::vector<size_t>* rightPartitionIndex = new std::vector<size_t>();

    // Test if the splitting feature is categorical
    std::vector<size_t> categorialCols = *(*trainingData).getCatCols();
    if (
        std::find(
          categorialCols.begin(),
          categorialCols.end(),
          getSplitFeature()
        ) != categorialCols.end()
    ){

      // If the splitting feature is categorical, split by (==) or (!=)
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it
      ) {
        if ((*xNew)[getSplitFeature()][*it] == getSplitValue()) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }

    } else {

      // For non-categorical, split to left (<) and right (>=) according to the
      // split value
      for (
          std::vector<size_t>::iterator it = (*updateIndex).begin();
          it != (*updateIndex).end();
          ++it
      ) {
        if ((*xNew)[getSplitFeature()][*it] < getSplitValue()) {
          (*leftPartitionIndex).push_back(*it);
        } else {
          (*rightPartitionIndex).push_back(*it);
        }
      }

    }

    // Recursively get predictions from its children
    if ((*leftPartitionIndex).size() > 0) {
      (*getLeftChild()).get_idx_in_leaf(
          outputPrediction,
          leftPartitionIndex,
          xNew,
          trainingData
      );
    }
    if ((*rightPartitionIndex).size() > 0) {
      (*getRightChild()).get_idx_in_leaf(
          outputPrediction,
          rightPartitionIndex,
          xNew,
          trainingData
      );
    }

    delete(leftPartitionIndex);
    delete(rightPartitionIndex);

  }

}
