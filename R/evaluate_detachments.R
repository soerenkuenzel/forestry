#' @include quantileForest.R
#' @importFrom caret createFolds
NULL

#' compute detachment indices' percentiles
#' @name evaluate_detachments-forestry
#' @title compute detachment indices' percentiles
#' @rdname evaluate_detachments-forestry
#' @description Computes and returns the detachments percentiles of new observations
#' in specified dimensions.
#' @inheritParams compute_detachments
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param feat.name a list of features for computing the levels with respect to.
#' @param verbose Print out the steps in the algorithm.
#' @param num.CV Number of folds in the CV to compute the detachements
#' @return A data frame of quantiles of in response variable conditional on the
#' test observations.
#' @examples
#' # Set seed for reproductivity
#' set.seed(292313)
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 10)
#'
#' # Select, for example, sepal length as the response variable
#' index <- which(colnames(iris) == "Sepal.Length")
#' x_train <- iris[-test_idx, -index]
#' y_train <- iris[-test_idx, index]
#' x_test <- iris[test_idx, -index]
#' rf <- forestry(x = x_train, y = y_train)
#'
#' features <- c("Sepal.Width", "Petal.Length", "Petal.Width", "Species")
#'
#' # Evaluate the test observations' detachment indices
#' trust <- evaluate_detachments(object = rf,
#'                               feature.new = x_test,
#'                               feature = features,
#'                               p = 1)
#' @export
evaluate_detachments <- function(object,
                                 feature.new,
                                 feat.name,
                                 p = 1,
                                 verbose = TRUE,
                                 num.CV = 2) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  x_train <- slot(object, "processed_dta")$processed_x
  y_train <- slot(object, "processed_dta")$y

  feature.new <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  eval <- data.frame(1:nrow(feature.new))

  for (feat in feat.name) {
    # Compute detachment indices for new data
    detachments <- compute_detachments(object = object,
                                       feature.new = feature.new,
                                       detachment.feat = feat,
                                       p = p)

    # Compute detachment indices for the training data using OOB observations:
    folds <- caret::createFolds(y_train, k = num.CV, list = TRUE,
                                returnTrain = FALSE)

    # Create an empty vector for detechment indices for training observations
    x_train_detachments <- rep(NA, nrow(x_train))

    for (k in 1:num.CV) {
      if (verbose) {
        print(paste("Running fold", k, "out of", num.CV))
      }
      fold_ids <- folds[[k]]
      rf <- forestry(x = x_train[-fold_ids, ],
                     y = y_train[-fold_ids],
                     ntree = object@ntree,
                     replace = object@replace,
                     sample.fraction = object@sampsize / length(y_train),
                     mtry = object@mtry,
                     nodesizeAvg = object@nodesizeAvg,
                     nodesizeStrictSpl = object@nodesizeStrictSpl,
                     nodesizeStrictAvg = object@nodesizeStrictAvg,
                     minSplitGain = object@minSplitGain,
                     maxDepth = object@maxDepth,
                     splitratio = object@splitratio,
                     middleSplit = object@middleSplit,
                     maxObs = object@maxObs,
                     ridgeRF = object@ridgeRF,
                     linFeats = object@linFeats + 1,
                     overfitPenalty = object@overfitPenalty,
                     doubleTree = object@doubleTree)

      x_train_detachments[fold_ids] <-
        compute_detachments(object = rf,
                            feature.new = x_train[fold_ids, ],
                            detachment.feat = feat,
                            p = p)
    }

    # Get conditional percentiles for new data
    probs <- conditional_dist_bnd(object,
                                  processed_x = feature.new,
                                  train_vector = x_train_detachments,
                                  test_vector = detachments)
    probs <- as.data.frame(probs)

    colnames(probs)[1] <- feat
    eval <- cbind(eval, probs)
  }

  eval <- eval[ ,-1]
  return(eval)
}






evaluate_detachments_alt <- function(object,
                                 feature.new,
                                 feat.name,
                                 p = 1,
                                 verbose = TRUE,
                                 num.CV = 2) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  x_train <- slot(object, "processed_dta")$processed_x
  y_train <- slot(object, "processed_dta")$y

  feature.new <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  # Get appropropriate weight matrix
  y_weights <- predict(object = object,
                       feature.new = feature.new,
                       aggregation = "weightMatrix")$weightMatrix


  eval <- data.frame(1:nrow(feature.new))
  for (feat in feat.name) {
    if (verbose) {
      print(paste("Evaluating detachments for feature: ", feat))
    }
    # Compute detachment indices for new data
    detachments <- compute_detachments(object = object,
                                       feature.new = feature.new,
                                       detachment.feat = feat,
                                       p = p)

    # Compute detachemtn indices for the training data using OOB observations:
    folds <- caret::createFolds(y_train, k = num.CV, list = TRUE,
                                returnTrain = FALSE)

    # Create an empty vector for detechment indices for training observations
    x_train_detachments <- rep(NA, nrow(x_train))

    for (k in 1:num.CV) {
      if (verbose) {
        print(paste("Running fold", k, "out of", num.CV))
      }
      fold_ids <- folds[[k]]
      rf <- forestry(x = x_train[-fold_ids, ],
                     y = y_train[-fold_ids],
                     ntree = object@ntree,
                     replace = object@replace,
                     sample.fraction = object@sampsize / length(y_train),
                     mtry = object@mtry,
                     nodesizeAvg = object@nodesizeAvg,
                     nodesizeStrictSpl = object@nodesizeStrictSpl,
                     nodesizeStrictAvg = object@nodesizeStrictAvg,
                     minSplitGain = object@minSplitGain,
                     maxDepth = object@maxDepth,
                     splitratio = object@splitratio,
                     middleSplit = object@middleSplit,
                     maxObs = object@maxObs,
                     ridgeRF = object@ridgeRF,
                     linFeats = object@linFeats + 1,
                     overfitPenalty = object@overfitPenalty,
                     doubleTree = object@doubleTree)

      x_train_detachments[fold_ids] <-
        compute_detachments(object = rf,
                            feature.new = x_train[fold_ids, ],
                            detachment.feat = feat,
                            p = p)
    }

    # Get conditional percentiles for new data
    probs <- conditional_dist_bnd(object,
                                  processed_x = feature.new,
                                  train_vector = x_train_detachments,
                                  test_vector = detachments)
    probs <- as.data.frame(probs)

    colnames(probs)[1] <- feat
    eval <- cbind(eval, probs)
  }

  eval <- eval[ ,-1]
  return(eval)
}







