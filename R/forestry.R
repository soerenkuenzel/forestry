#' @useDynLib forestry, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats predict runif
NULL


#' @include R_preprocessing.R
#-- Sanity Checker -------------------------------------------------------------
#' @name training_data_checker
#' @title Training data check
#' @rdname training_data_checker-forestry
#' @description Check the input to forestry constructor
#' @inheritParams forestry
#' @noRd
training_data_checker <- function(x,
                                  y,
                                  ntree,
                                  replace,
                                  sampsize,
                                  mtry,
                                  nodesizeSpl,
                                  nodesizeAvg,
                                  nodesizeStrictSpl,
                                  nodesizeStrictAvg,
                                  minSplitGain,
                                  maxDepth,
                                  splitratio,
                                  nthread,
                                  middleSplit,
                                  maxObs,
                                  maxProp,
                                  doubleTree,
                                  splitFeats,
                                  linFeats,
                                  monotonicConstraints,
                                  sampleWeights,
                                  linear) {
  x <- as.data.frame(x)
  nfeatures <- ncol(x)

  # Check if the input dimension of x matches y
  if (nrow(x) != length(y)) {
    stop("The dimension of input dataset x doesn't match the output vector y.")
  }

  # Check if x and y contain missing values
  if (any(is.na(x))) {
    stop("x contains missing data.")
  }
  if (any(is.na(y))) {
    stop("y contains missing data.")
  }

  if (!is.logical(replace)) {
    stop("replace must be TRUE or FALSE.")
  }

  if (ntree <= 0 || ntree %% 1 != 0) {
    stop("ntree must be a positive integer.")
  }

  if (sampsize <= 0 || sampsize %% 1 != 0) {
    stop("sampsize must be a positive integer.")
  }

  if (max(splitFeats) > nfeatures || any(splitFeats < 1)) {
    stop("splitFeats must be a positive integer less than or equal to ncol(x).")
  }

  if (max(linFeats) > nfeatures || any(linFeats < 1)) {
    stop("linFeats must be a positive integer less than or equal to ncol(x).")
  }

  if (length(monotonicConstraints) != ncol(x)) {
    stop("monotoneConstraints must be the size of x")
  }

  if (any((monotonicConstraints != 1 ) & (monotonicConstraints != -1 ) & (monotonicConstraints != 0 ))) {
    stop("monotonicConstraints must be either 1, 0, or -1")
  }

  if (any(monotonicConstraints != 0) && linear) {
    stop("Cannot use linear splitting with monotoneConstraints")
  }

  if (!replace && sampsize > nrow(x)) {
    stop(
      paste(
        "You cannot sample without replacement with size more than",
        "total number of oberservations."
      )
    )
  }
  if (mtry <= 0 || mtry %% 1 != 0) {
    stop("mtry must be a positive integer.")
  }
  if (mtry > nfeatures) {
    stop("mtry cannot exceed total amount of features in x.")
  }
  if (mtry > length(splitFeats)) {
    stop("mtry cannot exceed total number of features specified in splitFeats.")
  }
  if (nodesizeSpl <= 0 || nodesizeSpl %% 1 != 0) {
    stop("nodesizeSpl must be a positive integer.")
  }
  if (nodesizeAvg <= 0 || nodesizeAvg %% 1 != 0) {
    stop("nodesizeAvg must be a positive integer.")
  }

  if (nodesizeStrictSpl <= 0 || nodesizeStrictSpl %% 1 != 0) {
    stop("nodesizeStrictSpl must be a positive integer.")
  }
  if (nodesizeStrictAvg <= 0 || nodesizeStrictAvg %% 1 != 0) {
    stop("nodesizeStrictAvg must be a positive integer.")
  }
  if (minSplitGain < 0) {
    stop("minSplitGain must be greater than or equal to 0.")
  }
  if (minSplitGain > 0 && !linear) {
    stop("minSplitGain cannot be set without setting linear to be true.")
  }
  if (maxDepth <= 0 || maxDepth %% 1 != 0) {
    stop("maxDepth must be a positive integer.")
  }
  if (length(sampleWeights) != ncol(x)) {
    stop("Must have sample weight length equal to columns in data")
  }
  if (min(sampleWeights < 0)) {
    stop("sampleWeights must be greater than 0")
  }
  if (min(sampleWeights) <= .001*max(sampleWeights)) {
    stop("MAX(sampleWeights):MIN(sampleWeights) must be < 1000")
  }

  sampleWeights <- (sampleWeights / sum(sampleWeights))

  if (maxObs < 1 || maxObs > nrow(x)) {
    stop("maxObs must be greater than one and less than N")
  }

  if (maxProp <= 0 || maxProp > 1) {
    stop("maxProp must be between 0 and 1")
  }

  # We want maxProp to take precedent over maxObs
  if (maxProp != 1 && maxObs != nrow(x)) {
    warning("maxProp is set != 1, setting maxObs = nrow(x)")
    maxObs <- nrow(x)
  }

  # if the splitratio is 1, then we use adaptive rf and avgSampleSize is the
  # equal to the total sampsize
  if (splitratio == 0 || splitratio == 1) {
    splitSampleSize <- sampsize
    avgSampleSize <- sampsize
  } else {
    splitSampleSize <- splitratio * sampsize
    avgSampleSize <- floor(sampsize - splitSampleSize)
    splitSampleSize <- floor(splitSampleSize)
  }

  if (nodesizeStrictSpl > splitSampleSize) {
    warning(
      paste(
        "nodesizeStrictSpl cannot exceed splitting sample size.",
        "We have set nodesizeStrictSpl to be the maximum"
      )
    )
    nodesizeStrictSpl <- splitSampleSize
  }
  if (nodesizeStrictAvg > avgSampleSize) {
    warning(
      paste(
        "nodesizeStrictAvg cannot exceed averaging sample size.",
        "We have set nodesizeStrictAvg to be the maximum"
      )
    )
    nodesizeStrictAvg <- avgSampleSize
  }
  if (doubleTree) {
    if (splitratio == 0 || splitratio == 1) {
      warning("Trees cannot be doubled if splitratio is 1. We have set
              doubleTree to FALSE")
      doubleTree <- FALSE
    } else {
      if (nodesizeStrictAvg > splitSampleSize) {
        warning(
          paste(
            "nodesizeStrictAvg cannot exceed splitting sample size.",
            "We have set nodesizeStrictAvg to be the maximum"
          )
        )
        nodesizeStrictAvg <- splitSampleSize
      }
      if (nodesizeStrictSpl > avgSampleSize) {
        warning(
          paste(
            "nodesizeStrictSpl cannot exceed averaging sample size.",
            "We have set nodesizeStrictSpl to be the maximum"
          )
        )
        nodesizeStrictSpl <- avgSampleSize
      }
    }
  }

  if (splitratio < 0 || splitratio > 1) {
    stop("splitratio must in between 0 and 1.")
  }

  if (nthread < 0 || nthread %% 1 != 0) {
    stop("nthread must be a nonegative integer.")
  }

  if (nthread > 0) {
    #' @import parallel
    if (tryCatch(
      nthread > parallel::detectCores(),
      error = function(x) {
        FALSE
      }
    )) {
      stop(paste0(
        "nthread cannot exceed total cores in the computer: ",
        detectCores()
      ))
    }
  }

  if (!is.logical(middleSplit)) {
    stop("middleSplit must be TRUE or FALSE.")
  }
  return(list("x" = x,
              "y" = y,
              "ntree" = ntree,
              "replace" = replace,
              "sampsize" = sampsize,
              "mtry" = mtry,
              "nodesizeSpl" = nodesizeSpl,
              "nodesizeAvg" = nodesizeAvg,
              "nodesizeStrictSpl" = nodesizeStrictSpl,
              "nodesizeStrictAvg" = nodesizeStrictAvg,
              "minSplitGain" = minSplitGain,
              "maxDepth" = maxDepth,
              "splitratio" = splitratio,
              "nthread" = nthread,
              "middleSplit" = middleSplit,
              "maxObs" = maxObs,
              "maxProp" = maxProp,
              "doubleTree" = doubleTree,
              "splitFeats" = splitFeats,
              "linFeats" = linFeats,
              "monotonicConstraints" = monotonicConstraints,
              "sampleWeights" = sampleWeights))
}

#' @title Test data check
#' @name testing_data_checker-forestry
#' @description Check the testing data to do prediction
#' @param feature.new A data frame of testing predictors.
#' @noRd
testing_data_checker <- function(feature.new) {
  feature.new <- as.data.frame(feature.new)
  if (any(is.na(feature.new))) {
    stop("x contains missing data.")
  }
}

# -- Random Forest Constructor -------------------------------------------------
setClass(
  Class = "forestry",
  slots = list(
    forest = "externalptr",
    dataframe = "externalptr",
    processed_dta = "list",
    R_forest = "list",
    featureNames = "character",
    categoricalFeatureCols = "list",
    categoricalFeatureMapping = "list",
    ntree = "numeric",
    replace = "logical",
    sampsize = "numeric",
    mtry = "numeric",
    nodesizeSpl = "numeric",
    nodesizeAvg = "numeric",
    nodesizeStrictSpl = "numeric",
    nodesizeStrictAvg = "numeric",
    minSplitGain = "numeric",
    maxDepth = "numeric",
    splitratio = "numeric",
    middleSplit = "logical",
    y = "vector",
    maxObs = "numeric",
    maxProp = "numeric",
    linear = "logical",
    splitFeats = "numeric",
    linFeats = "numeric",
    monotonicConstraints = "numeric",
    sampleWeights = "numeric",
    overfitPenalty = "numeric",
    doubleTree = "logical"
  )
)

setClass(
  Class = "multilayerForestry",
  slots = list(
    forest = "externalptr",
    dataframe = "externalptr",
    processed_dta = "list",
    R_forest = "list",
    featureNames = "character",
    categoricalFeatureCols = "list",
    categoricalFeatureMapping = "list",
    ntree = "numeric",
    nrounds = "numeric",
    eta = "numeric",
    replace = "logical",
    sampsize = "numeric",
    mtry = "numeric",
    nodesizeSpl = "numeric",
    nodesizeAvg = "numeric",
    nodesizeStrictSpl = "numeric",
    nodesizeStrictAvg = "numeric",
    minSplitGain = "numeric",
    maxDepth = "numeric",
    splitratio = "numeric",
    middleSplit = "logical",
    y = "vector",
    maxObs = "numeric",
    maxProp = "numeric",
    linear = "logical",
    splitFeats = "numeric",
    linFeats = "numeric",
    monotonicConstraints = "numeric",
    sampleWeights = "numeric",
    overfitPenalty = "numeric",
    doubleTree = "logical"
  )
)


#' @title forestry
#' @rdname forestry
#' @param x A data frame of all training predictors.
#' @param y A vector of all training responses.
#' @param ntree The number of trees to grow in the forest. The default value is
#'   500.
#' @param replace An indicator of whether sampling of training data is with
#'   replacement. The default value is TRUE.
#' @param sampsize The size of total samples to draw for the training data. If
#'   sampling with replacement, the default value is the length of the training
#'   data. If samplying without replacement, the default value is two-third of
#'   the length of the training data.
#' @param sample.fraction If this is given, then sampsize is ignored and set to
#'   be round(length(y) * sample.fraction). It must be a real number between 0
#'   and 1
#' @param mtry The number of variables randomly selected at each split point.
#'   The default value is set to be one third of total number of features of the
#'   training data.
#' @param nodesizeSpl Minimum observations contained in terminal nodes. The
#'   default value is 3.
#' @param nodesizeAvg Minimum size of terminal nodes for averaging dataset. The
#'   default value is 3.
#' @param nodesizeStrictSpl Minimum observations to follow strictly in terminal
#'   nodes. The default value is 1.
#' @param nodesizeStrictAvg Minimum size of terminal nodes for averaging dataset
#'   to follow strictly. The default value is 1.
#' @param minSplitGain Minimum loss reduction to split a node further in a tree.
#'   specifically this is the percentage R squared increase which each potential
#'   split must give to be considered. The default value is 0.
#' @param maxDepth Maximum depth of a tree. The default value is 99.
#' @param splitratio Proportion of the training data used as the splitting
#'   dataset. It is a ratio between 0 and 1. If the ratio is 1, then essentially
#'   splitting dataset becomes the total entire sampled set and the averaging
#'   dataset is empty. If the ratio is 0, then the splitting data set is empty
#'   and all the data is used for the averaging data set (This is not a good
#'   usage however since there will be no data available for splitting).
#' @param seed Seed for random number generator.
#' @param verbose Flag to indicate if training process is verbose.
#' @param nthread Number of threads to train and predict the forest. The default
#'   number is 0 which represents using all cores.
#' @param splitrule Only variance is implemented at this point and it
#'   specifies the loss function according to which the splits of random forest
#'   should be made.
#' @param middleSplit Flag to indicate whether the split value takes the average
#'   of two feature values. If false, it will take a point based on a uniform
#'   distribution between two feature values. The default value is FALSE.
#' @param doubleTree Indicate whether the number of tree is doubled as averaging
#'   and splitting data can be exchanged to create decorrelated trees.
#'   The default value is FALSE.
#' @param reuseforestry Pass in a `forestry` object which will recycle the
#'   dataframe the old object created. It will save some space working on the
#'   same dataset.
#' @param maxObs The max number of observations to split on. If set to a number
#'   less than nrow(x), at each split point, maxObs split points will be
#'   randomly sampled to test as potential splitting points instead of every
#'   feature value (default).
#' @param maxProp A complementary option to `maxObs`, `maxProp` allows one to
#'   specify the proportion of possible split points which are downsampled at
#'   each point to test potential splitting points. For example, a value of .35
#'   will randomly select 35% of the possible splitting points to be potential
#'   splitting poimts at each split. If values of `maxProp` and `maxObs` are
#'   both supplied, the value of `maxProp` will take precedent.
#'   At the lower levels of the tree, we will select
#'   Max(`maxProp`* n, nodesizeSpl) splitting observations.
#' @param saveable If TRUE, then RF is created in such a way that it can be
#'   saved and loaded using save(...) and load(...). Setting it to TRUE
#'   (default) will, however, take longer and it will use more memory. When
#'   training many RF, it makes a lot of sense to set this to FALSE to save
#'   time and memory.
#' @param linear Fit the model with a split function optimizing for a linear
#'   aggregation function instead of a constant aggregation function. The default
#'   value is FALSE.
#' @param splitFeats Specify which features to split on when creating a tree
#'   (defaults to use all features).
#' @param linFeats Specify which features to split linearly on when using
#'   linear (defaults to use all numerical features)
#' @param monotonicConstraints Specifies monotonic relationships between the
#'   continuous features and the outcome. Supplied as a vector of length p with
#'   entries in 1,0,-1 which 1 indicating an increasing monotonic relationship,
#'   -1 indicating a decreasing monotonic relationship, and 0 indicating no
#'   relationship. Constraints supplied for categorical will be ignored.
#' @param sampleWeights Specify weights for weighted uniform distribution used
#'   to randomly sample features.
#' @param overfitPenalty Value to determine how much to penalize magnitude of
#'   coefficients in ridge regression when using linear. The default value is 1.
#' @return A `forestry` object.
#' @description forestry is a fast implementation of a variety of tree-based
#'   estimators. Implemented estimators include CART trees, randoms forests,
#'   boosted trees and forests, and linear trees and forests. All estimators are
#'   implemented to scale well with very large datasets.
#' @details For Linear Random Forests, set the linear option to TRUE and
#'   specify lambda for ridge regression with overfitPenalty parameter. For
#'   gradient boosting and gradient boosting forests, see mulitlayer-forestry.
#' @seealso \code{\link{predict.forestry}}
#' @seealso \code{\link{multilayer-forestry}}
#' @seealso \code{\link{predict-multilayer-forestry}}
#' @seealso \code{\link{getVI}}
#' @seealso \code{\link{getOOB}}
#' @seealso \code{\link{make_savable}}
#' @examples
#' set.seed(292315)
#' library(forestry)
#' test_idx <- sample(nrow(iris), 3)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train)
#' weights = predict(rf, x_test, aggregation = "weightMatrix")$weightMatrix
#'
#' weights %*% y_train
#' predict(rf, x_test)
#'
#' set.seed(49)
#' library(forestry)
#'
#' n <- c(100)
#' a <- rnorm(n)
#' b <- rnorm(n)
#' c <- rnorm(n)
#' y <- 4*a + 5.5*b - .78*c
#' x <- data.frame(a,b,c)
#'
#' forest <- forestry(
#'           x,
#'           y,
#'           ntree = 10,
#'           replace = TRUE,
#'           nodesizeStrictSpl = 5,
#'           nodesizeStrictAvg = 5,
#'           linear = TRUE
#'           )
#'
#' predict(forest, x)
#' @export
forestry <- function(x,
                     y,
                     ntree = 500,
                     replace = TRUE,
                     sampsize = if (replace)
                       nrow(x)
                     else
                       ceiling(.632 * nrow(x)),
                     sample.fraction = NULL,
                     mtry = max(floor(ncol(x) / 3), 1),
                     nodesizeSpl = 3,
                     nodesizeAvg = 3,
                     nodesizeStrictSpl = 1,
                     nodesizeStrictAvg = 1,
                     minSplitGain = 0,
                     maxDepth = round(nrow(x) / 2) + 1,
                     splitratio = 1,
                     seed = as.integer(runif(1) * 1000),
                     verbose = FALSE,
                     nthread = 0,
                     splitrule = "variance",
                     middleSplit = FALSE,
                     maxObs = length(y),
                     maxProp = 1,
                     linear = FALSE,
                     splitFeats = 1:(ncol(x)),
                     linFeats = 1:(ncol(x)),
                     monotonicConstraints = rep(0, ncol(x)),
                     sampleWeights = rep((1/ncol(x)), ncol(x)),
                     overfitPenalty = 1,
                     doubleTree = FALSE,
                     reuseforestry = NULL,
                     saveable = TRUE) {
  x <- as.data.frame(x)

  # only if sample.fraction is given, update sampsize
  if (!is.null(sample.fraction))
    sampsize <- ceiling(sample.fraction * nrow(x))
  splitFeats <- unique(splitFeats)
  linFeats <- unique(linFeats)

  # Preprocess the data
  updated_variables <-
    training_data_checker(
      x = x,
      y = y,
      ntree = ntree,
      replace = replace,
      sampsize = sampsize,
      mtry = mtry,
      nodesizeSpl = nodesizeSpl,
      nodesizeAvg = nodesizeAvg,
      nodesizeStrictSpl = nodesizeStrictSpl,
      nodesizeStrictAvg = nodesizeStrictAvg,
      minSplitGain = minSplitGain,
      maxDepth = maxDepth,
      splitratio = splitratio,
      nthread = nthread,
      middleSplit = middleSplit,
      maxObs = maxObs,
      maxProp = maxProp,
      doubleTree = doubleTree,
      splitFeats = splitFeats,
      linFeats = linFeats,
      monotonicConstraints = monotonicConstraints,
      sampleWeights = sampleWeights,
      linear = linear)

  for (variable in names(updated_variables)) {
    assign(x = variable, value = updated_variables[[variable]],
           envir = environment())
  }

  # Total number of obervations
  nObservations <- length(y)
  numColumns <- ncol(x)
  # Update linear features to be zero-indexed
  splitFeats = splitFeats - 1
  linFeats = linFeats - 1

  if (is.null(reuseforestry)) {
    preprocessedData <- preprocess_training(x, y)
    processed_x <- preprocessedData$x
    categoricalFeatureCols <-
      preprocessedData$categoricalFeatureCols
    categoricalFeatureMapping <-
      preprocessedData$categoricalFeatureMapping

    categoricalFeatureCols_cpp <- unlist(categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      # If we have monotonic constraints on any categorical features we need to
      # zero these out as we cannot do monotonicity with categorical features

      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    # Create rcpp object
    # Create a forest object
    forest <- tryCatch({
      rcppDataFrame <- rcpp_cppDataFrameInterface(processed_x,
                                                  y,
                                                  categoricalFeatureCols_cpp,
                                                  splitFeats,
                                                  linFeats,
                                                  nObservations,
                                                  numColumns,
                                                  sampleWeights,
                                                  monotonicConstraints)

      rcppForest <- rcpp_cppBuildInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        splitFeats,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        replace,
        sampsize,
        mtry,
        splitratio,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        maxProp,
        sampleWeights,
        monotonicConstraints,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )
      processed_dta <- list(
        "processed_x" = processed_x,
        "y" = y,
        "categoricalFeatureCols_cpp" = categoricalFeatureCols_cpp,
        "splittingFeaturesCols_cpp" = splitFeats,
        "linearFeatureCols_cpp" = linFeats,
        "nObservations" = nObservations,
        "numColumns" = numColumns
      )
      R_forest <- list()

      return(
        new(
          "forestry",
          forest = rcppForest,
          dataframe = rcppDataFrame,
          processed_dta = processed_dta,
          R_forest = R_forest,
          featureNames = colnames(x),
          categoricalFeatureCols = categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          maxProp = maxProp,
          sampleWeights = sampleWeights,
          linear = linear,
          splitFeats = splitFeats,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree
        )
      )
    },
    error = function(err) {
      print(err)
      return(NULL)
    })

  } else {
    categoricalFeatureCols_cpp <-
      unlist(reuseforestry@categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    categoricalFeatureMapping <-
      reuseforestry@categoricalFeatureMapping

    # Create rcpp object
    # Create a forest object
    forest <- tryCatch({
      rcppForest <- rcpp_cppBuildInterface(
        x,
        y,
        categoricalFeatureCols_cpp,
        splitFeats,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        replace,
        sampsize,
        mtry,
        splitratio,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        maxProp,
        sampleWeights,
        monotonicConstraints,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        reuseforestry@dataframe
      )

      return(
        new(
          "forestry",
          forest = rcppForest,
          dataframe = reuseforestry@dataframe,
          processed_dta = reuseforestry@processed_dta,
          R_forest = reuseforestry@R_forest,
          featureNames = colnames(x),
          categoricalFeatureCols = reuseforestry@categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          maxProp = maxProp,
          sampleWeights = sampleWeights,
          linear = linear,
          splitFeats = splitFeats,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree
        )
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })

  }

  return(forest)
}

# -- Multilayer Random Forest Constructor --------------------------------------
#' @name multilayer-forestry
#' @title Multilayer forestry
#' @rdname multilayer-forestry
#' @description Constructs a gradient boosted random forest.
#' @inheritParams forestry
#' @param nrounds Number of iterations used for gradient boosting.
#' @param eta Step size shrinkage used in gradient boosting update.
#' @return A trained model object of class "multilayerForestry".
#' @seealso \code{\link{forestry}}
#' @export
multilayerForestry <- function(x,
                     y,
                     ntree = 500,
                     nrounds = 1,
                     eta = 0.3,
                     replace = FALSE,
                     sampsize = nrow(x),
                     sample.fraction = NULL,
                     mtry = ncol(x),
                     nodesizeSpl = 3,
                     nodesizeAvg = 3,
                     nodesizeStrictSpl = max(round(nrow(x)/128), 1),
                     nodesizeStrictAvg = max(round(nrow(x)/128), 1),
                     minSplitGain = 0,
                     maxDepth = 99,
                     splitratio = 1,
                     seed = as.integer(runif(1) * 1000),
                     verbose = FALSE,
                     nthread = 0,
                     splitrule = "variance",
                     middleSplit = TRUE,
                     maxObs = length(y),
                     maxProp = 1,
                     linear = FALSE,
                     splitFeats = 1:(ncol(x)),
                     linFeats = 1:(ncol(x)),
                     monotonicConstraints = rep(0, ncol(x)),
                     sampleWeights = rep((1/ncol(x)), ncol(x)),
                     overfitPenalty = 1,
                     doubleTree = FALSE,
                     reuseforestry = NULL,
                     saveable = TRUE) {
  # only if sample.fraction is given, update sampsize
  if (!is.null(sample.fraction))
    sampsize <- ceiling(sample.fraction * nrow(x))
  splitFeats <- unique(splitFeats)
  linFeats <- unique(linFeats)

  x <- as.data.frame(x)
  # Preprocess the data
  training_data_checker(x, y, ntree,replace, sampsize, mtry, nodesizeSpl,
                        nodesizeAvg, nodesizeStrictSpl, nodesizeStrictAvg,
                        minSplitGain, maxDepth, splitratio, nthread, middleSplit,
                        maxObs, maxProp, doubleTree, splitFeats,
                        linFeats,monotonicConstraints, sampleWeights)
  # Total number of obervations
  nObservations <- length(y)
  numColumns <- ncol(x)
  # Update linear features to be zero-indexed
  splitFeats = splitFeats - 1
  linFeats = linFeats - 1

  if (is.null(reuseforestry)) {
    preprocessedData <- preprocess_training(x, y)
    processed_x <- preprocessedData$x
    categoricalFeatureCols <-
      preprocessedData$categoricalFeatureCols
    categoricalFeatureMapping <-
      preprocessedData$categoricalFeatureMapping

    categoricalFeatureCols_cpp <- unlist(categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      monotonicConstraints[categoricalFeatureCols_cpp] <- 0
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    # Create rcpp object
    # Create a forest object
    multilayerForestry <- tryCatch({
      rcppDataFrame <- rcpp_cppDataFrameInterface(processed_x,
                                                  y,
                                                  categoricalFeatureCols_cpp,
                                                  splitFeats,
                                                  linFeats,
                                                  nObservations,
                                                  numColumns,
                                                  sampleWeights,
                                                  monotonicConstraints)

      rcppForest <- rcpp_cppMultilayerBuildInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
        splitFeats,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        nrounds,
        eta,
        replace,
        sampsize,
        mtry,
        splitratio,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        maxProp,
        sampleWeights,
        monotonicConstraints,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )
      processed_dta <- list(
        "processed_x" = processed_x,
        "y" = y,
        "categoricalFeatureCols_cpp" = categoricalFeatureCols_cpp,
        "splittingFeaturesCols_cpp" = splitFeats,
        "linearFeatureCols_cpp" = linFeats,
        "nObservations" = nObservations,
        "numColumns" = numColumns
      )
      R_forest <- list()

      return(
        new(
          "multilayerForestry",
          forest = rcppForest,
          dataframe = rcppDataFrame,
          processed_dta = processed_dta,
          R_forest = R_forest,
          featureNames = colnames(x),
          categoricalFeatureCols = categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          nrounds = nrounds,
          eta = eta,
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          maxProp = maxProp,
          sampleWeights = sampleWeights,
          monotonicConstraints = monotonicConstraints,
          linear = linear,
          splitFeats = splitFeats,
          linFeats = linFeats,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree
        )
      )
    },
    error = function(err) {
      print(err)
      return(NULL)
    })

  } else {
    categoricalFeatureCols_cpp <-
      unlist(reuseforestry@categoricalFeatureCols)
    if (is.null(categoricalFeatureCols_cpp)) {
      categoricalFeatureCols_cpp <- vector(mode = "numeric", length = 0)
    } else {
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    categoricalFeatureMapping <-
      reuseforestry@categoricalFeatureMapping

    # Create rcpp object
    # Create a forest object
    multilayerForestry <- tryCatch({
      rcppForest <- rcpp_cppMultilayerBuildInterface(
        x,
        y,
        categoricalFeatureCols_cpp,
        splitFeats,
        linFeats,
        nObservations,
        numColumns,
        ntree,
        nrounds,
        eta,
        replace,
        sampsize,
        mtry,
        splitratio,
        nodesizeSpl,
        nodesizeAvg,
        nodesizeStrictSpl,
        nodesizeStrictAvg,
        minSplitGain,
        maxDepth,
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        maxProp,
        sampleWeights,
        linear,
        overfitPenalty,
        doubleTree,
        TRUE,
        reuseforestry@dataframe
      )

      return(
        new(
          "multilayerForestry",
          forest = rcppForest,
          dataframe = reuseforestry@dataframe,
          processed_dta = reuseforestry@processed_dta,
          R_forest = reuseforestry@R_forest,
          categoricalFeatureCols = reuseforestry@categoricalFeatureCols,
          categoricalFeatureMapping = categoricalFeatureMapping,
          ntree = ntree * (doubleTree + 1),
          replace = replace,
          sampsize = sampsize,
          mtry = mtry,
          nodesizeSpl = nodesizeSpl,
          nodesizeAvg = nodesizeAvg,
          nodesizeStrictSpl = nodesizeStrictSpl,
          nodesizeStrictAvg = nodesizeStrictAvg,
          minSplitGain = minSplitGain,
          maxDepth = maxDepth,
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          maxProp = maxProp,
          sampleWeights = sampleWeights,
          linear = linear,
          splitFeats = splitFeats,
          linFeats = linFeats,
          monotonicConstraints = monotonicConstraints,
          overfitPenalty = overfitPenalty,
          doubleTree = doubleTree
        )
      )
    }, error = function(err) {
      print(err)
      return(NULL)
    })

  }

  return(multilayerForestry)
}

# -- Predict Method ------------------------------------------------------------
#' predict-forestry
#' @name predict-forestry
#' @rdname predict-forestry
#' @description Return the prediction from the forest.
#' @param object A trained model object of class "forestry".
#' @param feature.new A data frame of testing predictors.
#' @param aggregation How shall the leaf be aggregated. The default is to return
#'   the mean of the leave `average`. Other options are `weightMatrix` and `coefs`
#'   to return the ridge regression coefficients when doing Linear Random Forest.
#' @param localVariableImportance Returns a matrix providing local variable
#'   importance for each prediction.
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @details Allows for different methods of prediction on new data.
#' @seealso \code{\link{forestry}}
#' @export
predict.forestry <- function(object,
                             feature.new,
                             aggregation = "average",
                             localVariableImportance = FALSE, ...) {
  # Preprocess the data
  testing_data_checker(feature.new)

  processed_x <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  if (localVariableImportance && (aggregation != "weightMatrix")) {
    stop("Aggregation must be set to weightMatrix if localVariableImportance is true.")
  }

  if ((!(object@linear)) && (aggregation == "coefs")) {
    stop("Aggregation can only be linear with setting the parameter linear = TRUE.")
  }

  rcppPrediction <- tryCatch({
    rcpp_cppPredictInterface(object@forest, processed_x, aggregation, localVariableImportance)
  }, error = function(err) {
    print(err)
    return(NULL)
  })

  # In the case aggregation is set to "linear"
  # rccpPrediction is a list with an entry $coef
  # which gives pointwise regression coeffficients averaged across the forest
  if (aggregation == "coefs") {
    if(length(object@linFeats) == 1) {
      feature.new <- data.frame(feature.new)
    }
    coef_names <- colnames(feature.new)[object@linFeats + 1]
    coef_names <- c(coef_names, "Intercept")
    colnames(rcppPrediction$coef) <- coef_names
  }

  if (aggregation == "average") {
    return(rcppPrediction$prediction)
  } else if (aggregation == "weightMatrix") {
    return(rcppPrediction)
  } else if (aggregation == "coefs") {
    return(rcppPrediction)
  }
}


# -- Multilayer Predict Method -------------------------------------------------------
#' predict-multilayer-forestry
#' @name predict-multilayer-forestry
#' @rdname predict-multilayer-forestry
#' @description Return the prediction from the forest.
#' @param object A `multilayerForestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param aggregation How shall the leaf be aggregated. The default is to return
#'   the mean of the leave `average`. Other options are `weightMatrix`.
#' @param ... additional arguments.
#' @return A vector of predicted responses.
#' @seealso \code{\link{forestry}}
#' @export
predict.multilayerForestry <- function(object,
                             feature.new,
                             aggregation = "average",
                             ...) {
    # Preprocess the data
    testing_data_checker(feature.new)

    processed_x <- preprocess_testing(feature.new,
                                      object@featureNames,
                                      object@categoricalFeatureCols,
                                      object@categoricalFeatureMapping)

    rcppPrediction <- tryCatch({
      rcpp_cppMultilayerPredictInterface(object@forest, processed_x, aggregation)
    }, error = function(err) {
      print(err)
      return(NULL)
    })

    if (aggregation == "average") {
      return(rcppPrediction$prediction)
    } else if (aggregation == "weightMatrix") {
      return(rcppPrediction)
    }
  }



# -- Calculate OOB Error -------------------------------------------------------
#' getOOB-forestry
#' @name getOOB-forestry
#' @rdname getOOB-forestry
#' @description Calculate the out-of-bag error of a given forest.
#' @param object A trained model object of class "forestry".
#' @param noWarning Flag to not display warnings.
#' @aliases getOOB, forestry-method
#' @return The out-of-bag error of the forest.
#' @seealso \code{\link{forestry}}
#' @export
getOOB <- function(object,
                   noWarning) {
    # TODO (all): find a better threshold for throwing such warning. 25 is
    # currently set up arbitrarily.
    if (!object@replace &&
        object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                        object@sampsize) < 10) {
      if (!noWarning) {
        warning(paste(
          "Samples are drawn without replacement and sample size",
          "is too big!"
        ))
      }
      return(NA)
    }

    rcppOOB <- tryCatch({
      return(rcpp_OBBPredictInterface(object@forest))
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppOOB)
  }

# -- Calculate OOB Predictions -------------------------------------------------
#' getOOBpreds-forestry
#' @name getOOBpreds-forestry
#' @rdname getOOBpreds-forestry
#' @description Calculate the out-of-bag predictions of a given forest.
#' @param object A trained model object of class "forestry".
#' @param noWarning Flag to not display warnings.
#' @aliases getOOBpreds, forestry-method
#' @return The vector of all training observations, with their out of bag
#'  predictions. Note each observation is out of bag for different trees, and so
#'  the predictions will be more or less stable based on the observation. Some
#'  observations may not be out of bag for any trees, and here the predictions
#'  are returned as NA.
#' @seealso \code{\link{forestry}}
#' @export
getOOBpreds <- function(object,
                        noWarning) {

  if (!object@replace &&
      object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                      object@sampsize) < 10) {
    if (!noWarning) {
      warning(paste(
        "Samples are drawn without replacement and sample size",
        "is too big!"
      ))
    }
    return(NA)
  }

  rcppOOBpreds <- tryCatch({
    return(rcpp_OBBPredictionsInterface(object@forest))
  }, error = function(err) {
    print(err)
    return(NA)
  })
  rcppOOBpreds[is.nan(rcppOOBpreds)] <- NA
  return(rcppOOBpreds)
}

# -- Calculate Variable Importance ---------------------------------------------
#' getVI-forestry
#' @rdname getVI-forestry
#' @description Calculate variable importance for `forestry` object as
#'   introduced by Breiman (2001). Returns a list of percentage increases in
#'   out-of-bag error when shuffling each feature values and getting
#'   out-of-bag error.
#' @param object A trained model object of class "forestry".
#' @param noWarning Flag to not display warnings or display warnings.
#' @seealso \code{\link{forestry}}
#' @export
getVI <- function(object,
                  noWarning = FALSE) {

    # Keep warning for small sample size
    if (!object@replace &&
        object@ntree * (rcpp_getObservationSizeInterface(object@dataframe) -
                        object@sampsize) < 10) {
      if (!noWarning) {
        warning(paste(
          "Samples are drawn without replacement and sample size",
          "is too big!"
        ))
      }
      return(NA)
    }
    rcppVI <- tryCatch({
      VI <- rcpp_VariableImportanceInterface(object@forest)[[1]]
      names(VI) <- colnames(object@processed_dta$processed_x)
      return(VI)
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppVI)
  }



# -- Add More Trees ------------------------------------------------------------
#' addTrees-forestry
#' @rdname addTrees-forestry
#' @description Add more trees to an existing forest.
#' @param object A trained model object of class "forestry".
#' @param ntree Number of new trees to add.
#' @return A trained model object of class "forestry".
#' @export
addTrees <- function(object,
                     ntree) {
    if (ntree <= 0 || ntree %% 1 != 0) {
      stop("ntree must be a positive integer.")
    }

    tryCatch({
      rcpp_AddTreeInterface(object@forest, ntree)
      object@ntree = object@ntree + ntree
      return(object)
    }, error = function(err) {
      print(err)
      return(NA)
    })

  }


# -- Translate C++ to R --------------------------------------------------------
#' @title Cpp to R translator
#' @description Add more trees to the existing forest.
#' @param object external CPP pointer that should be translated from Cpp to an R
#'   object
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @noRd
CppToR_translator <- function(object) {
    tryCatch({
      return(rcpp_CppToR_translator(object))
    }, error = function(err) {
      print(err)
      return(NA)
    })
  }

# -- Save RF -----------------------------------------------------
#' save RF
#' @rdname saveForestry-forestry
#' @description This wrapper function checks the forestry object, makes it
#'  saveable if needed, and then saves it.
#' @param object an object of class `forestry`
#' @param file a filename in which to store the `forestry` object
#' @param ... additional arguments useful for specifying compression type and level
#' @export
saveForestry <- function(object, filename, ...){
    # First we need to make sure the object is saveable
    forest <- make_savable(object)
    base::save(forest, file = filename, ...)
}

# -- Load RF -----------------------------------------------------
#' load RF
#' @rdname loadForestry-forestry
#' @description This wrapper function checks the forestry object, makes it
#'  saveable if needed, and then saves it.
#' @param filename a filename in which to store the `forestry` object
#' @export
loadForestry <- function(filename){
  # First we need to make sure the object is saveable
  name <- base::load(file = filename, envir = environment())
  rf <- get(name)
  rf <- relinkCPP_prt(rf)
  return(rf)
}


# -- relink forest CPP ptr -----------------------------------------------------
#' relink CPP ptr
#' @rdname relinkCPP
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @param object an object of class `forestry`
#' @export
relinkCPP_prt <- function(object) {
    # 1.) reconnect the data.frame to a cpp data.frame
    # 2.) reconnect the forest.
    tryCatch({
      forest_and_df_ptr <- rcpp_reconstructree(
        x = object@processed_dta$processed_x,
        y = object@processed_dta$y,
        catCols = object@processed_dta$categoricalFeatureCols_cpp,
        splitCols = object@processed_dta$splittingFeaturesCols_cpp,
        linCols = object@processed_dta$linearFeatureCols_cpp,
        numRows = object@processed_dta$nObservations,
        numColumns = object@processed_dta$numColumns,
        R_forest = object@R_forest,
        replace = object@replace,
        sampsize = object@sampsize,
        splitratio = object@splitratio,
        mtry = object@mtry,
        nodesizeSpl = object@nodesizeSpl,
        nodesizeAvg = object@nodesizeAvg,
        nodesizeStrictSpl = object@nodesizeStrictSpl,
        nodesizeStrictAvg = object@nodesizeStrictAvg,
        minSplitGain = object@minSplitGain,
        maxDepth = object@maxDepth,
        seed = sample(.Machine$integer.max, 1),
        nthread = 0, # will use all threads available.
        verbose = FALSE,
        middleSplit = object@middleSplit,
        maxObs = object@maxObs,
        maxProp = object@maxProp,
        sampleWeights = object@sampleWeights,
        monotonicConstraints = object@monotonicConstraints,
        linear = object@linear,
        overfitPenalty = object@overfitPenalty,
        doubleTree = object@doubleTree)

      object@forest <- forest_and_df_ptr$forest_ptr
      object@dataframe <- forest_and_df_ptr$data_frame_ptr

    }, error = function(err) {
      print('Problem when trying to create the forest object in Cpp')
      print(err)
      return(NA)
    })

    return(object)
  }




# -- relink forest CPP ptr -----------------------------------------------------
#' make_savable
#' @name make_savable
#' @rdname make_savable
#' @description When a `foresty` object is saved and then reloaded ,the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed.
#' @param object A trained model object of class "forestry".
#' @examples
#' set.seed(323652639)
#' x <- iris[, -1]
#' y <- iris[, 1]
#' forest <- forestry(x, y, ntree = 3)
#' y_pred_before <- predict(forest, x)
#'
#' forest <- make_savable(forest)
#' save(forest, file = "forest.Rda")
#' rm(forest)
#' load("forest.Rda", verbose = FALSE)
#' forest <- relinkCPP_prt(forest)
#'
#' y_pred_after <- predict(forest, x)
#' testthat::expect_equal(y_pred_before, y_pred_after, tolerance = 0.000001)
#' file.remove("forest.Rda")
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @seealso \code{\link{forestry}}
#' @aliases make_savable,forestry-method
#' @export
make_savable <- function(object) {
    object@R_forest <- CppToR_translator(object@forest)

    return(object)
  }




