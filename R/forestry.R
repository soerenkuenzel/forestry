#' @include R_preprocessing.R
#-- Sanity Checker -------------------------------------------------------------
#' @name training_data_checker
#' @title Training data check
#' @rdname training_data_checker-forestry
#' @description Check the input to forestry constructor
#' @inheritParams forestry
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
                                  splitratio,
                                  nthread,
                                  middleSplit,
                                  doubleTree) {
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

  # if the splitratio is 1, then we use adaptive rf and avgSampleSize is the
  # equal to the total sampsize
  if (splitratio == 0 || splitratio == 1) {
    splitSampleSize <- sampsize
    avgSampleSize <- sampsize
  } else {
    splitSampleSize <- splitratio * sampsize
    avgSampleSize <- sampsize - splitSampleSize
  }

  if (nodesizeStrictSpl > splitSampleSize) {
    warning(
      paste(
        "nodesizeStrictSpl cannot exceed splitting sample size.",
        "We have set nodesizeStrictSpl to be the maximum"
      )
    )
    nodesizeStrictSpl <<- splitSampleSize
  }
  if (nodesizeStrictAvg > avgSampleSize) {
    warning(
      paste(
        "nodesizeStrictAvg cannot exceed averaging sample size.",
        "We have set nodesizeStrictAvg to be the maximum"
      )
    )
    nodesizeStrictAvg <<- avgSampleSize
  }

  if (doubleTree) {
    if (splitratio == 0 || splitratio == 1) {
      warning("Trees cannot be doubled if splitratio is 1. We have set
              doubleTree to FALSE")
      doubleTree <<- FALSE
    } else {
      if (nodesizeStrictAvg > splitSampleSize) {
        warning(
          paste(
            "nodesizeStrictAvg cannot exceed splitting sample size.",
            "We have set nodesizeStrictAvg to be the maximum"
          )
        )
        nodesizeStrictAvg <<- splitSampleSize
      }
      if (nodesizeStrictSpl > avgSampleSize) {
        warning(
          paste(
            "nodesizeStrictSpl cannot exceed averaging sample size.",
            "We have set nodesizeStrictSpl to be the maximum"
          )
        )
        nodesizeStrictSpl <<- avgSampleSize
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
}

#' @title Test data check
#' @name testing_data_checker-forestry
#' @description Check the testing data to do prediction
#' @param feature.new A data frame of testing predictors.
testing_data_checker <- function(feature.new) {
  feature.new <- as.data.frame(feature.new)
  if (any(is.na(feature.new))) {
    stop("x contains missing data.")
  }
}

# -- Random Forest Constructor -------------------------------------------------
#' @title forestry class
#' @name forestry-class
#' @description `forestry` object implementing the most basic version of a
#'   random forest.
#' @slot forest An external pointer pointing to a C++ forestry object
#' @slot dataframe An external pointer pointing to a C++ DataFrame object
#' @slot processed_dta The R version of the training data frame. This will be an
#'   emapty dataframe, if the forest was created with the `saveable = FALSE`
#'   option. It is only used to reconstruct the forest after saving and loading
#'   it.
#' @slot R_forest This is a list containing for each tree a new list element.
#'   The list elements are in turn two lists: One containing the slit variable
#'   and the other one containing the split value. This is only saved, when
#'   `saveable = FALSE` and it will be used to reconstruct the C++ tree after
#'   saving and then reloding the tree.
#' @slot y A vector of all training responses.
#' @slot categoricalFeatureCols A list of index for all categorical data. Used
#'   for trees to detect categorical columns.
#' @slot categoricalFeatureMapping A list of encoding details for each
#'   categorical column, including all unique factor values and their
#'   corresponding numeric representation.
#' @slot ntree The number of trees to grow in the forest. The default value is
#'   500.
#' @slot replace An indicator of whether sampling of training data is with
#'   replacement. The default value is TRUE.
#' @slot sampsize The size of total samples to draw for the training data. If
#'   sampling with replacement, the default value is the length of the training
#'   data. If samplying without replacement, the default value is two-third of
#'   the length of the training data.
#' @slot mtry The number of variables randomly selected at each split point. The
#'   default value is set to be one third of total number of features of the
#'   training data.
#' @slot nodesizeSpl The minimum observations contained in terminal nodes. The
#'   default value is 3.
#' @slot nodesizeAvg Minimum size of terminal nodes for averaging dataset. The
#'   default value is 3.
#' @slot nodesizeStrictSpl Minimum observations to follow strictly in terminal
#'   nodes. The default value is 1.
#' @slot nodesizeStrictAvg Minimum size of terminal nodes for averaging dataset
#'   to follow strictly. The default value is 1.
#' @slot splitratio Proportion of the training data used as the splitting
#'   dataset. It is a ratio between 0 and 1. If the ratio is 1, then essentially
#'   splitting dataset becomes the total entire sampled set and the averaging
#'   dataset is empty. If the ratio is 0, then the splitting data set is empty
#'   and all the data is used for the averaging data set (This is not a good
#'   usage however since there will be no data available for splitting).
#' @slot middleSplit if the split value is taking the average of two feature
#'   values. If false, it will take a point based on a uniform distribution
#'   between two feature values. (Default = FALSE)
#' @slot maxObs The max number of observations to split on (Default = nrows(y))
#' @slot doubleTree if the number of tree is doubled as averaging and splitting
#'   data can be exchanged to create decorrelated trees. (Default = FALSE)
setClass(
  Class = "forestry",
  slots = list(
    forest = "externalptr",
    dataframe = "externalptr",
    processed_dta = "list",
    R_forest = "list",
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
    splitratio = "numeric",
    middleSplit = "logical",
    y = "vector",
    maxObs = "numeric",
    ridgeRF = "logical",
    overfitPenalty = "numeric",
    doubleTree = "logical"
  )
)


#' @title forestry-Constructor
#' @name forestry-forestry
#' @rdname forestry-forestry
#' @description Initialize a `forestry` object.
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
#' @param sample.fraction if this is given, then sampsize is ignored and set to
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
#' @param splitratio Proportion of the training data used as the splitting
#'   dataset. It is a ratio between 0 and 1. If the ratio is 1, then essentially
#'   splitting dataset becomes the total entire sampled set and the averaging
#'   dataset is empty. If the ratio is 0, then the splitting data set is empty
#'   and all the data is used for the averaging data set (This is not a good
#'   usage however since there will be no data available for splitting).
#' @param seed random seed
#' @param verbose if training process in verbose mode
#' @param nthread Number of threads to train and predict the forest. The default
#'   number is 0 which represents using all cores.
#' @param splitrule only variance is implemented at this point and it contains
#'   specifies the loss function according to which the splits of random forest
#'   should be made
#' @param middleSplit if the split value is taking the average of two feature
#'   values. If false, it will take a point based on a uniform distribution
#'   between two feature values. (Default = FALSE)
#' @param doubleTree if the number of tree is doubled as averaging and splitting
#'   data can be exchanged to create decorrelated trees. (Default = FALSE)
#' @param reuseforestry pass in an `forestry` object which will recycle the
#'   dataframe the old object created. It will save some space working on the
#'   same dataset.
#' @param maxObs The max number of observations to split on
#' @param saveable If TRUE, then RF is created in such a way that it can be
#'   saved and loaded using save(...) and load(...). Setting it to TRUE
#'   (default) will, however, take longer and it will use more memory. When
#'   training many RF, it makes a lot of sense to set this to FALSE to save
#'   time and memory.
#' @param ridgeRF Fit the model with a ridge regression or not
#' @param overfitPenalty Value to determine how much to penalize magnitude of
#' coefficients in ridge regression
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
#'           ridgeRF = TRUE
#'           )
#'
#' predict(forest, x)
#' @export forestry
setGeneric(
  name = "forestry",
  def = function(x,
                 y,
                 ntree,
                 replace,
                 sampsize,
                 sample.fraction,
                 mtry,
                 nodesizeSpl,
                 nodesizeAvg,
                 nodesizeStrictSpl,
                 nodesizeStrictAvg,
                 splitratio,
                 seed,
                 verbose,
                 nthread,
                 splitrule,
                 middleSplit,
                 maxObs,
                 ridgeRF,
                 overfitPenalty,
                 doubleTree,
                 reuseforestry,
                 saveable) {
    standardGeneric("forestry")
  }
)

#' @title forestry-Constructor
#' @rdname forestry-forestry
#' @aliases forestry forestry-method
#' @importFrom Rcpp evalCpp
#' @useDynLib forestry
#' @return A `forestry` object.
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
                     splitratio = 1,
                     seed = as.integer(runif(1) * 1000),
                     verbose = FALSE,
                     nthread = 0,
                     splitrule = "variance",
                     middleSplit = FALSE,
                     maxObs = length(y),
                     ridgeRF = FALSE,
                     overfitPenalty = 1,
                     doubleTree = FALSE,
                     reuseforestry = NULL,
                     saveable = TRUE) {
  # only if sample.fraction is given, update sampsize
  if (!is.null(sample.fraction))
    sampsize <- ceiling(sample.fraction * nrow(x))

  x <- as.data.frame(x)
  # Preprocess the data
  training_data_checker(x, y, ntree,replace, sampsize, mtry, nodesizeSpl,
                        nodesizeAvg, nodesizeStrictSpl, nodesizeStrictAvg,
                        splitratio, nthread, middleSplit, doubleTree)
  # Total number of obervations
  nObservations <- length(y)
  numColumns <- ncol(x)

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
      categoricalFeatureCols_cpp <- categoricalFeatureCols_cpp - 1
    }

    # Create rcpp object
    # Create a forest object
    forest <- tryCatch({
      rcppDataFrame <- rcpp_cppDataFrameInterface(processed_x,
                                                  y,
                                                  categoricalFeatureCols_cpp,
                                                  nObservations,
                                                  numColumns)

      rcppForest <- rcpp_cppBuildInterface(
        processed_x,
        y,
        categoricalFeatureCols_cpp,
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
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        ridgeRF,
        overfitPenalty,
        doubleTree,
        TRUE,
        rcppDataFrame
      )
      processed_dta <- list(
        "processed_x" = processed_x,
        "y" = y,
        "categoricalFeatureCols_cpp" = categoricalFeatureCols_cpp,
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
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          ridgeRF = ridgeRF,
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
        seed,
        nthread,
        verbose,
        middleSplit,
        maxObs,
        ridgeRF,
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
          splitratio = splitratio,
          middleSplit = middleSplit,
          maxObs = maxObs,
          ridgeRF = ridgeRF,
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


# -- Predict Method ------------------------------------------------------------
#' predict-forestry
#' @name predict-forestry
#' @rdname predict-forestry
#' @description Return the prediction from the forest.
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param aggregation How shall the leaf be aggregated. The default is to return
#'   the mean of the leave `average`. Other options are `weightMatrix`.
#' @return A vector of predicted responses.
#' @aliases predict,forestry-method
#' @exportMethod predict
setMethod(
  f = "predict",
  signature = "forestry",
  definition = function(object,
                        feature.new,
                        aggregation = "average") {
    # Preprocess the data
    testing_data_checker(feature.new)

    processed_x <- preprocess_testing(feature.new,
                                      object@categoricalFeatureCols,
                                      object@categoricalFeatureMapping)

    rcppPrediction <- tryCatch({
      rcpp_cppPredictInterface(object@forest, processed_x, aggregation)
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
)


# -- Calculate OOB Error -------------------------------------------------------
#' @title getOOB-forestry
#' @name getOOB-forestry
#' @rdname getOOB-forestry
#' @description Calculate the out-of-bag error of a given forest.
#' @param object A `forestry` object.
#' @param noWarning flag to not display warnings
#' @aliases getOOB
setGeneric(
  name = "getOOB",
  def = function(object,
                 noWarning = FALSE) {
    standardGeneric("getOOB")
  }
)

#' @title getOOB-forestry
#' @description Calculate the out-of-bag error of a given forest.
#' @param object A `forestry` object.
#' @param noWarning flag to not display warnings
#' @aliases getOOB,forestry-method
#' @return The OOB error of the forest.
#' @exportMethod getOOB
setMethod(
  f = "getOOB",
  signature = "forestry",
  definition = function(object,
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
)

# -- Calculate Variable Importance ---------------------------------------------
#' @title getVI-forestry
#' @name getVI-forestry
#' @rdname getVI-forestry
#' @description Calculate increase in OOB for each shuffled feature for forest.
#' @param object A `forestry` object.
#' @param noWarning flag to not display warnings
#' @aliases getVI
setGeneric(
  name = "getVI",
  def = function(object,
                 noWarning = FALSE) {
    standardGeneric("getVI")
  }
)

#' @title getVI-forestry
setMethod(
  f = "getVI",
  signature = "forestry",
  definition = function(object,
                        noWarning) {
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
      return(rcpp_VariableImportanceInterface(object@forest))
    }, error = function(err) {
      print(err)
      return(NA)
    })

    return(rcppVI)
  }
)


# -- Add More Trees ------------------------------------------------------------
#' @title addTrees-forestry
#' @name addTrees-forestry
#' @rdname addTrees-forestry
#' @description Add more trees to the existing forest.
#' @param object A `forestry` object.
#' @param ntree Number of new trees to add
#' @aliases addTrees
setGeneric(
  name = "addTrees",
  def = function(object,
                 ntree) {
    standardGeneric("addTrees")
  }
)

#' @title addTrees-forestry
#' @description Add more trees to the existing forest.
#' @param object A `forestry` object.
#' @param ntree Number of new trees to add
#' @aliases addTrees,forestry-method
#' @exportMethod addTrees
#' @return A `forestry` object
setMethod(
  f = "addTrees",
  signature = "forestry",
  definition = function(object,
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
)


# -- Auto-Tune -----------------------------------------------------------------
#' @title autoforestry-forestry
#' @name autoforestry-forestry
#' @rdname autoforestry-forestry
#' @description Autotune a forestry based on the input dataset. The methodology
#'   is based on paper `Hyperband: A Novel Bandit-Based Approach to
#'   Hyperparameter Optimization` by Lisha Li, et al.
#' @inheritParams forestry
#' @param sampsize The size of total samples to draw for the training data.
#' @param num_iter Maximum iterations/epochs per configuration. Default is 1024.
#' @param eta Downsampling rate. Default value is 2.
#' @param verbose if tuning process in verbose mode
setGeneric(
  name = "autoforestry",
  def = function(x,
                 y,
                 sampsize,
                 num_iter,
                 eta,
                 verbose,
                 seed,
                 nthread) {
    standardGeneric("autoforestry")
  }
)

#' @title autoforestry-forestry
#' @description Autotune a forestry based on the input dataset. The methodology
#'   is based on paper `Hyperband: A Novel Bandit-Based Approach to
#'   Hyperparameter Optimization` by Lisha Li, et al.
#' @inheritParams forestry
#' @param sampsize The size of total samples to draw for the training data.
#' @param num_iter Maximum iterations/epochs per configuration. Default is 1024.
#' @param eta Downsampling rate. Default value is 2.
#' @param verbose if tuning process in verbose mode
#' @param seed random seed
#' @param nthread Number of threads to train and predict theforest. The default
#'   number is 0 which represents using all cores.
#' @aliases autoforestry,forestry-method
#' @return A `forestry` object
#' @export autoforestry
#' @import stats
autoforestry <- function(x,
                         y,
                         sampsize = as.integer(nrow(x) * 0.75),
                         num_iter = 1024,
                         eta = 2,
                         verbose = FALSE,
                         seed = 24750371,
                         nthread = 0) {
  if (verbose) {
    print("Start auto-tuning.")
  }

  # Creat a dummy tree just to reuse its data.
  dummy_tree <-
    forestry(
      x,
      y,
      ntree = 1,
      nodesizeSpl = nrow(x),
      nodesizeAvg = nrow(x)
    )

  # Number of unique executions of Successive Halving (minus one)
  s_max <- as.integer(log(num_iter) / log(eta))

  # Total number of iterations (without reuse) per execution of
  # successive halving (n,r)
  B <- (s_max + 1) * num_iter

  if (verbose) {
    print(
      paste(
        "Hyperband will run successive halving in",
        s_max,
        "times, with",
        B,
        "iterations per execution."
      )
    )
  }

  # Begin finite horizon hyperband outlerloop
  models <- vector("list", s_max + 1)
  models_OOB <- vector("list", s_max + 1)

  set.seed(seed)

  for (s in s_max:0) {
    if (verbose) {
      print(paste("Hyperband successive halving round", s_max + 1 - s))
    }

    # Initial number of configurations
    n <- as.integer(ceiling(B / num_iter / (s + 1) * eta ^ s))

    # Initial number of iterations to run configurations for
    r <- num_iter * eta ^ (-s)

    if (verbose) {
      print(paste(">>> Total number of configurations:", n))
      print(paste(
        ">>> Number of iterations per configuration:",
        as.integer(r)
      ))
    }

    # Begin finite horizon successive halving with (n,r)
    # Generate parameters:
    allConfigs <- data.frame(
      mtry = sample(1:ncol(x), n, replace = TRUE),
      min_node_size_spl = NA, #sample(1:min(30, nrow(x)), n, replace = TRUE),
      min_node_size_ave = NA, #sample(1:min(30, nrow(x)), n, replace = TRUE),
      splitratio = runif(n, min = 0.1, max = 1),
      replace = sample(c(TRUE, FALSE), n, replace = TRUE),
      middleSplit = sample(c(TRUE, FALSE), n, replace = TRUE)
    )

    min_node_size_spl_raw <- floor(allConfigs$splitratio * sampsize *
                                     rbeta(n, 1, 3))
    allConfigs$min_node_size_spl <- ifelse(min_node_size_spl_raw == 0, 1,
                                           min_node_size_spl_raw)
    min_node_size_ave <- floor((1 - allConfigs$splitratio) * sampsize *
                                 rbeta(n, 1, 3))
    allConfigs$min_node_size_ave <- ifelse(min_node_size_ave == 0, 1,
                                           min_node_size_ave)

    if (verbose) {
      print(paste(">>>", n, " configurations have been generated."))
    }

    val_models <- vector("list", nrow(allConfigs))
    r_old <- 1
    for (j in 1:nrow(allConfigs)) {
      tryCatch({
        val_models[[j]] <- forestry(
          x = x,
          y = y,
          ntree = r_old,
          mtry = allConfigs$mtry[j],
          nodesizeSpl = allConfigs$min_node_size_spl[j],
          nodesizeAvg = allConfigs$min_node_size_ave[j],
          splitratio = allConfigs$splitratio[j],
          replace = allConfigs$replace[j],
          sampsize = sampsize,
          nthread = nthread,
          middleSplit = allConfigs$middleSplit[j],
          reuseforestry = dummy_tree
        )
      }, error = function(err) {
        val_models[[j]] <- NULL
      })
    }

    if (s != 0) {
      for (i in 0:(s - 1)) {
        # Run each of the n_i configs for r_i iterations and keep best
        # n_i/eta
        n_i <- as.integer(n * eta ^ (-i))
        r_i <- as.integer(r * eta ^ i)
        r_new <- r_i - r_old

        # if (verbose) {
        #   print(paste("Iterations", i))
        #   print(paste("Total number of configurations:", n_i))
        #   print(paste("Number of iterations per configuration:", r_i))
        # }

        val_losses <- vector("list", nrow(allConfigs))

        # Iterate to evaluate each parameter combination and cut the
        # parameter pools in half every iteration based on its score
        for (j in 1:nrow(allConfigs)) {
          if (r_new > 0 && !is.null(val_models[[j]])) {
            val_models[[j]] <- addTrees(val_models[[j]], r_new)
          }
          if (!is.null(val_models[[j]])) {
            val_losses[[j]] <- getOOB(val_models[[j]], noWarning = TRUE)
            if (is.na(val_losses[[j]])) {
              val_losses[[j]] <- Inf
            }
          } else {
            val_losses[[j]] <- Inf
          }
        }

        r_old <- r_i

        val_losses_idx <-
          sort(unlist(val_losses), index.return = TRUE)
        val_top_idx <- val_losses_idx$ix[0:as.integer(n_i / eta)]
        allConfigs <- allConfigs[val_top_idx,]
        val_models <- val_models[val_top_idx]
        gc()
        rownames(allConfigs) <- 1:nrow(allConfigs)

        # if (verbose) {
        #   print(paste(length(val_losses_idx$ix) - nrow(allConfigs),
        #               "configurations have been eliminated."))
        # }
      }
    }
    # End finite horizon successive halving with (n,r)
    if (!is.null(val_models[[1]])) {
      best_OOB <- getOOB(val_models[[1]], noWarning = TRUE)
      if (is.na(best_OOB)) {
        stop()
        best_OOB <- Inf
      }
    } else {
      stop()
      best_OOB <- Inf
    }
    if (verbose) {
      print(paste(">>> Successive halving ends and the best model is saved."))
      print(paste(">>> OOB:", best_OOB))
    }

    if (!is.null(val_models[[1]]))
      models[[s + 1]] <- val_models[[1]]
    models_OOB[[s + 1]] <- best_OOB

  }

  # End finite horizon hyperband outlerloop and sort by performance
  model_losses_idx <- sort(unlist(models_OOB), index.return = TRUE)

  if (verbose) {
    print(
      paste(
        "Best model is selected from best-performed model in",
        s_max,
        "successive halving, with OOB",
        models_OOB[model_losses_idx$ix[1]]
      )
    )
  }

  return(models[[model_losses_idx$ix[1]]])
}

# -- Translate C++ to R --------------------------------------------------------
#' @title Cpp to R translator
#' @name CppToR_translator
#' @description Translates the forest to a list which can then be used with the
#'   RToCPP_translator to create an CPP forest object again
#' @param object A `forestry` object.
setGeneric(
  name = "CppToR_translator",
  def = function(object) {
    standardGeneric("CppToR_translator")
  }
)

#' @title CppToR_translator
#' @description Add more trees to the existing forest.
#' @exportMethod CppToR_translator
#' @inheritParams CppToR_translator
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
setMethod(
  f = "CppToR_translator",
  signature = "externalptr",
  definition = function(object) {
    tryCatch({
      return(rcpp_CppToR_translator(object))
    }, error = function(err) {
      print(err)
      return(NA)
    })
  }
)

# -- relink forest CPP ptr -----------------------------------------------------
#' relink CPP ptr
#' @rdname relink
#' @name relinkCPP_prt
#' @rdname relinkCPP
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @param object an object of class `forestry`
#' @exportMethod relinkCPP_prt
setGeneric(
  name = "relinkCPP_prt",
  def = function(object) {
    standardGeneric("relinkCPP_prt")
  }
)

#' relink CPP ptr
#' @name relinkCPP_prt-forestry
#' @rdname relinkCPP_prt-forestry
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @inheritParams relinkCPP_prt
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @aliases relinkCPP_prt,forestry-method
#' @exportMethod relinkCPP_prt
setMethod(
  f = "relinkCPP_prt",
  signature = "forestry",
  definition = function(object) {
    # 1.) reconnect the data.frame to a cpp data.frame
    # 2.) reconnect the forest.
    tryCatch({
      forest_and_df_ptr <- rcpp_reconstructree(
        x = object@processed_dta$processed_x,
        y = object@processed_dta$y,
        catCols = object@processed_dta$categoricalFeatureCols_cpp,
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
        seed = sample(.Machine$integer.max, 1),
        nthread = 0, # will use all threads available.
        verbose = FALSE,
        middleSplit = object@middleSplit,
        maxObs = object@maxObs,
        ridgeRF = object@ridgeRF,
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
)



# -- relink forest CPP ptr -----------------------------------------------------
#' make_savable
#' @name make_savable
#' @rdname make_savable
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @param object an object of class `forestry`
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
#' testthat::expect_equal(y_pred_before, y_pred_after)
#' file.remove("forest.Rda")
setGeneric(
  name = "make_savable",
  def = function(object) {
    standardGeneric("make_savable")
  }
)

#' make_savable
#' @name make_savable-forestry
#' @rdname make_savable-forestry
#' @description When a `foresty` object is saved and then reloaded the Cpp
#'   pointers for the data set and the Cpp forest have to be reconstructed
#' @inheritParams make_savable
#' @return A list of lists. Each sublist contains the information to span a
#'   tree.
#' @aliases make_savable,forestry-method
#' @exportMethod make_savable
setMethod(
  f = "make_savable",
  signature = "forestry",
  definition = function(object) {
    object@R_forest <- CppToR_translator(object@forest)

    return(object)
  }
)



