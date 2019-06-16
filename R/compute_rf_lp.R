#' @include forestry.R

# ---Computing lp distances-----------------------------------------------------
#' comptute_lp
#' @name compute_lp-forestry
#' @title compute lp distances
#' @rdname compute_lp-forestry
#' @description return lp ditances of selected test observations.
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param distance.feat A string denoting the feature for computing lp distances with respect to.
#' @param p A positive real number determining the norm p-norm used.
#' @return A vector lp distances.
#' @examples
#' # Set seed for reproductivity
#' set.seed(292313)
#'
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 10)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train)
#' predictions <- predict(rf, x_test)
#'
#' # Compute the l-2 distances in the "Petal.Length" dimension
#' distances <- compute_lp(object = rf,
#'                         feature.new = x_test,
#'                         distance.feat = "Petal.Length",
#'                         p = 2)
#' @export
compute_lp <- function(object, feature.new, distance.feat, p) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }


  if (!(distance.feat %in% colnames(feature.new))) {
    stop("The submitted feature is not in the set of possible features")
  }

  # Preprocess the data
  testing_data_checker(feature.new)
  processed_x <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  # Get the feature column index in the R dataframe
  feat.num = which(colnames(processed_x) == distance.feat)

  # Set the aggregation to be average, and Local varaiable importace as False
  localVariableImportance = FALSE
  aggregation = "average"

  rcppPrediction <- tryCatch({
    rcpp_cppPredictInterface(object@forest,
                             processed_x,
                             aggregation,
                             localVariableImportance,
                             p,
                             feat.num)

  }, error = function(err) {
    print(err)
    return(NULL)
  })

  return(rcppPrediction$prediction)
}


