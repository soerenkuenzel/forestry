#' @include forestry.R

# ---Computing detachment indices-----------------------------------------------------
#' comptute_lp
#' @name compute_detachments-forestry
#' @title compute detachment indices
#' @rdname compute_detachments-forestry
#' @description return detachment ditances of selected test observations.
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param detachment.feat A string denoting the feature for computing detachment indices with respect to.
#' @param p A positive real number determining the norm p-norm used.
#' @return A vector detachment indices.
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
#' # Compute the l-2 detachment indices in the "Petal.Length" dimension
#' detachments <- compute_detachments(object = rf,
#'                                    feature.new = x_test,
#'                                    detachment.feat = "Petal.Length",
#'                                    p = 2)
#' @export
compute_detachments <- function(object, feature.new, detachment.feat, p) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }


  if (!(detachment.feat %in% colnames(feature.new))) {
    stop("The submitted feature is not in the set of possible features")
  }

  # Preprocess the data
  testing_data_checker(feature.new)
  processed_x <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  # Get the feature column index in the R dataframe
  feat.num = which(colnames(processed_x) == detachment.feat)

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


