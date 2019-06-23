#' @include compute_detachments.R

#' get_conditional_quantiles
#' @name get_conditional_quantiles-forestry
#' @title compute conditional quantiles
#' @rdname get_conditional_quantiles-forestry
#' @description Computes and returns the condidtional quantiles for a random
#' forest.
#' @inheritParams compute_detachments
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param probs probabilities to be computed at each x
#' @return A data frame of quantiles of in response variable conditional on the
#' test observations.
#' @examples
#' # Set seed for reproductivity
#' set.seed(292313)
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 10)
#'
#' # Select, for example, sepal sength as the response variable
#' index <- which(colnames(iris) == "Sepal.Length")
#' x_train <- iris[-test_idx, -index]
#' y_train <- iris[-test_idx, index]
#' x_test <- iris[test_idx, -index]
#' rf <- forestry(x = x_train, y = y_train)
#'
#' # Compute the  0.25, 0.5, and 0.75 conditional quantiles for sepal length
#' quants <- get_conditional_quantiles(object = rf,
#'                                     feature.new = x_test,
#'                                     quantiles = c(0.25,0.5, 0.75))
#'
#' # Compare the quantiles to the preditcions
#' predict(rf, x_test)
#' @export
get_conditional_quantiles <- function(object,
                                      feature.new,
                                      probs) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  train_y <- slot(object, "processed_dta")$y

  feature.new <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  y_weights <- predict(object = object,
                       feature.new = feature.new,
                       aggregation = "weightMatrix")$weightMatrix

  order_of_y <- order(train_y)
  quants <- matrix(NA, nrow = nrow(feature.new), ncol = length(probs))
  colnames(quants) <- probs

  for (quantile_prop in probs) {
    sum_total <- rep(0, nrow(feature.new))
    quantiles <- rep(-Inf, nrow(feature.new))

    for (i in 1:length(train_y)) {
      # For training observation i, get its rank in y
      ord <- order_of_y[i]
      sum_total <- sum_total + y_weights[ ,ord]
      quantiles[sum_total <= quantile_prop] <- train_y[ord]
    }

    quants[, as.character(quantile_prop)] <- quantiles
  }

  quants <- as.data.frame(quants)
  colnames(quants) <- paste0("q", colnames(quants))
  rownames(quants) <- paste0("s", rownames(quants))
  return(quants)
}


#' get_conditional_distribution
#' @name get_conditional_distribution-forestry
#' @title compute conditional distribution
#' @rdname get_conditional_distribution-forestry
#' @description Computes and returns the condidtional distributions.
#' @inheritParams compute_detachments
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param vals values at which values the conditional cdf will be computed
#' @return A data frame of conditional cumulative probabilities
#' @examples
#' # Set seed for reproductivity
#' set.seed(292313)
#'
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 10)
#'
#' # Select, for example, sepal sength as the response variable
#' index <- which(colnames(iris) == "Sepal.Length")
#' x_train <- iris[-test_idx, -index]
#' y_train <- iris[-test_idx, index]
#' x_test <- iris[test_idx, -index]
#' rf <- forestry(x = x_train, y = y_train)
#'
#' vals <- rep(mean(y_train), 10)
#'
#' # Compute the conditional probabilities associated with values
#' probs <- get_conditional_distribution(rf, feature.new = x_test, vals)
#'
#' @export

get_conditional_distribution <- function(object,
                                     feature.new,
                                     vals) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  # Preprocess the data
  testing_data_checker(feature.new)
  processed_x <- preprocess_testing(feature.new,
                                    object@featureNames,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  train_y <- slot(object, "processed_dta")$y

  return (conditional_dist_bnd(object,
                               processed_x,
                               train_y,
                               vals))

}

conditional_dist_bnd <- function(object,
                                 processed_x,
                                 train_vector,
                                 test_vector,
                                 isCategoricalDimension = FALSE){

  # Ensure that categorical outcomes are encoded
  if(is.factor(train_vector) | is.character(train_vector)){
    trainer <- preprocess_training(train_vector, train_vector)
    train_vector <- trainer$x[ ,1]
    test_vector <- preprocess_testing(test_vector,
                                      featureNames = "x",
                                      trainer$categoricalFeatureCols,
                                      trainer$categoricalFeatureMapping)[ ,1]
    isCategoricalDimension <- TRUE
  }

  rcppPrediction <- tryCatch({
    rcpp_cppPredictInterface(object@forest,
                             processed_x,
                             aggregation = "average",
                             localVariableImportance = FALSE,
                             trainVec = train_vector,
                             testVec = test_vector,
                             isCatDimension = isCategoricalDimension)
  }, error = function(err) {
    print(err)
    return(NULL)
  })

  probs <- rcppPrediction$prediction
  probs[probs > 1] <- 1
  probs[probs < 0] <- 0
  return (as.data.frame(probs))
}








