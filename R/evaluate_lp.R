#' @include quantileForest.R

#' compute lp levels
#' @name evaluate_lp-forestry
#' @title compute trust levels
#' @rdname evaluate_lp-forestry
#' @description Computes and returns the lp levels of new observations in
#' specified dimensions.
#' @inheritParams compute_lp
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param feature a list of features for computing the levels with respect to.
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
#' features <- c("Sepal.Width", "Petal.Length", "Petal.Width")
#'
#' # Evaluate the test observations' lp distances
#' trust <- evaluate_lp(object = rf,
#'                      feature.new = x_test,
#'                      feature = features,
#'                      p = 1)
#' #' @export
evaluate_lp <- function(object, feature.new, feature, p = 1){

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  x_train <- slot(object, "processed_dta")$processed_x
  y_train <- slot(object, "processed_dta")$y

  feature.new <- preprocess_testing(feature.new,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)

  eval <- data.frame(1:nrow(feature.new))
  for (feat in feature){
    # Compute lp distances for new data
    lp_distances <- compute_lp(object = object,
                               feature.new = feature.new,
                               feature = feat,
                               p = p)
    # Set seed for reproductivity
    set.seed(24750371)
    # Compute lp distance for the training data using OOB observations:
    k_CV <- 10
    folds <- caret::createFolds(y_train, k = k_CV, list = TRUE,
                               returnTrain = FALSE)
    # Create a vector of lp distances for training observations to be filled
    x_train_lp <- rep(NA, nrow(x_train))

    for(k in 1:k_CV){
      fold_ids <- folds[[k]]
      rf <- forestry(x = x_train[-fold_ids, ], y = y_train[-fold_ids])
      x_train_lp[fold_ids] <- compute_lp(object = rf,
                                         feature.new = x_train[fold_ids, ],
                                         feature = feat,
                                         p = p)
    }

    # Train a random forest with lp distances as the response variable
    lp_rf <- forestry(x = x_train, y = x_train_lp)

    # Get conditional probabilities for new data
    probs <- get_conditional_distribution(object = lp_rf,
                                          feature.new = feature.new,
                                          vals = lp_distances)
    colnames(probs)[1] <- feat
    eval <- cbind(eval, probs)
  }

  eval <- eval[ ,-1]
  return(eval)
}
