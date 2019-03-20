
#' @include forestry.R

# ---Computing lp distances-----------------------------------------------------
#' comptute_lp
#' @name compute_lp-forestry
#' @title compute lp distances
#' @rdname compute_lp-forestry
#' @description return lp ditances of selected test observations.
#' @param object A `forestry` object.
#' @param test_set A data frame of testing predictors.
#' @param feature A string denoting the dimension for computing lp distances.
#' @param p A positive real number determining the norm p-norm used.
#' @param ... additional arguments.
#' @return A vector lp distances.
#' @export
compute_lp <- function(object, test_set, feature, p){

  # Check the object is a "forestry" object
  # TODO

  # Extract the training data from the forest
  train_set <- slot(object, "processed_dta")$processed_x
  y_weights <- predict(object = object,
                       feature.new = test_set,
                       aggregation = "weightMatrix")$weightMatrix

  # Check that the test observations have correct features
  # TODO

  if (is.factor(as.data.frame(test_set)[1, feature])) {

    # Get categorical feature mapping
    mapping <- slot(object, "categoricalFeatureMapping")

    # replace the test_set categories with appropriate numbers
    # TODO

    diff_mat <- matrix(as.data.frame(test_set)[,feature],
                       nrow = nrow(test_set),
                       ncol = nrow(train_set),
                       byrow = TRUE) !=
                matrix(as.data.frame(train_set)[,feature],
                       nrow = nrow(test_set),
                       ncol = nrow(train_set),
                       byrow = FALSE)
    diff_mat[diff_mat] <- 1
  }
  else{
    diff_mat <- matrix(as.data.frame(test_set)[,feature],
                      nrow = nrow(test_set),
                      ncol = nrow(train_set),
                      byrow = TRUE) -
                matrix(as.data.frame(train_set)[,feature],
                      nrow = nrow(test_set),
                      ncol = nrow(train_set),
                      byrow = FALSE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat)^(p)

  # Compute final Lp distances
  distances = apply(y_weights * diff_mat, 1, sum)^(1/p)

  return(distances)
}



