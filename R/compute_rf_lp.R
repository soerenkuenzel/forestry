
#' @include forestry.R

# ---Computing lp distances-----------------------------------------------------
#' comptute_lp
#' @name compute_lp
#' @rdname compute_lp
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
  if(class(object)!="forestry"){
    stop("The object submitted is not a forestry random forest")
  }

  # Ensure the test set is a data frame
  test_set = as.data.frame(test_set)

  # Extract the training data from the forest
  train_set <- slot(object, "processed_dta")$processed_x

  # Check the submitted feature is in the set of possible features
  if(!(feature %in% colnames(train_set))){
    stop("The submitted feature is not in the set of possible features")
  }

  y_weights <- predict(object = object,
                       feature.new = test_set,
                       aggregation = "weightMatrix")$weightMatrix

  if (is.factor(test_set[1, feature])) {

    # Get categorical feature mapping
    mapping <-slot(object, "categoricalFeatureMapping")

    # Change factor values to corresponding integer levels
    factor_vals = mapping[[1]][2][[1]]
    map <- function(x) {return(which(factor_vals == x)[1])}
    test_set[,feature] <- unlist(lapply(test_set[,feature], map))

    diff_mat <- matrix(test_set[,feature],
                       nrow = nrow(test_set),
                       ncol = nrow(train_set),
                       byrow = TRUE) !=
                matrix(train_set[,feature],
                       nrow = nrow(test_set),
                       ncol = nrow(train_set),
                       byrow = FALSE)
    diff_mat[diff_mat] <- 1
  }
  else{
    diff_mat <- matrix(test_set[,feature],
                      nrow = nrow(test_set),
                      ncol = nrow(train_set),
                      byrow = TRUE) -
                matrix(train_set[,feature],
                      nrow = nrow(test_set),
                      ncol = nrow(train_set),
                      byrow = FALSE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat)^(p)

  # Compute final Lp distances
  distances <- apply(y_weights * diff_mat, 1, sum)^(1/p)

  return(distances)
}



