
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
#' @examples
#'
#' # Set seed for reproductivity
#' set.seed(292313)
#'
#' # Use Iris Data
#' test_idx <- sample(nrow(iris), 11)
#' x_train <- iris[-test_idx, -1]
#' y_train <- iris[-test_idx, 1]
#' x_test <- iris[test_idx, -1]
#'
#' rf <- forestry(x = x_train, y = y_train)
#' predict(rf, x_test)
#'
#' # Compute the l2 distances in the "Petal.Length" dimension
#' distances_2 <- compute_lp(object = rf,
#'                           test = x_test,
#'                           feature = "Petal.Length",
#'                           p = 2)
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

  # Ensure that the Lp distances for a factor are between 0 and 1
  if(is.factor(test_set[1, feature])){
    f <- function(x){if (x>1){x = 1} else if (x<0){x = 0} else{}}
    distances = apply(distances,f)}

  return(distances)
}



