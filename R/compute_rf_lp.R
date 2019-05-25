#' @include forestry.R

# ---Computing lp distances-----------------------------------------------------
#' comptute_lp
#' @name compute_lp-forestry
#' @title compute lp distances
#' @rdname compute_lp-forestry
#' @description return lp ditances of selected test observations.
#' @param object A `forestry` object.
#' @param feature.new A data frame of testing predictors.
#' @param feature A string denoting the dimension for computing lp distances.
#' @param p A positive real number determining the norm p-norm used.
#' @return A vector of lp distances.
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
#'                           feature.new = x_test,
#'                           feature = "Petal.Length",
#'                           p = 2)
#' @export
compute_lp <- function(object, feature.new, feature, p){

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  train_set <- slot(object, "processed_dta")$processed_x

  if (!(feature %in% colnames(train_set))) {
    stop("The submitted feature is not in the set of possible features")
  }

  feature.new <- preprocess_testing(feature.new,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)


  # Compute distances
  y_weights <- predict(object = object,
                       feature.new = feature.new,
                       aggregation = "weightMatrix")$weightMatrix

  distances <- compute_lp_bnd(y_weights = y_weights,
                              train_vec = train_set[,feature],
                              test_vec = feature.new[,feature],
                              p = p)
  return(distances)
}


# Backend function for compute_lp
compute_lp_bnd <- function(y_weights, train_vec, test_vec, p){
  # get difference matrix
  if (is.factor(test_vec)) {
    diff_mat <- matrix(test_vec,
                       nrow = length(test_vec),
                       ncol = length(train_vec),
                       byrow = FALSE) !=
                matrix(train_vec,
                       nrow = length(test_vec),
                       ncol = length(train_vec),
                       byrow = TRUE)
    diff_mat[diff_mat] <- 1
  } else {
    diff_mat <- matrix(test_vec,
                       nrow = length(test_vec),
                       ncol = length(train_vec),
                       byrow = FALSE) -
                matrix(train_vec,
                       nrow = length(test_vec),
                       ncol = length(train_vec),
                       byrow = TRUE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat) ^ p

  # Compute final Lp distances
  distances <- apply(y_weights * diff_mat, 1, sum) ^ (1 / p)

  # Ensure that the Lp distances for a factor are between 0 and 1
  if (is.factor(test_vec)) {
    distances[distances < 0] <- 0
    distances[distances > 1] <- 1
  }
  return(distances)
}








