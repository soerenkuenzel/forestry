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
#'                           test = x_test,
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

  if (is.factor(feature.new[1, feature])) {
    diff_mat <- matrix(feature.new[,feature],
                       nrow = nrow(feature.new),
                       ncol = nrow(train_set),
                       byrow = TRUE) !=
                matrix(train_set[,feature],
                       nrow = nrow(feature.new),
                       ncol = nrow(train_set),
                       byrow = FALSE)
    diff_mat[diff_mat] <- 1
  } else {
    diff_mat <- matrix(feature.new[,feature],
                       nrow = nrow(feature.new),
                       ncol = nrow(train_set),
                       byrow = TRUE) -
                matrix(train_set[,feature],
                       nrow = nrow(feature.new),
                       ncol = nrow(train_set),
                       byrow = FALSE)
  }

  # Raise absoulte differences to the pth power
  diff_mat <- abs(diff_mat) ^ p

  # Compute final Lp distances
  distances <- apply(y_weights * diff_mat, 1, sum) ^ (1 / p)

  # Ensure that the Lp distances for a factor are between 0 and 1
  if (is.factor(feature.new[1, feature])) {
    distances[distances < 0] <- 0
    distances[distances > 1] <- 1
  }


  return(distances)
}



