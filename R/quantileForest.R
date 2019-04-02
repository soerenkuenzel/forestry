#' @include compute_rf_lp.R

#' comptute_lp
#' @name get_quantiles-forestry
#' @title compute quantiles
#' @rdname get_quantiles-forestry
#' @description return lp ditances of selected test observations.
#' @inheritParams compute_lp
#' @param quantiles quantiles to be computed at each x
#' @return A vector of lp distances.
#' @export
get_quantiles <- function(object, feature.new,
                          quantiles = c(.05, .5, .95)) {

  # Checks and parsing:
  if (class(object) != "forestry") {
    stop("The object submitted is not a forestry random forest")
  }

  feature.new <- as.data.frame(feature.new)
  train_y <- slot(object, "processed_dta")$y

  feature.new <- preprocess_testing(feature.new,
                                    object@categoricalFeatureCols,
                                    object@categoricalFeatureMapping)


  # Compute distances
  y_weights <- predict(object = object,
                       feature.new = feature.new,
                       aggregation = "weightMatrix")$weightMatrix


  order_of_y <- order(train_y)
  quants <- matrix(NA, nrow = nrow(feature.new), ncol = length(quantiles))
  colnames(quants) <- quantiles
  for (quantile_prop in quantiles) {
    # quantile_prop = quantiles[2]
    sum_total <- rep(0, nrow(feature.new))
    quantile <- rep(-Inf, nrow(feature.new))
    for (i in 1:length(train_y)) {
      # i = 1
      ord <- order_of_y[i]
      sum_total <- sum_total + y_weights[ ,ord]


      quantile[sum_total < quantile_prop] <-
        (train_y[ord] + train_y[order_of_y[i]]) / 2
    }
    quants[, as.character(quantile_prop)] <- quantile
  }

  return(quants)
}
