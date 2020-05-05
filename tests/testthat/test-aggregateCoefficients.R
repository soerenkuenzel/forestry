test_that("Tests if coefficient aggregation for linear works", {
  context('Tests linear coefficient aggregation')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  x
  y
  iris


  set.seed(275)

  # Test linear forestry
  forest <- forestry(
    x,
    y,
    ntree = 1,
    linear = TRUE,
    overfitPenalty = .5
  )

  # Test prediction using the coefficients from ridge regression
  y_pred <- predict(forest, x, aggregation = "coefs")
  padded_x <- cbind(x, rep(1, nrow(x)))

  # Check using coefficients for prediction
  coefficient_pred <- as.matrix(padded_x[17,]) %*% as.matrix((y_pred$coef[17,]))
  y_pred$predictions[17]
  coefficient_pred[1]

  expect_equal(y_pred$predictions[17], coefficient_pred[1], tolerance = 0.1)


  # Now check one independent feature case
  x <- iris[, 1]
  y <- iris[, 2]
  # Test linear forestry
  forest <- forestry(
    x,
    y,
    ntree = 1,
    linear = TRUE,
    overfitPenalty = .5
  )

  # Test prediction using the coefficients from ridge regression
  y_pred <- predict(forest, x, aggregation = "coefs")
  padded_x <- cbind(data.frame(x), rep(1, length(x)))

  # Check using coefficients for prediction
  coefficient_pred <- as.matrix(padded_x[25,]) %*% as.matrix((y_pred$coef[25,]))
  coefficient_pred[1]
  y_pred$predictions[25]

  expect_equal(y_pred$predictions[25], coefficient_pred[1], tolerance = 0.1)
})
