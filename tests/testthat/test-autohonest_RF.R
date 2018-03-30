library(testthat)

test_that("Tests that autotuning is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  # Set seed for reproductivity
  set.seed(24750371)
  tuned_forest <- autohonestRF(x = x,
                               y = y,
                               num_iter = 9,
                               eta = 3,
                               nthread = 2)

  y_pred <- predict(tuned_forest, x)

  expect_lt(mean((y_pred - y) ^ 2), .3)


})
