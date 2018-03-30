library(testthat)
test_that("Test non-continuous split", {

  x <- iris[, -1]
  y <- iris[, 1]
  context("Non-continuous split")
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
    forest <- forestry(
      x,
      y,
      ntree = 500,
      replace = TRUE,
      sampsize = nrow(x),
      mtry = 4,
      nodesizeStrictSpl = 5,
      nthread = 2,
      splitrule = "variance",
      splitratio = 1,
      nodesizeStrictAvg = 5,
      middleSplit = TRUE
    )

  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  expect_lt(mean((y_pred - y) ^ 2), .1)

})
