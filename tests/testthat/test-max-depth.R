library(testthat)
test_that("Tests that maxDepth parameter is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Maximum depth for a tree')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    maxDepth = 4,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )
  # Test predict
  y_pred <- predict(forest, x)
  sum((y_pred - y) ^ 2)
  # Mean Square Error
  expect_equal(sum((y_pred - y) ^ 2), 11.25824, tolerance = 0.5)
})
