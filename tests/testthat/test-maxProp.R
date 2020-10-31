library(testthat)
test_that("Tests that maxProp parameter is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Sample 30% of possible split points')
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
    maxDepth = 10,
    nthread = 2,
    maxProp = .3
  )
  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  expect_equal(sum((y_pred - y) ^ 2), 12.61745, tolerance = 0.5)
})
