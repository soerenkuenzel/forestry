library(testthat)
test_that("Tests if ridgeRF works with minSplitGain", {
  context('Tests RidgeRF and the parameter minSplitGain')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  set.seed(231428176)
  forest <- forestry(
    x,
    y,
    ntree = 200,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    ridgeRF = TRUE,
    minSplitGain = 0.9,
    overfitPenalty = 1000
  )
  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  sum((y_pred - y) ^ 2)

  expect_equal(sum((y_pred - y) ^ 2), 42.8, tolerance = 0.2)
})
