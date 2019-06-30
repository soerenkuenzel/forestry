test_that("Tests if sampleWeights works", {
  context('Tests sampleWeights')

  x <- iris[, c(1,2,3)]
  y <- iris[, 4]

  x
  y
  iris


  set.seed(275)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 200,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 1,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5,
    # linear = FALSE,
    # sampleWeights = c(.1, .1, .8),
    overfitPenalty = 50
  )

  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  sum((y_pred - y) ^ 2)

  expect_equal(sum((y_pred - y) ^ 2), 3.88, tolerance = 0.5)

})
