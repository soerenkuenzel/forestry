test_that("Tests that random forest is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Weight matrix')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 5,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  # Test predict
  y_pred <- predict(forest, x[1:5,], aggregation = 'indices')
  y_pred <- predict(forest, x[1:5,])

  # Mean Square Error
  sum((y_pred - y[1:5]) ^ 2)
  expect_equal(sum((y_pred - y[1:5]) ^ 2), 0.1891317, tolerance = 1e-6)
})
