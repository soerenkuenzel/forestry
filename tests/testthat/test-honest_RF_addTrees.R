test_that("Tests adding more trees", {

  x <- iris[, -1]
  y <- iris[, 1]

  # Set seed for reproductivity
  set.seed(24750371)

  # Test honestRF (mimic RF)
    forest <- honestRF(
      x,
      y,
      ntree = 500,
      replace = TRUE,
      sampsize = nrow(x),
      mtry = 3,
      nodesizeSpl = 5,
      nthread = 2,
      splitrule = "variance",
      splitratio = 1,
      nodesizeAvg = 5
    )

  # Test add more trees
  forest <- addTrees(forest, 100)
  expect_equal(forest@ntree, 600, tolerance=1e-4)

  # Test predict
  y_pred <- predict(forest, x)

  # Mean Square Error
  expect_lt(mean((y_pred - y) ^ 2), 0.1)

})
