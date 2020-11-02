library(testthat)
test_that("Tests that Monotone splits parameter is working correctly", {
  x <- data.frame(V1 = runif(100, min = 0, max = 10))
  y <- .2*x[,1] + rnorm(100)

  context('Positive monotone splits')
  # Set seed for reproductivity
  set.seed(24750371)

  # Simulate some data that should be positive monotone

  monotone_forest <- forestry(
    x,
    y,
    ntree = 500,
    nodesizeStrictSpl = 5,
    maxDepth = 10,
    monotonicConstraints = c(1)
  )
  # Test predictions are monotonic increasing in the first feature
  pred_means <- sapply(c(1:9), function(x) {mean(predict(monotone_forest,
                                                    feature.new = data.frame(V1 = rep(x, 100))))})

  # Mean Square Error
  expect_equal(all.equal(order(pred_means), 1:9), TRUE)

  # Now try monotone decreasing in second feature- even though there is no real signal here
  monotone_forest <- forestry(
    x,
    y,
    ntree = 500,
    nodesizeStrictSpl = 3,
    maxDepth = 10,
    monotonicConstraints = c(-1)
  )
  # Test predictions are monotonic increasing in the first feature
  pred_means <- sapply(c(1,3,5,9), function(x) {mean(predict(monotone_forest,
                                                         feature.new = data.frame(V1 = rep(x, 100))))})

  # Mean Square Error
  # expect_equal(all.equal(order(pred_means), 4:1), TRUE)
})
