test_that("Tests that quantile forest is working correctly", {

  context('Quantile Forest')

  # Set seed for reproductivity
  set.seed(24750371)
  test_idx <- sample(nrow(iris), 10)

  index <- which(colnames(iris) == "Sepal.Length")
  x_train <- iris[-test_idx, -index]
  y_train <- iris[-test_idx, index]
  x_test <- iris[test_idx, -index]

  # Test forestry (mimic RF)
  forest <- forestry(
    x_train,
    y_train,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )


  quantiles <- get_conditional_quantiles(
    object = forest,
    feature.new = x_test,
    probs = c(0, .05, .6, .5, .8, .95, 1)
  )

  probs <- get_conditional_distribution(
    object = forest,
    feature.new = x_test,
    vals = rep(mean(y_train), 10)
  )


  # Tests for quantile function
  expect_equal(dim(quantiles), c(10, 7))
  expect_equal(as.numeric(quantiles[1,5]), 5, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[2,1]), 4.9, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[5,3]), 4.8, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[10,6]), 6.4, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[8,7]), 7.9, tolerance = 1e-2)
  expect_identical(quantiles[7,1], -Inf)

  # Tests for distribution function
  expect_equal(dim(probs), c(10,1))
  expect_equal(probs$probs,
               c(0.99999 ,0.50436, 0.01402, 0.99999, 0.99999,
                 0.00272, 0.97495, 0.12749, 0.01148, 0.69632),
               tolerance = 1e-2)
})








