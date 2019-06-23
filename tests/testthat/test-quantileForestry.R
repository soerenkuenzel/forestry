test_that("Tests that quantile forest is working correctly", {

  context('Quantile Forest')

  # Set seed for reproductivity
  set.seed(24750371)
  test_idx <- sample(nrow(iris), 10)
  index <- which(colnames(iris) == "Petal.Length")
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

  probs_1 <- get_conditional_distribution(
    object = forest,
    feature.new = x_test,
    vals = rep(mean(y_train), 10)
  )


  # Grow another tree with a categorical response
  index <- which(colnames(iris) == "Species")
  x_train <- iris[-test_idx, -index]
  y_train <- iris[-test_idx, index]
  x_test <- iris[test_idx, -index]
  rf <- forestry(x_train, y_train)

  probs_2 <- get_conditional_distribution(
    object = rf,
    feature.new = x_test,
    vals = c("setosa", "versicolor", "versicolor", "setosa", "virginica",
             "rose", "setosa", "setosa", "virginica","versicolor")
  )

  # Tests for quantile function
  expect_equal(dim(quantiles), c(10, 7))
  expect_equal(as.numeric(quantiles[1,5]), 1.6, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[2,1]), 4.5, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[5,3]), 1.4, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[10,6]), 4.5, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[8,7]), 6.9, tolerance = 1e-2)
  expect_identical(quantiles[4,1], -Inf)

  # Tests for distribution function
  expect_equal(probs_1$probs,
               c(1.00000, 0.00000, 0.00000, 1.00000, 1.00000,
                 0.00000, 0.57339, 0.00000, 0.00000, 0.07963),
               tolerance = 1e-2)
  expect_equal(probs_2$probs,
               c(0.991999, 0.117433, 0.029416, 0.998923, 1.000000,
                 0.971333, 0.040533, 0.924916, 1.000000, 0.985833),
               tolerance = 1e-2)

})


c(1.00000, 0.00000, 0.00000, 1.00000, 1.00000,
  0.00000, 0.57339, 0.00000, 0.00000, 0.07963)


