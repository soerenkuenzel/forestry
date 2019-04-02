test_that("Tests that quantile forest is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Quantile Forest')
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
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  quantiles <- get_quantiles(
    object = forest,
    feature.new = x[1:3, ],
    quantiles = c(0, .05, .6, .5, .8, .95, 1)
  )

  expect_equal(as.numeric(quantiles[1,5]), 5.2, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[1,2]), 4.6, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[3,3]), 4.6, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[2,6]), 5, tolerance = 1e-2)
  expect_equal(as.numeric(quantiles[1,2]), 4.6, tolerance = 1e-2)
})
