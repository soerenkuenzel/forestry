test_that("Tests if OOB calculation is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]
  context('OOB calculation')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  # Test OOB
  expect_lt(mean((getOOB(forest) - 15.79849)^2), .1)

  # Test a very extreme setting
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = FALSE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  expect_warning(
    testOOB <- getOOB(forest, FALSE),
    "Samples are drawn without replacement and sample size is too big!"
  )

  expect_equal(testOOB, NA, tolerance = 1e-4)
})
