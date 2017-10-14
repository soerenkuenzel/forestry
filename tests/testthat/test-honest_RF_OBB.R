test_that("Tests if OOB calculation is working correctly", {
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

  # Test OOB
  expect_equal(getOOB(forest), 15.79849, tolerance = 1e-2)

  # Test a very extreme setting
  forest <- honestRF(
    x,
    y,
    ntree = 500,
    replace = FALSE,
    sampsize = nrow(x),
    mtry = 3,
    nodesizeSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeAvg = 5
  )

  expect_warning(
    testOOB <- getOOB(forest),
    "Samples are drawn without replacement and sample size is too big!"
  )

  expect_equal(testOOB, NA, tolerance = 1e-4)
})
