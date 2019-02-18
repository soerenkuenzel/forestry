library(testthat)
test_that("Tests if variable importance works", {
  context('Tests Rf Variable Importance')

  set.seed(56)
  x <- iris[, -1]
  y <- iris[, 1]

  # Test forestry (mimic RF)
  forest <- forestry(x, y)

  vi <- getVI(forest)

  expect_equal(unlist(vi), c(0.1850195, 1.1332576, 0.5532491, 0.4080045),
               tolerance = 5e-2)
})
