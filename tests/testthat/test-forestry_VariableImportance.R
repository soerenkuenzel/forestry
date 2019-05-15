library(testthat)
test_that("Tests if variable importance works", {
  context('Tests Rf Variable Importance')

  set.seed(56)
  x <- iris[, -1]
  y <- iris[, 1]

  # Test forestry (mimic RF)
  forest <- forestry(x, y, ntree = 1000)

  vi <- getVI(forest)
  names(vi) <- NULL
  expect_equal(vi, c(0.1904870, 1.1977401, 0.5023229, 0.4000065),
               tolerance = 0.3)
})
