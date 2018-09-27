test_that("Tests if variable importance works", {
  context('Tests Rf Variable Importance')

  set.seed(56)
  x <- iris[, -1]
  y <- iris[, 1]

  # Test forestry (mimic RF)
  forest <- forestry(x, y)

  vi <- getVI(forest)

  expect_equal(unlist(vi), c(.199, 1.138, .561, .371), tolerance = 5e-2)
})
