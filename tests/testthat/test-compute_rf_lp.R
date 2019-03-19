test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  test_idx <- sample(nrow(iris), 11)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train)
  predict(rf, x_test)

  # Compute the l1 distances in the "Species" dimension
  distances_1 <- compute_lp(object = rf,
                         test = x_test,
                         feature = "Species",
                         p = 1)

  # Compute the l2 distances in the "Petal.Length" dimension
  distances_2 <- compute_lp(object = rf,
                            test = x_test,
                            feature = "Petal.Length",
                            p = 2)

  # What else can we expect?
  expect_equal(length(distances_1), nrow(x_test), tolerance = 1e-2)
  expect_equal(length(distances_2), nrow(x_test), tolerance = 1e-2)
})

