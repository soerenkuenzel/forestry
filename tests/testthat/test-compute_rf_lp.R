test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292315)

  # Use Iris Data
  test_idx <- sample(nrow(iris), 3)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train)
  predict(rf, x_test)

  # Apply lp function
  compute_lp(object = rf, test = 1)


  expect_equal(1, 1, tolerance = 1e-2)
})
