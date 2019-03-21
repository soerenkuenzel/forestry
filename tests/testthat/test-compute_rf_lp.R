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

  expect_identical(length(distances_1), nrow(x_test))
  expect_identical(length(distances_2), nrow(x_test))

  #set tolerance
  tol = 1e-2
  expect_gte(sum(distances_1), 0 - tol)
  expect_lte(sum(distances_1), length(test_idx) + tol)
  expect_equal(distances_2,
               c(2.628971, 2.360160, 2.177702, 2.574676, 2.404899,
                 2.212701, 2.091241, 2.622013, 2.276196, 2.465682, 2.801573),
               tolerance = tol)
})

