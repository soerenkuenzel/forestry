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

  # Compute the l1 distances in the "Species" dimension
  distances_1 <- compute_lp(object = rf,
                            feature.new = x_test,
                            feature = "Species",
                            p = 1)

  # Compute the l2 distances in the "Petal.Length" dimension
  distances_2 <- compute_lp(object = rf,
                            feature.new = x_test,
                            feature = "Petal.Length",
                            p = 2)

  expect_identical(length(distances_1), nrow(x_test))
  expect_identical(length(distances_2), nrow(x_test))

  #set tolerance
  expect_equal(distances_1,
               c(0.6757558, 0.5375544, 0.6937144, 0.6265924, 0.5884993,
                 0.6233176, 0.5467013, 0.8047591, 0.7466187, 0.6254624,
                 0.8397300),
               tolerance = 1e-2)
  expect_equal(distances_2,
               c(2.628971, 2.360160, 2.177702, 2.574676, 2.404899,
                 2.212701, 2.091241, 2.622013, 2.276196, 2.465682, 2.801573),
               tolerance = 1e-2)
})

