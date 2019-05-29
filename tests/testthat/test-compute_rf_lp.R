test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  # Add a location column to iris dataset
  iris$location <- as.factor(sample(c("north", "south", "east", "west"),
                                    nrow(iris),
                                    replace = TRUE))
  test_idx <- sample(nrow(iris), 10)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train, nthread = 1)

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

  expect_equal(distances_1,
               c(0.02586165, 0.08362668, 0.04007290, 0.04278755, 0.16482126,
                 0.04179358, 0.04066656, 0.05106356, 0.06894505, 0.14341744),
               tolerance = 1e-2)
  expect_equal(distances_2,
               c(0.5735928, 0.5648549, 0.6683935, 0.6109477, 0.5571071,
                 0.6334010, 0.4082906, 0.5991195, 0.6761388, 0.5996498),
               tolerance = 1e-2)

})

