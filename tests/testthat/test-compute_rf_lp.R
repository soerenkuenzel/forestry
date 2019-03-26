test_that("Tests that compute the lp distances works correctly", {

  context('Test lp distances')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  # Add a location column to iris dataset
  iris$location <- as.factor(sample(c("north", "south", "east", "west"),
                                    nrow(iris),
                                    replace = TRUE))
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

  expect_equal(distances_1,
               c(0.5713034, 0.3983406, 0.8628993, 0.5934071, 0.6393168,
                 0.6590699, 0.8023141, 0.7427544, 0.7753897, 0.6775414,
                 0.8607893),
               tolerance = 1e-2)
  expect_equal(distances_2,
               c(1.951685, 1.921449, 2.345230, 2.279486, 2.396665, 1.959820,
                 2.199482, 2.391815, 2.266701, 2.422162, 2.553033),
               tolerance = 1e-2)
})

