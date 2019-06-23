test_that("Tests that compute the detachment indices are working correctly", {

  context('Test detachment indices')

  # Set seed for reproductivity
  set.seed(292313)

  # Use Iris Data
  # Add another numeric column
  iris$num <- sample(c(1, 2, 3, 4),
                     nrow(iris), replace = TRUE)
  # Add another categorical column
  iris$location <- as.factor(sample(c("north", "south", "east", "west"),
                             nrow(iris), replace = TRUE))

  test_idx <- sample(nrow(iris), 10)
  x_train <- iris[-test_idx, -1]
  y_train <- iris[-test_idx, 1]
  x_test <- iris[test_idx, -1]

  # Create a random forest
  rf <- forestry(x = x_train, y = y_train)

  # Compute the 4.5th order detachments in the "Species" dimension
  detachments_1 <- compute_detachments(object = rf,
                            feature.new = x_test,
                            detachment.feat = "Species",
                            p = 4.5)

  # Compute the 2nd order detachments in the "Petal.Length" dimension
  detachments_2 <- compute_detachments(object = rf,
                            feature.new = x_test,
                            detachment.feat = "Sepal.Width",
                            p = 2)

  # Compute the 3rd order detachments in the "Petal.Length" dimension
  detachments_3 <- compute_detachments(object = rf,
                            feature.new = x_test,
                            detachment.feat = "Sepal.Width",
                            p = 0.7)

  # Assertions:
  expect_identical(length(detachments_1), nrow(x_test))
  expect_identical(length(detachments_2), nrow(x_test))
  expect_identical(length(detachments_3), nrow(x_test))

  expect_equal(detachments_1,
               c(0.5064062, 0.6135494, 0.6850906, 0.6110753, 0.4255796,
                 0.7071884, 0.3208159, 0.3319960, 0.4214267, 0.6171963),
               tolerance = 1e-5)
  expect_equal(detachments_2,
               c(0.2572747, 0.4026521, 0.2384931, 0.3999552, 0.2194508,
                 0.2179426, 0.3334502, 0.2518049, 0.2844727, 0.8506433),
               tolerance = 1e-5)
  expect_equal(detachments_3,
               c(0.1608303, 0.2221634, 0.1264818, 0.2757391, 0.1215103,
                 0.1370966, 0.2199645, 0.1301863, 0.1795759, 0.7679039),
               tolerance = 1e-5)
})

