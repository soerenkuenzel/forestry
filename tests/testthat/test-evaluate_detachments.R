test_that("Tests that evaluate_detachments is working correctly", {
  context('Test evaluate detachments')

  # Set seed for reproductivity
  set.seed(24750371)
  test_idx <- sample(nrow(iris), 5)

  index <- which(colnames(iris) == "Sepal.Length")
  x_train <- iris[-test_idx, -index]
  y_train <- iris[-test_idx, index]
  x_test <- iris[test_idx, -index]
  rf <- forestry(x = x_train, y = y_train, nthread = 1)

  # Select features to compute lp distances with respect to.
  features <- c("Sepal.Width", "Species")

  trust <- evaluate_detachments(object = rf,
                                feature.new = x_test,
                                feat.name = features,
                                p = 1,
                                verbose = FALSE)

  # Assertions
  expect_equal(trust$Sepal.Width,
               c(0.2304842, 0.4566183, 0.4179750, 0.4714608, 0.1065377),
               tolerance = 1e-2)
  expect_equal(trust$Species,
               c(0.3374607, 0.2840422, 0.4327882, 0.5381107, 0.3480015),
               tolerance = 1e-2)

})
