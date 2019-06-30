test_that("Tests that evaluate_detachments is working correctly", {
  context('Test evaluate detachments')

  # Set seed for reproductivity
  set.seed(24750371)
  test_idx <- sample(nrow(iris), 5)

  index <- which(colnames(iris) == "Sepal.Length")
  x_train <- iris[-test_idx, -index]
  y_train <- iris[-test_idx, index]
  x_test <- iris[test_idx, -index]
  rf <- forestry(x = x_train, y = y_train, ntree = 50, nthread = 1)

  # Select features to compute lp distances with respect to.
  features <- c("Sepal.Width", "Species")

  trust <- evaluate_detachments(object = rf,
                                feature.new = x_test,
                                feat.name = features,
                                p = 1,
                                verbose = FALSE,
                                num.CV = 2)


  # Assertions
  expect_equal(trust$Sepal.Width,
               c(0.2031064, 0.4957503, 0.4437268, 0.3337928, 0.1000588),
               tolerance = 1e-2)
  expect_equal(trust$Species,
               c(0.3453855, 0.4221628, 0.4566939, 0.4370547, 0.4604309),
               tolerance = 1e-2)

})



