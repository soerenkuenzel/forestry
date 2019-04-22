test_that("Tests that evaluate_lp is working correctly", {
  context('Test evaluate lp')

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

  trust <- evaluate_lp(object = rf,
                       feature.new = x_test,
                       feature = features,
                       p = 1)

  expect_equal(trust$Sepal.Width,
               c(0.3724928, 0.4318362, 0.2068011, 0.5407588, 0.2155091),
               tolerance = 5e-2)
  expect_equal(trust$Species,
               c(0.3935524, 0.5599691, 0.3910131, 0.6238988, 0.4652994),
               tolerance = 5e-2)

})
