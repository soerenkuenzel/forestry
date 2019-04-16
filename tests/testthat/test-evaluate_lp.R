test_that("Tests that evaluate_lp is working correctly", {
  context('Test evaluate lp')

  # Set seed for reproductivity
  set.seed(24750371)
  test_idx <- sample(nrow(iris), 10)

  index <- which(colnames(iris) == "Sepal.Length")
  x_train <- iris[-test_idx, -index]
  y_train <- iris[-test_idx, index]
  x_test <- iris[test_idx, -index]
  rf <- forestry(x = x_train, y = y_train)

  # Select features to compute lp distances with respect to.
  features <- c("Sepal.Width", "Petal.Length", "Petal.Width")

  trust <- evaluate_lp(object = rf,
                       feature.new = x_test,
                       feature = features,
                       p = 1)

  expect_equal(trust$Sepal.Width,
               c(0.1918844, 0.3973499, 0.4674050, 0.3807916, 0.1425799,
                 0.8756526, 0.3768030, 0.4693375, 0.6818932, 0.1696313),
               tolerance = 3e-2)
  expect_equal(trust$Petal.Length,
               c(0.08645036, 0.10509911, 0.02036391, 0.52422787, 0.08345800,
                 0.60929847, 0.43330681, 0.38149094, 0.50697562, 0.19027894),
               tolerance = 3e-2)
  expect_equal(trust$Petal.Width,
               c(0.05329916, 0.17491883, 0.69592820, 0.48759552, 0.04331170,
                 1.00000000, 0.02673648, 0.77352125, 0.54334786, 0.04020543),
               tolerance = 3e-2)
})
