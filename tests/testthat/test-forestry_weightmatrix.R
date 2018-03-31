library(testthat)
test_that("Tests that random forest is working correctly", {
  x <- iris[, -1]
  y <- iris[, 1]

  context('Weight matrix')
  # Set seed for reproductivity
  set.seed(24750371)

  # Test forestry (mimic RF)
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 5,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 5
  )

  # Test predict
  full_predicitons <- predict(forest, x[1:10,], aggregation = 'weightmatrix')
  y_pred <- predict(forest, x[1:10,])
  expect_equal(full_predicitons$predictions, y_pred)

  expect_equal(full_predicitons$weightMatrix %*% as.matrix(y),
               as.matrix(y_pred), tolerance = 1e-5)

  full_predicitons <- predict(forest, x, aggregation = 'weightmatrix')
  y_pred <- predict(forest, x)
  expect_equal(full_predicitons$predictions, y_pred)

  expect_equal(full_predicitons$weightMatrix %*% as.matrix(y),
               as.matrix(y_pred), tolerance = 1e-5)



})
