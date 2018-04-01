library(testthat)
test_that("Tests that saving RF and laoding it works", {
  context("Save and Load RF")

  set.seed(238943202)
  x <- iris[, -1]
  y <- iris[, 1]

  # Check that saving TRUE / FALSE saves and does not save the training data.
  forest <- forestry(x,
                     y,
                     ntree = 3,
                     saveable = FALSE)
  testthat::expect_equal(forest@processed_dta, list())
  testthat::expect_equal(forest@forest_R, list())

  forest <- forestry(x,
                     y,
                     ntree = 3,
                     saveable = TRUE)
  testthat::expect_equal(forest@processed_dta$y[2], 4.9)
  expect_equal(forest@forest_R[[3]]$var_id[1:5],
               c(3, 4, 2, 3, 0))

  # Check that saving the forest works well.
  expect_equal(CppToR_translator(forest@forest)[[3]]$var_id[1:5],
               c(3, 4, 2, 3, 0))

  # ----------------------------------------------------------------------------
  # translate Ranger and randomForest
  # rfr <-
  #   ranger::ranger(
  #     y ~ .,
  #     data = x,
  #     write.forest = TRUE,
  #     respect.unordered.factors = 'partition'
  #   )
  # rfr$forest$child.nodeIDs[[1]]
  # rfr$forest$split.varIDs[[1]]
  # rfr$forest$split.values[[1]]
  #
  # rfr$forest$is.ordered
  # rfr$forest$independent.variable.names
})
