library(testthat)
test_that("Tests that saving RF and laoding it works", {
  context("Save and Load RF")

  set.seed(238943202)
  x <- iris[, -1]
  y <- iris[, 1]

  #-- Translating C++ to R ------------------------------------------------
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
  expect_length(forest@forest_R[[3]]$var_id[1:5],
                5)

  # Check that saving the forest works well.
  expect_length(CppToR_translator(forest@forest)[[3]]$var_id[1:5],
                5)

  #-- Translating R to C++ ------------------------------------------------
  save(forest, file = "tests/testthat/forest.Rda")
  load("tests/testthat/forest.Rda", verbose = TRUE)
  str(forest)

  forest@dataframe
  forest@forest
  forest <- relinkCPP_prt(forest)
  forest@dataframe
  forest@forest

  #

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
