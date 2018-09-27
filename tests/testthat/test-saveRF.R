library(testthat)
test_that("Tests that saving RF and laoding it works", {
  context("Save and Load RF")

  set.seed(238943202)
  x <- iris[,-1]
  y <- iris[, 1]

  #-- Translating C++ to R ------------------------------------------------
  # Check that saving TRUE / FALSE saves and does not save the training data.
  forest <- forestry(x,
                     y,
                     ntree = 3,
                     saveable = FALSE)
  # testthat::expect_equal(forest@processed_dta, list())
  testthat::expect_equal(forest@R_forest, list())

  forest <- forestry(
    x,
    y,
    sample.fraction = 1,
    splitratio = .03,
    ntree = 3,
    saveable = TRUE
  )


  testthat::expect_equal(forest@processed_dta$y[2], 4.9)
  # Check that saving the forest works well.
  testthat::expect_length(CppToR_translator(forest@forest)[[3]]$var_id[1:5],
                          5)

  #-- Translating from R to C++ and back ---------------------------------------
  set.seed(238943202)
  x <- iris[,-1]
  y <- iris[, 1]

  forest <- forestry(
    x,
    y,
    sample.fraction = 1,
    splitratio = 1,
    ntree = 3,
    saveable = TRUE,
    replace = FALSE
  )

  forest <- make_savable(forest)
  before <- forest@R_forest[[1]]
  y_pred_before <- predict(forest, x)

  forest <- make_savable(forest)
  forest <- relinkCPP_prt(forest)

  after <- CppToR_translator(forest@forest)[[1]]
  # y_pred_after <- predict(forest, x)


  for (i in 1:6) {
    testthat::expect_equal(after[[i]], before[[i]])
  }


  # -- Actual saving and loading -----------------------------------------------
  y_pred_before <- predict(forest, x)
  forest <- make_savable(forest)
  save(forest, file = "forest.Rda")
  rm(forest)
  load("forest.Rda", verbose = FALSE)
  forest <- relinkCPP_prt(forest)
  y_pred_after <- predict(forest, x)
  testthat::expect_equal(y_pred_before, y_pred_after, tolerance = 1e-6)

  file.remove("forest.Rda")
})
