test_that("Tests that backward compatibility", {
  x <- iris[, -1]
  y <- iris[, 1]

  # ----------------------------------------------------------------------------
  context("Backward compatibility: honest_RF")
  set.seed(24750371)
  forest_old <- honestRF(
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

  set.seed(24750371)
  forest_new <- forestry(
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

  y_pred_old <- predict(forest_old, x)
  y_pred_new <- predict(forest_new, x)

  expect_equal(y_pred_new, y_pred_old, tolerance = 1e-5)

  # ----------------------------------------------------------------------------
  context("Backward compatibility: autohonest_RF")
  set.seed(24750371)
  tuned_forest_old <- autohonestRF(
    x = x,
    y = y,
    num_iter = 9,
    eta = 3,
    nthread = 2
  )

  set.seed(24750371)
  tuned_forest_new <- autoforestry(
    x = x,
    y = y,
    num_iter = 9,
    eta = 3,
    nthread = 2
  )

  y_pred_old <- predict(tuned_forest_old, x)
  y_pred_new <- predict(tuned_forest_new, x)

  expect_equal(y_pred_new, y_pred_old, tolerance = 1e-5)
})
