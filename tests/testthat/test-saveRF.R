library(testthat)
test_that("Tests that saving RF and laoding it works", {
  context("Save and Load RF")

  set.seed(238943202)
  # ----------------------------------------------------------------------------
  # Save RF
  x <- iris[,-1]
  y <- iris[, 1]
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

  forest

  # ----------------------------------------------------------------------------
  # load RF



  # ----------------------------------------------------------------------------
  # translate Ranger and randomForest
#
#   # Test predict
#   y_pred <- predict(forest, x)
#
#   cars
#   x <- as.data.frame(x)
#   x$Species <- as.character(x$Species)
#
#   unordered <- ifelse(rbinom(nrow(x), 1, .5) == 1, 'sa', 'go')
#   unordered[1:10] <- 1
#
#   unordered[40:50] <- 22
#   x$uu <- unordered
#   mode(x$uu)
#   rfr <-
#     ranger::ranger(
#       y ~ .,
#       data = x,
#       write.forest = TRUE,
#       respect.unordered.factors = 'partition'
#     )
#   rfr$forest$child.nodeIDs[[1]]
#   rfr$forest$split.varIDs[[1]]
#   rfr$forest$split.values[[1]]
#   rfr$forest$is.ordered
#
#   rfr$forest$independent.variable.names
})

