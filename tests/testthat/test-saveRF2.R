# set.seed(238943202)
# x <- iris[, -1]
# y <- iris[, 1]
#
# forest <- forestry(
#   x,
#   y,
#   sample.fraction = 1,
#   splitratio = 1,
#   ntree = 3,
#   saveable = TRUE,
#   replace = FALSE
# )
# y_pred_before <- predict(forest, x)
#
# forest <- relinkCPP_prt(forest)
# y_pred_after <- predict(forest, x)

