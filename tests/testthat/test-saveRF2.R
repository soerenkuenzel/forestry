library(forestry)
set.seed(238943202)
x <- iris[1:10, -1]
y <- iris[1:10, 1]

forest <- forestry(
  x,
  y,
  sample.fraction = 1,
  splitratio = 1,
  ntree = 1,
  nodesizeSpl = 1000,
  saveable = TRUE,
  replace = FALSE
)

forest@forest_R
forest@categoricalFeatureCols
(y_pred_before <- predict(forest, x))

testthat::expect_equal(relinkCPP_prt(forest), forest)
forest@forest
forest <- relinkCPP_prt(forest)
forest@forest
forest@categoricalFeatureCols
table(forest@forest_R[[1]][[5]])
testthat::expect_equal(relinkCPP_prt(forest), forest)

forest@forest_R

# (y_pred_after <- predict(forest, x))
