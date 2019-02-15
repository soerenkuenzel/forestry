library(forestry)
library(ggplot2)
library(tictoc)

set.seed(45)
x <- EuStockMarkets[, c(1, 2, 3)]
y <- EuStockMarkets[, 4]


r <- microbenchmark(
  # Test ridge RF with lambda
  Rforest <- forestry(
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
    nodesizeStrictAvg = 5,
    ridgeRF = TRUE,
    overfitPenalty = 1000
  ),
  #Test normal lambda
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
    nodesizeStrictAvg = 5,
    ridgeRF = FALSE,
    overfitPenalty = 1000
  ),
  times = 4
)

boxplot(r, t)

