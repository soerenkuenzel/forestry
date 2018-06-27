library(forestry)
library(Cubist)
library(ggplot2)
library(reshape2)

set.seed(45)
x <- EuStockMarkets[, c(1, 2, 3)]
y <- EuStockMarkets[, 4]

results <- data.frame(matrix(ncol = 3, nrow = 0))

for (l in c(.1, 3, 5, 7, 9, 10, 15)) {

  # Test ridge RF with lambda
  forest <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = 25,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = 25,
    ridgeRF = TRUE,
    overfitPenalty = l
  )

  cubistTree <- cubist(
    x = x,
    y = y
  )

  y_predCubist <- predict(cubistTree, x)
  y_predRidge <- predict(forest, x)

  results <- rbind(results, c(l, sum((y_predRidge - y) ^ 2), sum((y_predCubist - y) ^ 2)))
}
colnames(results) <- c("Lambda", "RidgeSplitPredict", "Cubist")
resultsm <- melt(results, id.var = "Lambda")

ggplot(data=resultsm, aes(Lambda, value ,colour=variable))+
  geom_point(alpha = 0.9)+
  scale_colour_manual("EU Stock Market Data (n = 1860)", values = c("red","blue"))+
  labs(x="Lambda", y="MSE")+
  ylim(0,20000000)


