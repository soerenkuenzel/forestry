library(forestry)
library(ggplot2)
library(reshape2)

set.seed(45)
x <- EuStockMarkets[, c(1, 2, 3)]
y <- EuStockMarkets[, 4]


results <- data.frame()

results <- data.frame(matrix(ncol = 3, nrow = 0))

for (n in c(50, 100, 250, 500, 1000, 1200, 1500)) {

  s <- sample(1:1860, n, replace = FALSE)

  xn <- x[s,]
  yn <- y[s]

  start <- Sys.time()
  # Test ridge RF with lambda
  Rforest <- forestry(
    xn,
    yn,
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
  )
  end1 <- Sys.time()

  #Test normal lambda
  forest <- forestry(
    xn,
    yn,
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
  )

  end2 <- Sys.time()

  ridgetime <- end1 - start
  rftime <- end2 - end1

  results <- rbind(results, c(n, rftime, ridgetime))
}
colnames(results) <- c("n", "RFMSE", "RidgeMSE")
resultsm <- melt(results, id.var = "n")

ggplot(data=resultsm, aes(n, value ,colour=variable))+
  geom_point(alpha = 0.9)+
  scale_colour_manual("RF Variant", values = c("red","blue"))+
  labs(x="n", y="Training time")
