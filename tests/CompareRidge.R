library(forestry)
library(ggplot2)
library(reshape2)

set.seed(45)

n <- 200

a <- rnorm(n)
b <- rnorm(n)
c <- rnorm(n)
d <- rnorm(n)

x <- data.frame(a,b,c,d)

y <- 5*a + 6*b - .5*c -7.8*d + rnorm(n, sd = 5)

sm <- 10
lg <- 35


results <- data.frame(matrix(ncol = 5, nrow = 0))

for (l in c(.3,1,3,5,10,15)) {

  # Test ridge RF with lambda
  ridgeLN <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = lg,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = lg,
    ridgeRF = TRUE,
    overfitPenalty = l
  )

  ridgeSN <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = sm,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = sm,
    ridgeRF = TRUE,
    overfitPenalty = l
  )

  rfLN <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = lg,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = lg,
    ridgeRF = FALSE,
    overfitPenalty = l
  )

  rfSN <- forestry(
    x,
    y,
    ntree = 500,
    replace = TRUE,
    sample.fraction = .8,
    mtry = 3,
    nodesizeStrictSpl = sm,
    nthread = 2,
    splitrule = "variance",
    splitratio = 1,
    nodesizeStrictAvg = sm,
    ridgeRF = FALSE,
    overfitPenalty = l
  )

  y_predRidgeLN <- predict(ridgeLN, x)
  y_predRidgeSN <- predict(ridgeSN, x)
  y_predRfLN <- predict(rfLN, x)
  y_predRfSN <- predict(rfSN, x)

  results <- rbind(results, c(l,
                              sum((y_predRidgeLN - y) ^ 2),
                              sum((y_predRidgeSN - y) ^ 2),
                              sum((y_predRfLN - y) ^ 2),
                              sum((y_predRfSN - y) ^ 2)))
}

colnames(results) <- c("Lambda",
                       "RidgeLN",
                       "RidgeSN",
                       "RF LN",
                       "RF SN")

resultsm <- melt(results, id.var = "Lambda")

ggplot(data=resultsm, aes(Lambda, value ,colour=variable))+
  geom_point(alpha = 0.9)+
  geom_line()+
  ggtitle("f(x) = 5 x_1 + 6 x_2 - .5 x_3 - 7.8 x_4")+
  theme_minimal()+
  theme(legend.position = "bottom")+
  scale_colour_manual("",values = c("red",
                                    "red3",
                                    "dodgerblue",
                                    "dodgerblue4"))+
  labs(x="Lambda", y="MSE")+
  ylim(2000,17000)


