library(forestry)
library(ggplot2)
library(reshape2)
library(microbenchmark)


set.seed(49)

#Construct Simulated Data
n <- 100
p <- 200

f <- rnorm(n)
x <- data.frame(f)
for (feat in 1:(p-1)) {
  f <- rnorm(n)
  x <- cbind(x, f)
}

y <- rnorm(n)

results <- data.frame(matrix(ncol = 3, nrow = 0))

testps <- c(5, 20, 30, 40, 50, 70, 80, 90, 120, 150)

for (num in testps) {

  s <- sample(1:p, num, replace = FALSE)
  xn <- x[,s]
  yn <- y

  m <- microbenchmark(list = alist(
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
      ridgeRF = FALSE,
      overfitPenalty = 3
    ),

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
      ridgeRF = TRUE,
      overfitPenalty = 3
    )
  ), times = 1
  )
  sm <- summary(m, unit = "s")
  results <- rbind(results, c(num, sm$mean[1], sm$mean[2]))
}
colnames(results) <- c("p", "RF", "Ridge")
#results

m <- lm(results$RF ~ results$p)
a <- signif(coef(m)[1], digits = 2)
b <- signif(coef(m)[2], digits = 2)
textlab <- paste("y = ",b,"x + ",a, sep="")


m <- lm(results$Ridge ~ results$p)
a <- signif(coef(m)[1], digits = 2)
b <- signif(coef(m)[2], digits = 2)
textlab2 <- paste("y = ",b,"x + ",a, sep="")

resultsm <- melt(results, id.var = "p")

ggplot(data=resultsm, aes(p, value ,colour=variable))+
  geom_point(alpha = 0.9)+
  #geom_smooth(method = "lm", se = FALSE)+
  scale_colour_manual("n = 100 Splitting on 10 random features", values = c("red","blue"))+
  labs(x="p", y="Time (s)")#+
#annotate("text", x = 150, y = .5, label = textlab, color="black", size = 3, parse=FALSE)+
#annotate("text", x = 150, y = 5, label = textlab2, color="black", size = 3, parse=FALSE)

results
