library(forestry)
library(ggplot2)
library(reshape2)
library(microbenchmark)


set.seed(45)

#Construct Simulated Data
n <- 52000
p <- 5
trees <- 100

f <- rnorm(n)
x <- data.frame(f)
for (feat in 1:(p-1)) {
  f <- rnorm(n)
  x <- cbind(x, f)
}

y <- rnorm(n)

results <- data.frame(matrix(ncol = 3, nrow = 0))

testns <- c(500, 600, 700, 800)#, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 10000, 20000, 30000, 40000, 50000)

for (num in testns) {

  s <- sample(1:n, num, replace = FALSE)
  xn <- x[s,]
  yn <- y[s]

  m <- microbenchmark(list = alist(
      # Test ridge RF with lambda
      Rforest <- forestry(
        xn,
        yn,
        ntree = trees,
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
        ntree = trees,
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
    ), times = 8
  )
  sm <- summary(m, unit = "s")
  results <- rbind(results, c(num, sm$mean[1], sm$mean[2]))
}
  colnames(results) <- c("n", "RF", "Ridge")
  results

  m <- lm(results$RF ~ results$n)
  a <- signif(coef(m)[1], digits = 2)
  b <- signif(coef(m)[2], digits = 2)
  textlab <- paste("y = ",b,"x + ",a, sep="")


  m <- lm(results$Ridge ~ results$n)
  a <- signif(coef(m)[1], digits = 2)
  b <- signif(coef(m)[2], digits = 2)
  textlab2 <- paste("y = ",b,"x + ",a, sep="")

resultsm <- melt(results, id.var = "n")

ggplot(data=resultsm, aes(n, value ,colour=variable))+
  geom_point(alpha = 0.9)+
  theme(legend.position = "bottom")+
  #geom_smooth(method = "lm", se = FALSE)+
  scale_colour_manual("Fast Armadillo Performance on p = 5", values = c("red","blue"))+
  labs(x="n", y="Time (s)")#+
  #annotate("text", x = 150, y = .5, label = textlab, color="black", size = 3, parse=FALSE)+
  #annotate("text", x = 150, y = 5, label = textlab2, color="black", size = 3, parse=FALSE)

results
