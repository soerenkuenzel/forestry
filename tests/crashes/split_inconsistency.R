# devtools::load_all()
set.seed(43213)
n <- 100
x <- data.frame(x = runif(n = n, min = 0, max = 2))
y <- ifelse(x < 1, 0, 1)
plot(x[,1], y)

forest_lin <- forestry(
  x = x,
  y = y,
  ntree = 1,
  replace = FALSE,
  sampsize = n,
  nodesizeStrictSpl = 34,
  ridgeRF = TRUE,
  overfitPenalty = 10000
)

plot(x[,1], predict(forest_lin, feature.new = x))

# forest_const <- forestry(
#   x,
#   y,
#   ntree = 1,
#   replace = FALSE,
#   sampsize = 300,
#   nodesizeStrictSpl = 100,
#   ridgeRF = FALSE
# )
#
# plot(x, predict(forest_const, feature.new = x))



