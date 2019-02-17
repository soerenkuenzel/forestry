setwd("~/Dropbox/forestry/")

devtools::load_all()
library(microbenchmark)
set.seed(292315)
library(forestry)
current_commit <- system("git rev-parse HEAD", intern = TRUE)
current_date <- Sys.Date()
current_time <- Sys.time()

# Setup iris test --------------------------------------------------------------

test_idx <- sample(nrow(iris), 3)
x_train <- iris[-test_idx, -1]
y_train <- iris[-test_idx, 1]
x_test <- iris[test_idx, -1]

rf <- forestry(x = x_train, y = y_train)

forestry_iris <- function() forestry(x = x_train, y = y_train)
weightmatrix_iris <- function() predict(rf, x_test,
                                        aggregation = "weightMatrix")
predict_iris <- function() predict(rf, x_test)

# Setup large n and d ----------------------------------------------------------
n <- 1000
dim <- 20

feat_large <- matrix(rnorm(n * dim), ncol = dim)
y_large <- rnorm(n)
rf_large <- forestry(x = feat_large, y = y_large)

forestry_large <- function() forestry(x = feat_large, y = y_large)
weightmatrix_large <- function() predict(rf_large, feat_large,
                                        aggregation = "weightMatrix")
predict_large <- function() predict(rf_large, feat_large)


# Setup ridge RF ---------------------------------------------------------------
n <- 10000
a <- rnorm(n)
b <- rnorm(n)
c <- rnorm(n)
y <- 4*a + 5.5*b - .78*c
x <- data.frame(a,b,c)

rf_ridge <- forestry(
  x,
  y,
  ntree = 10,
  replace = TRUE,
  nodesizeStrictSpl = 5,
  nodesizeStrictAvg = 5,
  ridgeRF = TRUE
)
train_ridge <- function() forestry(
  x,
  y,
  ntree = 10,
  replace = TRUE,
  nodesizeStrictSpl = 5,
  nodesizeStrictAvg = 5,
  ridgeRF = TRUE
)
predict_ridge <- function() predict(rf_ridge, x)



# XXX run everything -----------------------------------------------------------

mcb <- microbenchmark(forestry_iris(),
               weightmatrix_iris(),
               predict_iris(),
               train_ridge(),
               predict_ridge(),
               forestry_large(),
               weightmatrix_large(),
               predict_large(),
               times = 10,
               unit = "s")

mcb_s <- summary(mcb)

write.table(x = cbind(current_time, current_commit, mcb_s),
          file = "tests/Comparisons/speed_tests/speed_snapshots.csv",
          sep = ",",
          append = TRUE)





