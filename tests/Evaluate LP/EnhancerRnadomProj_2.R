# 04-28-2019
# Using lp distances on the Enhancer dataset with random projections

# Dependencies
library(dplyr)
library(roxygen2)
library(forestry)
library(ggplot2)
library(caret)

# projection function

compute_proj <- function(x, num_proj = 100, var_imp, supervise = TRUE){
  # Create one hot encodings for factor variables
  dummy <- caret::dummyVars(" ~ .", data = x)
  x_new <- data.frame(predict(dummy, newdata = x))

  # Standardize the data
  x_mean <- matrix(apply(x_new, 2, mean),
                   nrow = dim(x_new)[1],
                   ncol = dim(x_new)[2],
                   byrow = TRUE)
  x_sd <- matrix(apply(x_new, 2, sd),
                 nrow = dim(x_new)[1],
                 ncol = dim(x_new)[2],
                 byrow = TRUE)
  x_z <- as.matrix((x_new - x_mean) / x_sd)
  x_z[is.na(x_z)] <- 0

  # Create random matrix
  A <- matrix(rnorm(num_proj * ncol(x_new), mean = 0, sd = 1),
              nrow = num_proj,
              ncol = ncol(x_new))

  if(supervise){
    var_imp[var_imp < 0] <- 0
    A_sup <- 1/length(var_imp) * matrix(var_imp,
                                        nrow = dim(x_new)[1],
                                        ncol = dim(x_new)[2],
                                        byrow = TRUE)
    A <- A * A_sup
  }

  # Generate Random Projections
  proj <- apply(as.matrix(x_z) %*% t(A), 1, mean)
  return(cbind(x, proj))
}


# Load the data
data_path <-"/Users/rowancassius/Dropbox/Enhancer"
setwd(data_path)
load("enhancer.Rdata")

n <- 500
m <- 100

mini.train.id <- sample(train.id, n, replace = FALSE)
mini.test.id <- sample(test.id, m, replace = FALSE)

# Initial Random Forest
rf <- forestry(x = X[mini.train.id, ],
               y = Y[mini.train.id],
               ntree = 500,
               verbose = TRUE)

var_imp <- getVI(rf)
var_imp <- unlist(var_imp)

# Evaluation procedure==========================================================

trust <- data.frame(1:100)
for(i in 1:2){
  # Projection Random Forest
  X.proj <- compute_proj(X,
                         num_proj = 100,
                         var_imp,
                         supervise = FALSE)

  train.proj <- X.proj[mini.train.id, ]
  test.proj <- X.proj[mini.test.id, ]

  # Random Forest trained with projections
  rf_proj <- forestry(x = train.proj,
                      y = Y[mini.train.id],
                      ntree = 500,
                      verbose = TRUE)
  paste("Forest ", i, " done")

  # Get trust probs
  trust_helper <- evaluate_lp(object = rf_proj,
                              feature.new = test.proj,
                              c("proj"))

  trust <- cbind(trust, trust_helper)[, -1]
  paste("Projection ", i, " done")
}

# Temporary
trust <- trust_helper

# average the quantiles
probs <- apply(trust, 1, mean)

tr <- trust
tr[tr < 0.95] <- 0
tr[tr >= 0.95] <- 1

num_flags <- apply(tr, 1, sum)

pred <- predict(rf_proj, test.proj)
# pred <- predict(rf, X[mini.test.id, ])

# Make prediction rule
pred[pred <= 0.5] <- 0
pred[pred  > 0.5] <- 1

eval <- cbind(trust, abs(pred - Y[mini.test.id]))
eval <- as.data.frame(eval)
colnames(eval) <- c("prob", "abs_error")

# Determine Trust Flags
trust_ids <- which(eval$prob <= 0.90)
eval$trust <- rep("distrust", 100)
eval$trust[trust_ids] <- "trust"

# Separate Summaries
summary(eval[trust_ids, ]$abs_error)
summary(eval[-trust_ids, ]$abs_error)

# Overall absolute error Summary
summary(eval$abs_error)

# Visualize the difference in error distributions
ggplot(data = eval, mapping = aes(x = abs_error, fill = trust, color = trust)) +
  geom_histogram(position="identity", alpha=0.3) +
  theme(legend.position="top")


# Compare to Original Prediction errors=========================================
y_hat <- predict(rf, X[mini.train.id, ])
y_hat[pred <= 0.5] <- 0
y_hat[pred > 0.5] <- 1
err <- as.data.frame(abs(y_hat - Y[mini.test.id]))
colnames(err) <- "abs_error"
summary(err$abs_error)
