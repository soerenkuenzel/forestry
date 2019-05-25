# 05-22-2019
# Using lp distances on the Enhancer dataset with random projections

# Dependencies
library(dplyr)
library(roxygen2)
library(forestry)
library(ggplot2)
library(caret)

# projection function

compute_proj <- function(x, num_proj = 100, var_imp, supervise = FALSE){
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
  # Each column of proj is a vector of random projection data
  proj <- as.matrix(x_z) %*% t(A)
  return(proj)
}


# Load Enhancer data
data_path <-"/Users/rowancassius/Dropbox/Enhancer"
setwd(data_path)
load("enhancer.Rdata")

n <- 500
m <- 100

# Set seed for reproductivity
set.seed(24750371)

mini.train.id <- sample(train.id, n, replace = FALSE)
mini.test.id <- sample(test.id, m, replace = FALSE)

x_train <- X[mini.train.id, ]
x_test <- X[mini.test.id, ]
y_train <- Y[mini.train.id]
y_test <- Y[mini.test.id]

# Initial Random Forest
forest <- forestry(x = x_train,
                   y = y_train,
                   ntree = 500,
                   verbose = TRUE)

# var_imp <- getVI(rf)
# var_imp <- unlist(var_imp)
var_imp <- matrix(1:ncol(X))

n_proj <- 100
# compute projections
X.proj <- compute_proj(X,
                       num_proj = n_proj,
                       var_imp,
                       supervise = FALSE)

train.proj <- X.proj[mini.train.id, ]
test.proj <- X.proj[mini.test.id, ]

# ===================================

y_weights <- predict(object = forest,
                     feature.new = X[mini.test.id, ],
                     aggregation = "weightMatrix")$weightMatrix

# Compute lp distances for new data wrt to each projection
test_proj_lp <- data.frame(1:nrow(x_test))
for (i in 1:n_proj){
  dist_vec <- compute_lp_bnd(y_weights = y_weights,
                             train_vec = train.proj[ ,i],
                             test_vec = test.proj[ ,i],
                             p = 1)
  test_proj_lp <- cbind(test_proj_lp, dist_vec)
}
test_proj_lp <- test_proj_lp[ ,-1]


# Set seed for reproductivity
set.seed(24750371)

# Compute lp distances for the training data:
k_CV <- 10
folds <- caret::createFolds(y_train, k = k_CV, list = TRUE,
                            returnTrain = FALSE)

# Create an allocation matrix for training projection lp distances
train_proj_lp <- matrix(data = NA, nrow = nrow(x_train), ncol = n_proj)

for(k in 1:k_CV){
  #Select fold and train a random forest with OOB
  fold_ids <- folds[[k]]
  rf <- forestry(x = x_train[-fold_ids, ],
                 y = y_train[-fold_ids])
  fold_weights <- predict(object = rf,
                          feature.new = x_train[fold_ids, ],
                          aggregation = "weightMatrix")$weightMatrix

  # Compute lp distances with respect to each projection dimension
  for (i in 1:n_proj){
    train_proj_lp[fold_ids, i] <-
      compute_lp_bnd(y_weights = fold_weights,
                     train_vec = train.proj[-fold_ids, ][ ,i],
                     test_vec = train.proj[fold_ids, ][ ,i],
                     p = 1)
  }
}

# Get lp projection distances quantiles
trust <- matrix(data = NA, nrow = nrow(x_test), ncol = n_proj)
for (i in 1:n_proj){
  trust[ ,i] <- get_conditional_dist_bnd(y_weights = y_weights,
                                         train_y = train_proj_lp[ ,i],
                                         vals = test_proj_lp[ ,i])
}

# Make predictions using the original forest
pred <- predict(forest, x_test)
# Make prediction rule
pred[pred <= 0.5] <- 0
pred[pred  > 0.5] <- 1


# ========== Method 1, use median of the probs==================================
med_prob <- apply(trust, 1, median)
quant <- quantile(med_prob, probs = 0.90)

eval <- cbind(med_prob, abs_error = abs(pred - y_test))
eval <- as.data.frame(eval)

# Determine Trust Flags
trust_ids <- which(eval$med_prob <= quant)

# ========== Method 2, get median/location of top half of entries===============
sorted <- t(apply(trust, 1, sort))
top_med_prob <- apply(sorted[ ,51:100], 1, median)
quant <- quantile(top_med_prob, probs = 0.90)

eval <- cbind(top_med_prob, abs_error = abs(pred - y_test))
eval <- as.data.frame(eval)

# Determine Trust Flags
trust_ids <- which(eval$top_med_prob <= quant)


# ========== Method 3, count flags==============================================
tr <- trust
tr[tr < 0.95] <- 0
tr[tr >= 0.95] <- 1
num_flags <- apply(tr, 1, sum)

eval <- cbind(num_flags, abs_error = abs(pred - y_test))
eval <- as.data.frame(eval)

# Determine Trust Flags
trust_ids <- which(eval$flags < 10)

#============Test Results of the trusting procedure=============================
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






