library(forestry)
x <- iris[, -1]
y <- iris[, 1]

# Set seed for reproductivity
set.seed(24750371)
tuned_forest <- autoforestry(x = x,
                             y = y,
                             num_iter = 9,
                             eta = 3,
                             nthread = 2)
print("or this")
  y_pred <- predict(tuned_forest, x)
