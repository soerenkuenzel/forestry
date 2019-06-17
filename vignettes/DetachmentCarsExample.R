# Using Detachment Indices on a subset of the Cars dataset

# Dependencies
library(dplyr)
library(roxygen2)
library(forestry)
library(ggplot2)

# Load Data
data_path = "/Users/rowancassius/Desktop/Forestry/vignettes"
setwd(data_path)
cars_train_sample <- readRDS(file = "cars_train_sample.rds")
cars_test_sample <- readRDS(file = "cars_test_sample.rds")

# train random forest
rf_forestry <- forestry(x = cars_train_sample %>% select(-y),
                        y = cars_train_sample$y,
                        ntree = 500,
                        verbose = TRUE)

# Get variable importances
var_imp <- getVI(rf_forestry)
var_imp <- unlist(var_imp)
#' By Observation it looks like registration date and powerPS are the most
#' important variables

# Evaluate:
trust <- evaluate_detachments(object = rf_forestry,
                              feature.new = cars_test_sample %>% select(-y),
                              feat.name = c("DateOfRegistration", "powerPS"))

pred <- predict(rf_forestry,
                feature.new = cars_test_sample %>% select(-y))

y_true <- cars_test_sample$y
eval <- cbind(trust,
              data.frame(cars_test_sample$DateOfRegistration),
              data.frame(cars_test_sample$powerPS),
              pred,
              y_true,
              abs(y_true-pred))

colnames(eval)[c(1,2,3,4,7)] <- c("DoR_prob","power_prob","DateOfRegistration",
                                  "powerPS", "abs_error")

# trust data under the condition that registration and power are < 0.95
trust_ids <- which(eval$DoR_prob < 0.95 & eval$power_prob < 0.95)
eval$trust <- rep("distrust", 100)
eval$trust[trust_ids] <- "trust"

# Compare absolute error ditributions
summary(eval[trust_ids, ]$abs_error)
summary(eval[-trust_ids, ]$abs_error)

# Visualize the difference in error distributions
ggplot(data = eval, mapping = aes(x = abs_error, fill = trust, color = trust)) +
  geom_histogram(position ="identity", alpha = 0.5) +
  theme(legend.position = "top")






